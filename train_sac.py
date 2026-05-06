"""
SAC: No-VMM (raw sensors) vs VMM-augmented observations
---------------------------------------------------------
SAC-NoVMM : obs = [left, right, front] plus movement/safety reward only
SAC-VMM   : obs = ultrasonic + IMU + simulated egocentric visual RND features
            reward += rnd_novelty * VMM_NOVELTY_SCALE each step

Action    : [theta_norm, distance_norm]
            theta_norm maps to a rover-local heading in [-75deg, +75deg]
            distance_norm maps to a turn-then-drive local target distance.

VMM novelty is RND (Random Network Distillation) on a 2D egocentric "camera"
embedding: a forward fan of ray depths plus deterministic endpoint texture.
This is a simulator stand-in for the real VMM path, where MobileNetV3 embeds
camera frames and RND runs on those visual embeddings.

Multi-seed: each method runs on N_SEEDS independent obstacle layouts;
coverage curves are averaged and plotted with mean +/- std shaded bands.

Safety layer: SAC outputs a 2D rover-local target intent, then the simulator
executes it through the same turn-then-drive deterministic safety semantics as
the hardware stack.

Training is continuous — single map, coverage is logged as hidden evaluation,
not used as a reward in the SAC training factories.

Run:
    python train_sac.py            # train + plot
    python train_sac.py --preview  # visualise both envs first
"""

import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from apartment_env import ApartmentContinuousEnv, generate_apartment
# -- Config ---------------------------------------------------------------------
SEEDS         = [42]             # one furniture layout per seed; results averaged
TRAIN_STEPS   = 100_000
EVAL_EVERY    = 5_000
BUFFER_SIZE   = 100_000
BATCH_SIZE    = 1024
LR            = 3e-4
GAMMA         = 0.99
HIDDEN        = [256, 256]
PREVIEW_STEPS = 300

# Small per-trigger penalty: enough to stop the policy using the bumper as a
# free turning mechanism, small enough not to block near-obstacle coverage.
R_SAFETY = -0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _benchmark_metrics(inner):
    coverage = inner._coverage() * 100
    collisions = int(inner._collisions)
    bumper_total = int(inner._bumper_triggers)
    total_reward = float(inner._total_reward)
    steps = int(inner.step_count)
    elapsed_s = steps * 0.05
    return {
        "coverage": coverage,
        "collisions": collisions,
        "bumper_total": bumper_total,
        "total_reward": total_reward,
        "coverage_per_collision": coverage / max(collisions, 1),
        "collisions_per_coverage": collisions / max(coverage, 1e-6),
        "coverage_per_1k_steps": coverage / max(steps / 1000.0, 1e-6),
        "reward_per_1k_steps": total_reward / max(steps / 1000.0, 1e-6),
        "collisions_per_1k_steps": collisions / max(steps / 1000.0, 1e-6),
        "bumper_per_1k_steps": bumper_total / max(steps / 1000.0, 1e-6),
        "elapsed_s": elapsed_s,
    }

# -- Safety penalty wrapper -----------------------------------------------------

class SafetyPenaltyWrapper(gym.Wrapper):
    """Keep the simulated deterministic safety layer active during training."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.use_bumper = True
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, term, trunc, info


# -- VMM observation wrapper ----------------------------------------------------

_VMM_NOVELTY_SCALE = 0.60  # stronger pull toward unexplored rooms
_SIM_CAMERA_RAYS   = 17
_SIM_CAMERA_FOV    = np.radians(70.0)
_RND_INPUT_DIM     = _SIM_CAMERA_RAYS * 3 + 2
_RND_HIDDEN        = 64
_RND_OUTPUT_DIM    = 64
_RND_LR            = 5e-6  # very slow fit — prevents predictor overfitting, keeps novelty alive
_RND_WARMUP        = 100  # kick in sooner so bonus drives early exploration
_MEMORY_SIZE       = 1000
_MEMORY_KNOWN_DIST = 0.08
_MEMORY_UPDATE_RATE = 0.05
_MEMORY_NORM_DIST  = 0.18
_MEMORY_SMOOTH_WINDOW = 5
_MEMORY_SMOOTH_RESET_DIST = 0.30
_RND_WEIGHT        = 0.65
_MEMORY_WEIGHT     = 0.35
_NOVELTY_PERSIST_WINDOW = 3
_SPIN_GATE_DISPLACEMENT = 0.05
_SPIN_GATE_YAW_DELTA    = np.radians(20.0)
_SPIN_GATE_SCALE        = 0.3
_VMM_MODE               = "rnd_memory"  # "rnd", "memory", "rnd_memory"
_VMM_USE_SMOOTHING      = True


class _RNDTarget(torch.nn.Module):
    """Fixed random network — never trained."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(_RND_INPUT_DIM, _RND_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(_RND_HIDDEN,    _RND_OUTPUT_DIM),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x): return self.net(x)


class _RNDPredictor(torch.nn.Module):
    """Trained to predict target output; high error = novel state."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(_RND_INPUT_DIM, _RND_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(_RND_HIDDEN,    _RND_HIDDEN),  torch.nn.ReLU(),
            torch.nn.Linear(_RND_HIDDEN,    _RND_OUTPUT_DIM),
        )

    def forward(self, x): return self.net(x)


class _ClusterMemory:
    """Bounded visual-place clusters for explicit seen-before novelty."""
    def __init__(self, maxlen=_MEMORY_SIZE, known_dist=_MEMORY_KNOWN_DIST,
                 update_rate=_MEMORY_UPDATE_RATE):
        self.maxlen = maxlen
        self.known_dist = known_dist
        self.update_rate = update_rate
        self._centroids = []
        self._counts = []
        self._last_seen = []

    @property
    def size(self):
        return len(self._centroids)

    def query(self, x):
        with torch.no_grad():
            z = torch.nn.functional.normalize(x.detach().squeeze(0), dim=0)
            if not self._centroids:
                return 1.0, z, None
            centroids = torch.stack(self._centroids).to(z.device)
            sims = centroids @ z
            best_idx = int(sims.argmax().item())
            return float(1.0 - sims[best_idx].item()), z, best_idx

    def update(self, z, dist, cluster_idx, step):
        z_cpu = z.detach().cpu()
        if cluster_idx is not None and dist < self.known_dist:
            old = self._centroids[cluster_idx]
            eta = min(self.update_rate, 1.0 / (self._counts[cluster_idx] + 1))
            centroid = torch.nn.functional.normalize((1.0 - eta) * old + eta * z_cpu, dim=0)
            self._centroids[cluster_idx] = centroid
            self._counts[cluster_idx] += 1
            self._last_seen[cluster_idx] = step
            return cluster_idx, False

        if len(self._centroids) >= self.maxlen:
            evict = int(np.argmin(self._last_seen))
            self._centroids.pop(evict)
            self._counts.pop(evict)
            self._last_seen.pop(evict)

        self._centroids.append(z_cpu)
        self._counts.append(1)
        self._last_seen.append(step)
        return len(self._centroids) - 1, True


class _TemporalEmbeddingSmoother:
    """Mean-pool recent embeddings before visual-place clustering."""
    def __init__(self, window=_MEMORY_SMOOTH_WINDOW, reset_dist=_MEMORY_SMOOTH_RESET_DIST):
        self._buf = deque(maxlen=window)
        self._reset_dist = reset_dist
        self.last_reset = False

    def reset(self):
        self._buf.clear()
        self.last_reset = False

    def push(self, x):
        z = torch.nn.functional.normalize(x.detach().squeeze(0), dim=0)
        self.last_reset = False
        if self._buf:
            prev = torch.nn.functional.normalize(torch.stack(list(self._buf)).mean(dim=0), dim=0)
            dist = float(1.0 - torch.dot(prev, z).item())
            if dist > self._reset_dist:
                self._buf.clear()
                self.last_reset = True
        self._buf.append(z.cpu())
        smooth = torch.stack(list(self._buf)).mean(dim=0)
        return torch.nn.functional.normalize(smooth, dim=0)


def _combine_novelty(rnd_norm, mem_norm, mode=_VMM_MODE):
    if mode == "rnd":
        return float(rnd_norm)
    if mode == "memory":
        return float(mem_norm)
    if mode == "rnd_memory":
        return float(np.clip(_RND_WEIGHT * rnd_norm + _MEMORY_WEIGHT * mem_norm, 0.0, 1.0))
    raise ValueError("VMM mode must be 'rnd', 'memory', or 'rnd_memory'")


class VMMObsWrapper(gym.Wrapper):
    """
    RND-based novelty wrapper — no coverage-grid oracle.

    Policy obs (12-dim):
        [left, right, front]          — HC-SR04 ultrasonic sensors
        [sin(θ), cos(θ)]              — IMU heading
        [novelty]                     — combined RND + memory novelty
        [fov_L, fov_C, fov_R]         — combined novelty in visual sectors
        [yaw_rate]                    — normalised gyro-z (IMU), execution feedback
        [vl, vr]                      — last executed wheel speeds

    RND input   : simulated egocentric camera embedding + IMU heading.
    Memory bank : nearest visual-place cluster distance over the same embeddings.
    Reward bonus: novelty * _VMM_NOVELTY_SCALE every step (after warmup).
    """

    def __init__(self, env, mode=None, use_smoothing=None):
        super().__init__(env)
        mode = _VMM_MODE if mode is None else mode
        use_smoothing = _VMM_USE_SMOOTHING if use_smoothing is None else use_smoothing
        if mode not in ("rnd", "memory", "rnd_memory"):
            raise ValueError("mode must be 'rnd', 'memory', or 'rnd_memory'")
        self._mode = mode
        self._use_smoothing = bool(use_smoothing)
        s = env.observation_space
        # obs = [left, right, front, sin(θ), cos(θ), novelty,
        #        fov_L, fov_C, fov_R, yaw_rate, vl, vr] — 12-dim
        self.observation_space = spaces.Box(
            low  = np.concatenate([s.low,  [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]]).astype(np.float32),
            high = np.concatenate([s.high, [ 1.0,  1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0]]).astype(np.float32),
        )
        self._target    = _RNDTarget().to(DEVICE)
        self._predictor = _RNDPredictor().to(DEVICE)
        self._opt       = torch.optim.Adam(self._predictor.parameters(), lr=_RND_LR)
        self._memory    = _ClusterMemory()
        self._smoother  = _TemporalEmbeddingSmoother()
        self._rnd_mean     = 0.0
        self._rnd_m2       = 0.0
        self._rnd_n        = 0
        self._novelty      = 0.0
        self._rnd_novelty  = 0.0
        self._mem_novelty  = 0.0
        self._cluster_id   = None
        self._new_cluster  = False
        self._reward_novelty = 0.0
        self._novelty_window = deque(maxlen=_NOVELTY_PERSIST_WINDOW)
        self._global_steps = 0  # never resets — warmup is global, not per-episode
        self._step_log = []

        # RND diagnostics — accumulated between eval checkpoints
        self._diag_raw_losses  = []   # raw RND loss every step
        self._diag_novelties   = []   # normalised novelty every step
        self._diag_rnd         = []   # RND component every step
        self._diag_mem         = []   # memory component every step
        self._diag_fov         = []   # [fov_L, fov_C, fov_R] every step
        self._diag_memory_size = []   # bank size every step
        self._diag_new_clusters = []  # cluster creations every step
        self._diag_smooth_window = []  # active smoothing window length
        self._diag_smooth_resets = []  # abrupt visual transition resets

        # Fixed visual probes: synthetic egocentric views with distinct texture
        # phases. They are diagnostics only; policy/RND training never sees room ids.
        probes = []
        for room_i in range(7):
            phase = 0.7 * room_i
            ray_features = []
            for ray_i in range(_SIM_CAMERA_RAYS):
                d = 0.75 + 0.2 * np.sin(phase + ray_i * 0.31)
                ray_features.extend([
                    np.clip(d, 0.0, 1.0),
                    0.5 + 0.5 * np.sin(phase + ray_i * 0.47),
                    0.5 + 0.5 * np.cos(phase + ray_i * 0.47),
                ])
            heading = 0.5 * room_i
            probes.append(ray_features + [np.sin(heading), np.cos(heading)])
        self._probe_states = torch.tensor(probes, dtype=torch.float32).to(DEVICE)

    def _unwrap_inner(self):
        inner = self.env
        while hasattr(inner, "env"):
            inner = inner.env
        return inner

    def _visual_texture(self, x, y, distance):
        """Deterministic 2D texture seen at a ray hit point.

        This stands in for camera appearance: rooms/furniture can look different
        even when range readings are similar, without exposing coordinates to SAC.
        """
        if distance >= 0.995:
            return 0.0, 0.0
        phase = 2.7 * x + 3.9 * y
        return 0.5 + 0.5 * np.sin(phase), 0.5 + 0.5 * np.cos(phase)

    def _sim_camera_embedding(self, center_offset=0.0):
        """Egocentric forward visual fan: depth plus endpoint texture per ray."""
        from apartment_env import _apt_ray_cast, APT_W, APT_H
        from rover_coverage_env import SENSOR_MAX
        inner = self._unwrap_inner()
        features = []
        offsets = np.linspace(-_SIM_CAMERA_FOV / 2, _SIM_CAMERA_FOV / 2, _SIM_CAMERA_RAYS)
        for offset in offsets:
            angle = inner.theta + center_offset + float(offset)
            d = _apt_ray_cast(inner.x, inner.y, angle, inner.obstacles, max_dist=SENSOR_MAX)
            norm_d = float(np.clip(d / SENSOR_MAX, 0.0, 1.0))
            hx = float(np.clip(inner.x + d * np.cos(angle), 0.0, APT_W))
            hy = float(np.clip(inner.y + d * np.sin(angle), 0.0, APT_H))
            tex_s, tex_c = self._visual_texture(hx, hy, norm_d)
            features.extend([norm_d, tex_s, tex_c])
        features.extend([np.sin(inner.theta + center_offset), np.cos(inner.theta + center_offset)])
        return np.array(features, dtype=np.float32)

    def _rnd_input(self, obs_3):
        _ = obs_3
        vec = self._sim_camera_embedding()
        return torch.tensor(vec).unsqueeze(0).to(DEVICE)

    def _rnd_error(self, x, update):
        with torch.no_grad():
            t_out = self._target(x)
        p_out = self._predictor(x)
        loss = torch.nn.functional.mse_loss(p_out, t_out)
        if update:
            self._opt.zero_grad(); loss.backward(); self._opt.step()
        return loss.item()

    def _normalise_rnd(self, raw, update_stats):
        if update_stats:
            # Welford running mean — familiar states trend toward 0, novel states spike to 1
            self._rnd_n += 1
            d = raw - self._rnd_mean
            self._rnd_mean += d / self._rnd_n
            self._rnd_m2   += d * (raw - self._rnd_mean)
        return float(np.clip(raw / (self._rnd_mean + 1e-8), 0.0, 1.0))

    def _compute_novelty(self, x):
        """Compute RND + memory novelty, update online models, return combined ∈ [0,1]."""
        raw = self._rnd_error(x, update=True)
        rnd_norm = self._normalise_rnd(raw, update_stats=True)

        z_smooth = (
            self._smoother.push(x)
            if self._use_smoothing
            else torch.nn.functional.normalize(x.detach().squeeze(0), dim=0)
        )
        mem_dist, _, cluster_idx = self._memory.query(z_smooth.unsqueeze(0).to(x.device))
        mem_norm = float(np.clip(mem_dist / _MEMORY_NORM_DIST, 0.0, 1.0))
        new_cluster = False
        if self._global_steps >= _RND_WARMUP:
            cluster_idx, new_cluster = self._memory.update(
                z_smooth, mem_dist, cluster_idx, self._global_steps)

        combined = _combine_novelty(rnd_norm, mem_norm, self._mode)
        if new_cluster:
            combined = max(combined, mem_norm)

        self._diag_raw_losses.append(raw)
        self._diag_novelties.append(combined)
        self._diag_rnd.append(rnd_norm)
        self._diag_mem.append(mem_norm)
        self._diag_memory_size.append(self._memory.size)
        self._diag_new_clusters.append(1.0 if new_cluster else 0.0)
        self._diag_smooth_window.append(len(self._smoother._buf))
        self._diag_smooth_resets.append(1.0 if self._smoother.last_reset else 0.0)
        self._rnd_novelty = rnd_norm
        self._mem_novelty = mem_norm
        self._cluster_id = cluster_idx
        self._new_cluster = new_cluster
        return combined

    def _fov_novelty(self):
        """Combined novelty evaluated on left/center/right crops of the current view."""
        sector_centers = [np.radians(35), 0.0, -np.radians(35)]
        result = []
        for rel_angle in sector_centers:
            vec = self._sim_camera_embedding(center_offset=rel_angle)
            x_t = torch.tensor(vec).unsqueeze(0).to(DEVICE)
            rnd_norm = self._normalise_rnd(self._rnd_error(x_t, update=False), update_stats=False)
            mem_dist, _, _ = self._memory.query(x_t)
            mem_norm = float(np.clip(mem_dist / _MEMORY_NORM_DIST, 0.0, 1.0))
            combined = _combine_novelty(rnd_norm, mem_norm, self._mode)
            result.append(combined)
        return np.array(result, dtype=np.float32)

    def _reward_novelty_from_motion(self, novelty, info):
        self._novelty_window.append(float(novelty))
        persistent = float(np.median(self._novelty_window))
        displacement = float(info.get("displacement", 0.0))
        yaw_delta_abs = float(info.get("yaw_delta_abs", 0.0))
        gate = _SPIN_GATE_SCALE if (
            displacement < _SPIN_GATE_DISPLACEMENT and yaw_delta_abs > _SPIN_GATE_YAW_DELTA
        ) else 1.0
        return persistent * gate, persistent, gate

    def _augment(self, obs, novelty):
        from rover_coverage_env import MAX_WHEEL_SPEED
        inner = self._unwrap_inner()
        fov   = self._fov_novelty()
        self._diag_fov.append(fov.copy())
        return np.array([
            obs[0], obs[1], obs[2],          # HC-SR04 sensors
            np.sin(inner.theta),             # IMU yaw heading
            np.cos(inner.theta),
            novelty,                         # RND novelty at current position
            fov[0], fov[1], fov[2],          # RND novelty in visual sectors
            inner.yaw_rate,                  # gyro-z: execution feedback from IMU
            inner._vl / MAX_WHEEL_SPEED,
            inner._vr / MAX_WHEEL_SPEED,
        ], dtype=np.float32)

    def rnd_checkpoint_stats(self):
        """Drain accumulated step-level diagnostics and return summary stats.
        Called by TrackCallback at each eval interval."""
        losses    = np.array(self._diag_raw_losses) if self._diag_raw_losses else np.array([0.0])
        novelties = np.array(self._diag_novelties)  if self._diag_novelties  else np.array([0.0])
        rnd_vals   = np.array(self._diag_rnd)        if self._diag_rnd        else np.array([0.0])
        mem_vals   = np.array(self._diag_mem)        if self._diag_mem        else np.array([0.0])
        fovs      = np.array(self._diag_fov)         if self._diag_fov        else np.zeros((1, 3))
        mem_sizes = np.array(self._diag_memory_size) if self._diag_memory_size else np.array([self._memory.size])
        new_clusters = np.array(self._diag_new_clusters) if self._diag_new_clusters else np.array([0.0])
        smooth_windows = np.array(self._diag_smooth_window) if self._diag_smooth_window else np.array([0.0])
        smooth_resets = np.array(self._diag_smooth_resets) if self._diag_smooth_resets else np.array([0.0])

        # Probe: evaluate RND on each fixed state — no gradient, no predictor update
        with torch.no_grad():
            t_out = self._target(self._probe_states)
            p_out = self._predictor(self._probe_states)
            probe_errors = torch.nn.functional.mse_loss(
                p_out, t_out, reduction="none"
            ).mean(dim=1).cpu().numpy()   # shape (7,)

        stats = {
            "rnd_loss_mean":    float(losses.mean()),
            "rnd_loss_std":     float(losses.std()),
            "novelty_mean":     float(novelties.mean()),
            "novelty_std":      float(novelties.std()),
            "rnd_novelty_mean":  float(rnd_vals.mean()),
            "rnd_novelty_std":   float(rnd_vals.std()),
            "mem_novelty_mean":  float(mem_vals.mean()),
            "mem_novelty_std":   float(mem_vals.std()),
            "fov_L_mean":       float(fovs[:, 0].mean()),
            "fov_C_mean":       float(fovs[:, 1].mean()),
            "fov_R_mean":       float(fovs[:, 2].mean()),
            "probe_errors":     probe_errors.tolist(),   # 7 values, one per room
            "rnd_running_mean": self._rnd_mean,
            "memory_size":       int(mem_sizes[-1]),
            "memory_size_mean":  float(mem_sizes.mean()),
            "new_clusters":      int(new_clusters.sum()),
            "new_cluster_rate":  float(new_clusters.mean()),
            "smooth_window_mean": float(smooth_windows.mean()),
            "smooth_resets":     int(smooth_resets.sum()),
            "smooth_reset_rate": float(smooth_resets.mean()),
        }
        # Reset accumulators
        self._diag_raw_losses.clear()
        self._diag_novelties.clear()
        self._diag_rnd.clear()
        self._diag_mem.clear()
        self._diag_fov.clear()
        self._diag_memory_size.clear()
        self._diag_new_clusters.clear()
        self._diag_smooth_window.clear()
        self._diag_smooth_resets.clear()
        return stats

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._novelty = 0.0
        self._reward_novelty = 0.0
        self._novelty_window.clear()
        self._smoother.reset()
        return self._augment(obs, self._novelty), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._global_steps += 1
        x = self._rnd_input(obs)
        self._novelty = self._compute_novelty(x)
        self._reward_novelty, persistent_novelty, motion_gate = self._reward_novelty_from_motion(
            self._novelty, info)
        if self._global_steps >= _RND_WARMUP:
            reward += self._reward_novelty * _VMM_NOVELTY_SCALE
        info["vmm_novelty"] = self._novelty
        info["reward_novelty"] = self._reward_novelty
        info["persistent_novelty"] = persistent_novelty
        info["motion_gate"] = motion_gate
        info["rnd_novelty"] = self._rnd_novelty
        info["mem_novelty"] = self._mem_novelty
        info["memory_size"] = self._memory.size
        info["cluster_id"] = self._cluster_id
        info["new_cluster"] = self._new_cluster
        self._step_log.append({
            "step": self._global_steps,
            "env_step": info.get("steps"),
            "coverage": info.get("coverage", 0.0) * 100,
            "collisions": info.get("collisions", 0),
            "vmm_novelty": self._novelty,
            "reward_novelty": self._reward_novelty,
            "persistent_novelty": persistent_novelty,
            "motion_gate": motion_gate,
            "rnd_novelty": self._rnd_novelty,
            "mem_novelty": self._mem_novelty,
            "memory_size": self._memory.size,
            "cluster_id": -1 if self._cluster_id is None else self._cluster_id,
            "new_cluster": int(self._new_cluster),
            "displacement": info.get("displacement", 0.0),
            "yaw_delta_abs": info.get("yaw_delta_abs", 0.0),
            "mode": self._mode,
            "smoothing": int(self._use_smoothing),
        })
        return self._augment(obs, self._novelty), reward, term, trunc, info

    def drain_step_log(self):
        rows = self._step_log
        self._step_log = []
        return rows


# -- Env factories ---------------------------------------------------------------

def make_no_vmm_env(furniture, seed, render_mode=None):
    env = ApartmentContinuousEnv(
        seed=seed, obstacles=furniture, render_mode=render_mode,
        reward_mode="world_feedback",
    )
    env.use_stuck_respawn = False
    return SafetyPenaltyWrapper(env)

def make_vmm_env(furniture, seed, render_mode=None, mode=None, use_smoothing=None):
    env = ApartmentContinuousEnv(
        seed=seed, obstacles=furniture, render_mode=render_mode,
        reward_mode="world_feedback",
    )
    env.use_stuck_respawn = False
    env = SafetyPenaltyWrapper(env)
    return VMMObsWrapper(env, mode=mode, use_smoothing=use_smoothing)


# -- Preview --------------------------------------------------------------------

def _preview(factory, label):
    import pygame
    env = factory(render_mode="human")
    env.reset()
    # Unwrap to the ApartmentContinuousEnv which owns render()
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    inner.render_mode = "human"
    inner.render()
    print(f"  [{label}]  close window or press Q to continue")
    for _ in range(PREVIEW_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                inner.close(); return
        env.step(env.action_space.sample())
        inner.render()
        time.sleep(0.033)
    inner.close()

def preview():
    seed = SEEDS[0]
    furniture = generate_apartment(np.random.default_rng(seed))
    print("\n-- Preview: SAC-NoVMM -----------------------------------")
    _preview(lambda render_mode=None: make_no_vmm_env(furniture, seed, render_mode), "SAC-NoVMM")
    print("\n-- Preview: SAC-VMM -------------------------------------")
    _preview(lambda render_mode=None: make_vmm_env(furniture, seed, render_mode), "SAC-VMM")
    print("\nPreview done. Starting training...\n")


# -- Callback -------------------------------------------------------------------

def _unwrap_inner(env):
    while hasattr(env, "env"):
        env = env.env
    return env

class TrackCallback(BaseCallback):
    def __init__(self, label, pbar, verbose=0):
        super().__init__(verbose)
        self.label        = label
        self.pbar         = pbar
        self.checkpoints  = []
        self.rnd_log      = []
        self.vmm_step_log = []
        self._next_eval   = EVAL_EVERY
        self._prev_step   = 0

    def _vmm_wrapper(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            if isinstance(env, VMMObsWrapper):
                return env
            env = env.env
        return env if isinstance(env, VMMObsWrapper) else None

    def _on_step(self) -> bool:
        self.pbar.update(self.num_timesteps - self._prev_step)
        self._prev_step = self.num_timesteps

        if self.num_timesteps >= self._next_eval:
            self._next_eval += EVAL_EVERY
            inner = _unwrap_inner(self.training_env.envs[0])
            m = _benchmark_metrics(inner)
            self.checkpoints.append((inner.step_count, m))

            vmm = self._vmm_wrapper()
            postfix = {"cov": f"{m['coverage']:.1f}%", "col": m["collisions"], "bumper": m["bumper_total"]}
            if vmm is not None:
                self.vmm_step_log.extend(vmm.drain_step_log())
                rnd = vmm.rnd_checkpoint_stats()
                rnd["step"] = inner.step_count
                self.rnd_log.append(rnd)
                postfix["novelty"] = f"{rnd['novelty_mean']:.2f}"
                postfix["rnd_loss"] = f"{rnd['rnd_loss_mean']:.4f}"
            self.pbar.set_postfix(postfix)
        return True


# -- Single-seed training -------------------------------------------------------

def _train(factory, label, seed, return_rnd=False):
    env = factory()
    env.reset(seed=seed)
    with tqdm(total=TRAIN_STEPS, desc=label, unit="step", dynamic_ncols=True) as pbar:
        cb = TrackCallback(label, pbar)
        model = SAC(
            "MlpPolicy", env,
            learning_rate=LR,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=1_000,
            train_freq=1,
            policy_kwargs={"net_arch": HIDDEN},
            device=DEVICE,
            verbose=0,
            seed=seed,
        )
        model.learn(total_timesteps=TRAIN_STEPS, callback=cb)
        vmm = cb._vmm_wrapper()
        if vmm is not None:
            cb.vmm_step_log.extend(vmm.drain_step_log())
        import pathlib
        pathlib.Path("results").mkdir(exist_ok=True)
        model.save(f"results/{label.strip().replace(' ', '_')}.zip")
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    try: inner.close()
    except Exception: pass
    if return_rnd:
        return cb.checkpoints, cb.rnd_log, cb.vmm_step_log
    return cb.checkpoints


def _run_boustrophedon(furniture, seed):
    from rover_coverage_env import SENSOR_MAX, MAX_WHEEL_SPEED, DT, AXLE_LENGTH, STUCK_LIMIT

    TURN_SPEED  = 0.45
    FWD_SPEED   = 0.55
    FRONT_BLOCK = 0.50
    SIDE_NUDGE  = 0.28
    omega       = (2 * MAX_WHEEL_SPEED * TURN_SPEED) / AXLE_LENGTH
    STEPS_90    = int(np.pi / 2 / omega / DT)
    SHIFT_STEPS = int(0.55 / (FWD_SPEED * MAX_WHEEL_SPEED * DT))
    ESCAPE_STEPS = int(np.pi * 2 / omega / DT)   # full 360°

    env = ApartmentContinuousEnv(seed=seed, obstacles=furniture)
    env.use_bumper        = True
    env.use_stuck_respawn = False
    obs, _ = env.reset(seed=seed)

    FORWARD, TURN1, SLIDE, TURN2, ESCAPE = range(5)
    state       = FORWARD
    state_steps = 0
    turn_dir    = 1.0
    stuck_count = 0
    prev_cell   = (-1, -1)

    def _act(obs):
        nonlocal state, state_steps, turn_dir, stuck_count, prev_cell
        front = float(obs[2]) * SENSOR_MAX
        left  = float(obs[0]) * SENSOR_MAX
        right = float(obs[1]) * SENSOR_MAX

        cur = env._cell(env.x, env.y)
        if cur != prev_cell:
            stuck_count = 0; prev_cell = cur
        else:
            stuck_count += 1

        def _a(c, s): return np.array([c, 0.0, s], dtype=np.float32)

        if state == ESCAPE:
            state_steps += 1
            if state_steps >= ESCAPE_STEPS:
                state = FORWARD; state_steps = 0
            return _a(1.0 if left >= right else -1.0, TURN_SPEED)

        if stuck_count >= STUCK_LIMIT // 2:
            stuck_count = 0; state = ESCAPE; state_steps = 0
            return _a(1.0 if left >= right else -1.0, TURN_SPEED)

        if state == TURN1:
            state_steps += 1
            if state_steps >= STEPS_90:
                state = SLIDE; state_steps = 0
            return _a(turn_dir, TURN_SPEED)

        if state == SLIDE:
            state_steps += 1
            if front < FRONT_BLOCK:
                state = ESCAPE; state_steps = 0
                return _a(1.0 if left >= right else -1.0, TURN_SPEED)
            if state_steps >= SHIFT_STEPS:
                state = TURN2; state_steps = 0
            return _a(0.0, FWD_SPEED)

        if state == TURN2:
            state_steps += 1
            if state_steps >= STEPS_90:
                state = FORWARD; state_steps = 0; turn_dir *= -1.0
            return _a(turn_dir, TURN_SPEED)

        # FORWARD
        if front < FRONT_BLOCK:
            turn_dir = 1.0 if left >= right else -1.0
            state = TURN1; state_steps = 0
            return _a(turn_dir, TURN_SPEED)

        if left  < SIDE_NUDGE: return _a(-0.25, FWD_SPEED)
        if right < SIDE_NUDGE: return _a( 0.25, FWD_SPEED)
        return _a(0.0, FWD_SPEED)

    checkpoints = []; next_eval = EVAL_EVERY
    label = f"Boustrophedon seed={seed}"
    with tqdm(total=TRAIN_STEPS, desc=label, unit="step", dynamic_ncols=True) as pbar:
        for _ in range(TRAIN_STEPS):
            obs, _, _, _, _ = env.step(_act(obs))
            pbar.update(1)
            if env.step_count >= next_eval:
                m = _benchmark_metrics(env)
                checkpoints.append((env.step_count, m))
                pbar.set_postfix({"cov": f"{m['coverage']:.1f}%",
                                  "col": m["collisions"]})
                next_eval += EVAL_EVERY
    env.close()
    return checkpoints


# -- Multi-seed training --------------------------------------------------------

def _save_csv(all_ckpts, label, out_dir):
    """Save per-seed checkpoint data as CSV files."""
    import csv, pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for seed, ckpts in zip(SEEDS, all_ckpts):
        path = out_dir / f"{label}_seed{seed}.csv"
        metric_keys = list(ckpts[0][1].keys()) if ckpts else []
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + metric_keys)
            writer.writeheader()
            for step, m in ckpts:
                writer.writerow({"step": step, **m})
        print(f"  Saved {path}")


def _save_rnd_csv(all_rnd_logs, out_dir):
    import csv, pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for seed, log in zip(SEEDS, all_rnd_logs):
        if not log:
            continue
        path = out_dir / f"rnd_training_seed{seed}.csv"
        scalar_keys = [
            "step", "rnd_loss_mean", "rnd_loss_std", "novelty_mean",
            "novelty_std", "rnd_novelty_mean", "rnd_novelty_std",
            "mem_novelty_mean", "mem_novelty_std", "fov_L_mean",
            "fov_C_mean", "fov_R_mean", "rnd_running_mean",
            "memory_size", "memory_size_mean", "new_clusters",
            "new_cluster_rate", "smooth_window_mean", "smooth_resets",
            "smooth_reset_rate",
        ]
        probe_keys = [f"probe_error_{i}" for i in range(len(log[0].get("probe_errors", [])))]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys + probe_keys)
            writer.writeheader()
            for entry in log:
                row = {k: entry.get(k) for k in scalar_keys}
                row.update({
                    key: value for key, value in zip(probe_keys, entry.get("probe_errors", []))
                })
                writer.writerow(row)
        print(f"  Saved {path}")


def _save_vmm_step_csv(all_vmm_step_logs, out_dir):
    import csv, pathlib
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for seed, rows in zip(SEEDS, all_vmm_step_logs):
        if not rows:
            continue
        path = out_dir / f"vmm_steps_seed{seed}.csv"
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {path}")


def train():
    """Run Boustrophedon, SAC-NoVMM, SAC-VMM on each seed."""
    all_no_vmm, all_vmm, all_boustr = [], [], []
    all_rnd_logs = []
    all_vmm_step_logs = []

    for seed in tqdm(SEEDS, desc="Seeds", unit="seed", dynamic_ncols=True):
        furniture = generate_apartment(np.random.default_rng(seed))
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  — apartment map")
        print(f"{'='*60}")

        print(f"\n-- Boustrophedon seed={seed} ------------------------------")
        all_boustr.append(_run_boustrophedon(furniture, seed))

        print(f"\n-- SAC-NoVMM  seed={seed} ---------------------------------")
        all_no_vmm.append(_train(
            lambda s=seed, f=furniture: make_no_vmm_env(f, s),
            f"SAC-NoVMM s{seed}", seed))

        print(f"\n-- SAC-VMM    seed={seed} ---------------------------------")
        ckpts, rnd_log, vmm_step_log = _train(
            lambda s=seed, f=furniture: make_vmm_env(
                f, s, mode=_VMM_MODE, use_smoothing=_VMM_USE_SMOOTHING),
            f"SAC-VMM-{_VMM_MODE}{'-smooth' if _VMM_USE_SMOOTHING else '-raw'} s{seed}",
            seed, return_rnd=True)
        all_vmm.append(ckpts)
        all_rnd_logs.append(rnd_log)
        all_vmm_step_logs.append(vmm_step_log)

    # Save all dataframes
    import json, pathlib
    _save_csv(all_boustr,  "boustrophedon", "results")
    _save_csv(all_no_vmm,  "sac_novmm",    "results")
    _save_csv(all_vmm,     "sac_vmm",      "results")
    _save_rnd_csv(all_rnd_logs, "results")
    _save_vmm_step_csv(all_vmm_step_logs, "results")

    log_path = pathlib.Path("results") / "rnd_logs.json"
    log_path.write_text(json.dumps(all_rnd_logs, indent=2))
    print(f"\nRND diagnostics saved -> {log_path}")

    return all_no_vmm, all_vmm, all_boustr, all_rnd_logs


# -- Plot (mean +/- std) --------------------------------------------------------

def _align(all_ckpts):
    """Stack per-seed checkpoint lists into arrays.
    Returns steps (T,) and values dict key->(N, T)."""
    steps = [s for s, _ in all_ckpts[0]]
    keys  = list(all_ckpts[0][0][1].keys())
    arrays = {k: np.array([[m[k] for _, m in ckpts] for ckpts in all_ckpts])
              for k in keys}
    return np.array(steps), arrays


def plot(all_no_vmm, all_vmm, all_boustr):
    from rover_coverage_env import DT
    methods = [
        ("Boustrophedon", all_boustr,  "#A0A0A0", "--"),
        ("SAC",           all_no_vmm,  "#4C9BE8", "-"),
        ("SAC + VMM",     all_vmm,     "#F4845F", "-"),
    ]
    panels = [
        ("Coverage %",      "coverage"),
        ("Collisions",      "collisions"),
        ("Bumper Triggers", "bumper_total"),
    ]

    # ── Combined comparison plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"Ablation: Boustrophedon vs SAC vs SAC+VMM  "
        f"({len(SEEDS)} seeds, mean ± std, {TRAIN_STEPS//1000}k steps)",
        fontsize=12, fontweight="bold")

    for label, all_ckpts, color, ls in methods:
        steps, vals = _align(all_ckpts)
        time_s = steps * DT   # physics steps → simulated seconds (fair comparison)
        for ax, (title, key) in zip(axes, panels):
            arr  = vals[key]
            mean = arr.mean(0)
            std  = arr.std(0)
            ax.plot(time_s, mean, color=color, lw=2, ls=ls, label=label)
            ax.fill_between(time_s, mean - std, mean + std, color=color, alpha=0.18)

    for ax, (title, _) in zip(axes, panels):
        ax.set_title(title); ax.set_xlabel("Simulated time (s)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "results/comparison.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Plot saved ->{out}")

    # ── Separate plot per method ───────────────────────────────────────────────
    for label, all_ckpts, color, ls in methods:
        steps, vals = _align(all_ckpts)
        time_s = steps * DT
        fig, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{label}  ({len(SEEDS)} seeds, mean ± std, {TRAIN_STEPS//1000}k steps)",
                     fontsize=12, fontweight="bold")
        for ax, (title, key) in zip(axes2, panels):
            arr  = vals[key]
            mean = arr.mean(0)
            std  = arr.std(0)
            ax.plot(time_s, mean, color=color, lw=2, ls=ls)
            ax.fill_between(time_s, mean - std, mean + std, color=color, alpha=0.25)
            for seed_i, seed in enumerate(SEEDS):
                ax.plot(time_s, arr[seed_i], color=color, lw=0.8, alpha=0.4,
                        label=f"seed {seed}")
            ax.set_title(title); ax.set_xlabel("Simulated time (s)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"results/{label.lower().replace(' ', '_').replace('+', 'plus')}.png"
        plt.savefig(fname, dpi=150); plt.close()
        print(f"Plot saved ->{fname}")

    # ── Summary table ─────────────────────────────────────────────────────────
    def final_mean_std(all_ckpts, key):
        vals = [ckpts[-1][1][key] for ckpts in all_ckpts]
        return np.mean(vals), np.std(vals)

    print(f"\n{'':35s} {'Boustrophedon':>18s} {'SAC':>18s} {'SAC+VMM':>18s}")
    print("-" * 92)
    for row_label, key, fmt in [
        ("Final coverage %",      "coverage",    ".1f"),
        ("Final collisions",      "collisions",  ".1f"),
        ("Final bumper triggers", "bumper_total", ".1f"),
        ("Coverage / collision",  "coverage_per_collision", ".4f"),
        ("Collisions / coverage", "collisions_per_coverage", ".1f"),
        ("Coverage / 1k steps",   "coverage_per_1k_steps", ".3f"),
    ]:
        bm, bs = final_mean_std(all_boustr, key)
        nm, ns = final_mean_std(all_no_vmm, key)
        vm, vs = final_mean_std(all_vmm,    key)
        print(f"  {row_label:33s} "
              f"{bm:{fmt}} ± {bs:.1f}   "
              f"{nm:{fmt}} ± {ns:.1f}   "
              f"{vm:{fmt}} ± {vs:.1f}")

    _plot_benchmark_metrics(methods)


def _plot_benchmark_metrics(methods):
    """Extra benchmark plots for exploration quality, not just final coverage."""
    panels = [
        ("Coverage / Collision", "coverage_per_collision"),
        ("Collisions / 1k Steps", "collisions_per_1k_steps"),
        ("Coverage / 1k Steps", "coverage_per_1k_steps"),
        ("Reward / 1k Steps", "reward_per_1k_steps"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Benchmark Efficiency and Safety Metrics", fontweight="bold")

    for label, all_ckpts, color, ls in methods:
        steps, vals = _align(all_ckpts)
        elapsed_s = vals.get("elapsed_s", np.tile(steps, (len(all_ckpts), 1)) * 0.05)
        time_s = elapsed_s.mean(0)
        for ax, (title, key) in zip(axes.flat, panels):
            arr = vals[key]
            mean = arr.mean(0)
            std = arr.std(0)
            ax.plot(time_s, mean, color=color, lw=2, ls=ls, label=label)
            ax.fill_between(time_s, mean - std, mean + std, color=color, alpha=0.16)

    for ax, (title, _) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.set_xlabel("Simulated time (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "results/benchmark_metrics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Benchmark metrics plot saved -> {out}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, all_ckpts, color, ls in methods:
        _, vals = _align(all_ckpts)
        cov = vals["coverage"]
        col = vals["collisions"]
        ax.plot(col.mean(0), cov.mean(0), color=color, lw=2, ls=ls, marker="o", ms=3, label=label)
    ax.set_title("Coverage vs Collisions")
    ax.set_xlabel("Collisions")
    ax.set_ylabel("Coverage %")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = "results/coverage_vs_collisions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Coverage/collision plot saved -> {out}")


# -- RND analysis plot ----------------------------------------------------------

PROBE_LABELS = ["R1 top-L", "R2 top-ML", "R3 top-MR", "R4 top-R",
                "R5 bot-L", "R6 bot-M",  "R7 bot-R"]

def plot_rnd(all_rnd_logs):
    """
    Four-panel RND diagnostic plot, averaged across seeds.

    Panel 1 — Raw RND loss over training: should decay as predictor learns
               familiar states.  If it stays flat the predictor isn't learning.

    Panel 2 — Combined/RND/memory novelty: separates learned predictor error
               from explicit nearest-neighbour novelty.

    Panel 3 — Directional FOV novelty (L/C/R): should diverge when the rover
               is near a wall (one direction blocked, others open).

    Panel 4 — Probe errors per room over time: rooms the rover visits frequently
               should show declining error; unvisited rooms stay high.  This is
               the clearest test that RND is actually discriminating space.
    """
    if not all_rnd_logs or not all_rnd_logs[0]:
        print("No RND logs to plot.")
        return

    from rover_coverage_env import DT
    n_ckpts = min(len(log) for log in all_rnd_logs)
    steps   = [entry["step"] * DT for entry in all_rnd_logs[0][:n_ckpts]]

    def _mean_std(key):
        arr = np.array([[entry[key] for entry in log[:n_ckpts]]
                        for log in all_rnd_logs])
        return arr.mean(0), arr.std(0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("RND Diagnostics (mean ± std across seeds)", fontweight="bold")

    # Panel 1 — raw loss
    ax = axes[0, 0]
    m, s = _mean_std("rnd_loss_mean")
    ax.plot(steps, m, color="#E07040", lw=2)
    ax.fill_between(steps, m - s, m + s, alpha=0.2, color="#E07040")
    ax.set_title("Raw RND Loss (should decay)"); ax.set_xlabel("Simulated time (s)")
    ax.grid(alpha=0.3)

    # Panel 2 — novelty score
    ax = axes[0, 1]
    m, s = _mean_std("novelty_mean")
    ax.plot(steps, m, color="#4C9BE8", lw=2, label="combined")
    ax.fill_between(steps, m - s, m + s, alpha=0.2, color="#4C9BE8")
    if "rnd_novelty_mean" in all_rnd_logs[0][0]:
        rm, _ = _mean_std("rnd_novelty_mean")
        mm, _ = _mean_std("mem_novelty_mean")
        ax.plot(steps, rm, color="#E07040", lw=1.5, ls="--", label="RND")
        ax.plot(steps, mm, color="#60A060", lw=1.5, ls=":", label="memory")
    ax.set_title("Novelty Components"); ax.set_xlabel("Simulated time (s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3 — directional FOV novelty
    ax = axes[1, 0]
    for key, label, col in [
        ("fov_L_mean", "Left",   "#A0C8F0"),
        ("fov_C_mean", "Center", "#F0A040"),
        ("fov_R_mean", "Right",  "#90D090"),
    ]:
        m, s = _mean_std(key)
        ax.plot(steps, m, lw=2, color=col, label=label)
        ax.fill_between(steps, m - s, m + s, alpha=0.15, color=col)
    ax.set_title("Directional FOV Novelty (L/C/R)"); ax.set_xlabel("Simulated time (s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    if "memory_size" in all_rnd_logs[0][0]:
        ax2 = ax.twinx()
        ms, _ = _mean_std("memory_size")
        ax2.plot(steps, ms, color="#303030", lw=1.2, alpha=0.65, label="Clusters")
        ax2.set_ylabel("Visual-place clusters")

    # Panel 4 — probe errors per room
    ax = axes[1, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(PROBE_LABELS)))
    probe_arr = np.array([[entry["probe_errors"] for entry in log[:n_ckpts]]
                          for log in all_rnd_logs])  # (seeds, ckpts, 7)
    for room_i, (label, col) in enumerate(zip(PROBE_LABELS, colors)):
        m = probe_arr[:, :, room_i].mean(0)
        s = probe_arr[:, :, room_i].std(0)
        ax.plot(steps, m, lw=1.5, color=col, label=label)
        ax.fill_between(steps, m - s, m + s, alpha=0.12, color=col)
    ax.set_title("Probe Error per Room (high=novel, low=familiar)")
    ax.set_xlabel("Simulated time (s)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "rnd_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"RND analysis plot saved -> {out}")


# -- Main -----------------------------------------------------------------------

if __name__ == "__main__":
    if "--vmm-mode" in sys.argv:
        idx = sys.argv.index("--vmm-mode")
        try:
            _VMM_MODE = sys.argv[idx + 1]
        except IndexError:
            raise SystemExit("--vmm-mode requires one of: rnd, memory, rnd_memory")
        if _VMM_MODE not in ("rnd", "memory", "rnd_memory"):
            raise SystemExit("--vmm-mode must be one of: rnd, memory, rnd_memory")
    if "--no-smoothing" in sys.argv:
        _VMM_USE_SMOOTHING = False

    if "--preview" in sys.argv:
        preview()

    import pathlib; pathlib.Path("results").mkdir(exist_ok=True)
    all_no_vmm, all_vmm, all_boustr, all_rnd_logs = train()
    plot(all_no_vmm, all_vmm, all_boustr)
    plot_rnd(all_rnd_logs)
