"""
SAC: No-VMM (raw sensors) vs VMM-augmented observations
---------------------------------------------------------
SAC-NoVMM : obs = [left, right, front] plus movement/safety reward only
SAC-VMM   : obs = ultrasonic + IMU + simulated egocentric visual RND features
            reward += rnd_novelty * VMM_NOVELTY_SCALE each step

VMM novelty is RND (Random Network Distillation) on a 2D egocentric "camera"
embedding: a forward fan of ray depths plus deterministic endpoint texture.
This is a simulator stand-in for the real VMM path, where MobileNetV3 embeds
camera frames and RND runs on those visual embeddings.

Multi-seed: each method runs on N_SEEDS independent obstacle layouts;
coverage curves are averaged and plotted with mean +/- std shaded bands.

Safety layer: simulator collision physics stays active; the SAC factories keep
the reflex bumper off so the replay buffer does not silently store unclamped
actions as if they were executed.

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
    """Bumper is OFF during training — SAC learns from its own actions.
    Collision physics (slide code) prevents wall penetration.
    R_COLLISION penalty in the env already discourages crashing."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.use_bumper = False   # do not intercept SAC actions
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


class VMMObsWrapper(gym.Wrapper):
    """
    RND-based novelty wrapper — no coverage-grid oracle.

    Policy obs (12-dim):
        [left, right, front]          — HC-SR04 ultrasonic sensors
        [sin(θ), cos(θ)]              — IMU heading
        [novelty]                     — RND error at current position
        [fov_L, fov_C, fov_R]         — RND error in left/center/right visual sectors
        [yaw_rate]                    — normalised gyro-z (IMU), execution feedback
        [vl, vr]                      — last executed wheel speeds

    RND input   : simulated egocentric camera embedding + IMU heading.
    Reward bonus: novelty * _VMM_NOVELTY_SCALE every step (after warmup).
    """

    def __init__(self, env):
        super().__init__(env)
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
        self._rnd_mean     = 0.0
        self._rnd_m2       = 0.0
        self._rnd_n        = 0
        self._novelty      = 0.0
        self._global_steps = 0  # never resets — warmup is global, not per-episode

        # RND diagnostics — accumulated between eval checkpoints
        self._diag_raw_losses  = []   # raw RND loss every step
        self._diag_novelties   = []   # normalised novelty every step
        self._diag_fov         = []   # [fov_L, fov_C, fov_R] every step

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

    def _compute_novelty(self, x):
        """Compute RND error, update predictor, return normalised novelty ∈ [0,1]."""
        with torch.no_grad():
            t_out = self._target(x)
        p_out = self._predictor(x)
        loss = torch.nn.functional.mse_loss(p_out, t_out)
        self._opt.zero_grad(); loss.backward(); self._opt.step()

        raw = loss.item()
        # Welford running mean — familiar states trend toward 0, novel states spike to 1
        self._rnd_n += 1
        d = raw - self._rnd_mean
        self._rnd_mean += d / self._rnd_n
        self._rnd_m2   += d * (raw - self._rnd_mean)
        norm = float(np.clip(raw / (self._rnd_mean + 1e-8), 0.0, 1.0))
        self._diag_raw_losses.append(raw)
        self._diag_novelties.append(norm)
        return norm

    def _fov_novelty(self):
        """RND novelty evaluated on left/center/right crops of the current view."""
        sector_centers = [np.radians(35), 0.0, -np.radians(35)]
        result = []
        for rel_angle in sector_centers:
            vec = self._sim_camera_embedding(center_offset=rel_angle)
            x_t = torch.tensor(vec).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                t_out = self._target(x_t)
                p_out = self._predictor(x_t)
                err   = torch.nn.functional.mse_loss(p_out, t_out).item()
            norm = float(np.clip(err / (self._rnd_mean + 1e-8), 0.0, 1.0))
            result.append(norm)
        return np.array(result, dtype=np.float32)

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
        fovs      = np.array(self._diag_fov)         if self._diag_fov        else np.zeros((1, 3))

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
            "fov_L_mean":       float(fovs[:, 0].mean()),
            "fov_C_mean":       float(fovs[:, 1].mean()),
            "fov_R_mean":       float(fovs[:, 2].mean()),
            "probe_errors":     probe_errors.tolist(),   # 7 values, one per room
            "rnd_running_mean": self._rnd_mean,
        }
        # Reset accumulators
        self._diag_raw_losses.clear()
        self._diag_novelties.clear()
        self._diag_fov.clear()
        return stats

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._novelty = 0.0
        return self._augment(obs, self._novelty), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._global_steps += 1
        x = self._rnd_input(obs)
        self._novelty = self._compute_novelty(x)
        if self._global_steps >= _RND_WARMUP:
            reward += self._novelty * _VMM_NOVELTY_SCALE
        info["vmm_novelty"] = self._novelty
        return self._augment(obs, self._novelty), reward, term, trunc, info


# -- Env factories ---------------------------------------------------------------

def make_no_vmm_env(furniture, seed, render_mode=None):
    env = ApartmentContinuousEnv(
        seed=seed, obstacles=furniture, render_mode=render_mode,
        reward_mode="world_feedback",
    )
    env.use_stuck_respawn = False
    return SafetyPenaltyWrapper(env)

def make_vmm_env(furniture, seed, render_mode=None):
    env = ApartmentContinuousEnv(
        seed=seed, obstacles=furniture, render_mode=render_mode,
        reward_mode="world_feedback",
    )
    env.use_stuck_respawn = False
    env = SafetyPenaltyWrapper(env)
    return VMMObsWrapper(env)


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
        import pathlib
        pathlib.Path("results").mkdir(exist_ok=True)
        model.save(f"results/{label.strip().replace(' ', '_')}.zip")
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    try: inner.close()
    except Exception: pass
    if return_rnd:
        return cb.checkpoints, cb.rnd_log
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
            "novelty_std", "fov_L_mean", "fov_C_mean", "fov_R_mean",
            "rnd_running_mean",
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


def train():
    """Run Boustrophedon, SAC-NoVMM, SAC-VMM on each seed."""
    all_no_vmm, all_vmm, all_boustr = [], [], []
    all_rnd_logs = []

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
        ckpts, rnd_log = _train(
            lambda s=seed, f=furniture: make_vmm_env(f, s),
            f"SAC-VMM   s{seed}", seed, return_rnd=True)
        all_vmm.append(ckpts)
        all_rnd_logs.append(rnd_log)

    # Save all dataframes
    import json, pathlib
    _save_csv(all_boustr,  "boustrophedon", "results")
    _save_csv(all_no_vmm,  "sac_novmm",    "results")
    _save_csv(all_vmm,     "sac_vmm",      "results")
    _save_rnd_csv(all_rnd_logs, "results")

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

    Panel 2 — Novelty score mean ± std: high and stable = good exploration
               signal throughout.  Collapsing to 0 = predictor over-fitted,
               novelty dead.

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
    ax.plot(steps, m, color="#4C9BE8", lw=2, label="mean novelty")
    ax.fill_between(steps, m - s, m + s, alpha=0.2, color="#4C9BE8")
    ms, ss = _mean_std("novelty_std")
    ax.plot(steps, ms, color="#4C9BE8", lw=1, ls="--", label="std novelty")
    ax.set_title("Novelty Score Distribution"); ax.set_xlabel("Simulated time (s)")
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
    if "--preview" in sys.argv:
        preview()

    import pathlib; pathlib.Path("results").mkdir(exist_ok=True)
    all_no_vmm, all_vmm, all_boustr, all_rnd_logs = train()
    plot(all_no_vmm, all_vmm, all_boustr)
    plot_rnd(all_rnd_logs)
