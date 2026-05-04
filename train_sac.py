"""
SAC: No-VMM (raw sensors) vs VMM-augmented observations
---------------------------------------------------------
SAC-NoVMM : obs = [left, right, front]                              (3 floats)
SAC-VMM   : obs = [left, right, front, sin(θ), cos(θ), rnd_novelty] (6 floats)
            reward += rnd_novelty * VMM_NOVELTY_SCALE each step

VMM novelty is RND (Random Network Distillation) on
[left, right, front, sin(theta), cos(theta)] — the same signals available
on the real rover (HC-SR04 sensors + IMU yaw).  High prediction error = the
current sensor/pose context is unfamiliar.  The predictor trains online.
This mirrors the RND branch of the real VMM (which uses MobileNetV3 embeddings
instead of raw sensor readings).

Multi-seed: each method runs on N_SEEDS independent obstacle layouts;
coverage curves are averaged and plotted with mean +/- std shaded bands.

Safety layer: embedded 8-ray bumper reflex always active.
R_SAFETY=-1.0 per trigger discourages using it as a free turn.

Training is continuous — single map, coverage accumulates across the full run.

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
TRAIN_STEPS   = 350_000
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

_VMM_NOVELTY_SCALE = 0.40  # strong enough to pull policy toward unexplored rooms
_RND_INPUT_DIM     = 7    # [left, right, front, sin(theta), cos(theta), x/W, y/H]
_RND_HIDDEN        = 64
_RND_OUTPUT_DIM    = 64
_RND_LR            = 3e-5  # slower fit keeps novelty signal alive longer
_RND_WARMUP        = 200  # more warmup so Welford mean stabilises before bonuses kick in


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
        [fov_L, fov_C, fov_R]         — RND error at lookahead in each sector
        [x/W, y/H]                    — normalised odometry position
        [yaw_rate]                    — normalised gyro-z (IMU), execution feedback

    RND input   : [left, right, front, sin(θ), cos(θ)] — direction-aware novelty.
    Reward bonus: novelty * _VMM_NOVELTY_SCALE every step (after warmup).
    """

    def __init__(self, env):
        super().__init__(env)
        s = env.observation_space
        # obs = [left, right, front, sin(θ), cos(θ), novelty,
        #        fov_L, fov_C, fov_R, x_norm, y_norm, yaw_rate] — 12-dim
        self.observation_space = spaces.Box(
            low  = np.concatenate([s.low,  [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]).astype(np.float32),
            high = np.concatenate([s.high, [ 1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0]]).astype(np.float32),
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

        # Fixed probe states: one per apartment room — same input every checkpoint
        # [left, right, front, sin(θ), cos(θ)] at a representative point per room
        # facing east (θ=0) so sin=0, cos=1.  Sensors set to "open space" (1.0).
        # 7-dim probes: [l, r, f, sin, cos, x/W, y/H] — each room has unique position
        from apartment_env import APT_W, APT_H
        room_positions = [
            (2.5/APT_W,  11.5/APT_H),   # room1 top-left
            (7.5/APT_W,  11.5/APT_H),   # room2 top-mid-L
            (12.0/APT_W, 11.5/APT_H),   # room3 top-mid-R
            (17.0/APT_W, 11.5/APT_H),   # room4 top-right
            (4.0/APT_W,  3.0/APT_H),    # room5 bot-left
            (11.0/APT_W, 3.0/APT_H),    # room6 bot-mid
            (17.0/APT_W, 3.0/APT_H),    # room7 bot-right
        ]
        headings = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        self._probe_states = torch.tensor([
            [1.0, 1.0, 1.0, np.sin(h), np.cos(h), px, py]
            for (px, py), h in zip(room_positions, headings)
        ], dtype=torch.float32).to(DEVICE)

    def _unwrap_inner(self):
        inner = self.env
        while hasattr(inner, "env"):
            inner = inner.env
        return inner

    def _rnd_input(self, obs_3):
        """Onboard signals + odometry position for position-aware novelty.

        Adding x/W, y/H means two rooms with identical sensor geometry
        are treated as distinct states — critical for multi-room exploration."""
        from apartment_env import APT_W, APT_H
        inner = self._unwrap_inner()
        vec = np.array([
            obs_3[0],                    # left sensor
            obs_3[1],                    # right sensor
            obs_3[2],                    # front sensor
            np.sin(inner.theta),         # heading sin
            np.cos(inner.theta),         # heading cos
            inner.x / APT_W,            # odometry x (available on real rover)
            inner.y / APT_H,            # odometry y
        ], dtype=np.float32)
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
        """RND novelty evaluated at the lookahead point in each forward sector.

        For each sector (left / center / right), cast the center ray and evaluate
        the RND predictor at the *hypothetical* sensor context the rover would
        experience if it moved in that direction.  High error = unfamiliar scene
        ahead in that sector.

        This is hardware-safe: on the real rover, the camera looks in the forward
        direction and the RND runs on a MobileNetV3 crop of that view — no oracle,
        no coverage grid.  Here we simulate that by constructing a synthetic obs
        from the lookahead position's sensor readings."""
        from apartment_env import _apt_ray_cast, APT_W, APT_H
        from rover_coverage_env import SENSOR_MAX
        inner     = self._unwrap_inner()
        FOV_RANGE = 2.0   # metres to project ahead per sector
        SECTOR_CENTERS = [np.radians(40), 0.0, -np.radians(40)]  # left, center, right

        result = []
        for rel_angle in SECTOR_CENTERS:
            angle     = inner.theta + rel_angle
            ray_hit   = _apt_ray_cast(inner.x, inner.y, angle, inner.obstacles,
                                      max_dist=FOV_RANGE)
            look_dist = min(ray_hit * 0.85, FOV_RANGE)   # stay clear of obstacle
            lx = np.clip(inner.x + look_dist * np.cos(angle), 0.0, APT_W)
            ly = np.clip(inner.y + look_dist * np.sin(angle), 0.0, APT_H)
            # Synthetic sensor context at lookahead point facing same direction
            vec = np.array([
                _apt_ray_cast(lx, ly, angle + np.pi / 2, inner.obstacles) / SENSOR_MAX,
                _apt_ray_cast(lx, ly, angle - np.pi / 2, inner.obstacles) / SENSOR_MAX,
                _apt_ray_cast(lx, ly, angle,             inner.obstacles) / SENSOR_MAX,
                np.sin(angle),
                np.cos(angle),
                lx / APT_W,
                ly / APT_H,
            ], dtype=np.float32)
            x_t = torch.tensor(vec).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                t_out = self._target(x_t)
                p_out = self._predictor(x_t)
                err   = torch.nn.functional.mse_loss(p_out, t_out).item()
            norm = float(np.clip(err / (self._rnd_mean + 1e-8), 0.0, 1.0))
            result.append(norm)
        return np.array(result, dtype=np.float32)

    def _augment(self, obs, novelty):
        from apartment_env import APT_W, APT_H
        inner = self._unwrap_inner()
        fov   = self._fov_novelty()
        self._diag_fov.append(fov.copy())
        return np.array([
            obs[0], obs[1], obs[2],          # HC-SR04 sensors
            np.sin(inner.theta),             # IMU yaw heading
            np.cos(inner.theta),
            novelty,                         # RND novelty at current position
            fov[0], fov[1], fov[2],          # RND novelty ahead: left/center/right
            inner.x / APT_W,                 # odometry x
            inner.y / APT_H,                 # odometry y
            inner.yaw_rate,                  # gyro-z: execution feedback from IMU
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
    env = ApartmentContinuousEnv(seed=seed, obstacles=furniture, render_mode=render_mode)
    env.use_stuck_respawn = False
    return SafetyPenaltyWrapper(env)

def make_vmm_env(furniture, seed, render_mode=None):
    env = ApartmentContinuousEnv(seed=seed, obstacles=furniture, render_mode=render_mode)
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
            m = {
                "coverage":     inner._coverage() * 100,
                "collisions":   inner._collisions,
                "bumper_total": inner._bumper_triggers,
            }
            self.checkpoints.append((self.num_timesteps, m))

            vmm = self._vmm_wrapper()
            postfix = {"cov": f"{m['coverage']:.1f}%", "col": m["collisions"], "bumper": m["bumper_total"]}
            if vmm is not None:
                rnd = vmm.rnd_checkpoint_stats()
                rnd["step"] = self.num_timesteps
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
                m = {
                    "coverage":     env._coverage() * 100,
                    "collisions":   env._collisions,
                    "bumper_total": env._bumper_triggers,
                }
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
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "coverage", "collisions", "bumper_total"])
            writer.writeheader()
            for step, m in ckpts:
                writer.writerow({"step": step, **m})
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

    log_path = pathlib.Path("results") / "rnd_logs.json"
    log_path.write_text(json.dumps(all_rnd_logs, indent=2))
    print(f"\nRND diagnostics saved → {log_path}")

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
        for ax, (title, key) in zip(axes, panels):
            arr  = vals[key]
            mean = arr.mean(0)
            std  = arr.std(0)
            ax.plot(steps, mean, color=color, lw=2, ls=ls, label=label)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)

    for ax, (title, _) in zip(axes, panels):
        ax.set_title(title); ax.set_xlabel("Steps")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "results/comparison.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"Plot saved → {out}")

    # ── Separate plot per method ───────────────────────────────────────────────
    for label, all_ckpts, color, ls in methods:
        steps, vals = _align(all_ckpts)
        fig, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{label}  ({len(SEEDS)} seeds, mean ± std, {TRAIN_STEPS//1000}k steps)",
                     fontsize=12, fontweight="bold")
        for ax, (title, key) in zip(axes2, panels):
            arr  = vals[key]
            mean = arr.mean(0)
            std  = arr.std(0)
            ax.plot(steps, mean, color=color, lw=2, ls=ls)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.25)
            for seed_i, seed in enumerate(SEEDS):
                ax.plot(steps, arr[seed_i], color=color, lw=0.8, alpha=0.4,
                        label=f"seed {seed}")
            ax.set_title(title); ax.set_xlabel("Steps")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f"results/{label.lower().replace(' ', '_').replace('+', 'plus')}.png"
        plt.savefig(fname, dpi=150); plt.close()
        print(f"Plot saved → {fname}")

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
    ]:
        bm, bs = final_mean_std(all_boustr, key)
        nm, ns = final_mean_std(all_no_vmm, key)
        vm, vs = final_mean_std(all_vmm,    key)
        print(f"  {row_label:33s} "
              f"{bm:{fmt}} ± {bs:.1f}   "
              f"{nm:{fmt}} ± {ns:.1f}   "
              f"{vm:{fmt}} ± {vs:.1f}")


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

    n_ckpts = min(len(log) for log in all_rnd_logs)
    steps   = [entry["step"] for entry in all_rnd_logs[0][:n_ckpts]]

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
    ax.set_title("Raw RND Loss (should decay)"); ax.set_xlabel("Steps")
    ax.grid(alpha=0.3)

    # Panel 2 — novelty score
    ax = axes[0, 1]
    m, s = _mean_std("novelty_mean")
    ax.plot(steps, m, color="#4C9BE8", lw=2, label="mean novelty")
    ax.fill_between(steps, m - s, m + s, alpha=0.2, color="#4C9BE8")
    ms, ss = _mean_std("novelty_std")
    ax.plot(steps, ms, color="#4C9BE8", lw=1, ls="--", label="std novelty")
    ax.set_title("Novelty Score Distribution"); ax.set_xlabel("Steps")
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
    ax.set_title("Directional FOV Novelty (L/C/R)"); ax.set_xlabel("Steps")
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
    ax.set_xlabel("Steps"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "rnd_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"RND analysis plot saved → {out}")


# -- Main -----------------------------------------------------------------------

if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview()

    import pathlib; pathlib.Path("results").mkdir(exist_ok=True)
    all_no_vmm, all_vmm, all_boustr, all_rnd_logs = train()
    plot(all_no_vmm, all_vmm, all_boustr)
    plot_rnd(all_rnd_logs)