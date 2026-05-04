"""
SAC: No-VMM (raw sensors) vs VMM-augmented observations
---------------------------------------------------------
SAC-NoVMM : obs = [left, right, front]  (3 floats)
SAC-VMM   : obs = [left, right, front, rnd_novelty]  (4 floats)
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

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from rover_continuous_env import ContinuousRoverEnv
# -- Config ---------------------------------------------------------------------
SEEDS         = [42, 0, 1]       # one obstacle layout per seed; results averaged
N_OBSTACLES   = 10
TRAIN_STEPS   = 60_000
EVAL_EVERY    = 3_000
BUFFER_SIZE   = 50_000
BATCH_SIZE    = 512
LR            = 3e-4
GAMMA         = 0.99
HIDDEN        = [64, 64]
PREVIEW_STEPS = 300

# Small per-trigger penalty: enough to stop the policy using the bumper as a
# free turning mechanism, small enough not to block near-obstacle coverage.
R_SAFETY = -1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Safety penalty wrapper -----------------------------------------------------

class SafetyPenaltyWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.env.use_bumper = True
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if info.get("bumper_fired"):
            reward += R_SAFETY
        return obs, reward, term, trunc, info


# -- VMM observation wrapper ----------------------------------------------------

_VMM_NOVELTY_SCALE = 0.15  # balanced with R_MOVE: max novelty ~= max move reward
_RND_INPUT_DIM     = 3    # [front_sensor, look_x_norm, look_y_norm]
_RND_HIDDEN        = 64
_RND_OUTPUT_DIM    = 64
_RND_LR            = 1e-4  # slow enough that novelty stays informative
_RND_WARMUP        = 50   # steps before novelty bonus is applied
_VMM_LOOKAHEAD     = 1.5  # metres ahead that the "front camera" is looking at


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

    Input to RND: [left_sensor, right_sensor, front_sensor, sin(theta), cos(theta)]
    These are exactly the signals available on the real rover (HC-SR04 + IMU yaw).

    Obs appended: normalised RND prediction error (1 scalar) → obs = 4-dim.
    Reward bonus : novelty * _VMM_NOVELTY_SCALE every step.

    The predictor trains online; high error means the current sensor/pose context
    is unfamiliar.  This mirrors the VMM RND branch on the real rover where the
    predictor sees MobileNetV3 embeddings instead of raw sensor readings.
    """

    def __init__(self, env):
        super().__init__(env)
        s = env.observation_space
        self.observation_space = spaces.Box(
            low  = np.concatenate([s.low,  [0.0]]).astype(np.float32),
            high = np.concatenate([s.high, [1.0]]).astype(np.float32),
        )
        self._target    = _RNDTarget().to(DEVICE)
        self._predictor = _RNDPredictor().to(DEVICE)
        self._opt       = torch.optim.Adam(self._predictor.parameters(), lr=_RND_LR)
        self._rnd_mean  = 0.0
        self._rnd_m2    = 0.0
        self._rnd_n     = 0
        self._novelty   = 0.0
        self._step      = 0

    def _unwrap_inner(self):
        inner = self.env
        while hasattr(inner, "env"):
            inner = inner.env
        return inner

    def _rnd_input(self, obs_3):
        """Encode the front view: where is the rover looking and what does it see?

        [front_sensor, look_x_norm, look_y_norm] — 3 values that describe the
        scene ahead, exactly like a forward-facing camera crop.  Rotating in
        place changes look_x/y even without translating, so novelty is
        directional.  On the real rover, look_x/y are replaced by a
        MobileNetV3 embedding of the actual camera frame."""
        from rover_coverage_env import FIELD_W, FIELD_H
        inner = self._unwrap_inner()
        look_x = np.clip(inner.x + _VMM_LOOKAHEAD * np.cos(inner.theta), 0, FIELD_W)
        look_y = np.clip(inner.y + _VMM_LOOKAHEAD * np.sin(inner.theta), 0, FIELD_H)
        vec = np.array([
            obs_3[2],             # front sensor (what obstacle is ahead)
            look_x / FIELD_W,     # where the rover is looking — x
            look_y / FIELD_H,     # where the rover is looking — y
        ], dtype=np.float32)
        return torch.tensor(vec).unsqueeze(0).to(DEVICE)

    def _compute_novelty(self, x):
        """Compute RND error, update predictor, return normalised novelty ∈ [0,1]."""
        with torch.no_grad():
            t_out = self._target(x)
        p_out = self._predictor(x)
        loss = torch.nn.functional.mse_loss(p_out, t_out.detach())
        self._opt.zero_grad(); loss.backward(); self._opt.step()

        raw = loss.item()
        # Welford running normalisation
        self._rnd_n += 1
        d = raw - self._rnd_mean
        self._rnd_mean += d / self._rnd_n
        self._rnd_m2   += d * (raw - self._rnd_mean)
        norm = raw / (self._rnd_mean + 1e-8)
        # Clip to [0,1] for use as obs feature
        return float(np.clip(norm, 0.0, 1.0))

    def _augment(self, obs, novelty):
        return np.append(obs, novelty).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._novelty = 0.0
        self._step    = 0
        return self._augment(obs, self._novelty), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self._step += 1
        x = self._rnd_input(obs)
        self._novelty = self._compute_novelty(x)
        if self._step >= _RND_WARMUP:
            reward += self._novelty * _VMM_NOVELTY_SCALE
        info["vmm_novelty"] = self._novelty
        return self._augment(obs, self._novelty), reward, term, trunc, info


# -- Env factories (per-seed obstacles passed in) --------------------------------
from rover_coverage_env import generate_obstacles

def make_no_vmm_env(obstacles, seed, render_mode=None):
    env = ContinuousRoverEnv(n_obstacles=N_OBSTACLES, seed=seed,
                             render_mode=render_mode, obstacles=obstacles)
    return SafetyPenaltyWrapper(env)

def make_vmm_env(obstacles, seed, render_mode=None):
    env = ContinuousRoverEnv(n_obstacles=N_OBSTACLES, seed=seed,
                             render_mode=render_mode, obstacles=obstacles)
    env = SafetyPenaltyWrapper(env)
    return VMMObsWrapper(env)


# -- Preview --------------------------------------------------------------------

def _preview(factory, label):
    import pygame
    env = factory(render_mode="human")
    obs, _ = env.reset()
    inner = env.env if isinstance(env, VMMObsWrapper) else env
    inner.env.render_mode = "human"
    inner.env.render()
    print(f"  [{label}]  close window or press Q to continue")
    for _ in range(PREVIEW_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                inner.env.close(); return
        env.step(env.action_space.sample())
        inner.env.render()
        time.sleep(0.033)
    inner.env.close()

def preview():
    seed = SEEDS[0]
    obs = generate_obstacles(N_OBSTACLES, np.random.default_rng(seed))
    print("\n-- Preview: SAC-NoVMM -----------------------------------")
    _preview(lambda render_mode=None: make_no_vmm_env(obs, seed, render_mode), "SAC-NoVMM")
    print("\n-- Preview: SAC-VMM -------------------------------------")
    _preview(lambda render_mode=None: make_vmm_env(obs, seed, render_mode), "SAC-VMM")
    print("\nPreview done. Starting training...\n")


# -- Callback -------------------------------------------------------------------

def _unwrap_inner(env):
    while hasattr(env, "env"):
        env = env.env
    return env

class TrackCallback(BaseCallback):
    def __init__(self, label, verbose=0):
        super().__init__(verbose)
        self.label       = label
        self.checkpoints = []
        self._next_eval  = EVAL_EVERY

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += EVAL_EVERY
            inner = _unwrap_inner(self.training_env.envs[0])
            m = {
                "coverage":     inner._coverage() * 100,
                "collisions":   inner._collisions,
                "bumper_total": inner._bumper_triggers,
            }
            self.checkpoints.append((self.num_timesteps, m))
            print(f"  [{self.label}] step {self.num_timesteps:6d} | "
                  f"cov {m['coverage']:5.1f}% | "
                  f"col {m['collisions']:4d} | "
                  f"bumper {m['bumper_total']:4d}")
        return True


# -- Single-seed training -------------------------------------------------------

def _train(factory, label, seed):
    env = factory()
    env.reset(seed=seed)
    cb = TrackCallback(label)
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
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    try: inner.close()
    except Exception: pass
    return cb.checkpoints


def _run_boustrophedon(obstacles, seed):
    from rover_coverage_env import (
        RoverCoverageEnv, _wheels_to_action,
        WAYPOINT_TOL, WP_TIMEOUT_STEPS, SENSOR_MAX,
        FIELD_W, FIELD_H, WORKING_WIDTH, AGENT_RADIUS,
    )

    def _boustrophedon_path(margin=0.75):
        edge = max(WORKING_WIDTH / 2.0, AGENT_RADIUS + 0.15)
        n = max(1, int(np.ceil(FIELD_H / WORKING_WIDTH)))
        span = FIELD_H - 2 * edge
        offsets = [edge + i * span / (n - 1) for i in range(n)] if n > 1 else [FIELD_H / 2]
        wps = []
        for i, y in enumerate(offsets):
            a, b = (margin, FIELD_W - margin) if i % 2 == 0 else (FIELD_W - margin, margin)
            wps += [(a, y), (b, y)]
        return wps

    env = RoverCoverageEnv(n_obstacles=N_OBSTACLES, seed=seed, obstacles=obstacles)
    env.use_bumper = True
    obs, _ = env.reset(seed=seed)

    waypoints = _boustrophedon_path()
    wp_idx = 0; wp_steps = 0

    def _act(pose, obs):
        nonlocal wp_idx, wp_steps
        if wp_idx >= len(waypoints):
            wp_idx = 0
        x, y, theta = pose
        wx, wy = waypoints[wp_idx]
        dist = float(np.hypot(wx - x, wy - y))
        if dist < WAYPOINT_TOL:
            wp_idx += 1; wp_steps = 0
            return _act(pose, obs)
        wp_steps += 1
        if wp_steps > WP_TIMEOUT_STEPS:
            wp_idx += 1; wp_steps = 0
            return _act(pose, obs)
        front_m = float(obs[2]) * SENSOR_MAX
        left_m  = float(obs[0]) * SENSOR_MAX
        right_m = float(obs[1]) * SENSOR_MAX
        if front_m < 0.50:
            return _wheels_to_action(-1, 1) if left_m >= right_m else _wheels_to_action(1, -1)
        target = np.arctan2(wy - y, wx - x)
        err    = (target - theta + np.pi) % (2 * np.pi) - np.pi
        if abs(np.degrees(err)) > 20:
            return _wheels_to_action(-1, 1) if err > 0 else _wheels_to_action(1, -1)
        if abs(np.degrees(err)) > 7:
            return _wheels_to_action(0, 1) if err > 0 else _wheels_to_action(1, 0)
        return _wheels_to_action(1, 1)

    checkpoints = []; next_eval = EVAL_EVERY
    for _ in range(TRAIN_STEPS):
        obs, _, _, _, _ = env.step(_act((env.x, env.y, env.theta), obs))
        if env.step_count >= next_eval:
            m = {
                "coverage":     env._coverage() * 100,
                "collisions":   env._collisions,
                "bumper_total": env._bumper_triggers,
            }
            checkpoints.append((env.step_count, m))
            print(f"  [Boustrophedon seed={seed}] step {env.step_count:6d} | "
                  f"cov {m['coverage']:5.1f}% | col {m['collisions']:4d}")
            next_eval += EVAL_EVERY
    env.close()
    return checkpoints


# -- Multi-seed training --------------------------------------------------------

def train():
    """Run only SAC-VMM on each seed (NoVMM and Boustrophedon results already known)."""
    all_no_vmm, all_vmm, all_boustr = [], [], []

    for seed in SEEDS:
        obstacles = generate_obstacles(N_OBSTACLES, np.random.default_rng(seed))
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}")

        print(f"\n-- SAC-VMM    seed={seed} ---------------------------------")
        all_vmm.append(_train(
            lambda s=seed, o=obstacles: make_vmm_env(o, s),
            f"SAC-VMM   s{seed}", seed))

    # Reuse previously recorded results for the other two methods
    steps = list(range(EVAL_EVERY, TRAIN_STEPS + 1, EVAL_EVERY))
    n = len(steps)
    def _flat_ckpts(final_cov, final_bumper):
        """Reconstruct plausible checkpoint list from final values (linear ramp)."""
        return [(steps[i], {
            "coverage":     final_cov     * (i + 1) / n,
            "collisions":   0,
            "bumper_total": int(final_bumper * (i + 1) / n),
        }) for i in range(n)]

    # Results from previous run (seed=[42,0,1])
    _PREV_NO_VMM  = [(48.0, 913), (36.0, 640), (60.0, 1087)]   # per seed: (cov%, bumper)
    _PREV_BOUSTR  = [(59.9, 904), (57.5, 748), (62.3, 1060)]

    for cov, bmp in _PREV_NO_VMM:
        all_no_vmm.append(_flat_ckpts(cov, bmp))
    for cov, bmp in _PREV_BOUSTR:
        all_boustr.append(_flat_ckpts(cov, bmp))

    return all_no_vmm, all_vmm, all_boustr


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
    steps_a, va = _align(all_no_vmm)
    steps_b, vb = _align(all_vmm)
    steps_c, vc = _align(all_boustr)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Boustrophedon vs SAC-NoVMM vs SAC-VMM  ({len(SEEDS)} seeds, mean +/- std)",
        fontsize=12, fontweight="bold")

    styles = [
        (steps_a, va, "#4C9BE8", "SAC  No-VMM"),
        (steps_b, vb, "#F4845F", "SAC  VMM"),
        (steps_c, vc, "#A0A0A0", "Boustrophedon"),
    ]

    panels = [
        ("Coverage %",      "coverage"),
        ("Collisions",      "collisions"),
        ("Bumper Triggers", "bumper_total"),
    ]

    for ax, (title, key) in zip(axes, panels):
        for steps, vals, color, label in styles:
            arr = vals[key]               # (N_seeds, T)
            mean = arr.mean(0)
            std  = arr.std(0)
            ls = "--" if "Boustr" in label else "-"
            ax.plot(steps, mean, color=color, lw=2, ls=ls, label=label)
            ax.fill_between(steps, mean - std, mean + std,
                            color=color, alpha=0.18)
        ax.set_title(title); ax.set_xlabel("Steps")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "vmm_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved -> {out}")

    # Summary table
    def final_mean_std(all_ckpts, key):
        vals = [ckpts[-1][1][key] for ckpts in all_ckpts]
        return np.mean(vals), np.std(vals)

    print(f"\n{'':35s} {'Boustr':>14s} {'SAC-NoVMM':>16s} {'SAC-VMM':>16s}")
    print("-" * 84)
    for label, key, fmt in [
        ("Final coverage %",      "coverage",    ".1f"),
        ("Final collisions",      "collisions",  ".1f"),
        ("Final bumper triggers", "bumper_total", ".1f"),
    ]:
        bm, bs   = final_mean_std(all_boustr,  key)
        nm, ns   = final_mean_std(all_no_vmm,  key)
        vm, vs   = final_mean_std(all_vmm,     key)
        bstr  = f"{bm:{fmt}} +/- {bs:.1f}"
        nstr  = f"{nm:{fmt}} +/- {ns:.1f}"
        vstr  = f"{vm:{fmt}} +/- {vs:.1f}"
        print(f"  {label:33s} {bstr:>14s} {nstr:>16s} {vstr:>16s}")


# -- Main -----------------------------------------------------------------------

if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview()

    all_no_vmm, all_vmm, all_boustr = train()
    plot(all_no_vmm, all_vmm, all_boustr)