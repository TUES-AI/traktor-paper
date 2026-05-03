"""
SAC comparison: Discrete (DQN) vs Continuous (SAC)
----------------------------------------------------
Discrete   : SB3 DQN  — off-policy, discrete action space (9 actions)
Continuous : SB3 SAC  — off-policy, continuous [curvature, speed]

Both envs use the same map seed, sensor observations, and reward signal.
Coverage % and embedding distance (VMM encoder on rendered frames) are
tracked every EVAL_EVERY steps via a custom callback.

Run:
    python3 train_sac.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from rover_coverage_env   import RoverCoverageEnv
from rover_continuous_env import ContinuousRoverEnv
from VMM.vmm              import Encoder, TRANSFORM

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
N_OBSTACLES = 5
TRAIN_STEPS = 60_000
EVAL_EVERY  = 3_000    # steps between evaluations
EVAL_STEPS  = 500      # steps per evaluation episode
BUFFER_SIZE = 50_000
BATCH_SIZE  = 256
LR          = 3e-4
GAMMA       = 0.99
HIDDEN      = [64, 64]

# ── Shared VMM encoder for embedding metrics ──────────────────────────────────
print("Loading VMM encoder...")
_encoder = Encoder().eval()

def embed(frame_rgb: np.ndarray) -> torch.Tensor:
    t = TRANSFORM(frame_rgb).unsqueeze(0)
    with torch.no_grad():
        return _encoder(t).squeeze(0)

def emb_dist(a, b) -> float:
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, EnvCls, is_continuous: bool) -> dict:
    env = EnvCls(render_mode="rgb_array", n_obstacles=N_OBSTACLES, seed=SEED)
    env.use_bumper = False
    obs, _ = env.reset(seed=SEED)

    prev_emb  = None
    emb_dists = []
    info      = {}

    for _ in range(EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)

        frame = env.render()
        if frame is not None:
            e = embed(frame)
            if prev_emb is not None:
                emb_dists.append(emb_dist(prev_emb, e))
            prev_emb = e

    env.close()
    return {
        "coverage":   info.get("coverage", 0.0) * 100,
        "collisions": info.get("collisions", 0),
        "mean_emb_dist": float(np.mean(emb_dists)) if emb_dists else 0.0,
    }


# ── Callback ──────────────────────────────────────────────────────────────────

class TrackCallback(BaseCallback):
    """Evaluates the policy every EVAL_EVERY steps and records metrics."""

    def __init__(self, EnvCls, is_continuous, label, verbose=0):
        super().__init__(verbose)
        self.EnvCls       = EnvCls
        self.is_continuous = is_continuous
        self.label        = label
        self.checkpoints  = []   # (step, coverage, collisions, mean_emb_dist)
        self._next_eval   = EVAL_EVERY

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += EVAL_EVERY
            metrics = evaluate(self.model, self.EnvCls, self.is_continuous)
            self.checkpoints.append((self.num_timesteps, metrics))
            print(f"  [{self.label}] step {self.num_timesteps:6d} | "
                  f"cov {metrics['coverage']:5.1f}% | "
                  f"col {metrics['collisions']:4d} | "
                  f"emb_dist {metrics['mean_emb_dist']:.4f}")
        return True


# ── Training ──────────────────────────────────────────────────────────────────

def train_discrete():
    print("\n── Training DQN (Discrete, 9 actions) ──────────────────")
    env = RoverCoverageEnv(n_obstacles=N_OBSTACLES, seed=SEED)
    env.use_bumper = False
    env.reset(seed=SEED)

    cb = TrackCallback(RoverCoverageEnv, False, "DQN-Discrete")

    model = DQN(
        "MlpPolicy", env,
        learning_rate=LR,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=1_000,
        train_freq=1,
        target_update_interval=500,
        policy_kwargs={"net_arch": HIDDEN},
        verbose=0,
        seed=SEED,
    )
    model.learn(total_timesteps=TRAIN_STEPS, callback=cb)
    env.close()
    return cb.checkpoints


def train_continuous():
    print("\n── Training SAC (Continuous [curvature, speed]) ─────────")
    env = ContinuousRoverEnv(n_obstacles=N_OBSTACLES, seed=SEED)
    env.use_bumper = False
    env.reset(seed=SEED)

    cb = TrackCallback(ContinuousRoverEnv, True, "SAC-Continuous")

    model = SAC(
        "MlpPolicy", env,
        learning_rate=LR,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=1_000,
        train_freq=1,
        policy_kwargs={"net_arch": HIDDEN},
        verbose=0,
        seed=SEED,
    )
    model.learn(total_timesteps=TRAIN_STEPS, callback=cb)
    env.close()
    return cb.checkpoints


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(disc_ckpts, cont_ckpts):
    def unpack(ckpts, key):
        steps = [s for s, _ in ckpts]
        vals  = [m[key] for _, m in ckpts]
        return steps, vals

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("DQN (Discrete) vs SAC (Continuous)", fontsize=13, fontweight="bold")

    pd = {"color": "#4C9BE8", "label": "DQN  Discrete (9 actions)", "lw": 2}
    pc = {"color": "#F4845F", "label": "SAC  Continuous [curv, speed]", "lw": 2}

    titles  = ["Coverage %", "Collisions", "Mean Embedding Distance"]
    keys    = ["coverage", "collisions", "mean_emb_dist"]
    ylabels = ["%", "count", "cosine dist"]

    for ax, title, key, ylabel in zip(axes, titles, keys, ylabels):
        xs_d, ys_d = unpack(disc_ckpts, key)
        xs_c, ys_c = unpack(cont_ckpts, key)
        ax.plot(xs_d, ys_d, **pd)
        ax.plot(xs_c, ys_c, **pc)
        ax.set_title(title); ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "sac_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    # Summary
    def last(ckpts, key): return ckpts[-1][1][key] if ckpts else 0
    print(f"\n{'':35s} {'DQN-Discrete':>14s} {'SAC-Continuous':>14s}")
    print("─" * 65)
    rows = [
        ("Final coverage %",      f"{last(disc_ckpts,'coverage'):.1f}",
                                   f"{last(cont_ckpts,'coverage'):.1f}"),
        ("Final collisions",       f"{last(disc_ckpts,'collisions')}",
                                   f"{last(cont_ckpts,'collisions')}"),
        ("Final mean emb dist",    f"{last(disc_ckpts,'mean_emb_dist'):.4f}",
                                   f"{last(cont_ckpts,'mean_emb_dist'):.4f}"),
    ]
    for label, vd, vc in rows:
        print(f"  {label:33s} {vd:>14s} {vc:>14s}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    disc_ckpts = train_discrete()
    cont_ckpts = train_continuous()
    plot(disc_ckpts, cont_ckpts)
