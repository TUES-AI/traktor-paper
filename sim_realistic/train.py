from __future__ import annotations

import argparse
import csv
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from sim_realistic.baselines import DeterministicExplorer
from sim_realistic.env import RealisticRoverEnv
from sim_realistic.wrappers import SACVMMWrapper, PredictiveVMMWrapper, DEVICE


OUT = pathlib.Path("sim_realistic/results")


def make_env(method: str, seed: int, max_steps: int):
    env = RealisticRoverEnv(seed=seed, max_steps=max_steps)
    if method == "sac_vmm":
        env = SACVMMWrapper(env)
    elif method == "predictive_sac":
        env = PredictiveVMMWrapper(env)
    return env


def unwrap(env):
    while hasattr(env, "env"):
        env = env.env
    return env


class MetricsCallback(BaseCallback):
    def __init__(self, label: str, every: int = 1000):
        super().__init__()
        self.label = label
        self.every = every
        self.rows = []
        self.t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            inner = unwrap(self.training_env.envs[0])
            row = {
                "step": self.num_timesteps,
                "coverage": inner.visited.mean(),
                "rooms_seen": len(inner.rooms_seen),
                "door_crossings": inner.door_crossings,
                "collisions": inner.collisions,
                "elapsed": time.time() - self.t0,
            }
            self.rows.append(row)
            print(self.label, row, flush=True)
        return True


def run_deterministic(seed: int, steps: int):
    env = RealisticRoverEnv(seed=seed, max_steps=steps)
    agent = DeterministicExplorer(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    for step in range(1, steps + 1):
        obs, _, _, trunc, _ = env.step(agent.act(obs))
        if step % 1000 == 0 or trunc:
            rows.append({
                "step": step,
                "coverage": env.visited.mean(),
                "rooms_seen": len(env.rooms_seen),
                "door_crossings": env.door_crossings,
                "collisions": env.collisions,
                "elapsed": 0.0,
            })
        if trunc:
            break
    return rows


def train_method(method: str, seed: int, steps: int, max_episode_steps: int):
    env = make_env(method, seed, max_episode_steps)
    cb = MetricsCallback(method)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=1000,
        batch_size=512,
        gamma=0.99,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
        seed=seed,
        device=DEVICE,
    )
    model.learn(total_timesteps=steps, callback=cb, progress_bar=True)
    model.save(OUT / f"{method}_seed{seed}.zip")
    return cb.rows


def save_rows(label: str, seed: int, rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}_seed{seed}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "coverage", "rooms_seen", "door_crossings", "collisions", "elapsed"])
        writer.writeheader(); writer.writerows(rows)
    return path


def plot(labels: list[str], seed: int):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ["coverage", "rooms_seen", "door_crossings", "collisions"]
    for label in labels:
        path = OUT / f"{label}_seed{seed}.csv"
        if not path.exists():
            continue
        arr = np.genfromtxt(path, delimiter=",", names=True)
        if arr.ndim == 0:
            continue
        for ax, metric in zip(axes, metrics):
            ax.plot(arr["step"], arr[metric], label=label)
            ax.set_title(metric); ax.grid(alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Realistic local-only sim: no oracle pose/coverage in policy input")
    plt.tight_layout()
    out = OUT / f"comparison_seed{seed}.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+", default=["deterministic", "sac_novmm", "sac_vmm", "predictive_sac"])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = []
    if "deterministic" in args.methods:
        rows = run_deterministic(args.seed, args.steps)
        save_rows("deterministic", args.seed, rows)
        done.append("deterministic")
    for method in args.methods:
        if method == "deterministic":
            continue
        episode_steps = args.episode_steps if args.episode_steps is not None else args.steps
        rows = train_method(method, args.seed, args.steps, episode_steps)
        save_rows(method, args.seed, rows)
        done.append(method)
    plot(done, args.seed)


if __name__ == "__main__":
    main()
