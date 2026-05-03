"""
Comparison: Discrete (9 actions) vs Continuous ([curvature, speed]) action space.

Metrics per step
----------------
1. Coverage %         — cells visited / total cells
2. Collisions         — cumulative
3. Embedding distance — cosine distance between consecutive rendered-frame
                        embeddings (VMM encoder).  Measures how much the
                        rover's view changes → proxy for visual exploration.
4. Unique clusters    — k-means(k=30) on all embeddings collected so far,
                        count of populated clusters.

Both agents use the same reactive avoidance logic and run on the same map
(identical seed).  The only difference is the action parameterisation.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rover_coverage_env     import RoverCoverageEnv,    ReactiveAgent
from rover_continuous_env   import ContinuousRoverEnv,  ContinuousReactiveAgent
from VMM.vmm                import Encoder, TRANSFORM
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
N_STEPS  = 1_500
SEED     = 42
N_OBS   = 5          # obstacles
CLUSTER_K = 30
DEVICE   = torch.device("cpu")

# ── Embedding encoder (shared) ────────────────────────────────────────────────
print("Loading VMM encoder...")
encoder = Encoder().to(DEVICE).eval()


def embed(frame_rgb: np.ndarray) -> torch.Tensor:
    t = TRANSFORM(frame_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return encoder(t).squeeze(0)   # (128,)


def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def unique_clusters(embeddings: list, k: int = CLUSTER_K) -> int:
    """Simple k-means cluster count on collected embeddings."""
    if len(embeddings) < k:
        return len(embeddings)
    mat = torch.stack(embeddings).numpy()
    from sklearn.cluster import MiniBatchKMeans
    labels = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(mat)
    return int(np.unique(labels).shape[0])


# ── Runner ────────────────────────────────────────────────────────────────────

def run(env, agent, label: str, use_continuous: bool):
    obs, _ = env.reset(seed=SEED)
    env.use_bumper = False

    coverage     = []
    collisions   = []
    emb_dists    = []
    cluster_hist = []
    embeddings   = []

    prev_emb = None
    info     = {}
    collided = False

    print(f"\n── {label} ──────────────────")

    for step in range(N_STEPS):
        # Render → embed
        frame = env.render()   # rgb_array (H, W, 3)
        if frame is not None:
            emb = embed(frame)
            if prev_emb is not None:
                emb_dists.append(cosine_dist(prev_emb, emb))
            embeddings.append(emb)
            prev_emb = emb

        # Action
        if use_continuous:
            raw = agent.act(obs, collided=collided)
            action = np.array(raw, dtype=np.float32)
        else:
            action = agent.act(obs, collided=collided)

        obs, _, _, _, info = env.step(action)
        collided = info.get("collided", False)

        coverage.append(info.get("coverage", 0.0) * 100)
        collisions.append(info.get("collisions", 0))

        # Cluster count every 100 steps (expensive)
        if (step + 1) % 100 == 0 and embeddings:
            c = unique_clusters(embeddings)
            cluster_hist.append((step + 1, c))
            print(f"  step {step+1:4d} | cov {coverage[-1]:5.1f}% | "
                  f"collisions {collisions[-1]:4d} | clusters {c}")

    return {
        "label":        label,
        "coverage":     coverage,
        "collisions":   collisions,
        "emb_dists":    emb_dists,
        "cluster_hist": cluster_hist,
        "embeddings":   embeddings,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        print("scikit-learn required: pip install scikit-learn")
        sys.exit(1)

    # Discrete
    env_d  = RoverCoverageEnv(render_mode="rgb_array", n_obstacles=N_OBS, seed=SEED)
    agent_d = ReactiveAgent(seed=SEED)
    res_d   = run(env_d, agent_d, "Discrete (9 actions)", use_continuous=False)
    env_d.close()

    # Continuous
    env_c   = ContinuousRoverEnv(render_mode="rgb_array", n_obstacles=N_OBS, seed=SEED)
    agent_c = ContinuousReactiveAgent(seed=SEED)
    res_c   = run(env_c, agent_c, "Continuous [curvature, speed]", use_continuous=True)
    env_c.close()

    # ── Plots ─────────────────────────────────────────────────────────────────
    steps = np.arange(1, N_STEPS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Discrete vs Continuous Action Space", fontsize=14, fontweight="bold")

    palette = {"d": "#4C9BE8", "c": "#F4845F"}

    # 1. Coverage
    ax = axes[0, 0]
    ax.plot(steps, res_d["coverage"], color=palette["d"], label="Discrete",    lw=1.5)
    ax.plot(steps, res_c["coverage"], color=palette["c"], label="Continuous",  lw=1.5)
    ax.set_title("Coverage %");  ax.set_xlabel("Step"); ax.set_ylabel("%")
    ax.legend(); ax.grid(alpha=0.3)

    # 2. Collisions
    ax = axes[0, 1]
    ax.plot(steps, res_d["collisions"], color=palette["d"], label="Discrete",   lw=1.5)
    ax.plot(steps, res_c["collisions"], color=palette["c"], label="Continuous", lw=1.5)
    ax.set_title("Cumulative Collisions"); ax.set_xlabel("Step"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(alpha=0.3)

    # 3. Embedding distance (rolling mean)
    ax = axes[1, 0]
    w  = 30
    def smooth(x): return np.convolve(x, np.ones(w)/w, mode="valid")
    if res_d["emb_dists"]:
        ax.plot(smooth(res_d["emb_dists"]), color=palette["d"], label="Discrete",   lw=1.5)
    if res_c["emb_dists"]:
        ax.plot(smooth(res_c["emb_dists"]), color=palette["c"], label="Continuous", lw=1.5)
    ax.set_title(f"Embedding Distance (rolling {w}-step mean)")
    ax.set_xlabel("Step"); ax.set_ylabel("Cosine distance"); ax.legend(); ax.grid(alpha=0.3)

    # 4. Unique clusters
    ax = axes[1, 1]
    for res, col in [(res_d, palette["d"]), (res_c, palette["c"])]:
        if res["cluster_hist"]:
            xs, ys = zip(*res["cluster_hist"])
            ax.plot(xs, ys, "o-", color=col, label=res["label"], lw=1.5, ms=5)
    ax.set_title(f"Unique Embedding Clusters (k={CLUSTER_K})")
    ax.set_xlabel("Step"); ax.set_ylabel("Clusters"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = "action_space_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'':30s} {'Discrete':>12s} {'Continuous':>12s}")
    print("─" * 56)
    def fmt(res):
        cov   = res["coverage"][-1]
        col   = res["collisions"][-1]
        mean_d = np.mean(res["emb_dists"]) if res["emb_dists"] else 0
        clust  = res["cluster_hist"][-1][1] if res["cluster_hist"] else 0
        return cov, col, mean_d, clust

    fd = fmt(res_d); fc = fmt(res_c)
    rows = [
        ("Final coverage %",         f"{fd[0]:.1f}",   f"{fc[0]:.1f}"),
        ("Total collisions",          f"{fd[1]}",       f"{fc[1]}"),
        ("Mean embedding dist",       f"{fd[2]:.4f}",   f"{fc[2]:.4f}"),
        (f"Unique clusters (k={CLUSTER_K})", f"{fd[3]}", f"{fc[3]}"),
    ]
    for label, vd, vc in rows:
        print(f"  {label:28s} {vd:>12s} {vc:>12s}")


if __name__ == "__main__":
    main()
