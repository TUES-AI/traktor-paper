#!/usr/bin/env python3
"""Run VMM (RND + clusters + temporal smoothing) on a directory of JPG frames.

Usage:
    python tools/run_vmm_on_frames.py --frames data/manual_runs/wasd_20260506_174217/frames
    python tools/run_vmm_on_frames.py --frames data/manual_runs/wasd_20260506_174217/frames --save results/vmm_run.mp4
    python tools/run_vmm_on_frames.py --frames data/manual_runs/wasd_20260506_174217/frames --fps 10 --save results/vmm_run.mp4

Controls (live window):
    SPACE  — pause / resume
    LEFT   — step back one frame  (when paused)
    RIGHT  — step forward one frame (when paused)
    Q / ESC — quit
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VMM.vmm import VMM, draw_overlay

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _frame_paths(frame_dir):
    return sorted(p for p in Path(frame_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Directory of ordered JPG frames.")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback speed (default: 5 = real-time capture rate).")
    parser.add_argument("--save", default=None, help="Optional path to save output video (e.g. results/vmm_run.mp4).")
    parser.add_argument("--no-display", action="store_true", help="Skip live window (only useful with --save).")
    parser.add_argument("--plot", default="results/vmm_clusters.png", help="Save cluster plot to this path.")
    args = parser.parse_args()

    paths = _frame_paths(args.frames)
    if not paths:
        raise SystemExit(f"No image frames found in {args.frames}")

    print(f"Loaded {len(paths)} frames from {args.frames}")
    print("Running VMM — press SPACE to pause, LEFT/RIGHT to step, Q to quit\n")

    vmm = VMM()

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        sample = cv2.imread(str(paths[0]))
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (w, h))
        print(f"Saving video -> {args.save}")

    # Pre-run VMM on all frames so we can scrub back/forward
    results = []
    embeddings = []
    print("Processing frames...")
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        r = vmm.observe(frame)
        with torch.no_grad():
            z = vmm._embed(frame).squeeze(0).cpu().numpy()
        embeddings.append(z)
        results.append((frame, r))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)} | bank={r['bank_size']} | novelty={r['novelty']:.3f}")

    print(f"\nDone. Bank size: {len(vmm.memory.bank)} clusters from {len(results)} frames.")
    print(f"Novel frames: {sum(1 for _, r in results if r['is_novel'])} / {len(results)}\n")

    delay_ms = max(1, int(1000 / args.fps))
    paused = False
    idx = 0

    while idx < len(results):
        frame, r = results[idx]
        history = [rr["mem_dist"] for _, rr in results[:idx+1]]
        vis = draw_overlay(frame, r, history)
        cv2.putText(vis, f"frame {idx+1}/{len(results)}", (vis.shape[1]-200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        if writer:
            writer.write(vis)

        if args.no_display:
            idx += 1
            continue

        cv2.imshow("VMM — frame replay", vis)
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 81 and paused:   # LEFT arrow
            idx = max(0, idx - 1)
        elif paused:
            pass  # hold on current frame
        else:
            idx += 1

    if writer:
        writer.release()
        print(f"Saved -> {args.save}")

    if not args.no_display:
        cv2.destroyAllWindows()

    novel_count = sum(1 for _, r in results if r["is_novel"])
    print(f"\n── Summary ───────────────────────────────")
    print(f"Total frames : {len(results)}")
    print(f"Novel frames : {novel_count}  ({100*novel_count/max(len(results),1):.1f}%)")
    print(f"Clusters     : {len(vmm.memory.bank)}")
    print(f"RND mean     : {vmm.rnd_norm.mean:.4f}")

    if args.plot and embeddings:
        _plot_clusters(vmm, results, embeddings, args.plot)


def _pca2(X):
    X = np.asarray(X, dtype=np.float32)
    X -= X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T, Vt[:2]


def _plot_clusters(vmm, results, embeddings, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cluster_ids  = np.array([r["cluster_id"] if r["cluster_id"] is not None else -1
                             for _, r in results])
    novelty_vals = np.array([r["novelty"]   for _, r in results])
    mem_dists    = np.array([r["mem_dist"]  for _, r in results])
    bank_sizes   = np.array([r["bank_size"] for _, r in results])
    is_novel     = np.array([r["is_novel"]  for _, r in results])
    frames_x     = np.arange(len(results))

    # PCA on frame embeddings; project cluster centroids into same space
    emb_matrix = np.vstack(embeddings)
    coords, Vt = _pca2(emb_matrix)
    mean_emb   = emb_matrix.mean(axis=0)

    centroids = np.vstack([c.cpu().numpy() for c in vmm.memory.bank])
    cent_coords = (centroids - mean_emb) @ Vt.T

    n_clusters  = len(vmm.memory.bank)
    cmap        = plt.cm.get_cmap("tab20", max(n_clusters, 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"VMM cluster analysis — {len(results)} frames, {n_clusters} clusters, "
        f"{int(is_novel.sum())} novel ({100*is_novel.mean():.1f}%)",
        fontweight="bold", fontsize=13,
    )

    # ── Panel 1: PCA of all frame embeddings colored by cluster ──────────────
    ax = axes[0, 0]
    for cid in range(n_clusters):
        mask = cluster_ids == cid
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=cmap(cid), s=12, alpha=0.5, linewidths=0)
    # centroids as large markers
    for cid, (cx, cy) in enumerate(cent_coords):
        ax.scatter(cx, cy, color=cmap(cid), s=120, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=5)
    ax.set_title("Frame embeddings PCA (colour = cluster, ★ = centroid)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)

    # ── Panel 2: novelty + mem_dist over time, novel frames highlighted ───────
    ax = axes[0, 1]
    ax.fill_between(frames_x, 0, is_novel.astype(float) * 1.05,
                    color="#f87171", alpha=0.25, label="novel frame")
    ax.plot(frames_x, mem_dists, lw=1.2, color="#3b82f6", label="mem dist")
    ax.plot(frames_x, novelty_vals, lw=1.2, color="#f59e0b", alpha=0.8, label="combined novelty")
    ax.axhline(0.12, color="#3b82f6", lw=0.8, ls="--", alpha=0.6, label="mem threshold")
    ax.axhline(0.50, color="#f59e0b", lw=0.8, ls="--", alpha=0.6, label="novelty threshold")
    ax.set_title("Novelty signals over time")
    ax.set_xlabel("Frame"); ax.set_ylabel("Score")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ── Panel 3: cluster growth over time ─────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(frames_x, bank_sizes, lw=2, color="#10b981")
    ax.set_title("Clusters discovered over time")
    ax.set_xlabel("Frame"); ax.set_ylabel("Cluster count")
    ax.grid(alpha=0.25)

    # ── Panel 4: cluster size distribution ────────────────────────────────────
    ax = axes[1, 1]
    counts = vmm.memory.bank and [vmm.memory.counts[i] for i in range(n_clusters)] or []
    if counts:
        sorted_counts = sorted(counts, reverse=True)
        bar_colors = [cmap(i) for i in range(n_clusters)]
        ax.bar(range(n_clusters), sorted_counts, color=bar_colors, edgecolor="none", width=1.0)
    ax.set_title("Frames per cluster (sorted)")
    ax.set_xlabel("Cluster rank"); ax.set_ylabel("Frame count")
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Cluster plot saved -> {out_path}")


if __name__ == "__main__":
    main()
