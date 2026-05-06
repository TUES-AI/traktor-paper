"""Analyze manual-drive frame logs with VMM novelty and cluster memory.

Usage:
    python tools/analyze_vmm_manual_drive.py --frames path/to/frames --out results/manual_vmm
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from VMM.vmm import VMM


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _frame_paths(frame_dir):
    return sorted(p for p in Path(frame_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _pca2(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def analyze(frame_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vmm = VMM()
    rows = []
    embeddings = []
    paths = _frame_paths(frame_dir)
    if not paths:
        raise SystemExit(f"No image frames found in {frame_dir}")

    for idx, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = vmm.observe(frame)
        embedding = vmm._embed(frame).squeeze(0).detach().cpu().numpy()
        embeddings.append(embedding)
        rows.append({
            "idx": idx,
            "file": path.name,
            "novelty": result["novelty"],
            "rnd_norm": result["rnd_norm"],
            "mem_dist": result["mem_dist"],
            "mem_norm": result["mem_norm"],
            "cluster_id": -1 if result["cluster_id"] is None else result["cluster_id"],
            "new_cluster": int(result["new_cluster"]),
            "cluster_count": result["bank_size"],
            "smooth_window": result["smooth_window"],
            "is_novel": int(result["is_novel"]),
        })

    csv_path = out_dir / "manual_vmm_frames.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(rows))
    novelty = np.array([r["novelty"] for r in rows], dtype=float)
    clusters = np.array([r["cluster_count"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, novelty, lw=1.5)
    axes[0].set_title("VMM Combined Novelty Over Manual Drive")
    axes[0].set_ylabel("Novelty")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, clusters, lw=1.5, color="#555555")
    axes[1].set_title("Visual-Place Clusters Discovered")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Clusters")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "manual_vmm_timeseries.png", dpi=150)
    plt.close(fig)

    if len(embeddings) >= 3:
        coords = _pca2(np.vstack(embeddings))
        cluster_ids = np.array([r["cluster_id"] for r in rows])
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=cluster_ids, cmap="tab20", s=16)
        ax.set_title("MobileNet Embeddings PCA, Colored by VMM Cluster")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="cluster_id")
        plt.tight_layout()
        fig.savefig(out_dir / "manual_vmm_embedding_pca.png", dpi=150)
        plt.close(fig)

    print(f"Saved {csv_path}")
    print(f"Saved plots in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Directory of ordered camera frames.")
    parser.add_argument("--out", default="results/manual_vmm", help="Output directory.")
    args = parser.parse_args()
    analyze(args.frames, args.out)


if __name__ == "__main__":
    main()
