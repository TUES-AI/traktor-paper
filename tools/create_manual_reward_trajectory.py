#!/usr/bin/env python3
"""Create a clean white-background trajectory sketch from hand-clicked points.

The click JSON is produced by tools/draw_manual_trajectory.py. This script
stretches the real per-step reward sequence over the manually drawn path
segments and colors each segment by average reward.

This figure is intentionally a paper placeholder: replace the hand-drawn path
with a cleaner map/trajectory from a stronger repeated run before final submit.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def load_rewards(path: Path) -> np.ndarray:
    rewards = []
    for line in path.read_text(errors='ignore').splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if 'reward' in obj:
            rewards.append(float(obj.get('reward') or 0.0))
    return np.asarray(rewards, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trace-json',
        default='paper/data/pcvm_theta_front_coverage_resume100_from150/figures/manual_ir_trace.json',
    )
    parser.add_argument(
        '--run-log',
        default='results/pcvm_theta_front_coverage_resume100_from150.jsonl',
    )
    parser.add_argument(
        '--out',
        default='paper/data/pcvm_theta_front_coverage_resume100_from150/figures/manual_ir_trace_white_reward_avg.png',
    )
    args = parser.parse_args()

    trace = json.loads(Path(args.trace_json).read_text())
    pts = np.asarray([[p['x'], p['y']] for p in trace['points_px']], dtype=float)
    x = pts[:, 0] - pts[0, 0]
    y = -(pts[:, 1] - pts[0, 1])
    scale = max(np.ptp(x), np.ptp(y)) or 1.0
    xy = np.column_stack([x / scale, y / scale])
    segments = np.stack([xy[:-1], xy[1:]], axis=1)

    rewards = load_rewards(Path(args.run_log))
    if len(rewards) == 0:
        rewards = np.zeros(len(segments))

    edges = np.linspace(0, len(rewards), len(segments) + 1)
    seg_reward = []
    for i in range(len(segments)):
        a = int(np.floor(edges[i]))
        b = int(np.ceil(edges[i + 1]))
        a = max(0, min(a, len(rewards) - 1))
        b = max(a + 1, min(b, len(rewards)))
        seg_reward.append(float(np.mean(rewards[a:b])))
    seg_reward = np.asarray(seg_reward)
    color_scale = max(float(np.percentile(np.abs(seg_reward), 90)), 0.25)

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=260)
    ax.set_facecolor('white')
    lc = LineCollection(
        segments,
        cmap='RdYlGn',
        norm=plt.Normalize(vmin=-color_scale, vmax=color_scale),
    )
    lc.set_array(seg_reward)
    lc.set_linewidth(4.2)
    lc.set_capstyle('butt')
    lc.set_joinstyle('miter')
    ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], s=13, c='white', edgecolors='black', linewidths=0.55, zorder=4)

    labels = [
        (0, 'start', 'limegreen', 'o', 115, (0.025, 0.025)),
        (11, 'corridor 1', '#d95f02', 'o', 95, (0.025, 0.025)),
        (23, 'room 1', '#1b9e77', 'o', 95, (0.025, 0.025)),
        # Pushed downward so labels do not overlap the path.
        (35, 'corridor 2', '#7570b3', 'o', 95, (0.025, -0.075)),
        (len(xy) - 1, 'room 2', 'black', '*', 160, (0.025, -0.075)),
    ]
    for idx, label, color, marker, size, offset in labels:
        idx = min(max(idx, 0), len(xy) - 1)
        ax.scatter([xy[idx, 0]], [xy[idx, 1]], s=size, c=color, edgecolors='black', marker=marker, zorder=6)
        ax.text(
            xy[idx, 0] + offset[0],
            xy[idx, 1] + offset[1],
            label,
            fontsize=9,
            weight='bold',
            color=color if color != 'black' else 'black',
        )

    cbar = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('avg reward', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='box')
    margin = 0.08
    ax.set_xlim(np.min(xy[:, 0]) - margin, np.max(xy[:, 0]) + margin)
    ax.set_ylim(np.min(xy[:, 1]) - margin, np.max(xy[:, 1]) + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.05)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
