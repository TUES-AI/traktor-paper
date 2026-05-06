#!/usr/bin/env python3
"""Plot a real PCVM rover run from JSON log lines.

The trajectory is reconstructed from execution reports, not from the model's
internal pose estimate, so stalls, zero-distance actions, and recovery turns are
visible instead of being smoothed into a clean path.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def iter_events(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def reconstruct(events):
    x = 0.0
    y = 0.0
    yaw = 0.0
    points = [(x, y)]
    colors = [0.0]
    stuck = []
    recoveries = []
    new_places = []

    for ev in events:
        step = int(ev.get('step') or len(points))
        backend = ev.get('backend') or {}
        novelty = float(backend.get('pcvm_novelty') or backend.get('predictive_novelty') or 0.0)
        if backend.get('pcvm_new_cluster'):
            new_places.append((x, y, step))

        recovery = ev.get('recovery')
        if recovery:
            report = recovery.get('recovery') or {}
            yaw += math.radians(float(report.get('yaw_deg') or 0.0))
            recoveries.append((x, y, step))

        execution = ev.get('execution')
        if not execution:
            stuck.append((x, y, step))
            points.append((x, y))
            colors.append(novelty)
            continue

        turn = execution.get('turn') or {}
        yaw += math.radians(float(turn.get('yaw_deg') or 0.0))
        distance_m = max(0.0, float(execution.get('clipped_distance_cm') or 0.0)) / 100.0
        reason = str(execution.get('reason') or '') + ' ' + str((execution.get('drive') or {}).get('reason') or '')
        if distance_m <= 0.01 or 'front_safety_stop' in reason or 'distance_clipped_to_zero' in reason:
            stuck.append((x, y, step))

        x += math.cos(yaw) * distance_m
        y += math.sin(yaw) * distance_m
        points.append((x, y))
        colors.append(novelty)

    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32), stuck, recoveries, new_places


def main():
    parser = argparse.ArgumentParser(description='Plot real PCVM run trajectory from log JSON.')
    parser.add_argument('log', type=Path)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    points, novelty, stuck, recoveries, new_places = reconstruct(list(iter_events(args.log)))
    if len(points) < 2:
        raise SystemExit('No rover step JSON events found in log')

    out = args.out or args.log.with_suffix('.trajectory.png')
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(1, len(points)):
        ax.plot(points[i - 1:i + 1, 0], points[i - 1:i + 1, 1], color=plt.cm.plasma(float(novelty[i])), lw=2.0)
    sc = ax.scatter(points[:, 0], points[:, 1], c=novelty, cmap='plasma', s=22, zorder=3)
    fig.colorbar(sc, ax=ax, label='PCVM novelty')

    if stuck:
        xs, ys, _steps = zip(*stuck)
        ax.scatter(xs, ys, marker='x', s=55, color='red', label='stuck / no drive', zorder=4)
    if recoveries:
        xs, ys, _steps = zip(*recoveries)
        ax.scatter(xs, ys, marker='^', s=60, color='black', label='recovery turn', zorder=4)
    if new_places:
        xs, ys, steps = zip(*new_places)
        ax.scatter(xs, ys, marker='*', s=120, color='limegreen', edgecolor='black', label='new cluster', zorder=5)
        for x, y, step in new_places:
            ax.annotate(str(step), (x, y), fontsize=8, xytext=(4, 4), textcoords='offset points')

    ax.scatter([0.0], [0.0], marker='o', s=90, color='white', edgecolor='black', label='start', zorder=6)
    ax.set_title('Real Rover Trajectory From Executed Motion')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.axis('equal')
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == '__main__':
    main()
