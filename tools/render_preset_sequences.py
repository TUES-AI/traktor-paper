#!/usr/bin/env python3
"""Render preset sequence intent diagram."""

from pathlib import Path

import matplotlib.pyplot as plt


SEQUENCES = {
    '1': ('short straight forward', [('drive', (0, 0), (45, 0))]),
    '2': ('left lane-offset', [
        ('rotate left', (0, 0), None),
        ('drive', (0, 0), (30, 18)),
        ('rotate right', (30, 18), None),
        ('drive', (30, 18), (62, 18)),
    ]),
    '3': ('right lane-offset', [
        ('rotate right', (0, 0), None),
        ('drive', (0, 0), (30, -18)),
        ('rotate left', (30, -18), None),
        ('drive', (30, -18), (62, -18)),
    ]),
    '4': ('left obstacle-bypass', [
        ('rotate left', (0, 0), None),
        ('drive', (0, 0), (24, 20)),
        ('rotate right', (24, 20), None),
        ('drive', (24, 20), (70, 5)),
        ('rotate left', (70, 5), None),
        ('drive', (70, 5), (92, 0)),
    ]),
    '5': ('right obstacle-bypass', [
        ('rotate right', (0, 0), None),
        ('drive', (0, 0), (24, -20)),
        ('rotate left', (24, -20), None),
        ('drive', (24, -20), (70, -5)),
        ('rotate right', (70, -5), None),
        ('drive', (70, -5), (92, 0)),
    ]),
}


def draw_rover(ax):
    ax.scatter([0], [0], color='black', s=35)
    ax.arrow(0, 0, 8, 0, head_width=2.0, head_length=3.0, color='red')
    ax.text(9, 2, 'start / +x forward', color='red', fontsize=9)


def main():
    out = Path('docs/preset_sequences.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    for ax, (key, (name, segments)) in zip(axes, SEQUENCES.items()):
        draw_rover(ax)
        point_i = 0
        for label, start, end in segments:
            sx, sy = start
            if end is None:
                ax.scatter([sx], [sy], color='orange', s=55, marker='x')
                ax.text(sx + 1, sy + 4, label, color='orange', fontsize=8)
                continue
            ex, ey = end
            ax.annotate('', xy=(ex, ey), xytext=(sx, sy), arrowprops={'arrowstyle': '->', 'linewidth': 2.4})
            ax.scatter([sx, ex], [sy, ey], color='black', s=22)
            ax.text(ex + 1, ey + 1, f'drive {point_i}', fontsize=8)
            point_i += 1
        ax.set_title(f'{key}: {name}')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-15, 110)
        ax.set_ylim(-45, 45)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x forward (cm, rough intent)')
        ax.set_ylabel('y left (cm, rough intent)')
    axes[-1].axis('off')
    fig.suptitle('Preset motor sequences: rotate-in-place markers + forward drive arrows, not odometry')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
