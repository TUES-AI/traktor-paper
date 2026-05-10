#!/usr/bin/env python3
"""Interactively draw a rough real-life rover trajectory over an image.

Controls:
  left click  : add point
  right click : undo last point
  c           : clear
  s           : save PNG + JSON
  q / escape  : save and quit
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        default='paper/data/hero_placeholder/raw/final_run_map.jpg',
        help='Background image path',
    )
    parser.add_argument(
        '--out-prefix',
        default='paper/data/pcvm_theta_front_coverage_resume100_from150/figures/manual_ir_trace',
        help='Output prefix; writes .png and .json',
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    img = ImageOps.exif_transpose(Image.open(image_path).convert('RGB'))
    w, h = img.size
    points = []

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title('Draw IRL trajectory: left=add, right=undo, s=save, c=clear, q=save+quit')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    line, = ax.plot([], [], '-o', color='red', linewidth=3, markersize=5, alpha=0.9)
    start = ax.scatter([], [], s=120, c='lime', edgecolors='black', zorder=5)
    end = ax.scatter([], [], s=160, c='black', marker='*', zorder=5)
    label = ax.text(10, 25, '', color='white', fontsize=11, bbox=dict(facecolor='black', alpha=0.55))

    def redraw():
        if points:
            xs, ys = zip(*points)
            line.set_data(xs, ys)
            start.set_offsets([[xs[0], ys[0]]])
            end.set_offsets([[xs[-1], ys[-1]]])
        else:
            line.set_data([], [])
            empty = np.empty((0, 2))
            start.set_offsets(empty)
            end.set_offsets(empty)
        label.set_text(f'{len(points)} points | left add, right undo, s save, c clear, q quit')
        fig.canvas.draw_idle()

    def save():
        png = out_prefix.with_suffix('.png')
        js = out_prefix.with_suffix('.json')
        data = {
            'background_image': str(image_path),
            'points_px': [{'x': float(x), 'y': float(y)} for x, y in points],
            'controls': 'left add, right undo, c clear, s save, q save+quit',
        }
        js.write_text(json.dumps(data, indent=2))
        fig.savefig(png, dpi=220, bbox_inches='tight', pad_inches=0.02)
        print(f'saved {png}')
        print(f'saved {js}')

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            points.append((event.xdata, event.ydata))
        elif event.button == 3 and points:
            points.pop()
        redraw()

    def onkey(event):
        if event.key == 's':
            save()
        elif event.key == 'c':
            points.clear()
            redraw()
        elif event.key in ('q', 'escape'):
            save()
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    redraw()
    plt.show()


if __name__ == '__main__':
    main()
