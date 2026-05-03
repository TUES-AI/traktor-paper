#!/usr/bin/env python3
"""Live top-down ultrasonic visualization over SSH.

Run from the Mac:
    /Volumes/SSD/v/py/bin/python tools/visualize_rover_ultrasonic.py
"""

import argparse
import json
import math
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROVER_LENGTH = 16.5
BODY_WIDTH = 8.5
TRACK_WIDTH = 4.5
TOTAL_WIDTH = BODY_WIDTH + 2 * TRACK_WIDTH
X_BACK = -ROVER_LENGTH / 2
X_FRONT = ROVER_LENGTH / 2

SENSORS = {
    'front': {'pos': (X_FRONT + 0.6, 0.0), 'yaw_deg': 0, 'color': 'red'},
    'left': {'pos': (X_BACK + 14.5, BODY_WIDTH / 2 + 0.6), 'yaw_deg': 90, 'color': 'green'},
    'right': {'pos': (X_BACK + 14.5, -BODY_WIDTH / 2 - 0.6), 'yaw_deg': -90, 'color': 'green'},
}


def start_ssh_stream(hz):
    remote = (
        'cd /home/yasen/traktor-paper; '
        f'PYTHONUNBUFFERED=1 python3 embedded/scripts/stream_ultrasonic_json.py --hz {hz}'
    )
    return subprocess.Popen(
        ['ssh', 'rover', remote],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )


def draw_rover(ax):
    ax.add_patch(Rectangle((X_BACK, -TOTAL_WIDTH / 2), ROVER_LENGTH, TOTAL_WIDTH, fill=False, edgecolor='black', linewidth=2))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2), ROVER_LENGTH, BODY_WIDTH, fill=True, color='#3b82f6', alpha=0.15))
    ax.add_patch(Rectangle((X_BACK, BODY_WIDTH / 2), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.35))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2 - TRACK_WIDTH), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.35))
    ax.arrow(0, 0, 5, 0, head_width=0.7, head_length=1.0, color='red')
    ax.text(5.4, 0.4, '+x forward', color='red')


def ray_endpoint(sensor, distance_cm):
    x, y = sensor['pos']
    yaw = math.radians(sensor['yaw_deg'])
    distance = 120.0 if distance_cm is None else min(float(distance_cm), 120.0)
    return x + math.cos(yaw) * distance, y + math.sin(yaw) * distance


def redraw(ax, fig, latest):
    ax.clear()
    draw_rover(ax)
    title_parts = []
    for name, sensor in SENSORS.items():
        value = latest.get(name)
        sx, sy = sensor['pos']
        ex, ey = ray_endpoint(sensor, value)
        color = sensor['color'] if value is not None else 'orange'
        ax.plot([sx, ex], [sy, ey], color=color, linewidth=3)
        ax.scatter([sx], [sy], color=color, s=40)
        label = 'NO_ECHO' if value is None else f'{value:.1f}cm'
        ax.text(ex, ey, f'{name}: {label}', color=color, fontsize=10)
        title_parts.append(f'{name}={label}')
    ax.set_title(' | '.join(title_parts))
    ax.set_xlabel('x forward (cm)')
    ax.set_ylabel('y left (cm)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-25, 140)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.25)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def main():
    parser = argparse.ArgumentParser(description='Visualize rover ultrasonic rays live over SSH.')
    parser.add_argument('--hz', type=float, default=5.0)
    args = parser.parse_args()

    proc = start_ssh_stream(args.hz)
    fig, ax = plt.subplots(num='Rover Ultrasonic Topdown')
    plt.ion()
    try:
        while plt.fignum_exists(fig.number):
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                plt.pause(0.02)
                continue
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue
            redraw(ax, fig, latest)
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
