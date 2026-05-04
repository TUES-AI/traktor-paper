#!/usr/bin/env python3
"""Click two rover-relative points, convert to `[theta1,d1,theta2,d2]`, optionally execute on rover."""

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


def normalize_angle(deg):
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def points_to_guide(points):
    if len(points) != 2:
        return None
    (x1, y1), (x2, y2) = points
    theta1 = math.degrees(math.atan2(y1, x1))
    d1 = math.hypot(x1, y1)
    vx2 = x2 - x1
    vy2 = y2 - y1
    theta2_abs = math.degrees(math.atan2(vy2, vx2))
    d2 = math.hypot(vx2, vy2)
    theta2 = normalize_angle(theta2_abs - theta1)
    return theta1, d1, theta2, d2


def start_ssh_stream(hz, timeout):
    remote = (
        'cd /home/yasen/traktor-paper; '
        f'PYTHONUNBUFFERED=1 python3 embedded/scripts/stream_ultrasonic_json.py --hz {hz} --timeout {timeout}'
    )
    return subprocess.Popen(
        ['ssh', 'rover', remote],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )


def draw_rover(ax):
    ax.add_patch(Rectangle((X_BACK, -TOTAL_WIDTH / 2), ROVER_LENGTH, TOTAL_WIDTH, fill=False, edgecolor='black', linewidth=1.5))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2), ROVER_LENGTH, BODY_WIDTH, fill=True, color='#3b82f6', alpha=0.12))
    ax.add_patch(Rectangle((X_BACK, BODY_WIDTH / 2), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.18))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2 - TRACK_WIDTH), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.18))
    ax.arrow(0, 0, 8, 0, head_width=1.2, head_length=1.8, color='red')
    ax.text(8.5, 1.0, '+x forward', color='red')


def ray_endpoint(sensor, distance_cm):
    x, y = sensor['pos']
    yaw = math.radians(sensor['yaw_deg'])
    distance = 120.0 if distance_cm is None else min(float(distance_cm), 120.0)
    return x + math.cos(yaw) * distance, y + math.sin(yaw) * distance


def execute_guide(args, guide):
    theta1, d1, theta2, d2 = guide
    remote = (
        'cd /home/yasen/traktor-paper; '
        'PYTHONUNBUFFERED=1 python3 embedded/scripts/execute_two_vector_guide.py '
        f'--theta1 {theta1:.4f} --d1 {d1:.4f} --theta2 {theta2:.4f} --d2 {d2:.4f} '
        f'--speed {args.speed:.4f} --turn-speed {args.turn_speed:.4f} --cm-per-second {args.cm_per_second:.4f}'
    )
    print('executing:', remote, flush=True)
    proc = subprocess.run(['ssh', 'rover', remote], text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)
    print(f'exit_code={proc.returncode}', flush=True)


def redraw(ax, fig, points, latest, view_range_cm):
    ax.clear()
    draw_rover(ax)

    title_parts = []
    for name, sensor in SENSORS.items():
        value = latest.get(name)
        sx, sy = sensor['pos']
        ex, ey = ray_endpoint(sensor, value)
        color = sensor['color'] if value is not None else 'orange'
        ax.plot([sx, ex], [sy, ey], color=color, linewidth=3, alpha=0.8)
        ax.scatter([sx], [sy], color=color, s=40)
        label = 'NO_ECHO' if value is None else f'{value:.1f}cm'
        ax.text(ex, ey, f'{name}: {label}', color=color, fontsize=9)
        title_parts.append(f'{name}={label}')

    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, color='black', s=70, zorder=10)
        for idx, (x, y) in enumerate(points, start=1):
            ax.text(x + 1.5, y + 1.5, f'P{idx}', color='black', fontsize=11)
        path_x = [0.0] + [p[0] for p in points]
        path_y = [0.0] + [p[1] for p in points]
        ax.plot(path_x, path_y, color='black', linewidth=2, linestyle='--')

    guide = points_to_guide(points)
    if guide:
        theta1, d1, theta2, d2 = guide
        ax.text(
            0.02,
            0.98,
            f'[theta1,d1,theta2,d2] = [{theta1:+.1f}, {d1:.1f}, {theta2:+.1f}, {d2:.1f}]\npress e to execute, c to clear, q to quit',
            transform=ax.transAxes,
            va='top',
            ha='left',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'black'},
        )
    else:
        ax.text(
            0.02,
            0.98,
            'click two target points relative to rover center\n+x forward, +y left | c clear | q quit',
            transform=ax.transAxes,
            va='top',
            ha='left',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'black'},
        )

    ax.set_title(' | '.join(title_parts))
    ax.set_xlabel('x forward (cm)')
    ax.set_ylabel('y left (cm)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-view_range_cm, view_range_cm)
    ax.set_ylim(-view_range_cm, view_range_cm)
    ax.grid(True, alpha=0.25)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def stop_stream(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    parser = argparse.ArgumentParser(description='Draw two local target points and execute them on the rover.')
    parser.add_argument('--hz', type=float, default=4.0)
    parser.add_argument('--timeout', type=float, default=0.03)
    parser.add_argument('--speed', type=float, default=85.0, help='Executor drive PWM, not part of SAC guide.')
    parser.add_argument('--turn-speed', type=float, default=80.0, help='Executor turn PWM, not part of SAC guide.')
    parser.add_argument('--cm-per-second', type=float, default=40.0, help='Distance/time calibration used by deterministic executor.')
    parser.add_argument('--view-range-cm', type=float, default=140.0, help='Symmetric map extent around the rover center.')
    args = parser.parse_args()

    points = []
    latest = {}
    proc = start_ssh_stream(args.hz, args.timeout)
    fig, ax = plt.subplots(num='Draw Two-Vector Guide')
    plt.ion()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if len(points) >= 2:
            points.clear()
        points.append((float(event.xdata), float(event.ydata)))
        redraw(ax, fig, points, latest, args.view_range_cm)

    def on_key(event):
        nonlocal proc
        if event.key == 'c':
            points.clear()
            redraw(ax, fig, points, latest, args.view_range_cm)
        elif event.key == 'q':
            plt.close(fig)
        elif event.key == 'e':
            guide = points_to_guide(points)
            if guide is None:
                print('need exactly two points before executing', flush=True)
                return
            stop_stream(proc)
            proc = None
            execute_guide(args, guide)
            proc = start_ssh_stream(args.hz, args.timeout)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        while plt.fignum_exists(fig.number):
            if proc is not None and proc.stdout is not None:
                line = proc.stdout.readline()
                if line:
                    try:
                        latest = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            redraw(ax, fig, points, latest, args.view_range_cm)
            plt.pause(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        stop_stream(proc)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
