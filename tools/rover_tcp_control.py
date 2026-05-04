#!/usr/bin/env python3
"""Robust Mac control UI using plain TCP JSON-lines."""

import argparse
import json
import math
import os
import socket
import subprocess
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SSH_OPTS = [
    '-o', 'ConnectTimeout=8',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-o', 'IdentitiesOnly=yes',
    '-i', os.path.expanduser('~/.ssh/rover-ssh'),
]

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
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
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
    return theta1, d1, normalize_angle(theta2_abs - theta1), d2


def stop_server(args):
    remote = r'''
python3 - <<'PY'
import os, signal
needles = ('embedded/scripts/rover_tcp_server.py', 'embedded/scripts/rover_ws_server.py')
protected = {os.getpid(), os.getppid()}
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid in protected:
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode('utf-8', 'ignore')
    except Exception:
        continue
    if any(needle in cmd for needle in needles):
        os.kill(pid, signal.SIGTERM)
PY
'''
    subprocess.run(['ssh', *SSH_OPTS, args.ssh_host, remote], text=True, capture_output=True, timeout=12)


def start_server(args):
    remote = (
        'cd /home/yasen/traktor-paper; '
        f'nohup python3 embedded/scripts/rover_tcp_server.py --port {args.port} --telemetry-hz {args.telemetry_hz} '
        '> /tmp/rover_tcp_server.log 2>&1 &'
    )
    proc = subprocess.run(['ssh', *SSH_OPTS, args.ssh_host, remote], text=True, capture_output=True, timeout=12)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or 'ssh server start failed')


class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.file = None
        self.lock = threading.Lock()

    def connect(self, timeout=4.0):
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.settimeout(1.0)
        self.file = self.sock.makefile('rwb', buffering=0)

    def close(self):
        try:
            if self.file:
                self.file.close()
        except OSError:
            pass
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        self.file = None
        self.sock = None

    def send(self, obj):
        line = (json.dumps(obj) + '\n').encode('utf-8')
        with self.lock:
            self.file.write(line)
            self.file.flush()

    def recv(self):
        line = self.file.readline()
        if not line:
            raise ConnectionError('server closed')
        return json.loads(line.decode('utf-8'))


def wait_connect(args):
    deadline = time.time() + args.connect_timeout
    last = None
    while time.time() < deadline:
        client = TCPClient(args.rover_host, args.port)
        try:
            client.connect()
            return client
        except Exception as exc:
            last = exc
            client.close()
            time.sleep(0.4)
    raise RuntimeError(f'could not connect: {last}')


def draw_rover(ax):
    ax.add_patch(Rectangle((X_BACK, -TOTAL_WIDTH / 2), ROVER_LENGTH, TOTAL_WIDTH, fill=False, edgecolor='black', linewidth=1.4))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2), ROVER_LENGTH, BODY_WIDTH, fill=True, color='#3b82f6', alpha=0.12))
    ax.add_patch(Rectangle((X_BACK, BODY_WIDTH / 2), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.18))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2 - TRACK_WIDTH), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.18))
    ax.arrow(0, 0, 9, 0, head_width=1.5, head_length=2, color='red')


def ray_endpoint(sensor, value):
    x, y = sensor['pos']
    yaw = math.radians(sensor['yaw_deg'])
    dist = 180 if value is None else min(float(value), 180)
    return x + math.cos(yaw) * dist, y + math.sin(yaw) * dist


def redraw(ax, fig, state, args):
    ax.clear()
    draw_rover(ax)
    latest = state['telemetry']
    for name, sensor in SENSORS.items():
        value = latest.get(name)
        sx, sy = sensor['pos']
        ex, ey = ray_endpoint(sensor, value)
        color = sensor['color'] if value is not None else 'orange'
        ax.plot([sx, ex], [sy, ey], color=color, linewidth=3, alpha=0.75)
        ax.scatter([sx], [sy], color=color, s=35)
        ax.text(ex, ey, f'{name}: {"NO_ECHO" if value is None else f"{value:.1f}cm"}', color=color, fontsize=9)
    pts = state['points']
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color='black', s=75)
        ax.plot([0] + [p[0] for p in pts], [0] + [p[1] for p in pts], color='black', linestyle='--', linewidth=2)
    guide = points_to_guide(pts)
    status = state.get('status', '')
    if guide:
        text = f'guide=[{guide[0]:+.1f}, {guide[1]:.1f}, {guide[2]:+.1f}, {guide[3]:.1f}]\ne execute | WASD | x stop | c clear | q quit\n{status}'
    else:
        text = f'click two points | WASD | x stop | c clear | q quit\n{status}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', bbox={'facecolor': 'white', 'alpha': 0.88})
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-args.view_range_cm, args.view_range_cm)
    ax.set_ylim(-args.view_range_cm, args.view_range_cm)
    ax.set_xlabel('x forward (cm)')
    ax.set_ylabel('y left (cm)')
    ax.grid(True, alpha=0.25)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def receiver(client, state):
    while state['running']:
        try:
            msg = client.recv()
        except socket.timeout:
            continue
        except Exception as exc:
            state['status'] = f'disconnected: {exc}'
            state['connected'] = False
            break
        kind = msg.get('type')
        if kind == 'telemetry':
            state['telemetry'] = msg
            state['status'] = 'connected'
        elif kind == 'execute_started':
            state['status'] = f'executing log={msg.get("log_path")}'
        elif kind == 'execute_done':
            rep = msg.get('report', {})
            state['status'] = f'done {rep.get("stopped_reason")} log={rep.get("log_path")}'
        elif kind == 'error':
            state['status'] = f'error {msg.get("where")}: {msg.get("error")}'
        else:
            state['status'] = json.dumps(msg, sort_keys=True)


def safe_send(client, state, obj):
    try:
        client.send(obj)
    except Exception as exc:
        state['status'] = f'send failed: {exc}'


def main():
    parser = argparse.ArgumentParser(description='Plain TCP rover control UI.')
    parser.add_argument('--ssh-host', default='yasen@192.168.100.42')
    parser.add_argument('--rover-host', default='192.168.100.42')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--telemetry-hz', type=float, default=5.0)
    parser.add_argument('--connect-timeout', type=float, default=12.0)
    parser.add_argument('--view-range-cm', type=float, default=260.0)
    parser.add_argument('--manual-speed', type=float, default=100.0)
    parser.add_argument('--manual-duration', type=float, default=0.45)
    parser.add_argument('--guide-speed', type=float, default=85.0)
    parser.add_argument('--turn-speed', type=float, default=80.0)
    parser.add_argument('--cm-per-second', type=float, default=40.0)
    parser.add_argument('--restart-server', action='store_true')
    args = parser.parse_args()

    print(f'using ssh={args.ssh_host}, tcp={args.rover_host}:{args.port}', flush=True)

    if args.restart_server:
        stop_server(args)
        time.sleep(0.5)
    try:
        client = wait_connect(args)
        print('connected to existing TCP server', flush=True)
    except RuntimeError:
        print('starting TCP server over SSH...', flush=True)
        start_server(args)
        client = wait_connect(args)

    state = {'running': True, 'connected': True, 'telemetry': {}, 'points': [], 'status': 'connected'}
    threading.Thread(target=receiver, args=(client, state), daemon=True).start()
    fig, ax = plt.subplots(num='Rover TCP Control')
    plt.ion()

    def on_click(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            if len(state['points']) >= 2:
                state['points'].clear()
            state['points'].append((float(event.xdata), float(event.ydata)))

    def drive(left, right):
        safe_send(client, state, {'type': 'drive', 'left': left, 'right': right, 'speed': args.manual_speed, 'duration': args.manual_duration})

    def on_key(event):
        if event.key == 'q':
            state['running'] = False
            plt.close(fig)
        elif event.key == 'c':
            state['points'].clear()
        elif event.key == 'x':
            safe_send(client, state, {'type': 'stop'})
        elif event.key == 'w':
            drive('forward', 'forward')
        elif event.key == 's':
            drive('backward', 'backward')
        elif event.key == 'a':
            drive('backward', 'forward')
        elif event.key == 'd':
            drive('forward', 'backward')
        elif event.key == 'e':
            guide = points_to_guide(state['points'])
            if guide is None:
                state['status'] = 'need two points'
                return
            safe_send(client, state, {
                'type': 'execute', 'theta1': guide[0], 'd1': guide[1], 'theta2': guide[2], 'd2': guide[3],
                'speed': args.guide_speed, 'turn_speed': args.turn_speed, 'cm_per_second': args.cm_per_second,
            })

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    try:
        while state['running'] and plt.fignum_exists(fig.number):
            redraw(ax, fig, state, args)
            plt.pause(0.05)
    finally:
        safe_send(client, state, {'type': 'stop'})
        client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
