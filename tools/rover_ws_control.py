#!/usr/bin/env python3
"""Mac control UI for rover WebSocket server.

Starts the Pi server over SSH, then uses one persistent WebSocket connection for
telemetry, WASD, and two-point guide execution.
"""

import argparse
import base64
import json
import math
import os
import socket
import struct
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


def start_server_over_ssh(args):
    remote = (
        'cd /home/yasen/traktor-paper; '
        f'nohup python3 embedded/scripts/rover_ws_server.py --port {args.port} --telemetry-hz {args.telemetry_hz} '
        '> /tmp/rover_ws_server.log 2>&1 &'
    )
    proc = subprocess.run(['ssh', *SSH_OPTS, args.ssh_host, remote], text=True, capture_output=True, timeout=12)
    if proc.returncode != 0:
        raise RuntimeError(f'failed to start rover server over ssh: {proc.stderr.strip() or proc.stdout.strip()}')


def stop_server_over_ssh(args):
    remote = r'''
python3 - <<'PY'
import os, signal
needle = 'embedded/scripts/rover_ws_server.py'
protected = {os.getpid()}
pid = os.getppid()
while pid and pid not in protected:
    protected.add(pid)
    try:
        with open(f'/proc/{pid}/stat') as f:
            pid = int(f.read().split()[3])
    except Exception:
        break
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid in protected:
        continue
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmd = f.read().replace(b'\0', b' ').decode('utf-8', 'ignore')
    except Exception:
        continue
    if needle in cmd:
        os.kill(pid, signal.SIGTERM)
PY
'''
    subprocess.run(['ssh', *SSH_OPTS, args.ssh_host, remote], text=True, capture_output=True, timeout=12)


class WSClient:
    def __init__(self, host, port, timeout=4.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.lock = threading.Lock()

    def connect(self):
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        key = base64.b64encode(os.urandom(16)).decode('ascii')
        request = (
            f'GET / HTTP/1.1\r\n'
            f'Host: {self.host}:{self.port}\r\n'
            'Upgrade: websocket\r\n'
            'Connection: Upgrade\r\n'
            f'Sec-WebSocket-Key: {key}\r\n'
            'Sec-WebSocket-Version: 13\r\n\r\n'
        )
        sock.sendall(request.encode('ascii'))
        response = b''
        while b'\r\n\r\n' not in response:
            response += sock.recv(4096)
        if b'101 Switching Protocols' not in response:
            raise RuntimeError(response.decode('utf-8', errors='replace'))
        self.sock = sock

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def send_json(self, obj):
        payload = json.dumps(obj).encode('utf-8')
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = struct.pack('!BB', 0x81, 0x80 | length)
        elif length < 65536:
            header = struct.pack('!BBH', 0x81, 0x80 | 126, length)
        else:
            header = struct.pack('!BBQ', 0x81, 0x80 | 127, length)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        with self.lock:
            self.sock.sendall(header + mask + masked)

    def recv_json(self):
        header = self._recv_exact(2)
        b1, b2 = header
        opcode = b1 & 0x0F
        length = b2 & 0x7F
        if opcode == 0x8:
            raise ConnectionError('websocket closed')
        if length == 126:
            length = struct.unpack('!H', self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack('!Q', self._recv_exact(8))[0]
        payload = self._recv_exact(length) if length else b''
        if opcode != 0x1:
            return None
        return json.loads(payload.decode('utf-8'))

    def _recv_exact(self, n):
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError('socket closed')
            data += chunk
        return data


def draw_rover(ax):
    ax.add_patch(Rectangle((X_BACK, -TOTAL_WIDTH / 2), ROVER_LENGTH, TOTAL_WIDTH, fill=False, edgecolor='black', linewidth=1.4))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2), ROVER_LENGTH, BODY_WIDTH, fill=True, color='#3b82f6', alpha=0.12))
    ax.add_patch(Rectangle((X_BACK, BODY_WIDTH / 2), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.18))
    ax.add_patch(Rectangle((X_BACK, -BODY_WIDTH / 2 - TRACK_WIDTH), ROVER_LENGTH, TRACK_WIDTH, fill=True, color='gray', alpha=0.18))
    ax.arrow(0, 0, 9, 0, head_width=1.5, head_length=2.0, color='red')
    ax.text(9.5, 1.2, '+x forward', color='red')


def ray_endpoint(sensor, distance_cm):
    x, y = sensor['pos']
    yaw = math.radians(sensor['yaw_deg'])
    distance = 180.0 if distance_cm is None else min(float(distance_cm), 180.0)
    return x + math.cos(yaw) * distance, y + math.sin(yaw) * distance


def redraw(ax, fig, state, args):
    ax.clear()
    draw_rover(ax)
    latest = state['telemetry']
    points = state['points']
    title_parts = []
    for name, sensor in SENSORS.items():
        value = latest.get(name)
        sx, sy = sensor['pos']
        ex, ey = ray_endpoint(sensor, value)
        color = sensor['color'] if value is not None else 'orange'
        ax.plot([sx, ex], [sy, ey], color=color, linewidth=3, alpha=0.75)
        ax.scatter([sx], [sy], color=color, s=35)
        label = 'NO_ECHO' if value is None else f'{value:.1f}cm'
        ax.text(ex, ey, f'{name}: {label}', color=color, fontsize=9)
        title_parts.append(f'{name}={label}')
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, color='black', s=75, zorder=10)
        ax.plot([0.0] + [p[0] for p in points], [0.0] + [p[1] for p in points], color='black', linewidth=2, linestyle='--')
        for i, (x, y) in enumerate(points, 1):
            ax.text(x + 2, y + 2, f'P{i}', fontsize=11)
    guide = points_to_guide(points)
    status = state.get('status', '')
    if guide:
        theta1, d1, theta2, d2 = guide
        text = f'guide=[{theta1:+.1f}, {d1:.1f}, {theta2:+.1f}, {d2:.1f}]\ne execute | c clear | WASD move | x stop | q quit\n{status}'
    else:
        text = f'click two points | WASD move | x stop | q quit\n{status}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left', bbox={'facecolor': 'white', 'alpha': 0.87, 'edgecolor': 'black'})
    ax.set_title(' | '.join(title_parts))
    ax.set_xlabel('x forward (cm)')
    ax.set_ylabel('y left (cm)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-args.view_range_cm, args.view_range_cm)
    ax.set_ylim(-args.view_range_cm, args.view_range_cm)
    ax.grid(True, alpha=0.25)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def receiver_loop(ws, state):
    while state['running']:
        try:
            msg = ws.recv_json()
        except Exception as exc:
            state['status'] = f'connection error: {exc}'
            break
        if not msg:
            continue
        kind = msg.get('type')
        if kind == 'telemetry':
            state['telemetry'] = msg
        elif kind == 'execute_started':
            state['status'] = f'executing, log={msg.get("log_path")}'
        elif kind == 'execute_done':
            report = msg.get('report', {})
            state['status'] = f'done: {report.get("stopped_reason")}, log={report.get("log_path")}'
        elif kind == 'error':
            state['status'] = f'error: {msg.get("where")}: {msg.get("error")}'
        elif kind in ('hello', 'drive_ack', 'stop_ack', 'warn'):
            state['status'] = json.dumps(msg, sort_keys=True)


def wait_for_server(args):
    deadline = time.time() + args.connect_timeout
    last_exc = None
    while time.time() < deadline:
        ws = WSClient(args.rover_host, args.port, timeout=3.0)
        try:
            ws.connect()
            return ws
        except Exception as exc:
            last_exc = exc
            ws.close()
            time.sleep(0.4)
    raise RuntimeError(f'could not connect to rover websocket: {last_exc}')


def safe_send(ws, state, msg):
    try:
        ws.send_json(msg)
        return True
    except Exception as exc:
        state['status'] = f'send failed: {exc}'
        return False


def main():
    parser = argparse.ArgumentParser(description='Persistent WebSocket rover control UI.')
    parser.add_argument('--ssh-host', default='yasen@192.168.100.42')
    parser.add_argument('--rover-host', default='192.168.100.42')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--telemetry-hz', type=float, default=6.0)
    parser.add_argument('--connect-timeout', type=float, default=12.0)
    parser.add_argument('--view-range-cm', type=float, default=220.0)
    parser.add_argument('--manual-speed', type=float, default=100.0)
    parser.add_argument('--manual-duration', type=float, default=0.45)
    parser.add_argument('--guide-speed', type=float, default=85.0)
    parser.add_argument('--turn-speed', type=float, default=80.0)
    parser.add_argument('--cm-per-second', type=float, default=40.0)
    parser.add_argument('--restart-server', action='store_true')
    args = parser.parse_args()

    print(f'using ssh={args.ssh_host}, websocket={args.rover_host}:{args.port}', flush=True)

    if args.restart_server:
        print('stopping old rover websocket server over ssh...', flush=True)
        stop_server_over_ssh(args)
        time.sleep(0.5)
    try:
        ws = wait_for_server(args)
        print('connected to existing rover websocket server', flush=True)
    except RuntimeError:
        print('starting rover websocket server over ssh...', flush=True)
        start_server_over_ssh(args)
        ws = wait_for_server(args)
    state = {'running': True, 'telemetry': {}, 'points': [], 'status': 'connected'}
    threading.Thread(target=receiver_loop, args=(ws, state), daemon=True).start()

    fig, ax = plt.subplots(num='Rover WebSocket Control')
    plt.ion()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if len(state['points']) >= 2:
            state['points'].clear()
        state['points'].append((float(event.xdata), float(event.ydata)))

    def send_drive(left, right):
        safe_send(ws, state, {'type': 'drive', 'left': left, 'right': right, 'speed': args.manual_speed, 'duration': args.manual_duration})

    def on_key(event):
        if event.key == 'q':
            state['running'] = False
            plt.close(fig)
        elif event.key == 'c':
            state['points'].clear()
        elif event.key == 'x':
            safe_send(ws, state, {'type': 'stop'})
        elif event.key == 'w':
            send_drive('forward', 'forward')
        elif event.key == 's':
            send_drive('backward', 'backward')
        elif event.key == 'a':
            send_drive('backward', 'forward')
        elif event.key == 'd':
            send_drive('forward', 'backward')
        elif event.key == 'e':
            guide = points_to_guide(state['points'])
            if guide is None:
                state['status'] = 'need exactly two points before execute'
                return
            theta1, d1, theta2, d2 = guide
            safe_send(ws, state, {
                'type': 'execute',
                'theta1': theta1,
                'd1': d1,
                'theta2': theta2,
                'd2': d2,
                'speed': args.guide_speed,
                'turn_speed': args.turn_speed,
                'cm_per_second': args.cm_per_second,
            })

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    try:
        while state['running'] and plt.fignum_exists(fig.number):
            redraw(ax, fig, state, args)
            plt.pause(0.05)
    finally:
        try:
            safe_send(ws, state, {'type': 'stop'})
        except Exception:
            pass
        ws.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
