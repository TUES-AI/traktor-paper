#!/usr/bin/env python3
"""Rover-hosted browser UI for local targets and cheap lidar display.

Open from iPad/Mac:
    http://192.168.100.42:8080

This intentionally uses Flask + Server-Sent Events instead of a websocket
package because the Pi currently has Flask installed but not websocket helpers.
"""

import argparse
import json
import math
import queue
import ssl
import threading
import time
from pathlib import Path

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150
from flask import Flask, Response, jsonify, request


ROVER_LENGTH_CM = 16.5
BODY_WIDTH_CM = 8.5
TRACK_WIDTH_CM = 4.5
TOTAL_WIDTH_CM = BODY_WIDTH_CM + 2 * TRACK_WIDTH_CM
X_BACK_CM = -ROVER_LENGTH_CM / 2
X_FRONT_CM = ROVER_LENGTH_CM / 2

SENSORS = {
    'front': {'x': X_FRONT_CM + 0.6, 'y': 0.0, 'yaw_deg': 0.0},
    'left': {'x': X_BACK_CM + 14.5, 'y': BODY_WIDTH_CM / 2 + 0.6, 'yaw_deg': 90.0},
    'right': {'x': X_BACK_CM + 14.5, 'y': -BODY_WIDTH_CM / 2 - 0.6, 'yaw_deg': -90.0},
}

TURN_PWM = 65.0
MAX_TURN_PWM = 100.0
MIN_TURN_PROGRESS_DEG = 0.8
TURN_STALL_SECONDS = 1.2
DRIVE_PWM = 90.0
DT = 0.05
TURN_TOLERANCE_DEG = 5.0
MAX_TURN_SECONDS = 12.0
MAX_DRIVE_SECONDS = 4.0
CM_PER_SECOND = 40.0

app = Flask(__name__)


HTML = r"""
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <title>Rover Web Trainer</title>
  <style>
    body { margin: 0; background: #111827; color: #e5e7eb; font-family: -apple-system, system-ui, sans-serif; }
    #top { padding: 10px 14px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    button { font-size: 18px; padding: 10px 14px; border-radius: 10px; border: 0; background: #ef4444; color: white; }
    #status { font-size: 15px; color: #d1d5db; }
    canvas { display: block; width: 100vw; height: calc(100vh - 64px); background: #f8fafc; touch-action: none; }
  </style>
</head>
<body>
  <div id="top">
    <button onclick="stopRover()">STOP</button>
    <span id="status">connecting...</span>
  </div>
  <canvas id="map"></canvas>
<script>
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
let telemetry = {yaw_deg: 0, pose: {x_cm: 0, y_cm: 0}, distances: {}, lidar: [], lidar_map: [], target: null, state: 'idle'};
let rangeCm = 260;

function resize() {
  canvas.width = Math.floor(canvas.clientWidth * window.devicePixelRatio);
  canvas.height = Math.floor(canvas.clientHeight * window.devicePixelRatio);
  draw();
}
window.addEventListener('resize', resize);

function cmToPx(x, y) {
  const s = Math.min(canvas.width, canvas.height) / (2 * rangeCm);
  return [canvas.width / 2 + x * s, canvas.height / 2 - y * s];
}

function worldToPx(x, y) {
  const pose = telemetry.pose || {x_cm: 0, y_cm: 0};
  return cmToPx(x - pose.x_cm, y - pose.y_cm);
}

function pxToCm(px, py) {
  const s = Math.min(canvas.width, canvas.height) / (2 * rangeCm);
  return [(px - canvas.width / 2) / s, -(py - canvas.height / 2) / s];
}

function drawGrid() {
  ctx.strokeStyle = '#d1d5db'; ctx.lineWidth = 1;
  ctx.fillStyle = '#6b7280'; ctx.font = `${13 * window.devicePixelRatio}px sans-serif`;
  const pose = telemetry.pose || {x_cm: 0, y_cm: 0};
  const startX = Math.floor((pose.x_cm - rangeCm) / 20) * 20;
  const endX = Math.ceil((pose.x_cm + rangeCm) / 20) * 20;
  const startY = Math.floor((pose.y_cm - rangeCm) / 20) * 20;
  const endY = Math.ceil((pose.y_cm + rangeCm) / 20) * 20;
  for (let v = startY; v <= endY; v += 20) {
    let [x1, y1] = worldToPx(pose.x_cm - rangeCm, v), [x2, y2] = worldToPx(pose.x_cm + rangeCm, v);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  for (let v = startX; v <= endX; v += 20) {
    let [x1, y1] = worldToPx(v, pose.y_cm - rangeCm), [x2, y2] = worldToPx(v, pose.y_cm + rangeCm);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
}

function drawRover() {
  const yaw = telemetry.yaw_deg * Math.PI / 180;
  const s = Math.min(canvas.width, canvas.height) / (2 * rangeCm);
  ctx.save();
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate(-yaw);
  ctx.strokeStyle = '#111827'; ctx.fillStyle = 'rgba(59,130,246,0.22)'; ctx.lineWidth = 2;
  ctx.fillRect(-8.25*s, -8.75*s, 16.5*s, 17.5*s);
  ctx.strokeRect(-8.25*s, -8.75*s, 16.5*s, 17.5*s);
  ctx.fillStyle = 'rgba(75,85,99,0.45)';
  ctx.fillRect(-8.25*s, 4.25*s, 16.5*s, 4.5*s);
  ctx.fillRect(-8.25*s, -8.75*s, 16.5*s, 4.5*s);
  ctx.strokeStyle = '#dc2626'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(22*s, 0); ctx.stroke();
  ctx.restore();
}

function drawLidar() {
  for (const p of telemetry.lidar_map || []) {
    const [x, y] = worldToPx(p.x, p.y);
    ctx.fillStyle = '#2563eb';
    ctx.beginPath(); ctx.arc(x, y, 2.8 * window.devicePixelRatio, 0, 2*Math.PI); ctx.fill();
  }
  for (const p of telemetry.lidar || []) {
    const [x, y] = worldToPx(p.x, p.y);
    ctx.fillStyle = p.name === 'front' ? '#ef4444' : '#16a34a';
    ctx.beginPath(); ctx.arc(x, y, 5 * window.devicePixelRatio, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = '#111827'; ctx.font = `${12 * window.devicePixelRatio}px sans-serif`;
    ctx.fillText(`${p.name} ${p.distance_cm.toFixed(0)}cm`, x + 7, y - 7);
  }
}

function drawTarget() {
  if (!telemetry.target) return;
  const [x, y] = worldToPx(telemetry.target.x_cm, telemetry.target.y_cm);
  ctx.strokeStyle = '#ec4899'; ctx.fillStyle = '#ec4899'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(canvas.width/2, canvas.height/2); ctx.lineTo(x, y); ctx.stroke();
  ctx.beginPath(); ctx.arc(x, y, 8 * window.devicePixelRatio, 0, 2*Math.PI); ctx.fill();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid(); drawTarget(); drawLidar(); drawRover();
}

canvas.addEventListener('pointerdown', async (ev) => {
  const r = canvas.getBoundingClientRect();
  const [rx, ry] = pxToCm((ev.clientX - r.left) * window.devicePixelRatio, (ev.clientY - r.top) * window.devicePixelRatio);
  const pose = telemetry.pose || {x_cm: 0, y_cm: 0};
  await fetch('/target', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x_cm: pose.x_cm + rx, y_cm: pose.y_cm + ry})
  });
});

async function stopRover() { await fetch('/stop', {method: 'POST'}); }

const es = new EventSource('/events');
es.onmessage = (ev) => {
  telemetry = JSON.parse(ev.data);
  const p = telemetry.pose || {x_cm: 0, y_cm: 0};
  statusEl.textContent = `state=${telemetry.state} pose=(${p.x_cm.toFixed(0)},${p.y_cm.toFixed(0)}) yaw=${telemetry.yaw_deg.toFixed(1)} front=${telemetry.distances.front ?? 'NO'} left=${telemetry.distances.left ?? 'NO'} right=${telemetry.distances.right ?? 'NO'}`;
  draw();
};
es.onerror = () => { statusEl.textContent = 'event stream disconnected'; };
resize();
</script>
</body>
</html>
"""


class RoverWebState:
    def __init__(self):
        self.rover = RoverAPI(camera_enabled=False)
        self.imu = MPU9150(bus=1, address=0x68)
        self.safety = SafetyController(self.rover, imu=self.imu, config=SafetyConfig())
        self.lock = threading.RLock()
        self.clients = []
        self.running = True
        self.yaw_deg = 0.0
        self.last_imu_t = time.monotonic()
        self.distances = {'front': None, 'left': None, 'right': None}
        self.lidar = []
        self.lidar_map = []
        self.target = None
        self.state = 'idle'
        self.worker = None
        self.pose_x_cm = 0.0
        self.pose_y_cm = 0.0
        self.drive_active = False
        self.last_pose_t = time.monotonic()

    def calibrate(self):
        self.safety.calibrate_gyro()

    def close(self):
        self.running = False
        with self.lock:
            self.rover.stop_motors()
            self.imu.close()
            self.rover.close()

    def broadcast(self):
        msg = json.dumps({
            'yaw_deg': self.yaw_deg,
            'distances': self.distances,
            'pose': {'x_cm': self.pose_x_cm, 'y_cm': self.pose_y_cm},
            'lidar': self.lidar,
            'lidar_map': self.lidar_map[-500:],
            'target': self.target,
            'state': self.state,
        })
        for q in list(self.clients):
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def telemetry_loop(self):
        while self.running:
            with self.lock:
                now = time.monotonic()
                imu = self.imu.read_all()
                gyro_z = imu['gyro']['z'] - self.safety._gyro_z_bias
                self.yaw_deg += gyro_z * (now - self.last_imu_t)
                self.last_imu_t = now
                if self.drive_active:
                    dt = now - self.last_pose_t
                    yaw = math.radians(self.yaw_deg)
                    self.pose_x_cm += math.cos(yaw) * CM_PER_SECOND * dt
                    self.pose_y_cm += math.sin(yaw) * CM_PER_SECOND * dt
                self.last_pose_t = now
                self.distances = self.safety.read_distances()
                self.lidar = self.compute_lidar_points(self.distances)
                self.add_lidar_map_points(self.lidar)
                self.broadcast()
            time.sleep(0.1)

    def compute_lidar_points(self, distances):
        pts = []
        yaw = math.radians(self.yaw_deg)
        for name, sensor in SENSORS.items():
            d = distances.get(name)
            if d is None:
                continue
            local_yaw = math.radians(sensor['yaw_deg'])
            lx = sensor['x'] + math.cos(local_yaw) * d
            ly = sensor['y'] + math.sin(local_yaw) * d
            wx = self.pose_x_cm + math.cos(yaw) * lx - math.sin(yaw) * ly
            wy = self.pose_y_cm + math.sin(yaw) * lx + math.cos(yaw) * ly
            pts.append({'name': name, 'x': wx, 'y': wy, 'distance_cm': d})
        return pts

    def add_lidar_map_points(self, points):
        for p in points:
            if p['distance_cm'] > 180.0:
                continue
            if self.lidar_map:
                last = self.lidar_map[-1]
                if abs(last['x'] - p['x']) < 1.5 and abs(last['y'] - p['y']) < 1.5 and last['name'] == p['name']:
                    continue
            self.lidar_map.append({'name': p['name'], 'x': p['x'], 'y': p['y'], 't': time.time()})
        if len(self.lidar_map) > 1000:
            self.lidar_map = self.lidar_map[-1000:]

    def world_to_rover_local(self, x_cm, y_cm):
        yaw = math.radians(self.yaw_deg)
        dx = x_cm - self.pose_x_cm
        dy = y_cm - self.pose_y_cm
        # World map axes stay fixed on the screen. Convert clicked world vector
        # into current rover body frame: +x forward, +y left.
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return lx, ly

    def local_to_world(self, x_cm, y_cm):
        yaw = math.radians(self.yaw_deg)
        wx = self.pose_x_cm + math.cos(yaw) * x_cm - math.sin(yaw) * y_cm
        wy = self.pose_y_cm + math.sin(yaw) * x_cm + math.cos(yaw) * y_cm
        return wx, wy

    def set_target(self, x_cm, y_cm):
        with self.lock:
            local_x, local_y = self.world_to_rover_local(float(x_cm), float(y_cm))
            self.target = {'x_cm': float(x_cm), 'y_cm': float(y_cm), 'local_x_cm': local_x, 'local_y_cm': local_y}
            self.rover.stop_motors()
            self.state = f'target_set local=({local_x:.1f},{local_y:.1f})'
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self.execute_target, daemon=True)
            self.worker.start()

    def set_executor_status(self, status):
        with self.lock:
            self.state = status
            self.drive_active = status.startswith('driving_')
            if self.drive_active:
                self.last_pose_t = time.monotonic()

    def clip_distance(self, theta_deg, distance_cm):
        distances = self.distances
        allowed = distance_cm
        if abs(theta_deg) <= 45 and distances['front'] is not None:
            allowed = min(allowed, max(20.0, distances['front'] - 20.0))
        if theta_deg > 25 and distances['left'] is not None:
            allowed = min(allowed, max(20.0, distances['left'] - 20.0))
        if theta_deg < -25 and distances['right'] is not None:
            allowed = min(allowed, max(20.0, distances['right'] - 20.0))
        return allowed

    def execute_target(self):
        with self.lock:
            target = dict(self.target)
        executor = LocalTargetExecutor(
            self.safety,
            config=LocalTargetExecutorConfig(
                turn_pwm=TURN_PWM,
                max_turn_pwm=MAX_TURN_PWM,
                drive_pwm=DRIVE_PWM,
                dt=DT,
                turn_tolerance_deg=TURN_TOLERANCE_DEG,
                min_turn_progress_deg=MIN_TURN_PROGRESS_DEG,
                turn_stall_seconds=TURN_STALL_SECONDS,
                max_turn_seconds=MAX_TURN_SECONDS,
                max_drive_seconds=MAX_DRIVE_SECONDS,
                cm_per_second=CM_PER_SECOND,
            ),
            status_callback=self.set_executor_status,
        )
        executor.execute_local_target(target['local_x_cm'], target['local_y_cm'])
        with self.lock:
            self.state = 'idle'
            self.drive_active = False
            self.rover.stop_motors()

    def turn_to(self, theta_deg):
        direction = 'left' if theta_deg >= 0 else 'right'
        left_cmd, right_cmd = ('backward', 'forward') if direction == 'left' else ('forward', 'backward')
        target = abs(theta_deg) if direction == 'left' else -abs(theta_deg)
        yaw = 0.0
        last = time.monotonic()
        start = last
        pwm = TURN_PWM
        last_progress_time = start
        last_abs_yaw = 0.0
        with self.lock:
            self.state = f'turning_{direction}_{theta_deg:.1f} pwm={pwm:.0f}'
            self.rover.drive(left_cmd, right_cmd, left_speed=pwm, right_speed=pwm)
        try:
            while time.monotonic() - start < MAX_TURN_SECONDS:
                with self.lock:
                    turn_safe, side, reason = self.safety.is_turn_safe(direction, self.distances)
                    if not turn_safe:
                        side_label = 'NO_ECHO' if side is None else f'{side:.1f}'
                        self.state = f'turn_safety_stop {direction}={side_label}'
                        self.rover.stop_motors()
                        return {'ok': False, 'reason': reason, 'yaw_deg': yaw, 'target_deg': target}
                    now = time.monotonic()
                    imu = self.imu.read_all()
                    gyro_z = imu['gyro']['z'] - self.safety._gyro_z_bias
                yaw += gyro_z * (now - last)
                abs_yaw = abs(yaw)
                if abs_yaw > last_abs_yaw + 0.8:
                    last_progress_time = now
                    last_abs_yaw = abs_yaw
                if now - last_progress_time > TURN_STALL_SECONDS:
                    if pwm < MAX_TURN_PWM:
                        pwm = MAX_TURN_PWM
                        last_progress_time = now
                        with self.lock:
                            self.state = f'turning_{direction}_{theta_deg:.1f} pwm={pwm:.0f} full_boost'
                            self.rover.drive(left_cmd, right_cmd, left_speed=pwm, right_speed=pwm)
                    elif abs_yaw < 3.0:
                        return {'ok': False, 'reason': f'stalled yaw={yaw:.1f}/{target:.1f}', 'yaw_deg': yaw, 'target_deg': target}
                last = now
                if direction == 'left' and yaw >= target - TURN_TOLERANCE_DEG:
                    return {'ok': True, 'reason': 'target_reached', 'yaw_deg': yaw, 'target_deg': target}
                if direction == 'right' and yaw <= target + TURN_TOLERANCE_DEG:
                    return {'ok': True, 'reason': 'target_reached', 'yaw_deg': yaw, 'target_deg': target}
                time.sleep(DT)
            return {'ok': False, 'reason': f'max_turn_time yaw={yaw:.1f}/{target:.1f}', 'yaw_deg': yaw, 'target_deg': target}
        finally:
            with self.lock:
                self.rover.stop_motors()
            time.sleep(0.15)

    def drive_for(self, distance_cm):
        seconds = max(0.45, min(MAX_DRIVE_SECONDS, distance_cm / CM_PER_SECOND))
        start = time.monotonic()
        with self.lock:
            self.state = f'driving_{distance_cm:.1f}cm'
            self.drive_active = True
            self.last_pose_t = time.monotonic()
            self.rover.drive('forward', 'forward', left_speed=DRIVE_PWM, right_speed=DRIVE_PWM)
        try:
            while time.monotonic() - start < seconds:
                with self.lock:
                    safe, _, _ = self.safety.is_front_safe(DRIVE_PWM, self.distances)
                    if not safe:
                        self.state = 'front_safety_stop'
                        return False
                time.sleep(DT)
            return True
        finally:
            with self.lock:
                self.drive_active = False
                self.rover.stop_motors()
            time.sleep(0.15)

    def stop(self):
        with self.lock:
            self.rover.stop_motors()
            self.state = 'stopped'


STATE = None


def ensure_self_signed_cert(cert_dir):
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = cert_dir / 'cert.pem'
    key_file = cert_dir / 'key.pem'
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file

    import subprocess

    subprocess.run([
        'openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
        '-keyout', str(key_file),
        '-out', str(cert_file),
        '-days', '365',
        '-subj', '/CN=192.168.100.42',
        '-addext', 'subjectAltName=IP:192.168.100.42,DNS:rover',
    ], check=True)
    return cert_file, key_file


@app.route('/')
def index():
    return HTML


@app.route('/events')
def events():
    q = queue.Queue(maxsize=8)
    STATE.clients.append(q)

    def gen():
        try:
            while STATE.running:
                msg = q.get()
                yield f'data: {msg}\n\n'
        finally:
            if q in STATE.clients:
                STATE.clients.remove(q)
    return Response(gen(), mimetype='text/event-stream')


@app.route('/target', methods=['POST'])
def target():
    data = request.get_json(force=True)
    STATE.set_target(float(data['x_cm']), float(data['y_cm']))
    return jsonify({'ok': True})


@app.route('/stop', methods=['POST'])
def stop():
    STATE.stop()
    return jsonify({'ok': True})


def main():
    global STATE
    parser = argparse.ArgumentParser(description='Run rover browser target UI.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8443)
    parser.add_argument('--http', action='store_true', help='Serve plain HTTP instead of HTTPS.')
    parser.add_argument('--cert-dir', default='/tmp/rover_web_trainer_tls')
    args = parser.parse_args()
    STATE = RoverWebState()
    try:
        print('calibrating gyro...', flush=True)
        STATE.calibrate()
        threading.Thread(target=STATE.telemetry_loop, daemon=True).start()
        if args.http:
            print(f'open http://192.168.100.42:{args.port}', flush=True)
            app.run(host=args.host, port=args.port, threaded=True)
        else:
            cert_file, key_file = ensure_self_signed_cert(args.cert_dir)
            print(f'open https://192.168.100.42:{args.port}', flush=True)
            print('browser warning is expected: self-signed local certificate', flush=True)
            app.run(host=args.host, port=args.port, threaded=True, ssl_context=(str(cert_file), str(key_file)))
    finally:
        STATE.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
