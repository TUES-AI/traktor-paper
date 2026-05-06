#!/usr/bin/env python3
"""WASD teleop with synchronized rover dataset logging.

Controls are designed for SSH terminals that cannot report simultaneous
key holds. Dedicated single-key combos avoid the need for chords:

- w: forward
- s: reverse
- q: forward + left arc
- e: forward + right arc
- a: spin left while stopped, steer left while moving
- d: spin right while stopped, steer right while moving
- x / space: full stop
- ESC / Ctrl+C: quit

The script logs:

- `telemetry.jsonl`: timestamps, ultrasonic, IMU, key/control state, motor command
- `frames/*.jpg`: camera frames at the requested FPS
- `manifest.json`: run configuration
"""

import argparse
import json
import math
import os
from pathlib import Path
from select import select
import sys
import termios
import threading
import time
import tty

import cv2

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from drivers.sensors.mpu9150 import MPU9150


def now_ns():
    return time.time_ns()


def now_s():
    return time.time()


def safe_json(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def set_raw_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    return fd, old_settings


def read_all_pending_keys(timeout=0.02):
    ready, _, _ = select([sys.stdin], [], [], timeout)
    if not ready:
        return []
    keys = [sys.stdin.read(1)]
    while select([sys.stdin], [], [], 0)[0]:
        keys.append(sys.stdin.read(1))
    result = []
    for k in keys:
        if k == ' ':
            result.append('space')
        else:
            result.append(k.lower())
    return result


class ControlState:
    """Key-set controller friendly to SSH terminals that cannot report
    simultaneous key holds. Every received key stays active for its own
    duration, so ``w`` + ``d`` interleaved on the wire still produces an
    arc-right-driving-forward state."""

    def __init__(self, forward_speed, reverse_speed, arc_inner_scale, spin_speed, steer_timeout, throttle_timeout):
        self.forward_speed = float(forward_speed)
        self.reverse_speed = float(reverse_speed)
        self.arc_inner_scale = float(arc_inner_scale)
        self.spin_speed = float(spin_speed)
        self.steer_timeout = float(steer_timeout)
        self.throttle_timeout = float(throttle_timeout)
        self.active = {}  # key -> last_seen_monotonic

    def handle_keys(self, keys):
        now = time.monotonic()
        for key in keys:
            self.active[key] = now

    def command(self):
        now = time.monotonic()
        for key in list(self.active):
            timeout = (
                self.steer_timeout if key in ('a', 'd') else self.throttle_timeout
            )
            if now - self.active[key] > timeout:
                del self.active[key]

        throttle = 0
        steer = 0

        if 'q' in self.active:
            throttle = 1
            steer = 1
        elif 'e' in self.active:
            throttle = 1
            steer = -1
        elif 'w' in self.active:
            throttle = 1
        elif 's' in self.active:
            throttle = -1

        if steer == 0:
            if 'a' in self.active:
                steer = 1
            elif 'd' in self.active:
                steer = -1

        if throttle == 0:
            if steer > 0:
                return 'backward', 'forward', self.spin_speed, self.spin_speed, 'slow_spin_left'
            if steer < 0:
                return 'forward', 'backward', self.spin_speed, self.spin_speed, 'slow_spin_right'
            return 'stop', 'stop', 0.0, 0.0, 'stop'

        direction = 'forward' if throttle > 0 else 'backward'
        speed = self.forward_speed if throttle > 0 else self.reverse_speed
        left_speed = speed
        right_speed = speed
        label = direction
        if steer > 0:
            left_speed = speed * self.arc_inner_scale
            label = f'{direction}_arc_left'
        elif steer < 0:
            right_speed = speed * self.arc_inner_scale
            label = f'{direction}_arc_right'
        return direction, direction, left_speed, right_speed, label


class DatasetLogger:
    def __init__(self, root, rover, imu, control, sensor_hz, camera_fps, jpeg_quality):
        self.root = Path(root)
        self.frames_dir = self.root / 'frames'
        self.rover = rover
        self.imu = imu
        self.control = control
        self.sensor_period = 1.0 / float(sensor_hz)
        self.camera_period = 1.0 / float(camera_fps)
        self.jpeg_quality = int(jpeg_quality)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_command = {}
        self.frame_index = 0
        self.telemetry_f = None
        self.threads = []

    def start(self, manifest):
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_f = (self.root / 'telemetry.jsonl').open('a', buffering=1)
        (self.root / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        self.threads = [
            threading.Thread(target=self._sensor_loop, name='sensor_logger', daemon=True),
            threading.Thread(target=self._camera_loop, name='camera_logger', daemon=True),
        ]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)
        if self.telemetry_f is not None:
            self.telemetry_f.close()

    def update_command(self, command):
        with self.lock:
            self.latest_command = dict(command)

    def _snapshot_command(self):
        with self.lock:
            return dict(self.latest_command)

    def _write_row(self, row):
        self.telemetry_f.write(json.dumps(row, sort_keys=True, default=safe_json) + '\n')

    def _sensor_loop(self):
        next_t = time.monotonic()
        while not self.stop_event.is_set():
            row = {
                'type': 'telemetry',
                't_wall_s': now_s(),
                't_ns': now_ns(),
                'command': self._snapshot_command(),
            }
            try:
                raw_ultra = self.rover.get_ultrasonic(timeout_seconds=0.03)
                row['ultrasonic_cm'] = {
                    'right': raw_ultra.get(1),
                    'left': raw_ultra.get(2),
                    'front': raw_ultra.get(3),
                    'raw': raw_ultra,
                }
            except Exception as exc:
                row['ultrasonic_error'] = repr(exc)
            try:
                row['imu'] = self.imu.read_all()
            except Exception as exc:
                row['imu_error'] = repr(exc)
            self._write_row(row)
            next_t += self.sensor_period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _camera_loop(self):
        next_t = time.monotonic()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while not self.stop_event.is_set():
            t0_ns = now_ns()
            try:
                frame = self.rover.get_camera_frame()
                rel = f'frames/frame_{self.frame_index:06d}.jpg'
                path = self.root / rel
                ok = cv2.imwrite(str(path), frame, params)
                row = {
                    'type': 'frame',
                    't_wall_s': now_s(),
                    't_ns': t0_ns,
                    'frame_index': self.frame_index,
                    'path': rel,
                    'shape': list(frame.shape),
                    'ok': bool(ok),
                    'command': self._snapshot_command(),
                }
                self.frame_index += 1
            except Exception as exc:
                row = {'type': 'frame_error', 't_wall_s': now_s(), 't_ns': t0_ns, 'error': repr(exc)}
            self._write_row(row)
            next_t += self.camera_period
            time.sleep(max(0.0, next_t - time.monotonic()))


def parse_args():
    parser = argparse.ArgumentParser(description='WASD teleop with dataset logging.')
    parser.add_argument('--out-root', default='data/manual_runs')
    parser.add_argument('--name', default=None)
    parser.add_argument('--sensor-hz', type=float, default=20.0)
    parser.add_argument('--camera-fps', type=float, default=5.0)
    parser.add_argument('--jpeg-quality', type=int, default=85)
    parser.add_argument('--forward-speed', type=float, default=85.0)
    parser.add_argument('--reverse-speed', type=float, default=60.0)
    parser.add_argument('--spin-speed', type=float, default=70.0)
    parser.add_argument('--arc-inner-scale', type=float, default=0.45)
    parser.add_argument('--steer-timeout', type=float, default=0.32)
    parser.add_argument('--throttle-timeout', type=float, default=0.22)
    parser.add_argument('--control-hz', type=float, default=20.0)
    return parser.parse_args()


def make_run_dir(out_root, name):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = name or f'wasd_{stamp}'
    return Path(out_root) / run_name


def main():
    args = parse_args()
    run_dir = make_run_dir(args.out_root, args.name)
    rover = RoverAPI(camera_enabled=True)
    imu = MPU9150(bus=1, address=0x68)
    control = ControlState(
        forward_speed=args.forward_speed,
        reverse_speed=args.reverse_speed,
        arc_inner_scale=args.arc_inner_scale,
        spin_speed=args.spin_speed,
        steer_timeout=args.steer_timeout,
        throttle_timeout=args.throttle_timeout,
    )
    logger = DatasetLogger(
        root=run_dir,
        rover=rover,
        imu=imu,
        control=control,
        sensor_hz=args.sensor_hz,
        camera_fps=args.camera_fps,
        jpeg_quality=args.jpeg_quality,
    )

    manifest = {
        'script': 'wasd_dataset_logger.py',
        'run_dir': str(run_dir),
        'controls': {
            'w': 'forward',
            's': 'reverse',
            'q': 'forward + left arc',
            'e': 'forward + right arc',
            'a': 'spin left while stopped / steer left while moving',
            'd': 'spin right while stopped / steer right while moving',
            'x_or_space': 'full stop',
            'esc': 'quit',
        },
        'args': vars(args),
    }

    print(__doc__.strip(), flush=True)
    print(f'dataset: {run_dir}', flush=True)
    print('Hold w/s/q/e to move, a/d to spin, x/space to stop, ESC to quit.', flush=True)

    fd = old_settings = None
    last_motor = None
    control_period = 1.0 / float(args.control_hz)
    try:
        logger.start(manifest)
        fd, old_settings = set_raw_terminal()
        rover.stop_motors()
        while True:
            keys = read_all_pending_keys(timeout=control_period)
            for key in keys:
                if key in ('\x1b', '\x03'):
                    print('quit', flush=True)
                    return 0
            if keys:
                control.handle_keys(keys)
                if 'x' in keys or 'space' in keys:
                    control.active.clear()

            left_dir, right_dir, left_speed, right_speed, label = control.command()
            command = {
                'label': label,
                'left_direction': left_dir,
                'right_direction': right_dir,
                'left_speed': left_speed,
                'right_speed': right_speed,
                'active_keys': sorted(control.active.keys()),
            }
            logger.update_command(command)
            motor_tuple = (left_dir, right_dir, round(left_speed, 2), round(right_speed, 2))
            if motor_tuple != last_motor:
                rover.drive(left_dir, right_dir, left_speed=left_speed, right_speed=right_speed)
                print(f'{label}: left={left_dir}@{left_speed:.0f} right={right_dir}@{right_speed:.0f}', flush=True)
                last_motor = motor_tuple
    finally:
        if fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        rover.stop_motors()
        logger.stop()
        imu.close()
        rover.close()
        print(f'dataset saved: {run_dir}', flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
