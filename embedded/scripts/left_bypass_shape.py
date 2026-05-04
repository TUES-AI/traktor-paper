#!/usr/bin/env python3
"""Run a continuous left-side obstacle bypass through SafetyController."""

import argparse
import json
import math
import time

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150


def accel_norm(reading):
    a = reading['accel']
    return math.sqrt(a['x'] ** 2 + a['y'] ** 2 + a['z'] ** 2)


class LeftBypassShape:
    def __init__(self, safety, speed, turn_speed, dt):
        self.safety = safety
        self.speed = speed
        self.turn_speed = turn_speed
        self.dt = dt
        self.yaw_deg = 0.0
        self.accel_min = None
        self.accel_max = None
        self.last_time = time.monotonic()

    def update_imu(self):
        if self.safety.imu is None:
            return
        now = time.monotonic()
        reading = self.safety.imu.read_all()
        gyro_z = reading['gyro']['z'] - self.safety._gyro_z_bias
        self.yaw_deg += gyro_z * (now - self.last_time)
        self.last_time = now
        norm = accel_norm(reading)
        self.accel_min = norm if self.accel_min is None else min(self.accel_min, norm)
        self.accel_max = norm if self.accel_max is None else max(self.accel_max, norm)

    def turn_to_relative_yaw(self, target_yaw, timeout):
        direction = 'left' if target_yaw > self.yaw_deg else 'right'
        left_cmd, right_cmd = ('backward', 'forward') if direction == 'left' else ('forward', 'backward')
        start = time.monotonic()
        self.last_time = time.monotonic()
        self.safety.rover.drive(left_cmd, right_cmd, left_speed=self.turn_speed, right_speed=self.turn_speed)
        reason = 'target_yaw'
        try:
            while time.monotonic() - start < timeout:
                distances = self.safety.read_distances()
                side = 'left' if direction == 'left' else 'right'
                side_cm = distances[side]
                if side_cm is None:
                    reason = f'{side}_no_echo'
                    break
                if side_cm < self.safety.config.side_turn_clear_cm:
                    reason = f'{side}_blocked'
                    break
                self.update_imu()
                if direction == 'left' and self.yaw_deg >= target_yaw:
                    break
                if direction == 'right' and self.yaw_deg <= target_yaw:
                    break
                time.sleep(self.dt)
            else:
                reason = 'timeout'
        finally:
            self.safety.rover.stop_motors()
        return {'phase': 'turn', 'direction': direction, 'target_yaw_deg': target_yaw, 'yaw_deg': self.yaw_deg, 'reason': reason}

    def forward_arc(self, curvature, seconds):
        start = time.monotonic()
        direction = 'right' if curvature < 0.0 else 'left'
        outer = 'left' if curvature < 0.0 else 'right'
        inner = 'right' if curvature < 0.0 else 'left'
        inner_ratio = max(0.25, min(1.0, 1.0 - abs(curvature)))
        inner_acc = 0.0
        last_command = None
        reason = 'duration_complete'
        self.last_time = time.monotonic()
        try:
            while time.monotonic() - start < seconds:
                distances = self.safety.read_distances()
                front_safe, front, threshold = self.safety.is_front_safe(self.speed, distances)
                if not front_safe:
                    reason = 'front_blocked'
                    break

                inner_acc += inner_ratio
                inner_on = inner_acc >= 1.0
                if inner_on:
                    inner_acc -= 1.0

                left = 'forward'
                right = 'forward'
                if inner == 'left' and not inner_on:
                    left = 'stop'
                if inner == 'right' and not inner_on:
                    right = 'stop'
                command = (left, right)
                if command != last_command:
                    self.safety.rover.drive(left, right, left_speed=self.speed, right_speed=self.speed)
                    last_command = command

                self.update_imu()
                time.sleep(self.dt)
        finally:
            self.safety.rover.stop_motors()
        return {
            'phase': 'forward_arc',
            'direction': direction,
            'outer_track': outer,
            'inner_track': inner,
            'seconds': seconds,
            'yaw_deg': self.yaw_deg,
            'reason': reason,
        }

    def run(self, entry_yaw, arc_curvature, arc_seconds, final_yaw, turn_timeout):
        report = {
            'start_distances': self.safety.read_distances(),
            'phases': [],
        }
        report['phases'].append(self.turn_to_relative_yaw(entry_yaw, turn_timeout))
        if not report['phases'][-1]['reason'] in ('target_yaw',):
            report['stopped_reason'] = 'entry_turn_failed'
            return self.finish(report)

        report['phases'].append(self.forward_arc(arc_curvature, arc_seconds))
        if report['phases'][-1]['reason'] != 'duration_complete':
            report['stopped_reason'] = 'arc_failed'
            return self.finish(report)

        report['phases'].append(self.turn_to_relative_yaw(final_yaw, turn_timeout))
        report['stopped_reason'] = 'complete' if report['phases'][-1]['reason'] == 'target_yaw' else 'final_turn_failed'
        return self.finish(report)

    def finish(self, report):
        report['end_distances'] = self.safety.read_distances()
        report['yaw_deg'] = self.yaw_deg
        report['accel_delta_g'] = 0.0 if self.accel_min is None else self.accel_max - self.accel_min
        return report


def main():
    parser = argparse.ArgumentParser(description='Continuous left-side bypass: rotate left, arc around, rotate back.')
    parser.add_argument('--entry-yaw', type=float, default=25.0)
    parser.add_argument('--arc-curvature', type=float, default=-0.35, help='Negative curves right after the initial left offset.')
    parser.add_argument('--arc-seconds', type=float, default=2.4)
    parser.add_argument('--final-yaw', type=float, default=0.0)
    parser.add_argument('--speed', type=float, default=85.0)
    parser.add_argument('--turn-speed', type=float, default=80.0)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--turn-timeout', type=float, default=4.0)
    parser.add_argument('--min-front-stop-cm', type=float, default=10.0)
    parser.add_argument('--max-front-stop-cm', type=float, default=35.0)
    parser.add_argument('--side-turn-clear-cm', type=float, default=20.0)
    args = parser.parse_args()

    print(json.dumps({'maneuver': vars(args)}, sort_keys=True), flush=True)
    rover = RoverAPI(camera_enabled=False)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(
        rover,
        imu=imu,
        config=SafetyConfig(
            min_front_stop_cm=args.min_front_stop_cm,
            max_front_stop_cm=args.max_front_stop_cm,
            side_turn_clear_cm=args.side_turn_clear_cm,
        ),
    )
    try:
        print(json.dumps({'gyro_z_bias': safety.calibrate_gyro()}, sort_keys=True), flush=True)
        maneuver = LeftBypassShape(safety, speed=args.speed, turn_speed=args.turn_speed, dt=args.dt)
        report = maneuver.run(args.entry_yaw, args.arc_curvature, args.arc_seconds, args.final_yaw, args.turn_timeout)
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        safety.close()
        imu.close()
        rover.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
