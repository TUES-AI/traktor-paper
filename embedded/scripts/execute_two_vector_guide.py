#!/usr/bin/env python3
"""Execute SAC-style `[theta1, d1, theta2, d2]` guide and log calibration data."""

import argparse
import json
import os
import time

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.safety import SafetyConfig, SafetyController
from control.two_vector_executor import TwoVectorExecutor, TwoVectorGuide
from drivers.sensors.mpu9150 import MPU9150


def main():
    parser = argparse.ArgumentParser(description='Execute a two-vector local guide through SafetyController.')
    parser.add_argument('--theta1', type=float, required=True, help='First segment heading, degrees relative to rover forward. Positive is left.')
    parser.add_argument('--d1', type=float, required=True, help='First segment distance in cm.')
    parser.add_argument('--theta2', type=float, required=True, help='Second heading relative to first heading, degrees. Positive is left.')
    parser.add_argument('--d2', type=float, required=True, help='Second segment distance in cm.')
    parser.add_argument('--speed', type=float, default=85.0, help='Executor-selected drive PWM percent, not SAC output.')
    parser.add_argument('--turn-speed', type=float, default=80.0, help='Executor-selected turn PWM percent, not SAC output.')
    parser.add_argument('--cm-per-second', type=float, default=40.0, help='Current rough distance-to-time calibration.')
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--log-dir', default='data/executor_logs')
    parser.add_argument('--min-front-stop-cm', type=float, default=10.0)
    parser.add_argument('--max-front-stop-cm', type=float, default=35.0)
    parser.add_argument('--side-turn-clear-cm', type=float, default=20.0)
    args = parser.parse_args()

    guide = TwoVectorGuide(args.theta1, args.d1, args.theta2, args.d2)
    log_name = 'two_vector_{ts}_t1_{t1:+.0f}_d1_{d1:.0f}_t2_{t2:+.0f}_d2_{d2:.0f}.csv'.format(
        ts=time.strftime('%Y%m%d_%H%M%S'),
        t1=args.theta1,
        d1=args.d1,
        t2=args.theta2,
        d2=args.d2,
    ).replace('+', 'p').replace('-', 'm')
    log_path = os.path.join(args.log_dir, log_name)

    print(json.dumps({'guide': guide.__dict__, 'log_path': log_path}, sort_keys=True), flush=True)
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
        executor = TwoVectorExecutor(
            safety,
            guide,
            log_path,
            speed_pct=args.speed,
            turn_speed_pct=args.turn_speed,
            cm_per_second=args.cm_per_second,
            dt=args.dt,
        )
        report = executor.execute()
        report['summary_path'] = executor.write_summary(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        safety.close()
        imu.close()
        rover.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
