#!/usr/bin/env python3
"""Execute one SAC-style local guide vector through SafetyController."""

import argparse
import json

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.guide_executor import GuideExecutor
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150


PRESETS = {
    'straight': (0.0, 1.2, 55.0),
    'avoid_left': (0.75, 1.8, 60.0),
    'avoid_right': (-0.75, 1.8, 60.0),
    'shallow_left': (0.35, 1.4, 55.0),
    'shallow_right': (-0.35, 1.4, 55.0),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Execute [curvature, horizon, speed] through the safety layer.')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='avoid_left')
    parser.add_argument('--curvature', type=float, help='Override curvature, [-1, 1]. Positive means left.')
    parser.add_argument('--horizon', type=float, help='Override horizon in seconds.')
    parser.add_argument('--speed', type=float, help='Override speed percent, [0, 100].')
    parser.add_argument('--tick', type=float, default=0.05)
    parser.add_argument('--turn-speed', type=float, default=70.0)
    parser.add_argument('--min-front-stop-cm', type=float, default=10.0)
    parser.add_argument('--max-front-stop-cm', type=float, default=35.0)
    parser.add_argument('--side-turn-clear-cm', type=float, default=20.0)
    parser.add_argument('--front-clear-to-resume-cm', type=float, default=30.0)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    curvature, horizon, speed = PRESETS[args.preset]
    if args.curvature is not None:
        curvature = args.curvature
    if args.horizon is not None:
        horizon = args.horizon
    if args.speed is not None:
        speed = args.speed

    guide = {'curvature': curvature, 'horizon_s': horizon, 'speed_pct': speed}
    print(json.dumps({'guide': guide, 'dry_run': args.dry_run}, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    rover = RoverAPI(camera_enabled=False)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(
        rover,
        imu=imu,
        config=SafetyConfig(
            min_front_stop_cm=args.min_front_stop_cm,
            max_front_stop_cm=args.max_front_stop_cm,
            side_turn_clear_cm=args.side_turn_clear_cm,
            front_clear_to_resume_cm=args.front_clear_to_resume_cm,
        ),
    )
    executor = GuideExecutor(safety, tick_seconds=args.tick, turn_speed=args.turn_speed)

    try:
        bias = safety.calibrate_gyro()
        print(json.dumps({'gyro_z_bias': bias}, sort_keys=True), flush=True)
        report = executor.execute(curvature, horizon, speed)
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        safety.close()
        imu.close()
        rover.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
