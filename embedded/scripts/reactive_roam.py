#!/usr/bin/env python3
"""Run reactive safety-first roaming.

Layering:
    RoverAPI/raw drivers -> SafetyController -> ReactiveRoamPolicy
"""

import argparse
import json

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.reactive_roam import ReactiveRoamPolicy
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150


def main():
    parser = argparse.ArgumentParser(description='Run safety-filtered reactive roaming.')
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--slow-speed', type=float, default=45.0)
    parser.add_argument('--fast-speed', type=float, default=65.0)
    parser.add_argument('--turn-speed', type=float, default=70.0)
    parser.add_argument('--fast-clear-cm', type=float, default=100.0)
    parser.add_argument('--min-front-stop-cm', type=float, default=10.0)
    parser.add_argument('--max-front-stop-cm', type=float, default=35.0)
    parser.add_argument('--side-turn-clear-cm', type=float, default=20.0)
    parser.add_argument('--front-clear-to-resume-cm', type=float, default=30.0)
    parser.add_argument('--wall-seen-cm', type=float, default=45.0)
    args = parser.parse_args()

    rover = RoverAPI(camera_enabled=False)
    imu = MPU9150(bus=1, address=0x68)
    config = SafetyConfig(
        min_front_stop_cm=args.min_front_stop_cm,
        max_front_stop_cm=args.max_front_stop_cm,
        side_turn_clear_cm=args.side_turn_clear_cm,
        front_clear_to_resume_cm=args.front_clear_to_resume_cm,
        wall_seen_cm=args.wall_seen_cm,
    )
    safety = SafetyController(rover, imu=imu, config=config)
    policy = ReactiveRoamPolicy(
        safety,
        slow_speed=args.slow_speed,
        fast_speed=args.fast_speed,
        turn_speed=args.turn_speed,
        fast_clear_cm=args.fast_clear_cm,
    )

    try:
        bias = safety.calibrate_gyro()
        print(f'gyro_z_bias={bias:+.4f} deg/s', flush=True)
        for report in policy.run(seconds=args.seconds):
            print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        safety.close()
        imu.close()
        rover.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
