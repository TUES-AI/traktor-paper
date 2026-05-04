#!/usr/bin/env python3
"""Execute one rover-local 2D target through the deterministic executor."""

import argparse
import json

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150


def main():
    parser = argparse.ArgumentParser(description='Execute local target: +x forward, +y left, centimeters.')
    parser.add_argument('--x-cm', type=float, required=True)
    parser.add_argument('--y-cm', type=float, required=True)
    parser.add_argument('--turn-pwm', type=float, default=65.0)
    parser.add_argument('--drive-pwm', type=float, default=90.0)
    args = parser.parse_args()

    rover = RoverAPI(camera_enabled=False)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(rover, imu=imu, config=SafetyConfig())
    try:
        print(json.dumps({'gyro_z_bias': safety.calibrate_gyro()}, sort_keys=True), flush=True)
        executor = LocalTargetExecutor(
            safety,
            config=LocalTargetExecutorConfig(turn_pwm=args.turn_pwm, drive_pwm=args.drive_pwm),
            status_callback=lambda s: print(json.dumps({'status': s}, sort_keys=True), flush=True),
        )
        print(json.dumps(executor.execute_local_target(args.x_cm, args.y_cm), sort_keys=True), flush=True)
    finally:
        safety.close()
        imu.close()
        rover.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
