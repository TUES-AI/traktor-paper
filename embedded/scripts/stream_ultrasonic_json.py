#!/usr/bin/env python3
"""Stream ultrasonic readings as JSON lines for remote visualization."""

import argparse
import json
import time

import _paths  # noqa: F401
from drivers.sensors.ultrasonic_array import UltrasonicArray
from hardware_pins import ULTRASONIC_1_PINS, ULTRASONIC_2_PINS, ULTRASONIC_3_PINS


def main():
    parser = argparse.ArgumentParser(description='Stream ultrasonic readings as JSON lines.')
    parser.add_argument('--hz', type=float, default=8.0)
    parser.add_argument('--timeout', type=float, default=0.02)
    args = parser.parse_args()

    sensors = UltrasonicArray(
        sensor1_trig=ULTRASONIC_1_PINS[0], sensor1_echo=ULTRASONIC_1_PINS[1],
        sensor2_trig=ULTRASONIC_2_PINS[0], sensor2_echo=ULTRASONIC_2_PINS[1],
        sensor3_trig=ULTRASONIC_3_PINS[0], sensor3_echo=ULTRASONIC_3_PINS[1],
        settle_seconds=0.05,
    )
    delay = 1.0 / args.hz if args.hz > 0 else 0.1
    try:
        while True:
            d = sensors.read_all(timeout_seconds=args.timeout)
            print(json.dumps({'t': time.time(), 'left': d[2], 'right': d[1], 'front': d[3]}), flush=True)
            time.sleep(delay)
    finally:
        sensors.cleanup()


if __name__ == '__main__':
    raise SystemExit(main())
