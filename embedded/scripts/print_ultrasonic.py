#!/usr/bin/env python3
"""Print raw ultrasonic readings for quick rover sensor checks."""

import argparse
import json
import time

import _paths  # noqa: F401
from api.rover_api import RoverAPI


def parse_args():
    p = argparse.ArgumentParser(description='Print rover ultrasonic sensor values.')
    p.add_argument('--interval', type=float, default=0.2)
    p.add_argument('--timeout', type=float, default=0.05)
    p.add_argument('--count', type=int, default=0, help='0 means run forever')
    return p.parse_args()


def main():
    args = parse_args()
    rover = RoverAPI(camera_enabled=False)
    try:
        i = 0
        while args.count <= 0 or i < args.count:
            i += 1
            raw = rover.get_ultrasonic(timeout_seconds=args.timeout)
            record = {
                'i': i,
                'time': time.time(),
                'right_cm': raw.get(1),
                'left_cm': raw.get(2),
                'front_cm': raw.get(3),
                'raw': raw,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            time.sleep(args.interval)
    finally:
        rover.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
