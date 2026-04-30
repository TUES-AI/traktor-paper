#!/usr/bin/env python3
"""Simple autonomous movement sequence using the high-level RoverAPI."""

import time

import _paths  # noqa: F401
from api import RoverAPI

FORWARD_1_SECONDS = 2.0
SPIN_SECONDS = 1.5
FORWARD_2_SECONDS = 2.0
SPEED = 60


def main():
    rover = RoverAPI()
    try:
        print('Step 1: forward')
        rover.drive('forward', 'forward', left_speed=SPEED, right_speed=SPEED)
        time.sleep(FORWARD_1_SECONDS)

        print('Step 2: spin left')
        rover.drive('backward', 'forward', left_speed=SPEED, right_speed=SPEED)
        time.sleep(SPIN_SECONDS)

        print('Step 3: forward')
        rover.drive('forward', 'forward', left_speed=SPEED, right_speed=SPEED)
        time.sleep(FORWARD_2_SECONDS)
    finally:
        rover.stop_motors()
        rover.close()
        print('Done: motors stopped and GPIO cleaned up.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
