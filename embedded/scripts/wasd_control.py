#!/usr/bin/env python3
"""Interactive keyboard motor control.

Keys:
- w/s: forward/backward
- a/d: spin left/right
- x: stop
- +/-: adjust speed
- 1-9: set 10-90 percent speed
- 0: set 100 percent speed
- q: quit
"""

import sys
import termios
import tty

import _paths  # noqa: F401
from api import RoverAPI

SPEED_STEP = 10
DEFAULT_SPEED = 60


def clamp_speed(value):
    return max(0, min(100, int(value)))


def read_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def apply_drive(rover, left_direction, right_direction, speed):
    return rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)


def main():
    rover = RoverAPI()
    speed = DEFAULT_SPEED
    left_direction = 'stop'
    right_direction = 'stop'

    print(__doc__.strip())
    print(f'Current speed: {speed}%')

    try:
        while True:
            key = read_key()

            if key in ('+', '='):
                speed = clamp_speed(speed + SPEED_STEP)
            elif key == '-':
                speed = clamp_speed(speed - SPEED_STEP)
            elif key.isdigit():
                speed = 100 if key == '0' else int(key) * 10
            else:
                key = key.lower()
                if key == 'w':
                    left_direction, right_direction = 'forward', 'forward'
                elif key == 's':
                    left_direction, right_direction = 'backward', 'backward'
                elif key == 'a':
                    left_direction, right_direction = 'backward', 'forward'
                elif key == 'd':
                    left_direction, right_direction = 'forward', 'backward'
                elif key == 'x':
                    left_direction, right_direction = 'stop', 'stop'
                elif key == 'q':
                    print('quit')
                    break
                else:
                    continue

            if left_direction == 'stop' and right_direction == 'stop':
                rover.stop_motors()
                print('stop')
            else:
                apply_drive(rover, left_direction, right_direction, speed)
                print(f'{left_direction}/{right_direction} at {speed}%')
    finally:
        rover.stop_motors()
        rover.close()
        print('Motors stopped and GPIO cleaned up.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
