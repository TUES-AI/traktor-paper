#!/usr/bin/env python3
"""Minimal WASD drive control.

Keys:
- w: forward
- s: backward
- a: spin left
- d: spin right
- x: stop
- q: quit

Hold a key to run. If no repeated key arrives for a short timeout, motors stop.
"""

import sys
import termios
import time
import tty
from select import select

import _paths  # noqa: F401
from drivers.motor.hbridge import DualHBridgeMotorDriver
from hardware_pins import LEFT_MOTOR_PINS, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_PINS, RIGHT_MOTOR_PINS

SPEED = 100
STOP_AFTER_SECONDS = 0.22

COMMANDS = {
    'w': ('forward', 'forward', 'forward'),
    's': ('backward', 'backward', 'backward'),
    'a': ('backward', 'forward', 'spin left'),
    'd': ('forward', 'backward', 'spin right'),
}


def set_raw_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    return fd, old_settings


def read_pending_key():
    ready, _, _ = select([sys.stdin], [], [], 0.02)
    if not ready:
        return None
    key = sys.stdin.read(1).lower()
    while select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1).lower()
    return key


def main():
    motor = DualHBridgeMotorDriver(
        left_in1=LEFT_MOTOR_PINS[0],
        left_in2=LEFT_MOTOR_PINS[1],
        right_in1=RIGHT_MOTOR_PINS[0],
        right_in2=RIGHT_MOTOR_PINS[1],
        left_pwm_pin=MOTOR_PWM_PINS[0],
        right_pwm_pin=MOTOR_PWM_PINS[1],
        pwm_frequency_hz=MOTOR_PWM_FREQUENCY_HZ,
    )

    print(__doc__.strip(), flush=True)
    print(f'pins: left={LEFT_MOTOR_PINS}, right={RIGHT_MOTOR_PINS}, pwm={MOTOR_PWM_PINS}', flush=True)
    print(f'speed: {SPEED}%', flush=True)
    print(f'hold-to-run timeout: {STOP_AFTER_SECONDS}s', flush=True)

    fd = None
    old_settings = None
    active_key = None
    last_key_time = 0.0
    try:
        fd, old_settings = set_raw_terminal()
        motor.stop()
        while True:
            now = time.monotonic()
            key = read_pending_key()

            if active_key and now - last_key_time > STOP_AFTER_SECONDS:
                motor.stop()
                print('stop', flush=True)
                active_key = None

            if key is None:
                continue

            if key == 'q':
                print('quit', flush=True)
                break
            if key == 'x':
                motor.stop()
                print('stop', flush=True)
                active_key = None
                continue
            if key not in COMMANDS:
                continue

            left_direction, right_direction, label = COMMANDS[key]
            if key != active_key:
                motor.drive(left_direction, right_direction, left_speed=SPEED, right_speed=SPEED)
                print(f'{label}: left={left_direction}, right={right_direction}', flush=True)
            active_key = key
            last_key_time = now
    finally:
        if fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        motor.cleanup()
        print('motors stopped and GPIO cleaned up', flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
