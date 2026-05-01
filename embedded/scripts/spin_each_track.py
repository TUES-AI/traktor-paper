#!/usr/bin/env python3
"""Spin left track for 10 seconds, then right track for 10 seconds."""

import argparse
import time

import _paths  # noqa: F401
from drivers.motor.hbridge import DualHBridgeMotorDriver
from hardware_pins import LEFT_MOTOR_PINS, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_PINS, RIGHT_MOTOR_PINS


def main():
    parser = argparse.ArgumentParser(description='Spin each track separately for a fixed duration.')
    parser.add_argument('--speed', type=float, default=50.0, help='PWM duty cycle, 0-100.')
    parser.add_argument('--seconds', type=float, default=10.0, help='Seconds per track.')
    parser.add_argument('--direction', default='forward', choices=('forward', 'backward'))
    args = parser.parse_args()

    driver = DualHBridgeMotorDriver(
        left_in1=LEFT_MOTOR_PINS[0],
        left_in2=LEFT_MOTOR_PINS[1],
        right_in1=RIGHT_MOTOR_PINS[0],
        right_in2=RIGHT_MOTOR_PINS[1],
        left_pwm_pin=MOTOR_PWM_PINS[0],
        right_pwm_pin=MOTOR_PWM_PINS[1],
        pwm_frequency_hz=MOTOR_PWM_FREQUENCY_HZ,
    )

    try:
        print(f'Left track {args.direction} at {args.speed}% for {args.seconds}s')
        driver.drive(args.direction, 'stop', left_speed=args.speed, right_speed=0)
        time.sleep(args.seconds)

        print('Stop 1s')
        driver.stop()
        time.sleep(1.0)

        print(f'Right track {args.direction} at {args.speed}% for {args.seconds}s')
        driver.drive('stop', args.direction, left_speed=0, right_speed=args.speed)
        time.sleep(args.seconds)
    finally:
        driver.cleanup()
        print('Motors stopped and GPIO cleaned up.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
