#!/usr/bin/env python3
"""Raw low-level motor profiler with no sensors and no safety layer."""

import argparse
import json
import time

import RPi.GPIO as GPIO

import _paths  # noqa: F401
from drivers.motor.hbridge import DualHBridgeMotorDriver
from hardware_pins import LEFT_MOTOR_PINS, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_PINS, RIGHT_MOTOR_PINS


TESTS = {
    'left_forward': ('forward', 'stop'),
    'left_backward': ('backward', 'stop'),
    'right_forward': ('stop', 'forward'),
    'right_backward': ('stop', 'backward'),
    'both_forward': ('forward', 'forward'),
    'both_backward': ('backward', 'backward'),
    'spin_left': ('backward', 'forward'),
    'spin_right': ('forward', 'backward'),
}


def build_motor():
    return DualHBridgeMotorDriver(
        left_in1=LEFT_MOTOR_PINS[0],
        left_in2=LEFT_MOTOR_PINS[1],
        right_in1=RIGHT_MOTOR_PINS[0],
        right_in2=RIGHT_MOTOR_PINS[1],
        left_pwm_pin=MOTOR_PWM_PINS[0],
        right_pwm_pin=MOTOR_PWM_PINS[1],
        pwm_frequency_hz=MOTOR_PWM_FREQUENCY_HZ,
    )


def pin_snapshot():
    pins = {
        'left_in1': LEFT_MOTOR_PINS[0],
        'left_in2': LEFT_MOTOR_PINS[1],
        'right_in1': RIGHT_MOTOR_PINS[0],
        'right_in2': RIGHT_MOTOR_PINS[1],
        'pwm': MOTOR_PWM_PINS[0],
    }
    return {name: GPIO.input(pin) for name, pin in pins.items()}


def run_test(motor, name, duration, speed, settle):
    left_direction, right_direction = TESTS[name]
    print(json.dumps({'test': name, 'phase': 'start', 'left': left_direction, 'right': right_direction, 'speed': speed}), flush=True)
    result = motor.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
    time.sleep(0.03)
    print(json.dumps({'test': name, 'phase': 'pins_after_drive', 'driver_result': result, 'pins': pin_snapshot()}), flush=True)
    time.sleep(duration)
    motor.stop()
    time.sleep(0.03)
    print(json.dumps({'test': name, 'phase': 'pins_after_stop', 'pins': pin_snapshot()}), flush=True)
    time.sleep(settle)


def main():
    parser = argparse.ArgumentParser(description='Run raw motor commands without sensors/safety.')
    parser.add_argument('--test', choices=sorted(TESTS) + ['all'], default='all')
    parser.add_argument('--duration', type=float, default=0.8)
    parser.add_argument('--speed', type=float, default=100.0)
    parser.add_argument('--settle', type=float, default=0.4)
    args = parser.parse_args()

    print(json.dumps({
        'pins': {
            'left': LEFT_MOTOR_PINS,
            'right': RIGHT_MOTOR_PINS,
            'pwm': MOTOR_PWM_PINS,
            'pwm_hz': MOTOR_PWM_FREQUENCY_HZ,
        },
        'duration_s': args.duration,
        'speed_pct': args.speed,
        'test': args.test,
    }), flush=True)

    motor = build_motor()
    try:
        motor.stop()
        tests = list(TESTS) if args.test == 'all' else [args.test]
        for name in tests:
            run_test(motor, name, args.duration, args.speed, args.settle)
    finally:
        motor.cleanup()
        print(json.dumps({'done': True, 'motors': 'stopped', 'gpio': 'cleaned'}), flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
