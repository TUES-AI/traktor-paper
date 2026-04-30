#!/usr/bin/env python3
"""Short motor smoke test: forward, reverse, then stop and cleanup."""

import time

import _paths  # noqa: F401
from drivers.motor.hbridge import DualHBridgeMotorDriver
from hardware_pins import LEFT_MOTOR_PINS, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_PINS, RIGHT_MOTOR_PINS

STEP_SECONDS = 2.0
TEST_SPEED = 70


def main():
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
        print(f'Forward at {TEST_SPEED}%')
        driver.drive('forward', 'forward', left_speed=TEST_SPEED, right_speed=TEST_SPEED)
        time.sleep(STEP_SECONDS)

        print(f'Reverse at {TEST_SPEED}%')
        driver.drive('backward', 'backward', left_speed=TEST_SPEED, right_speed=TEST_SPEED)
        time.sleep(STEP_SECONDS)
    finally:
        driver.cleanup()
        print('Motors stopped and GPIO cleaned up.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
