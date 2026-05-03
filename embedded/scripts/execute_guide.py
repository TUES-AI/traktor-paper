#!/usr/bin/env python3
"""Execute one deterministic local guide for real-world calibration.

Guide format:
    curvature: -1.0 right arc, 0.0 straight, +1.0 left arc
    horizon: seconds to execute
    speed: shared PWM duty cycle, 0-100

Because the current L298N wiring shares one PWM enable between both tracks,
curves are implemented by pulsing the inner track direction pins on/off.
"""

import argparse
import time

import _paths  # noqa: F401
from drivers.motor.hbridge import DualHBridgeMotorDriver
from hardware_pins import LEFT_MOTOR_PINS, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_PINS, RIGHT_MOTOR_PINS


PRESETS = {
    'straight': {'curvature': 0.0, 'horizon': 1.2, 'speed': 45.0, 'mode': 'arc'},
    'shallow_left': {'curvature': 0.35, 'horizon': 1.2, 'speed': 45.0, 'mode': 'arc'},
    'shallow_right': {'curvature': -0.35, 'horizon': 1.2, 'speed': 45.0, 'mode': 'arc'},
    'hard_left': {'curvature': 0.75, 'horizon': 1.0, 'speed': 45.0, 'mode': 'arc'},
    'hard_right': {'curvature': -0.75, 'horizon': 1.0, 'speed': 45.0, 'mode': 'arc'},
    'reverse': {'curvature': 0.0, 'horizon': 0.8, 'speed': 40.0, 'mode': 'reverse'},
    'spin_left': {'curvature': 1.0, 'horizon': 0.6, 'speed': 40.0, 'mode': 'spin_left'},
    'spin_right': {'curvature': -1.0, 'horizon': 0.6, 'speed': 40.0, 'mode': 'spin_right'},
}


def build_motor_driver():
    return DualHBridgeMotorDriver(
        left_in1=LEFT_MOTOR_PINS[0],
        left_in2=LEFT_MOTOR_PINS[1],
        right_in1=RIGHT_MOTOR_PINS[0],
        right_in2=RIGHT_MOTOR_PINS[1],
        left_pwm_pin=MOTOR_PWM_PINS[0],
        right_pwm_pin=MOTOR_PWM_PINS[1],
        pwm_frequency_hz=MOTOR_PWM_FREQUENCY_HZ,
    )


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def drive_arc(motor, curvature, horizon, speed, tick_seconds):
    curvature = clamp(float(curvature), -1.0, 1.0)
    speed = clamp(float(speed), 0.0, 100.0)
    horizon = max(0.0, float(horizon))

    turn_amount = abs(curvature)
    inner_ratio = clamp(1.0 - turn_amount, 0.0, 1.0)
    inner_acc = 0.0
    end_time = time.monotonic() + horizon

    print(
        f'arc: curvature={curvature:+.2f}, horizon={horizon:.2f}s, speed={speed:.1f}%, '
        f'inner_track_on_ratio={inner_ratio:.2f}',
        flush=True,
    )

    while time.monotonic() < end_time:
        if curvature == 0:
            left_direction, right_direction = 'forward', 'forward'
        elif curvature > 0:
            inner_acc += inner_ratio
            left_on = inner_acc >= 1.0
            if left_on:
                inner_acc -= 1.0
            left_direction = 'forward' if left_on else 'stop'
            right_direction = 'forward'
        else:
            inner_acc += inner_ratio
            right_on = inner_acc >= 1.0
            if right_on:
                inner_acc -= 1.0
            left_direction = 'forward'
            right_direction = 'forward' if right_on else 'stop'

        motor.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
        time.sleep(tick_seconds)


def drive_mode(motor, mode, curvature, horizon, speed, tick_seconds):
    if mode == 'arc':
        drive_arc(motor, curvature, horizon, speed, tick_seconds)
        return
    if mode == 'reverse':
        print(f'reverse: horizon={horizon:.2f}s, speed={speed:.1f}%', flush=True)
        motor.drive('backward', 'backward', left_speed=speed, right_speed=speed)
        time.sleep(horizon)
        return
    if mode == 'spin_left':
        print(f'spin_left: horizon={horizon:.2f}s, speed={speed:.1f}%', flush=True)
        motor.drive('backward', 'forward', left_speed=speed, right_speed=speed)
        time.sleep(horizon)
        return
    if mode == 'spin_right':
        print(f'spin_right: horizon={horizon:.2f}s, speed={speed:.1f}%', flush=True)
        motor.drive('forward', 'backward', left_speed=speed, right_speed=speed)
        time.sleep(horizon)
        return
    raise ValueError(f'Unknown mode: {mode}')


def print_observation_sheet(args, settings):
    print('\nObservation sheet to write down:', flush=True)
    print(f'- preset: {args.preset}', flush=True)
    print(f'- mode: {settings["mode"]}', flush=True)
    print(f'- curvature: {settings["curvature"]:+.2f}', flush=True)
    print(f'- horizon_s: {settings["horizon"]:.2f}', flush=True)
    print(f'- speed_pct: {settings["speed"]:.1f}', flush=True)
    print('- moved forward distance estimate:', flush=True)
    print('- yaw direction and rough angle:', flush=True)
    print('- did it match intended arc:', flush=True)
    print('- slip/stuck/vibration:', flush=True)
    print('- obstacle/safety issue:', flush=True)
    print('- notes:', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Execute one local guide for calibration.')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='shallow_left')
    parser.add_argument('--curvature', type=float, help='Override preset curvature, [-1, 1].')
    parser.add_argument('--horizon', type=float, help='Override preset horizon in seconds.')
    parser.add_argument('--speed', type=float, help='Override preset speed percent, [0, 100].')
    parser.add_argument('--tick', type=float, default=0.05, help='Software pulsing tick in seconds.')
    parser.add_argument('--dry-run', action='store_true', help='Print settings without moving motors.')
    args = parser.parse_args()

    settings = dict(PRESETS[args.preset])
    if args.curvature is not None:
        settings['curvature'] = args.curvature
        settings['mode'] = 'arc'
    if args.horizon is not None:
        settings['horizon'] = args.horizon
    if args.speed is not None:
        settings['speed'] = args.speed

    print(f'pins: left={LEFT_MOTOR_PINS}, right={RIGHT_MOTOR_PINS}, pwm={MOTOR_PWM_PINS}', flush=True)
    print_observation_sheet(args, settings)
    if args.dry_run:
        print('\ndry run: motors not started', flush=True)
        return 0

    motor = build_motor_driver()
    try:
        motor.stop()
        time.sleep(0.2)
        drive_mode(
            motor,
            settings['mode'],
            settings['curvature'],
            settings['horizon'],
            settings['speed'],
            args.tick,
        )
    finally:
        motor.cleanup()
        print('\ndone: motors stopped and GPIO cleaned up', flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
