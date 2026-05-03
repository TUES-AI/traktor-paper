#!/usr/bin/env python3
"""Run 100cm + turn repeated square-style calibration with front safety."""

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import _paths  # noqa: F401
from drivers.motor.hbridge import DualHBridgeMotorDriver
from drivers.sensors.ultrasonic_array import UltrasonicArray
from hardware_pins import LEFT_MOTOR_PINS, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_PINS, RIGHT_MOTOR_PINS
from hardware_pins import ULTRASONIC_1_PINS, ULTRASONIC_2_PINS, ULTRASONIC_3_PINS


def load_mpu9150():
    source = Path('/home/yasen/patatnik/embedded/drivers/sensors/mpu9150.py')
    spec = importlib.util.spec_from_file_location('mpu9150_driver', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MPU9150


def main():
    parser = argparse.ArgumentParser(description='Run square-ish guide sequence.')
    parser.add_argument('--drive-seconds', type=float, default=2.5)
    parser.add_argument('--drive-speed', type=float, default=55.0)
    parser.add_argument('--turn-direction', choices=('left', 'right'), default='right')
    parser.add_argument('--turn-target-deg', type=float, default=90.0)
    parser.add_argument('--turn-speed', type=float, default=55.0)
    parser.add_argument('--turn-max-seconds', type=float, default=6.0)
    parser.add_argument('--front-stop-cm', type=float, default=20.0)
    parser.add_argument('--turn-side-stop-cm', type=float, default=20.0)
    parser.add_argument('--dt', type=float, default=0.04)
    parser.add_argument('--out', default='/tmp/traktor_square_sequence.csv')
    args = parser.parse_args()

    MPU9150 = load_mpu9150()
    imu = MPU9150(bus=1, address=0x68)
    motor = DualHBridgeMotorDriver(
        left_in1=LEFT_MOTOR_PINS[0], left_in2=LEFT_MOTOR_PINS[1],
        right_in1=RIGHT_MOTOR_PINS[0], right_in2=RIGHT_MOTOR_PINS[1],
        left_pwm_pin=MOTOR_PWM_PINS[0], right_pwm_pin=MOTOR_PWM_PINS[1],
        pwm_frequency_hz=MOTOR_PWM_FREQUENCY_HZ,
    )
    sensors = UltrasonicArray(
        sensor1_trig=ULTRASONIC_1_PINS[0], sensor1_echo=ULTRASONIC_1_PINS[1],
        sensor2_trig=ULTRASONIC_2_PINS[0], sensor2_echo=ULTRASONIC_2_PINS[1],
        sensor3_trig=ULTRASONIC_3_PINS[0], sensor3_echo=ULTRASONIC_3_PINS[1],
        settle_seconds=0.05,
    )
    rows = []

    def row(phase, t_phase, yaw, bias_z, left_cmd, right_cmd, speed, front_cm=None, safety_stop=False):
        d = imu.read_all(); a = d['accel']; g = d['gyro']; o = d['orientation']; m = d.get('mag') or {'x': None, 'y': None, 'z': None}
        rows.append({
            'phase': phase, 't_phase_s': t_phase, 'yaw_phase_deg': yaw,
            'gyro_z_raw_dps': g['z'], 'gyro_z_unbiased_dps': g['z'] - bias_z,
            'gyro_x_dps': g['x'], 'gyro_y_dps': g['y'],
            'accel_x_g': a['x'], 'accel_y_g': a['y'], 'accel_z_g': a['z'],
            'roll_deg': o['roll'], 'pitch_deg': o['pitch'],
            'mag_x_uT': m['x'], 'mag_y_uT': m['y'], 'mag_z_uT': m['z'],
            'left_cmd': left_cmd, 'right_cmd': right_cmd, 'speed_pct': speed,
            'front_cm': front_cm, 'safety_stop': safety_stop,
        })

    def calibrate_bias():
        motor.stop(); samples=[]; start=time.monotonic()
        print('Calibrating gyro bias 1.2s. Keep still.', flush=True)
        while time.monotonic() - start < 1.2:
            samples.append(imu.read_all()['gyro']['z']); time.sleep(args.dt)
        bias = sum(samples) / max(1, len(samples))
        print(f'gyro_z_bias={bias:+.4f} deg/s', flush=True)
        return bias

    def forward(label, bias_z):
        print(f'{label}: forward {args.drive_speed}% for {args.drive_seconds}s', flush=True)
        motor.drive('forward', 'forward', left_speed=args.drive_speed, right_speed=args.drive_speed)
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            front_cm = sensors.read_sensor('front', timeout_seconds=0.01)
            safety_stop = front_cm is None or front_cm < args.front_stop_cm
            row(label, elapsed, 0.0, bias_z, 'forward', 'forward', args.drive_speed, front_cm, safety_stop)
            if safety_stop:
                front_text = 'NO_ECHO' if front_cm is None else f'{front_cm:.2f}cm'
                print(f'  safety stop: front={front_text} < {args.front_stop_cm:.2f}cm', flush=True)
                break
            if elapsed >= args.drive_seconds:
                break
            time.sleep(args.dt)
        motor.stop(); time.sleep(0.25)

    def turn(label, bias_z):
        left_cmd, right_cmd = ('backward', 'forward') if args.turn_direction == 'left' else ('forward', 'backward')
        print(f'{label}: turn {args.turn_direction} {args.turn_speed}% to {args.turn_target_deg}deg', flush=True)
        yaw = 0.0; last = time.monotonic(); start = last
        motor.drive(left_cmd, right_cmd, left_speed=args.turn_speed, right_speed=args.turn_speed)
        while True:
            now = time.monotonic(); dt = now - last; last = now
            gz = imu.read_all()['gyro']['z'] - bias_z; yaw += gz * dt; elapsed = now - start
            side_sensor = 'left' if args.turn_direction == 'left' else 'right'
            side_cm = sensors.read_sensor(side_sensor, timeout_seconds=0.01)
            side_stop = side_cm is None or side_cm < args.turn_side_stop_cm
            row(label, elapsed, yaw, bias_z, left_cmd, right_cmd, args.turn_speed, side_cm, side_stop)
            if side_stop:
                print(
                    f'  turn safety stop: {side_sensor}={"NO_ECHO" if side_cm is None else f"{side_cm:.2f}cm"} < {args.turn_side_stop_cm:.2f}cm',
                    flush=True,
                )
                break
            if abs(yaw) >= args.turn_target_deg:
                print(f'  reached yaw={yaw:+.1f}deg', flush=True); break
            if elapsed >= args.turn_max_seconds:
                print(f'  max turn time, yaw={yaw:+.1f}deg', flush=True); break
            time.sleep(args.dt)
        motor.stop(); time.sleep(0.25)

    try:
        print('Sequence: 100cm, turn, 100cm, turn, 100cm, turn, 100cm', flush=True)
        bias_z = calibrate_bias()
        forward('forward_1', bias_z); turn('turn_1', bias_z)
        forward('forward_2', bias_z); turn('turn_2', bias_z)
        forward('forward_3', bias_z); turn('turn_3', bias_z)
        forward('forward_4', bias_z)
    finally:
        motor.cleanup(); imu.close(); sensors.cleanup()
        if rows:
            out = Path(args.out)
            with out.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
            print(f'saved={out} rows={len(rows)}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
