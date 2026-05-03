#!/usr/bin/env python3
"""Run one forward -> 180 -> forward -> 180 calibration sequence."""

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
    local_path = Path(__file__).resolve().parents[1] / 'drivers' / 'sensors' / 'mpu9150.py'
    fallback_path = Path('/home/yasen/patatnik/embedded/drivers/sensors/mpu9150.py')
    source = local_path if local_path.exists() else fallback_path
    spec = importlib.util.spec_from_file_location('mpu9150_driver', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MPU9150


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


def main():
    parser = argparse.ArgumentParser(description='Run one forward/spin/return/spin sequence.')
    parser.add_argument('--forward-1-seconds', type=float, default=2.0)
    parser.add_argument('--forward-2-seconds', type=float, default=2.0)
    parser.add_argument('--forward-speed', type=float, default=55.0)
    parser.add_argument('--drive-direction', choices=('forward', 'backward'), default='forward')
    parser.add_argument('--spin-direction', choices=('left', 'right'), default='left')
    parser.add_argument('--spin-speed', type=float, default=55.0)
    parser.add_argument('--spin-target-deg', type=float, default=180.0)
    parser.add_argument('--spin-1-target-deg', type=float)
    parser.add_argument('--spin-2-target-deg', type=float)
    parser.add_argument('--spin-max-seconds', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.04)
    parser.add_argument('--calibration-seconds', type=float, default=1.2)
    parser.add_argument('--front-stop-cm', type=float, default=20.0)
    parser.add_argument('--turn-side-stop-cm', type=float, default=20.0)
    parser.add_argument('--out', default='/tmp/traktor_forward_spin_sequence.csv')
    args = parser.parse_args()

    MPU9150 = load_mpu9150()
    imu = MPU9150(bus=1, address=0x68)
    motor = build_motor_driver()
    sensors = UltrasonicArray(
        sensor1_trig=ULTRASONIC_1_PINS[0],
        sensor1_echo=ULTRASONIC_1_PINS[1],
        sensor2_trig=ULTRASONIC_2_PINS[0],
        sensor2_echo=ULTRASONIC_2_PINS[1],
        sensor3_trig=ULTRASONIC_3_PINS[0],
        sensor3_echo=ULTRASONIC_3_PINS[1],
        settle_seconds=0.05,
    )
    rows = []

    def read_row(phase, t_phase, yaw, bias_z, left_cmd, right_cmd, speed, front_cm=None, safety_stop=False):
        d = imu.read_all()
        a = d['accel']
        g = d['gyro']
        o = d['orientation']
        m = d.get('mag') or {'x': None, 'y': None, 'z': None}
        rows.append({
            'phase': phase,
            't_phase_s': t_phase,
            'yaw_phase_deg': yaw,
            'gyro_z_raw_dps': g['z'],
            'gyro_z_unbiased_dps': g['z'] - bias_z,
            'gyro_x_dps': g['x'],
            'gyro_y_dps': g['y'],
            'accel_x_g': a['x'],
            'accel_y_g': a['y'],
            'accel_z_g': a['z'],
            'roll_deg': o['roll'],
            'pitch_deg': o['pitch'],
            'mag_x_uT': m['x'],
            'mag_y_uT': m['y'],
            'mag_z_uT': m['z'],
            'left_cmd': left_cmd,
            'right_cmd': right_cmd,
            'speed_pct': speed,
            'front_cm': front_cm,
            'safety_stop': safety_stop,
        })

    def calibrate_bias():
        print(f'Calibrating gyro bias {args.calibration_seconds}s. Keep still.', flush=True)
        motor.stop()
        samples = []
        start = time.monotonic()
        while time.monotonic() - start < args.calibration_seconds:
            samples.append(imu.read_all()['gyro']['z'])
            time.sleep(args.dt)
        bias = sum(samples) / max(1, len(samples))
        print(f'gyro_z_bias={bias:+.4f} deg/s', flush=True)
        return bias

    def drive_forward(label, seconds, bias_z):
        print(f'{label}: {args.drive_direction} {args.forward_speed}% for {seconds:.2f}s', flush=True)
        motor.drive(args.drive_direction, args.drive_direction, left_speed=args.forward_speed, right_speed=args.forward_speed)
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            front_cm = sensors.read_sensor('front', timeout_seconds=0.01)
            safety_stop = front_cm is None or front_cm < args.front_stop_cm
            read_row(label, elapsed, 0.0, bias_z, args.drive_direction, args.drive_direction, args.forward_speed, front_cm, safety_stop)
            if args.drive_direction == 'forward' and safety_stop:
                front_text = 'NO_ECHO' if front_cm is None else f'{front_cm:.2f}cm'
                print(f'  safety stop: front={front_text} < {args.front_stop_cm:.2f}cm', flush=True)
                break
            if elapsed >= seconds:
                break
            time.sleep(args.dt)
        motor.stop()
        time.sleep(0.25)

    def spin_180(label, bias_z, target_deg):
        print(f'{label}: spin {args.spin_direction} {args.spin_speed}% to {target_deg} deg', flush=True)
        yaw = 0.0
        last = time.monotonic()
        start = last
        left_cmd, right_cmd = ('backward', 'forward') if args.spin_direction == 'left' else ('forward', 'backward')
        motor.drive(left_cmd, right_cmd, left_speed=args.spin_speed, right_speed=args.spin_speed)
        while True:
            now = time.monotonic()
            dt = now - last
            last = now
            d = imu.read_all()
            gz = d['gyro']['z'] - bias_z
            yaw += gz * dt
            elapsed = now - start
            side_sensor = 'left' if args.spin_direction == 'left' else 'right'
            side_cm = sensors.read_sensor(side_sensor, timeout_seconds=0.01)
            side_stop = side_cm is None or side_cm < args.turn_side_stop_cm
            read_row(label, elapsed, yaw, bias_z, left_cmd, right_cmd, args.spin_speed)
            if side_stop:
                print(
                    f'  turn safety stop: {side_sensor}={"NO_ECHO" if side_cm is None else f"{side_cm:.2f}cm"} < {args.turn_side_stop_cm:.2f}cm',
                    flush=True,
                )
                break
            if len(rows) % 15 == 0:
                print(f'  t={elapsed:4.2f}s yaw={yaw:+6.1f}deg gz={gz:+7.1f}dps', flush=True)
            if abs(yaw) >= target_deg:
                print(f'  reached yaw={yaw:+.1f}deg', flush=True)
                break
            if elapsed >= args.spin_max_seconds:
                print(f'  max spin time, yaw={yaw:+.1f}deg', flush=True)
                break
            time.sleep(args.dt)
        motor.stop()
        time.sleep(0.25)

    try:
        print(
            f'sequence f1={args.forward_1_seconds:.2f}s f2={args.forward_2_seconds:.2f}s '
            f'forward_speed={args.forward_speed}% spin_speed={args.spin_speed}%',
            flush=True,
        )
        bias_z = calibrate_bias()
        spin_1_target = args.spin_1_target_deg if args.spin_1_target_deg is not None else args.spin_target_deg
        spin_2_target = args.spin_2_target_deg if args.spin_2_target_deg is not None else args.spin_target_deg
        drive_forward('forward_1', args.forward_1_seconds, bias_z)
        spin_180('spin_1', bias_z, spin_1_target)
        drive_forward('forward_2', args.forward_2_seconds, bias_z)
        spin_180('spin_2', bias_z, spin_2_target)
    finally:
        motor.cleanup()
        imu.close()
        sensors.cleanup()
        if rows:
            out = Path(args.out)
            with out.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f'saved={out} rows={len(rows)}', flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
