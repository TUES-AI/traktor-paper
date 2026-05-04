#!/usr/bin/env python3
"""Interactive continuous preset motion sequences.

Keys:
- w: manual forward while held/repeated
- s: manual backward while held/repeated
- a: manual spin left while held/repeated
- d: manual spin right while held/repeated
- 1: short straight forward
- 2: left lane-offset
- 3: right lane-offset
- 4: left obstacle-bypass shape
- 5: right obstacle-bypass shape
- x: stop motors
- q: quit

WASD is raw/manual because you are watching the rover. Numbered presets run
through SafetyController using ultrasonic checks. Do not run this together with
other GPIO-owning scripts. Use long continuous segments; this chassis barely
moves with tiny start/stop ticks.
"""

import sys
import termios
import time
import tty
from select import select

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150
from hardware_pins import LEFT_MOTOR_PINS, MOTOR_PWM_FREQUENCY_HZ, MOTOR_PWM_PINS, RIGHT_MOTOR_PINS


SPEED = 100.0
TURN_SPEED = 65.0
PAUSE_SECONDS = 0.18
STOP_AFTER_SECONDS = 0.22
SAFETY_DT_SECONDS = 0.05
TURN_TOLERANCE_DEG = 5.0
MAX_TURN_SECONDS = 3.0

COMMANDS = {
    'forward': ('forward', 'forward'),
    'backward': ('backward', 'backward'),
    'spin_left': ('backward', 'forward'),
    'spin_right': ('forward', 'backward'),
}

MANUAL_KEYS = {
    'w': ('forward', 'manual forward'),
    's': ('backward', 'manual backward'),
    'a': ('spin_left', 'manual spin left'),
    'd': ('spin_right', 'manual spin right'),
}

SEQUENCES = {
    '1': {
        'name': 'short straight forward',
        'segments': [('forward', 1.0)],
        'shape': [(0, 0), (45, 0)],
    },
    '2': {
        'name': 'left lane-offset',
        'segments': [('turn_left', 25.0), ('forward', 0.85), ('turn_right', 25.0), ('forward', 0.65)],
        'shape': [(0, 0), (18, 18), (55, 18), (72, 0)],
    },
    '3': {
        'name': 'right lane-offset',
        'segments': [('turn_right', 25.0), ('forward', 0.85), ('turn_left', 25.0), ('forward', 0.65)],
        'shape': [(0, 0), (18, -18), (55, -18), (72, 0)],
    },
    '4': {
        'name': 'left obstacle-bypass',
        'segments': [('turn_left', 25.0), ('forward', 0.70), ('turn_right', 50.0), ('forward', 1.00), ('turn_left', 25.0), ('forward', 0.45)],
        'shape': [(0, 0), (16, 18), (45, 24), (72, 8), (92, 0)],
    },
    '5': {
        'name': 'right obstacle-bypass',
        'segments': [('turn_right', 25.0), ('forward', 0.70), ('turn_left', 50.0), ('forward', 1.00), ('turn_right', 25.0), ('forward', 0.45)],
        'shape': [(0, 0), (16, -18), (45, -24), (72, -8), (92, 0)],
    },
}


def set_raw_terminal():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    return fd, old_settings


def read_pending_key():
    ready, _, _ = select([sys.stdin], [], [], 0.03)
    if not ready:
        return None
    key = sys.stdin.read(1).lower()
    while select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1).lower()
    return key


def print_menu():
    print(__doc__.strip(), flush=True)
    print(f'pins: left={LEFT_MOTOR_PINS}, right={RIGHT_MOTOR_PINS}, pwm={MOTOR_PWM_PINS}', flush=True)
    print(f'speed: {SPEED:.0f}%', flush=True)
    print(f'sequence turn speed: {TURN_SPEED:.0f}%, IMU target tolerance: {TURN_TOLERANCE_DEG:.1f}deg', flush=True)
    print(f'WASD hold-to-run timeout: {STOP_AFTER_SECONDS:.2f}s', flush=True)
    print('WASD: raw manual motor control, no safety', flush=True)
    print('1-5: SafetyController ultrasonic checks enabled', flush=True)
    print('diagram: docs/preset_sequences.png', flush=True)
    for key, spec in sorted(SEQUENCES.items()):
        parts = ', '.join(f'{cmd}:{duration:.2f}s' for cmd, duration in spec['segments'])
        print(f'{key}: {spec["name"]} -> {parts}', flush=True)


def command_is_safe(safety, command):
    distances = safety.read_distances()
    if command == 'forward':
        safe, front, threshold = safety.is_front_safe(SPEED, distances)
        if not safe:
            return False, f'front_blocked front={front} threshold={threshold:.1f}', distances
        return True, 'front_clear', distances
    if command in ('spin_left', 'turn_left'):
        left = distances['left']
        if left is None:
            return False, 'left_no_echo', distances
        if left < safety.config.side_turn_clear_cm:
            return False, f'left_blocked left={left:.1f}', distances
        return True, 'left_clear', distances
    if command in ('spin_right', 'turn_right'):
        right = distances['right']
        if right is None:
            return False, 'right_no_echo', distances
        if right < safety.config.side_turn_clear_cm:
            return False, f'right_blocked right={right:.1f}', distances
        return True, 'right_clear', distances
    return True, 'manual_or_reverse', distances


def run_turn_segment(safety, command, target_deg):
    direction = 'left' if command == 'turn_left' else 'right'
    left, right = COMMANDS['spin_left' if direction == 'left' else 'spin_right']
    signed_target = abs(target_deg) if direction == 'left' else -abs(target_deg)
    yaw = 0.0
    last = time.monotonic()
    start = last
    print(f'  {command}: target={target_deg:.1f}deg speed={TURN_SPEED:.0f}%', flush=True)
    safe, reason, distances = command_is_safe(safety, command)
    if not safe:
        safety.rover.stop_motors()
        print(f'  SAFETY STOP before {command}: {reason}, distances={distances}', flush=True)
        return False

    safety.rover.drive(left, right, left_speed=TURN_SPEED, right_speed=TURN_SPEED)
    try:
        while time.monotonic() - start < MAX_TURN_SECONDS:
            safe, reason, distances = command_is_safe(safety, command)
            if not safe:
                print(f'  SAFETY STOP during {command}: {reason}, yaw={yaw:.1f}deg, distances={distances}', flush=True)
                return False
            now = time.monotonic()
            gyro_z = safety.imu.read_all()['gyro']['z'] - safety._gyro_z_bias if safety.imu is not None else 0.0
            yaw += gyro_z * (now - last)
            last = now
            if direction == 'left' and yaw >= signed_target - TURN_TOLERANCE_DEG:
                print(f'  turn reached: yaw={yaw:.1f}deg target={signed_target:.1f}deg', flush=True)
                return True
            if direction == 'right' and yaw <= signed_target + TURN_TOLERANCE_DEG:
                print(f'  turn reached: yaw={yaw:.1f}deg target={signed_target:.1f}deg', flush=True)
                return True
            time.sleep(SAFETY_DT_SECONDS)
        print(f'  TURN TIMEOUT: yaw={yaw:.1f}deg target={signed_target:.1f}deg', flush=True)
        return False
    finally:
        safety.rover.stop_motors()
        time.sleep(PAUSE_SECONDS)


def run_segment(safety, command, duration):
    if command in ('turn_left', 'turn_right'):
        return run_turn_segment(safety, command, duration)
    left, right = COMMANDS[command]
    print(f'  {command}: {duration:.2f}s', flush=True)
    start = time.monotonic()
    safe, reason, distances = command_is_safe(safety, command)
    if not safe:
        safety.rover.stop_motors()
        print(f'  SAFETY STOP before {command}: {reason}, distances={distances}', flush=True)
        return False
    safety.rover.drive(left, right, left_speed=SPEED, right_speed=SPEED)
    try:
        while time.monotonic() - start < duration:
            safe, reason, distances = command_is_safe(safety, command)
            if not safe:
                print(f'  SAFETY STOP during {command}: {reason}, distances={distances}', flush=True)
                return False
            time.sleep(SAFETY_DT_SECONDS)
    finally:
        safety.rover.stop_motors()
    time.sleep(PAUSE_SECONDS)
    return True


def run_sequence(safety, key):
    spec = SEQUENCES[key]
    print(f'RUN {key}: {spec["name"]}', flush=True)
    try:
        for command, duration in spec['segments']:
            if not run_segment(safety, command, duration):
                print(f'ABORT {key}: {spec["name"]}', flush=True)
                return
    finally:
        safety.rover.stop_motors()
    print(f'DONE {key}: {spec["name"]}', flush=True)


def start_manual(rover, manual_key):
    command, label = MANUAL_KEYS[manual_key]
    left, right = COMMANDS[command]
    rover.drive(left, right, left_speed=SPEED, right_speed=SPEED)
    print(label, flush=True)


def main():
    rover = RoverAPI(camera_enabled=False)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(rover, imu=imu, config=SafetyConfig())
    fd = None
    old_settings = None
    try:
        print_menu()
        print(f'calibrating gyro z bias: {safety.calibrate_gyro():.4f} deg/s', flush=True)
        fd, old_settings = set_raw_terminal()
        rover.stop_motors()
        active_manual_key = None
        last_manual_time = 0.0
        while True:
            now = time.monotonic()
            if active_manual_key and now - last_manual_time > STOP_AFTER_SECONDS:
                rover.stop_motors()
                print('manual stop', flush=True)
                active_manual_key = None

            key = read_pending_key()
            if key is None:
                continue
            if key == 'q':
                print('quit', flush=True)
                break
            if key == 'x':
                rover.stop_motors()
                print('stop', flush=True)
                active_manual_key = None
                continue
            if key in MANUAL_KEYS:
                if key != active_manual_key:
                    start_manual(rover, key)
                active_manual_key = key
                last_manual_time = now
                continue
            if key in SEQUENCES:
                active_manual_key = None
                rover.stop_motors()
                run_sequence(safety, key)
                continue
    finally:
        if fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        imu.close()
        rover.close()
        print('motors stopped and GPIO cleaned up', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
