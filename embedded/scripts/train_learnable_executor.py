#!/usr/bin/env python3
"""Train a tiny continuous-primitive executor from real rover feedback.

The policy does not output ticks. It outputs parameters for two continuous
primitives:

1. turn toward a dynamically clipped local target angle
2. drive forward toward a dynamically clipped local target distance

SafetyController remains the hard gate before/during motor execution.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass

import _paths  # noqa: F401
import torch
from api.rover_api import RoverAPI
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150


MIN_TURN_PWM = 55.0
MAX_TURN_PWM = 90.0
MIN_DRIVE_PWM = 65.0
MAX_DRIVE_PWM = 100.0
MIN_TURN_SECONDS = 0.35
MIN_DRIVE_SECONDS = 0.45
MAX_TURN_SECONDS = 3.0
MAX_DRIVE_SECONDS = 3.0
DT = 0.05


@dataclass
class Target:
    theta_deg: float
    distance_cm: float
    clipped: bool
    clip_reason: str


class PrimitivePolicy(torch.nn.Module):
    def __init__(self, input_dim=9, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
        )
        self.mean = torch.nn.Linear(hidden, 4)
        self.log_std = torch.nn.Parameter(torch.full((4,), -0.45))

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.log_std.clamp(-1.5, 0.6)


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def norm_distance(cm):
    if cm is None:
        return -1.0
    return clamp(cm / 200.0, 0.0, 1.5)


def accel_norm(accel):
    return math.sqrt(accel['x'] ** 2 + accel['y'] ** 2 + accel['z'] ** 2)


def choose_random_target(safety, min_distance_cm, max_distance_cm, theta_min_deg, theta_max_deg):
    theta = random.uniform(theta_min_deg, theta_max_deg)
    distance = random.uniform(min_distance_cm, max_distance_cm)
    return clip_target_to_sensors(theta, distance, safety.read_distances())


def clip_target_to_sensors(theta_deg, distance_cm, distances, margin_cm=20.0):
    clipped = False
    reason = 'none'
    relevant = []
    if abs(theta_deg) <= 45.0:
        relevant.append(('front', distances['front']))
    if theta_deg > 25.0:
        relevant.append(('left', distances['left']))
    if theta_deg < -25.0:
        relevant.append(('right', distances['right']))

    allowed = distance_cm
    for name, value in relevant:
        if value is None:
            allowed = min(allowed, 30.0)
            clipped = True
            reason = f'{name}_no_echo'
        else:
            allowed = min(allowed, max(20.0, value - margin_cm))
            if allowed < distance_cm:
                clipped = True
                reason = f'{name}_wall_margin'
    if allowed < 20.0:
        allowed = 20.0
        clipped = True
        reason = 'min_distance'
    return Target(theta_deg=theta_deg, distance_cm=allowed, clipped=clipped, clip_reason=reason)


def make_state(target, distances, imu_reading):
    gyro = imu_reading['gyro']
    accel = imu_reading['accel']
    return torch.tensor([
        target.theta_deg / 90.0,
        target.distance_cm / 200.0,
        norm_distance(distances['front']),
        norm_distance(distances['left']),
        norm_distance(distances['right']),
        gyro['z'] / 250.0,
        accel_norm(accel) / 2.0,
        1.0 if target.clipped else 0.0,
        1.0,
    ], dtype=torch.float32)


def scale_action(raw_action, target_distance_cm):
    squashed = torch.tanh(raw_action)
    values = squashed.detach().cpu().tolist()
    turn_pwm = MIN_TURN_PWM + (values[0] + 1.0) * 0.5 * (MAX_TURN_PWM - MIN_TURN_PWM)
    turn_time_scale = 0.75 + (values[1] + 1.0) * 0.5 * 0.9
    drive_pwm = MIN_DRIVE_PWM + (values[2] + 1.0) * 0.5 * (MAX_DRIVE_PWM - MIN_DRIVE_PWM)
    drive_time_scale = 0.75 + (values[3] + 1.0) * 0.5 * 0.9
    base_drive_seconds = target_distance_cm / 45.0
    return {
        'turn_pwm': clamp(turn_pwm, MIN_TURN_PWM, MAX_TURN_PWM),
        'turn_time_scale': turn_time_scale,
        'drive_pwm': clamp(drive_pwm, MIN_DRIVE_PWM, MAX_DRIVE_PWM),
        'drive_seconds': clamp(base_drive_seconds * drive_time_scale, MIN_DRIVE_SECONDS, MAX_DRIVE_SECONDS),
    }


def sample_policy(policy, state):
    mean, log_std = policy(state)
    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    raw = dist.rsample()
    log_prob = dist.log_prob(raw).sum()
    entropy = dist.entropy().sum()
    return raw, log_prob, entropy


def execute_turn(safety, theta_deg, params):
    direction = 'left' if theta_deg >= 0.0 else 'right'
    left_cmd, right_cmd = ('backward', 'forward') if direction == 'left' else ('forward', 'backward')
    target = abs(theta_deg)
    signed_target = target if direction == 'left' else -target
    yaw = 0.0
    last = time.monotonic()
    start = last
    max_seconds = clamp(target / 45.0 * params['turn_time_scale'], MIN_TURN_SECONDS, MAX_TURN_SECONDS)
    accel_min = None
    accel_max = None
    reason = 'target_reached'
    safety.rover.drive(left_cmd, right_cmd, left_speed=params['turn_pwm'], right_speed=params['turn_pwm'])
    try:
        while time.monotonic() - start < max_seconds:
            distances = safety.read_distances()
            side = distances[direction]
            if side is None:
                reason = f'{direction}_no_echo'
                break
            if side < safety.config.side_turn_clear_cm:
                reason = f'{direction}_blocked'
                break
            now = time.monotonic()
            reading = safety.imu.read_all()
            gyro_z = reading['gyro']['z'] - safety._gyro_z_bias
            yaw += gyro_z * (now - last)
            last = now
            an = accel_norm(reading['accel'])
            accel_min = an if accel_min is None else min(accel_min, an)
            accel_max = an if accel_max is None else max(accel_max, an)
            if direction == 'left' and yaw >= signed_target - 4.0:
                break
            if direction == 'right' and yaw <= signed_target + 4.0:
                break
            time.sleep(DT)
        else:
            reason = 'timeout'
    finally:
        safety.rover.stop_motors()
        time.sleep(0.15)
    return {
        'reason': reason,
        'yaw_deg': yaw,
        'target_yaw_deg': signed_target,
        'yaw_error_deg': abs(signed_target - yaw),
        'accel_delta_g': 0.0 if accel_min is None else accel_max - accel_min,
        'seconds': time.monotonic() - start,
    }


def execute_drive(safety, target, params):
    yaw = 0.0
    last = time.monotonic()
    start = last
    accel_min = None
    accel_max = None
    reason = 'duration_complete'
    target_current = target
    safety.rover.drive('forward', 'forward', left_speed=params['drive_pwm'], right_speed=params['drive_pwm'])
    try:
        while time.monotonic() - start < params['drive_seconds']:
            distances = safety.read_distances()
            target_current = clip_target_to_sensors(target_current.theta_deg, target_current.distance_cm, distances)
            safe, front, threshold = safety.is_front_safe(params['drive_pwm'], distances)
            if not safe:
                reason = f'front_blocked front={front} threshold={threshold:.1f}'
                break
            now = time.monotonic()
            reading = safety.imu.read_all()
            gyro_z = reading['gyro']['z'] - safety._gyro_z_bias
            yaw += gyro_z * (now - last)
            last = now
            an = accel_norm(reading['accel'])
            accel_min = an if accel_min is None else min(accel_min, an)
            accel_max = an if accel_max is None else max(accel_max, an)
            elapsed = time.monotonic() - start
            expected_total = max(MIN_DRIVE_SECONDS, target_current.distance_cm / 45.0)
            if elapsed >= expected_total:
                reason = 'dynamic_target_reached'
                break
            time.sleep(DT)
    finally:
        safety.rover.stop_motors()
        time.sleep(0.15)
    return {
        'reason': reason,
        'yaw_drift_deg': yaw,
        'accel_delta_g': 0.0 if accel_min is None else accel_max - accel_min,
        'seconds': time.monotonic() - start,
        'target_distance_final_cm': target_current.distance_cm,
        'target_clip_reason_final': target_current.clip_reason,
    }


def reward_from_results(turn, drive):
    reward = 1.0
    reward -= 0.035 * min(60.0, turn['yaw_error_deg'])
    reward -= 0.025 * min(50.0, abs(drive['yaw_drift_deg']))
    if turn['reason'] not in ('target_reached',):
        reward -= 1.0
    if drive['reason'] not in ('duration_complete', 'dynamic_target_reached'):
        reward -= 1.25
    if drive['accel_delta_g'] < 0.025:
        reward -= 0.8
    reward -= 0.1 * min(5.0, drive['accel_delta_g'])
    return reward


def ensure_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    f = open(path, 'a', newline='')
    fieldnames = [
        'episode', 'theta_deg', 'distance_cm', 'clipped', 'clip_reason',
        'turn_pwm', 'turn_time_scale', 'drive_pwm', 'drive_seconds',
        'turn_reason', 'turn_yaw_deg', 'turn_yaw_error_deg',
        'drive_reason', 'drive_yaw_drift_deg', 'drive_accel_delta_g',
        'reward', 'loss', 'checkpoint',
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return f, writer


def main():
    parser = argparse.ArgumentParser(description='Train learnable continuous-primitive executor on the real rover.')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--min-distance-cm', type=float, default=45.0)
    parser.add_argument('--max-distance-cm', type=float, default=120.0)
    parser.add_argument('--theta-min-deg', type=float, default=-180.0)
    parser.add_argument('--theta-max-deg', type=float, default=180.0)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--checkpoint', default='data/learned_executor_policy.pt')
    parser.add_argument('--log', default='data/learned_executor_training.csv')
    parser.add_argument('--reset-weights', action='store_true', help='Start from fresh random policy even if checkpoint exists.')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    rover = RoverAPI(camera_enabled=False)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(rover, imu=imu, config=SafetyConfig())
    policy = PrimitivePolicy()
    if args.reset_weights and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print(f'reset checkpoint: {args.checkpoint}', flush=True)
    if os.path.exists(args.checkpoint):
        policy.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        print(f'loaded checkpoint: {args.checkpoint}', flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    baseline = 0.0
    log_f, writer = ensure_log(args.log)
    try:
        print(f'gyro_z_bias={safety.calibrate_gyro():.4f}', flush=True)
        for episode in range(1, args.episodes + 1):
            target = choose_random_target(
                safety,
                min_distance_cm=args.min_distance_cm,
                max_distance_cm=args.max_distance_cm,
                theta_min_deg=args.theta_min_deg,
                theta_max_deg=args.theta_max_deg,
            )
            distances = safety.read_distances()
            state = make_state(target, distances, imu.read_all())
            raw_action, log_prob, entropy = sample_policy(policy, state)
            params = scale_action(raw_action, target.distance_cm)
            print(f'episode={episode} target=({target.theta_deg:.1f}deg,{target.distance_cm:.1f}cm,{target.clip_reason}) params={params}', flush=True)
            turn = execute_turn(safety, target.theta_deg, params)
            drive = execute_drive(safety, target, params) if turn['reason'] == 'target_reached' else {
                'reason': 'skipped_after_turn_failure', 'yaw_drift_deg': 0.0, 'accel_delta_g': 0.0,
                'seconds': 0.0, 'target_distance_final_cm': target.distance_cm, 'target_clip_reason_final': target.clip_reason,
            }
            reward = reward_from_results(turn, drive)
            baseline = 0.9 * baseline + 0.1 * reward
            advantage = reward - baseline
            loss = -(log_prob * advantage) - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            torch.save(policy.state_dict(), args.checkpoint)
            row = {
                'episode': episode,
                'theta_deg': target.theta_deg,
                'distance_cm': target.distance_cm,
                'clipped': target.clipped,
                'clip_reason': target.clip_reason,
                'turn_pwm': params['turn_pwm'],
                'turn_time_scale': params['turn_time_scale'],
                'drive_pwm': params['drive_pwm'],
                'drive_seconds': params['drive_seconds'],
                'turn_reason': turn['reason'],
                'turn_yaw_deg': turn['yaw_deg'],
                'turn_yaw_error_deg': turn['yaw_error_deg'],
                'drive_reason': drive['reason'],
                'drive_yaw_drift_deg': drive['yaw_drift_deg'],
                'drive_accel_delta_g': drive['accel_delta_g'],
                'reward': reward,
                'loss': float(loss.detach().cpu()),
                'checkpoint': args.checkpoint,
            }
            writer.writerow(row)
            log_f.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        log_f.close()
        safety.close()
        imu.close()
        rover.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
