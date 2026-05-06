#!/usr/bin/env python3
"""Run a SAC policy on the real rover using 2D local-target actions.

SAC action contract:
    [theta_norm, distance_norm]

Mapping:
    theta_deg   = theta_norm * max_theta_deg
    distance_cm = ((distance_norm + 1) / 2) * max_distance_cm

The action is not sent to motors directly. It is converted to a rover-local
target and executed through LocalTargetExecutor + SafetyController.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ULTRA_MAX_CM = 400.0
YAW_RATE_MAX_DPS = 180.0


def clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))


def norm_distance(cm):
    if cm is None:
        return 1.0
    return clamp(cm / ULTRA_MAX_CM, 0.0, 1.0)


class RealRoverObsBuilder:
    def __init__(self, rover, safety, use_camera_vmm=True):
        self.rover = rover
        self.safety = safety
        self.use_camera_vmm = use_camera_vmm
        self.yaw_deg = 0.0
        self.last_t = time.monotonic()
        self.last_yaw_rate = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.vmm = None
        if use_camera_vmm:
            try:
                from VMM.vmm import VMM
                self.vmm = VMM()
            except Exception as exc:
                print(json.dumps({'warning': 'vmm_unavailable', 'error': repr(exc)}), flush=True)

    def update_imu(self):
        now = time.monotonic()
        dt = max(1e-3, now - self.last_t)
        self.last_t = now
        if self.safety.imu is None:
            self.last_yaw_rate = 0.0
            return
        reading = self.safety.imu.read_all()
        gyro_z = reading['gyro']['z'] - self.safety._gyro_z_bias
        self.last_yaw_rate = gyro_z
        self.yaw_deg += gyro_z * dt

    def camera_novelty(self):
        if self.vmm is None:
            return 0.0
        try:
            frame = self.rover.get_camera_frame()
            result = self.vmm.observe(frame)
            return float(result.get('novelty', result.get('rnd_norm', 0.0)))
        except Exception as exc:
            print(json.dumps({'warning': 'camera_vmm_failed', 'error': repr(exc)}), flush=True)
            return 0.0

    def build(self, obs_dim):
        self.update_imu()
        distances = self.safety.read_distances()
        base3 = np.array([
            norm_distance(distances['left']),
            norm_distance(distances['right']),
            norm_distance(distances['front']),
        ], dtype=np.float32)
        if obs_dim == 3:
            return base3, distances

        if obs_dim == 12:
            yaw_rad = math.radians(self.yaw_deg)
            novelty = self.camera_novelty()
            # Real directional FOV novelty is not available yet; duplicate the
            # current camera novelty so a SAC-VMM model sees the expected shape.
            return np.array([
                base3[0], base3[1], base3[2],
                math.sin(yaw_rad), math.cos(yaw_rad),
                clamp(novelty, 0.0, 1.0),
                clamp(novelty, 0.0, 1.0), clamp(novelty, 0.0, 1.0), clamp(novelty, 0.0, 1.0),
                clamp(self.last_yaw_rate / YAW_RATE_MAX_DPS, -1.0, 1.0),
                self.last_action[0], self.last_action[1],
            ], dtype=np.float32), distances

        raise ValueError(f'Unsupported model observation dim {obs_dim}; expected 3 or 12')


def action_to_target(action, max_theta_deg, max_distance_cm, min_drive_cm):
    theta_norm = clamp(action[0], -1.0, 1.0)
    dist_norm = clamp(action[1], -1.0, 1.0)
    theta_deg = theta_norm * max_theta_deg
    distance_cm = ((dist_norm + 1.0) * 0.5) * max_distance_cm
    if distance_cm < min_drive_cm:
        distance_cm = 0.0
    theta_rad = math.radians(theta_deg)
    return {
        'theta_norm': theta_norm,
        'distance_norm': dist_norm,
        'theta_deg': theta_deg,
        'distance_cm': distance_cm,
        'x_cm': math.cos(theta_rad) * distance_cm,
        'y_cm': math.sin(theta_rad) * distance_cm,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Run SAC-VMM local-target actions on the real rover.')
    parser.add_argument('--model', required=True, help='Path to Stable-Baselines SAC .zip model')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--sleep', type=float, default=0.25)
    parser.add_argument('--max-theta-deg', type=float, default=75.0)
    parser.add_argument('--max-distance-cm', type=float, default=120.0)
    parser.add_argument('--min-drive-cm', type=float, default=10.0)
    parser.add_argument('--turn-pwm', type=float, default=65.0)
    parser.add_argument('--drive-pwm', type=float, default=90.0)
    parser.add_argument('--no-camera-vmm', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    from stable_baselines3 import SAC
    from api.rover_api import RoverAPI
    from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
    from control.safety import SafetyConfig, SafetyController
    from drivers.sensors.mpu9150 import MPU9150

    model = SAC.load(args.model)
    obs_dim = int(np.prod(model.observation_space.shape))
    print(json.dumps({
        'model': args.model,
        'obs_dim': obs_dim,
        'action_contract': '[theta_norm, distance_norm]',
        'dry_run': args.dry_run,
    }, sort_keys=True), flush=True)

    camera_enabled = (not args.no_camera_vmm) and obs_dim == 12
    rover = RoverAPI(camera_enabled=camera_enabled)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(rover, imu=imu, config=SafetyConfig())
    obs_builder = RealRoverObsBuilder(rover, safety, use_camera_vmm=camera_enabled)
    executor = LocalTargetExecutor(
        safety,
        config=LocalTargetExecutorConfig(turn_pwm=args.turn_pwm, drive_pwm=args.drive_pwm),
        status_callback=lambda s: print(json.dumps({'status': s}, sort_keys=True), flush=True),
    )

    try:
        print(json.dumps({'gyro_z_bias': safety.calibrate_gyro()}, sort_keys=True), flush=True)
        for step in range(args.steps):
            obs, distances = obs_builder.build(obs_dim)
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs_builder.last_action = np.asarray(action, dtype=np.float32).copy()
            target = action_to_target(action, args.max_theta_deg, args.max_distance_cm, args.min_drive_cm)
            print(json.dumps({
                'step': step,
                'distances': distances,
                'action': [float(action[0]), float(action[1])],
                'target': target,
            }, sort_keys=True), flush=True)
            if not args.dry_run:
                report = executor.execute_local_target(target['x_cm'], target['y_cm'])
                print(json.dumps({'execution': report}, sort_keys=True), flush=True)
            time.sleep(args.sleep)
    finally:
        safety.close()
        imu.close()
        rover.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
