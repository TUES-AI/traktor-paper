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
from collections import deque
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ULTRA_MAX_CM = 400.0
YAW_RATE_MAX_DPS = 180.0
PREDICTIVE_CAM_W = 32
PREDICTIVE_CAM_H = 16
PREDICTIVE_LATENT_DIM = 64
PREDICTIVE_CANDIDATES = np.array([
    [0.0, 0.25],
    [-0.7, 0.0],
    [0.7, 0.0],
    [-1.0, -1.0],
    [1.0, -1.0],
    [0.0, -0.6],
], dtype=np.float32)

DEFAULT_MODELS = {
    'no-vmm': 'results/SAC-NoVMM_s42.zip',
    'vmm': 'results/SAC-VMM___s42.zip',
    'mine': 'results/predictive_sac_seed42.zip',
}


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


class PredictiveStateModel:
    """Online predictive latent backend for the user's model.

    The policy observation shape matches sim_realistic.PredictiveVMMWrapper:
        latent(64) + novelty/surprise(2) + candidate_scores(6) + raw tail(7)

    The backend learns online from real transitions. It is intentionally local to
    the runner; no pretraining or ImageNet feature extractor is required.
    """

    def __init__(self, raw_dim):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, 256), nn.ReLU(), nn.Linear(256, PREDICTIVE_LATENT_DIM), nn.Tanh()
        ).to(self.device)
        self.transition = nn.Sequential(
            nn.Linear(PREDICTIVE_LATENT_DIM + 2, 128), nn.ReLU(), nn.Linear(128, PREDICTIVE_LATENT_DIM)
        ).to(self.device)
        self.inverse = nn.Sequential(
            nn.Linear(PREDICTIVE_LATENT_DIM * 2, 128), nn.ReLU(), nn.Linear(128, 2), nn.Tanh()
        ).to(self.device)
        self.rnd_target = nn.Sequential(
            nn.Linear(PREDICTIVE_LATENT_DIM, 128), nn.ReLU(), nn.Linear(128, 64)
        ).to(self.device)
        self.rnd_pred = nn.Sequential(
            nn.Linear(PREDICTIVE_LATENT_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64)
        ).to(self.device)
        for p in self.rnd_target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.transition.parameters()) + list(self.inverse.parameters()),
            lr=2e-4,
        )
        self.rnd_opt = torch.optim.Adam(self.rnd_pred.parameters(), lr=5e-5)
        self.rnd_n = 0
        self.rnd_mean = 0.0
        self.losses = deque(maxlen=500)

    def _tensor(self, x):
        return self.torch.as_tensor(x, dtype=self.torch.float32, device=self.device).unsqueeze(0)

    def encode(self, raw):
        with self.torch.no_grad():
            return self.encoder(self._tensor(raw)).squeeze(0).detach().cpu().numpy().astype(np.float32)

    def candidate_scores(self, z):
        with self.torch.no_grad():
            zt = self.torch.as_tensor(z, dtype=self.torch.float32, device=self.device).unsqueeze(0).repeat(len(PREDICTIVE_CANDIDATES), 1)
            at = self.torch.as_tensor(PREDICTIVE_CANDIDATES, dtype=self.torch.float32, device=self.device)
            pred = self.transition(self.torch.cat([zt, at], dim=1))
            score = self.torch.linalg.vector_norm(pred - zt, dim=1)
            score = score / (score.mean() + 1e-6)
            return self.torch.clamp(score / 3.0, 0, 1).detach().cpu().numpy().astype(np.float32)

    def rnd_score_update(self, z):
        import torch.nn.functional as F

        zt = self._tensor(z)
        with self.torch.no_grad():
            target = self.rnd_target(zt)
        pred = self.rnd_pred(zt)
        loss = F.mse_loss(pred, target)
        self.rnd_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()
        raw = float(loss.item())
        self.rnd_n += 1
        self.rnd_mean += (raw - self.rnd_mean) / self.rnd_n
        return float(np.clip(raw / (self.rnd_mean + 1e-8), 0.0, 3.0) / 3.0)

    def train_transition(self, raw, action, next_raw):
        import torch.nn.functional as F

        x = self._tensor(raw)
        xp = self._tensor(next_raw)
        a = self._tensor(action)
        z = self.encoder(x)
        zp = self.encoder(xp)
        pred = self.transition(self.torch.cat([z, a], dim=1))
        inv = self.inverse(self.torch.cat([z.detach(), zp.detach()], dim=1))
        transition_loss = F.mse_loss(pred, zp.detach())
        inverse_loss = F.mse_loss(inv, a)
        smooth_loss = 0.02 * self.torch.linalg.vector_norm(zp - z, dim=1).mean()
        loss = transition_loss + 0.2 * inverse_loss + smooth_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        raw_surprise = float(transition_loss.detach().item())
        self.losses.append(raw_surprise)
        mean = float(np.mean(self.losses)) + 1e-8
        return float(np.clip(raw_surprise / mean, 0.0, 3.0) / 3.0), float(loss.detach().item())


class PredictiveRoverObsBuilder(RealRoverObsBuilder):
    def __init__(self, rover, safety):
        super().__init__(rover, safety, use_camera_vmm=False)
        raw_dim = PREDICTIVE_CAM_W * PREDICTIVE_CAM_H + 3 + 4
        self.model = PredictiveStateModel(raw_dim)
        self.prev_raw = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.novelty = 0.0
        self.surprise = 0.0

    def _camera_raw(self):
        try:
            import cv2
            frame = self.rover.get_camera_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (PREDICTIVE_CAM_W, PREDICTIVE_CAM_H), interpolation=cv2.INTER_AREA)
            return (small.astype(np.float32).reshape(-1) / 255.0)
        except Exception as exc:
            print(json.dumps({'warning': 'predictive_camera_failed', 'error': repr(exc)}), flush=True)
            return np.zeros(PREDICTIVE_CAM_W * PREDICTIVE_CAM_H, dtype=np.float32)

    def raw_obs(self):
        self.update_imu()
        distances = self.safety.read_distances()
        sensors = np.array([
            norm_distance(distances['left']),
            norm_distance(distances['right']),
            norm_distance(distances['front']),
        ], dtype=np.float32)
        motion = np.array([
            (clamp(self.last_yaw_rate / YAW_RATE_MAX_DPS, -1.0, 1.0) + 1.0) * 0.5,
            0.0,
            (self.last_action[0] + 1.0) * 0.5,
            (self.last_action[1] + 1.0) * 0.5,
        ], dtype=np.float32)
        return np.concatenate([self._camera_raw(), sensors, motion]).astype(np.float32), distances

    def build_predictive(self, last_executed_action=None):
        raw, distances = self.raw_obs()
        if self.prev_raw is not None and last_executed_action is not None:
            self.surprise, loss = self.model.train_transition(self.prev_raw, last_executed_action, raw)
        else:
            loss = None
        z = self.model.encode(raw)
        self.novelty = self.model.rnd_score_update(z)
        cand = self.model.candidate_scores(z)
        tail = raw[-7:].astype(np.float32) * 2.0 - 1.0
        obs = np.concatenate([
            z,
            np.array([self.novelty, self.surprise], dtype=np.float32),
            cand,
            tail,
        ]).astype(np.float32)
        self.prev_raw = raw.copy()
        return obs, distances, {'predictive_loss': loss, 'predictive_novelty': self.novelty, 'predictive_surprise': self.surprise}


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--no-vmm', action='store_const', const='no-vmm', dest='mode')
    mode.add_argument('--vmm', action='store_const', const='vmm', dest='mode')
    mode.add_argument('--mine', action='store_const', const='mine', dest='mode')
    parser.set_defaults(mode='vmm')
    parser.add_argument('--mode', choices=['no-vmm', 'vmm', 'mine'], help='Observation backend/method to run')
    parser.add_argument('--model', default=None, help='Path to Stable-Baselines SAC .zip model')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--sleep', type=float, default=0.25)
    parser.add_argument('--max-theta-deg', type=float, default=75.0)
    parser.add_argument('--max-distance-cm', type=float, default=120.0)
    parser.add_argument('--min-drive-cm', type=float, default=10.0)
    parser.add_argument('--turn-pwm', type=float, default=65.0)
    parser.add_argument('--drive-pwm', type=float, default=90.0)
    parser.add_argument('--front-stop-cm', type=float, default=45.0)
    parser.add_argument('--front-clear-cm', type=float, default=55.0)
    parser.add_argument('--no-echo-is-wall', action='store_true')
    parser.add_argument('--no-camera-vmm', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model is None:
        args.model = DEFAULT_MODELS[args.mode]
    from stable_baselines3 import SAC
    from api.rover_api import RoverAPI
    from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
    from control.safety import SafetyConfig, SafetyController
    from drivers.sensors.mpu9150 import MPU9150

    model = SAC.load(args.model)
    obs_dim = int(np.prod(model.observation_space.shape))
    expected_obs_dim = {'no-vmm': 3, 'vmm': 12, 'mine': 79}[args.mode]
    if obs_dim != expected_obs_dim:
        raise ValueError(f'Mode {args.mode} expects obs_dim={expected_obs_dim}, model has obs_dim={obs_dim}')
    print(json.dumps({
        'model': args.model,
        'mode': args.mode,
        'obs_dim': obs_dim,
        'action_contract': '[theta_norm, distance_norm]',
        'dry_run': args.dry_run,
    }, sort_keys=True), flush=True)

    camera_enabled = (args.mode == 'vmm' and not args.no_camera_vmm) or args.mode == 'mine'
    rover = RoverAPI(camera_enabled=camera_enabled)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(
        rover,
        imu=imu,
        config=SafetyConfig(
            min_front_stop_cm=args.front_stop_cm,
            max_front_stop_cm=args.front_stop_cm,
            front_clear_to_resume_cm=args.front_clear_cm,
            no_echo_is_clear=not args.no_echo_is_wall,
        ),
    )
    if args.mode == 'mine':
        obs_builder = PredictiveRoverObsBuilder(rover, safety)
    else:
        obs_builder = RealRoverObsBuilder(rover, safety, use_camera_vmm=(args.mode == 'vmm' and not args.no_camera_vmm))
    executor = LocalTargetExecutor(
        safety,
        config=LocalTargetExecutorConfig(turn_pwm=args.turn_pwm, drive_pwm=args.drive_pwm),
        status_callback=lambda s: print(json.dumps({'status': s}, sort_keys=True), flush=True),
    )

    try:
        print(json.dumps({'gyro_z_bias': safety.calibrate_gyro()}, sort_keys=True), flush=True)
        last_executed_action = None
        for step in range(args.steps):
            if args.mode == 'mine':
                obs, distances, backend_info = obs_builder.build_predictive(last_executed_action)
            else:
                obs, distances = obs_builder.build(obs_dim)
                backend_info = {}
            action, _ = model.predict(obs, deterministic=args.deterministic)
            safe, front, threshold = safety.is_front_safe(args.drive_pwm, distances)
            if not safe:
                turn_dir = safety.freer_side(distances)
                print(json.dumps({
                    'step': step,
                    'distances': distances,
                    'safety_override': 'front_blocked_before_policy_execution',
                    'front_cm': front,
                    'threshold_cm': threshold,
                    'turn_dir': turn_dir,
                }, sort_keys=True), flush=True)
                if not args.dry_run:
                    report = safety.turn_until_clear(turn_dir, speed_pct=args.turn_pwm)
                    print(json.dumps({'recovery': report}, sort_keys=True), flush=True)
                time.sleep(args.sleep)
                continue
            obs_builder.last_action = np.asarray(action, dtype=np.float32).copy()
            target = action_to_target(action, args.max_theta_deg, args.max_distance_cm, args.min_drive_cm)
            print(json.dumps({
                'step': step,
                'distances': distances,
                'action': [float(action[0]), float(action[1])],
                'backend': backend_info,
                'target': target,
            }, sort_keys=True), flush=True)
            if not args.dry_run:
                report = executor.execute_local_target(target['x_cm'], target['y_cm'])
                print(json.dumps({'execution': report}, sort_keys=True), flush=True)
            last_executed_action = np.asarray(action, dtype=np.float32).copy()
            time.sleep(args.sleep)
    finally:
        safety.close()
        imu.close()
        rover.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
