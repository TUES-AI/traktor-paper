#!/usr/bin/env python3
"""SAC over active DINOv3 scan candidates.

This is the learned version of run_dino_scan_baseline.py. Each environment step
scans [-angle, 0, +angle], exposes candidate visual novelty/distances to SAC,
maps the scalar SAC action to one of the three candidates, executes it with the
same deterministic drive-until-front executor, and updates visual memory from the
post-action frame.
"""

import argparse
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150
from dino_scan_common import DINOScanner, distance_features, make_visual_cluster_sheet, read_distances, save_frame, summarize_log
from run_dino_scan_baseline import capture_scan


class DinoScanSACEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        self.rover = RoverAPI(camera_enabled=True)
        self.imu = MPU9150(bus=1, address=0x68)
        self.safety = SafetyController(
            self.rover,
            imu=self.imu,
            config=SafetyConfig(
                min_front_stop_cm=args.front_stop_cm,
                max_front_stop_cm=args.front_stop_cm,
                front_clear_to_resume_cm=args.front_clear_cm,
            ),
        )
        self.executor = LocalTargetExecutor(
            self.safety,
            config=LocalTargetExecutorConfig(
                turn_pwm=args.turn_pwm,
                drive_pwm=args.drive_pwm,
                until_front_stop_cm=args.front_stop_cm,
                max_drive_seconds=args.until_front_max_seconds,
                cm_per_second=args.cm_per_second,
            ),
            status_callback=lambda s: print(json.dumps({'status': s}, sort_keys=True), flush=True),
        )
        self.scanner = DINOScanner(input_size=args.input_size, known_dist=args.known_dist, norm_dist=args.norm_dist, warmup=args.memory_warmup)
        self.prefix = Path(args.save_prefix)
        self.log_path = self.prefix.with_suffix('.jsonl')
        self.frame_dir = Path(str(self.prefix) + '_frames')
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_f = self.log_path.open('a', buffering=1)
        self.step_count = 0
        self.last_choice_norm = 0.0
        self.pending_candidates = None
        self.pending_distances = None
        self.safety.calibrate_gyro()

    def close(self):
        self.rover.stop_motors()
        self.log_f.close()
        self.safety.close()
        self.imu.close()
        self.rover.close()

    def _obs_from_scan(self):
        self.step_count += 1
        self.pending_distances = read_distances(self.safety)
        candidates, recenter = capture_scan(self.rover, self.executor, self.scanner, self.args.angle_deg, self.frame_dir, self.step_count)
        self.pending_candidates = candidates
        nov = np.array([float(c['novelty']) for c in candidates], dtype=np.float32)
        dist = np.array([float(c['dist']) / 2.0 for c in candidates], dtype=np.float32)
        theta = np.array([float(c['theta_deg']) / max(1e-6, self.args.angle_deg) for c in candidates], dtype=np.float32)
        obs = np.concatenate([
            np.clip(nov, 0, 1),
            np.clip(dist, 0, 1),
            theta,
            distance_features(self.pending_distances),
            np.array([self.last_choice_norm], dtype=np.float32),
        ]).astype(np.float32)
        self._last_recenter = recenter
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_choice_norm = 0.0
        return self._obs_from_scan(), {}

    def _action_to_index(self, action):
        a = float(np.asarray(action).reshape(-1)[0])
        if a < -1.0 / 3.0:
            return 2  # right, theta negative
        if a > 1.0 / 3.0:
            return 1  # left, theta positive
        return 0  # center

    def step(self, action):
        idx = self._action_to_index(action)
        candidates = self.pending_candidates or []
        chosen = candidates[idx]
        self.last_choice_norm = float(chosen['theta_deg']) / max(1e-6, self.args.angle_deg)
        turn_to_choice = self.executor.turn_to(float(chosen['theta_deg']))
        if turn_to_choice.get('ok'):
            execution = {'turn': turn_to_choice, 'drive': self.executor.drive_until_front(self.args.front_stop_cm)}
        else:
            execution = {'turn': turn_to_choice, 'drive': None}
        time.sleep(0.2)
        post_frame = self.rover.get_camera_frame()
        post_path = self.frame_dir / f'step_{self.step_count:04d}_post.jpg'
        save_frame(post_frame, post_path)
        post = self.scanner.update_frame(post_frame, self.step_count)
        drive = execution.get('drive') or {}
        moved_cm = float(drive.get('estimated_distance_cm') or 0.0)
        motion_gate = moved_cm >= self.args.motion_gate_cm
        reward = 0.0
        reward += self.args.visual_novelty_weight * float(post['novelty']) if motion_gate else 0.0
        reward += self.args.new_cluster_bonus if (motion_gate and post['new_cluster']) else 0.0
        reward += self.args.distance_weight * min(1.0, moved_cm / max(1e-6, self.args.max_distance_cm))
        if drive.get('reason') == 'contact_or_stall':
            reward -= self.args.contact_penalty
        if moved_cm < self.args.zero_forward_cm:
            reward -= self.args.zero_forward_penalty
        record = {
            'step': self.step_count,
            'time': time.time(),
            'action': [float(x) for x in np.asarray(action).reshape(-1)],
            'chosen_index': idx,
            'chosen': chosen,
            'candidates': candidates,
            'distances': self.pending_distances,
            'execution': execution,
            'post_frame_path': str(post_path),
            'post_update': {k: v for k, v in post.items() if k != 'embedding'},
            'reward': float(reward),
            'reward_terms': {
                'motion_gate': bool(motion_gate),
                'visual_novelty_reward': self.args.visual_novelty_weight * float(post['novelty']) if motion_gate else 0.0,
                'new_cluster_bonus': self.args.new_cluster_bonus if (motion_gate and post['new_cluster']) else 0.0,
                'distance_reward': self.args.distance_weight * min(1.0, moved_cm / max(1e-6, self.args.max_distance_cm)),
                'contact_penalty': self.args.contact_penalty if drive.get('reason') == 'contact_or_stall' else 0.0,
                'zero_forward_penalty': self.args.zero_forward_penalty if moved_cm < self.args.zero_forward_cm else 0.0,
            },
        }
        self.log_f.write(json.dumps(record, sort_keys=True) + '\n')
        print(json.dumps({'step': self.step_count, 'chosen': chosen['name'], 'reward': round(float(reward), 3), 'visual_bank': post['bank_size']}, sort_keys=True), flush=True)
        terminated = False
        truncated = self.step_count >= self.args.steps
        obs = self._obs_from_scan() if not truncated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, {}


def parse_args():
    p = argparse.ArgumentParser(description='Train SAC on active DINOv3 scan candidates.')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--save-prefix', default='results/dino_scan_sac_50')
    p.add_argument('--resume', default=None)
    p.add_argument('--angle-deg', type=float, default=75.0)
    p.add_argument('--front-stop-cm', type=float, default=40.0)
    p.add_argument('--front-clear-cm', type=float, default=45.0)
    p.add_argument('--turn-pwm', type=float, default=60.0)
    p.add_argument('--drive-pwm', type=float, default=65.0)
    p.add_argument('--until-front-max-seconds', type=float, default=6.0)
    p.add_argument('--cm-per-second', type=float, default=40.0)
    p.add_argument('--input-size', type=int, default=336)
    p.add_argument('--known-dist', type=float, default=1.10)
    p.add_argument('--norm-dist', type=float, default=2.20)
    p.add_argument('--memory-warmup', type=int, default=0)
    p.add_argument('--learning-starts', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--buffer-size', type=int, default=1000)
    p.add_argument('--visual-novelty-weight', type=float, default=2.0)
    p.add_argument('--new-cluster-bonus', type=float, default=1.0)
    p.add_argument('--distance-weight', type=float, default=0.4)
    p.add_argument('--max-distance-cm', type=float, default=120.0)
    p.add_argument('--motion-gate-cm', type=float, default=3.0)
    p.add_argument('--zero-forward-cm', type=float, default=3.0)
    p.add_argument('--zero-forward-penalty', type=float, default=0.3)
    p.add_argument('--contact-penalty', type=float, default=1.0)
    return p.parse_args()


def main():
    from stable_baselines3 import SAC

    args = parse_args()
    env = DinoScanSACEnv(args)
    model_path = Path(args.save_prefix).with_suffix('.zip')
    try:
        if args.resume:
            model = SAC.load(args.resume, env=env)
        else:
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=3e-4,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                gamma=0.98,
                tau=0.02,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs={'net_arch': [64, 64]},
                verbose=1,
                device='cpu',
            )
        model.learn(total_timesteps=args.steps, log_interval=1, progress_bar=False)
        model.save(model_path)
    finally:
        env.close()
    sheet = make_visual_cluster_sheet(env.log_path, env.frame_dir, Path(str(Path(args.save_prefix)) + '_visual_clusters.png'))
    summary = summarize_log(env.log_path)
    summary['cluster_sheet'] = sheet
    Path(args.save_prefix).with_suffix('.out.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({'saved_model': str(model_path), **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
