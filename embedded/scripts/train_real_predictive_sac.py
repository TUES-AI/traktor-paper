#!/usr/bin/env python3
"""Train the predictive SAC rover policy online from real-world feedback.

This is the RLxF path for the user's model:

    real camera/sensors/action history
        -> online predictive latent backend
        -> SAC actor chooses [theta_norm, distance_norm]
        -> LocalTargetExecutor + SafetyController execute it
        -> reward from novelty/surprise + real execution feedback

No pretrained vision encoder or pretrained SAC actor is required. The output is a
Stable-Baselines SAC zip that can later be run with:

    embedded/scripts/run_real_sac_vmm.sh --mine --model <zip>
"""

import argparse
from collections import deque
import json
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import _paths  # noqa: F401
from run_sac_vmm_local_targets import (
    PCVM_OBS_DIM,
    PCVMRoverObsBuilder,
    PredictiveRoverObsBuilder,
    action_to_target,
)


class RealPredictiveSACEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, args):
        super().__init__()
        from api.rover_api import RoverAPI
        from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
        from control.safety import SafetyConfig, SafetyController
        from drivers.sensors.mpu9150 import MPU9150

        self.args = args
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        obs_dim = PCVM_OBS_DIM if args.backend in ('pcvm', 'pcvm-m') else 79
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.rover = RoverAPI(camera_enabled=True)
        self.imu = MPU9150(bus=1, address=0x68)
        self.safety = SafetyController(
            self.rover,
            imu=self.imu,
            config=SafetyConfig(
                min_front_stop_cm=args.front_stop_cm,
                max_front_stop_cm=args.front_stop_cm,
                front_clear_to_resume_cm=args.front_clear_cm,
                no_echo_is_clear=not args.no_echo_is_wall,
            ),
        )
        if args.backend in ('pcvm', 'pcvm-m'):
            self.obs_builder = PCVMRoverObsBuilder(self.rover, self.safety, mobilenet=(args.backend == 'pcvm-m'))
        else:
            self.obs_builder = PredictiveRoverObsBuilder(self.rover, self.safety)
        self.executor = LocalTargetExecutor(
            self.safety,
            config=LocalTargetExecutorConfig(turn_pwm=args.turn_pwm, drive_pwm=args.drive_pwm),
            status_callback=lambda s: print(json.dumps({'status': s}, sort_keys=True), flush=True),
        )
        self.step_count = 0
        self.last_action = None
        self.last_backend = {}
        self.last_distances = {}
        self.path_points = deque(maxlen=args.path_memory_size)
        self.last_reward_terms = {}
        self.safety.calibrate_gyro()

    def close(self):
        self.safety.close()
        self.imu.close()
        self.rover.close()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        self.path_points.clear()
        self.last_reward_terms = {}
        obs, distances, backend = self._build_obs(None)
        self._update_path_memory(backend)
        self.last_backend = backend
        self.last_distances = distances
        return obs, {'distances': distances, 'backend': backend}

    def _build_obs(self, action, execution_feedback=None):
        if self.args.backend in ('pcvm', 'pcvm-m'):
            return self.obs_builder.build_pcvm(action, execution_feedback)
        return self.obs_builder.build_predictive(action)

    def _recover_if_blocked(self, distances):
        safe, front, threshold = self.safety.is_front_safe(self.args.drive_pwm, distances)
        if safe:
            return None
        turn_dir = self.safety.freer_side(distances)
        report = self.safety.turn_until_clear(turn_dir, speed_pct=self.args.turn_pwm)
        return {
            'front_cm': front,
            'threshold_cm': threshold,
            'turn_dir': turn_dir,
            'recovery': report,
        }

    def _path_reward(self, backend):
        pose = backend.get('pcvm_pose')
        if not pose or len(pose) < 2:
            return 0.0, {'path_min_dist_m': None, 'path_revisit_penalty': 0.0, 'path_away_bonus': 0.0}
        x = float(pose[0])
        y = float(pose[1])
        if not self.path_points:
            return 0.0, {'path_min_dist_m': None, 'path_revisit_penalty': 0.0, 'path_away_bonus': 0.0}

        dists = [float(np.hypot(x - px, y - py)) for px, py in self.path_points]
        min_dist = min(dists)
        near = max(0.0, 1.0 - min_dist / max(1e-6, self.args.path_near_radius_m))
        away = min(1.0, min_dist / max(1e-6, self.args.path_far_radius_m))
        penalty = self.args.path_revisit_penalty * near
        bonus = self.args.path_away_bonus * away
        return bonus - penalty, {
            'path_min_dist_m': min_dist,
            'path_revisit_penalty': penalty,
            'path_away_bonus': bonus,
        }

    def _update_path_memory(self, backend):
        pose = backend.get('pcvm_pose')
        if pose and len(pose) >= 2:
            self.path_points.append((float(pose[0]), float(pose[1])))

    def _reward(self, execution, backend, recovery):
        reward = -0.03
        novelty_reward = self.args.novelty_weight * float(backend.get('predictive_novelty') or 0.0)
        surprise_reward = self.args.surprise_weight * float(backend.get('predictive_surprise') or 0.0)
        path_reward, path_terms = self._path_reward(backend)
        reward += novelty_reward + surprise_reward + path_reward
        self.last_reward_terms = {
            'base': -0.03,
            'novelty_reward': novelty_reward,
            'surprise_reward': surprise_reward,
            'path_reward': path_reward,
            **path_terms,
        }

        if recovery is not None:
            reward -= self.args.recovery_penalty
            self.last_reward_terms['recovery_penalty'] = self.args.recovery_penalty

        if execution is None:
            return reward

        drive = execution.get('drive') or {}
        clipped = float(execution.get('clipped_distance_cm') or 0.0)
        requested = max(1.0, float(execution.get('requested_distance_cm') or 1.0))
        distance_reward = self.args.distance_weight * min(1.0, clipped / requested)
        reward += distance_reward
        self.last_reward_terms['distance_reward'] = distance_reward
        if drive.get('ok'):
            reward += self.args.success_bonus
            self.last_reward_terms['success_bonus'] = self.args.success_bonus
        else:
            reward -= self.args.stop_penalty
            self.last_reward_terms['stop_penalty'] = self.args.stop_penalty
        if 'front_safety_stop' in str(drive.get('reason')):
            reward -= self.args.front_stop_penalty
            self.last_reward_terms['front_stop_penalty'] = self.args.front_stop_penalty
        if execution.get('reason') == 'distance_clipped_to_zero':
            reward -= self.args.zero_distance_penalty
            self.last_reward_terms['zero_distance_penalty'] = self.args.zero_distance_penalty
        return float(reward)

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        self.obs_builder.last_action = action.copy()

        recovery = self._recover_if_blocked(self.last_distances)
        target = action_to_target(action, self.args.max_theta_deg, self.args.max_distance_cm, self.args.min_drive_cm)
        execution = None
        if recovery is None:
            execution = self.executor.execute_local_target(target['x_cm'], target['y_cm'])

        obs, distances, backend = self._build_obs(action, {'execution': execution, 'recovery': recovery})
        reward = self._reward(execution, backend, recovery)
        self._update_path_memory(backend)
        info = {
            'step': self.step_count,
            'action': [float(action[0]), float(action[1])],
            'action_contract': '[theta_norm, distance_norm]',
            'target': target,
            'distances': distances,
            'backend': backend,
            'execution': execution,
            'recovery': recovery,
            'reward': reward,
            'reward_terms': self.last_reward_terms,
        }
        print(json.dumps(info, sort_keys=True), flush=True)
        self.last_action = action.copy()
        self.last_backend = backend
        self.last_distances = distances
        time.sleep(self.args.sleep)
        return obs, reward, False, False, info


def parse_args():
    parser = argparse.ArgumentParser(description='Train predictive SAC online on the real rover.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--backend', choices=['predictive', 'pcvm', 'pcvm-m'], default='pcvm')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--sleep', type=float, default=0.2)
    parser.add_argument('--max-theta-deg', type=float, default=75.0)
    parser.add_argument('--max-distance-cm', type=float, default=80.0)
    parser.add_argument('--min-drive-cm', type=float, default=10.0)
    parser.add_argument('--turn-pwm', type=float, default=65.0)
    parser.add_argument('--drive-pwm', type=float, default=75.0)
    parser.add_argument('--front-stop-cm', type=float, default=45.0)
    parser.add_argument('--front-clear-cm', type=float, default=55.0)
    parser.add_argument('--no-echo-is-wall', action='store_true')
    parser.add_argument('--learning-starts', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--buffer-size', type=int, default=5000)
    parser.add_argument('--novelty-weight', type=float, default=0.7)
    parser.add_argument('--surprise-weight', type=float, default=0.2)
    parser.add_argument('--distance-weight', type=float, default=0.35)
    parser.add_argument('--success-bonus', type=float, default=0.35)
    parser.add_argument('--stop-penalty', type=float, default=0.15)
    parser.add_argument('--front-stop-penalty', type=float, default=0.25)
    parser.add_argument('--recovery-penalty', type=float, default=0.35)
    parser.add_argument('--zero-distance-penalty', type=float, default=0.25)
    parser.add_argument('--path-revisit-penalty', type=float, default=0.45)
    parser.add_argument('--path-away-bonus', type=float, default=0.25)
    parser.add_argument('--path-near-radius-m', type=float, default=0.45)
    parser.add_argument('--path-far-radius-m', type=float, default=1.5)
    parser.add_argument('--path-memory-size', type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_path is None:
        if args.backend == 'pcvm-m':
            args.save_path = 'results/pcvm_m_sac_real.zip'
        elif args.backend == 'pcvm':
            args.save_path = 'results/pcvm_cnn_sac_real.zip'
        else:
            args.save_path = 'results/predictive_sac_real.zip'
    from stable_baselines3 import SAC

    env = RealPredictiveSACEnv(args)
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
                policy_kwargs={'net_arch': [128, 128]},
                verbose=1,
                device='cpu',
            )
        model.learn(total_timesteps=args.steps, log_interval=1, progress_bar=False)
        model.save(args.save_path)
        print(json.dumps({'saved': args.save_path}, sort_keys=True), flush=True)
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
