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
from pathlib import Path
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
    execution_feedback_to_action,
    clamp,
)


class RealPredictiveSACEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, args, dashboard=None):
        super().__init__()
        from api.rover_api import RoverAPI
        from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
        from control.safety import SafetyConfig, SafetyController
        from drivers.sensors.mpu9150 import MPU9150

        self.args = args
        self.dashboard = dashboard
        self.action_space = spaces.Box(
            low=np.full((1 if args.action_mode == 'theta_until_front' else 2,), -1.0, dtype=np.float32),
            high=np.full((1 if args.action_mode == 'theta_until_front' else 2,), 1.0, dtype=np.float32),
        )
        obs_dim = PCVM_OBS_DIM if args.backend in ('pcvm', 'pcvm-m', 'pcvm-t') else 79
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
        if args.backend in ('pcvm', 'pcvm-m', 'pcvm-t'):
            self.obs_builder = PCVMRoverObsBuilder(
                self.rover,
                self.safety,
                mobilenet=(args.backend == 'pcvm-m'),
                transformer=(args.backend == 'pcvm-t'),
            )
        else:
            self.obs_builder = PredictiveRoverObsBuilder(self.rover, self.safety)
        self.executor = LocalTargetExecutor(
            self.safety,
            config=LocalTargetExecutorConfig(
                turn_pwm=args.turn_pwm,
                drive_pwm=args.drive_pwm,
                until_front_stop_cm=args.until_front_cm,
                max_drive_seconds=args.until_front_max_seconds,
                cm_per_second=args.cm_per_second,
            ),
            status_callback=lambda s: print(json.dumps({'status': s}, sort_keys=True), flush=True),
        )
        self.step_count = 0
        self.last_action = None
        self.last_executed_action = None
        self.last_backend = {}
        self.last_distances = {}
        self.path_points = deque(maxlen=args.path_memory_size)
        self.recent_reward_poses = deque(maxlen=args.loop_memory_size)
        self.recent_recoveries = deque(maxlen=args.recovery_streak_window)
        self.last_reward_terms = {}
        self.log_f = None
        if args.log_path:
            log_path = Path(args.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_f = log_path.open('a', buffering=1)
        self.frame_dir = Path(args.frame_dir) if args.frame_dir else None
        if self.frame_dir is not None:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.safety.calibrate_gyro()

    def close(self):
        if self.log_f is not None:
            self.log_f.close()
            self.log_f = None
        self.safety.close()
        self.imu.close()
        self.rover.close()

    def _emit(self, record):
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self.log_f is not None:
            self.log_f.write(line + '\n')

    def _save_step_frame(self, step):
        if self.frame_dir is None:
            return None
        frame = getattr(self.obs_builder, 'last_frame', None)
        if frame is None:
            return None
        try:
            import cv2

            path = self.frame_dir / f'step_{int(step):04d}.jpg'
            cv2.imwrite(str(path), frame)
            return str(path)
        except Exception as exc:
            return f'frame_save_failed:{exc!r}'

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        self.last_executed_action = None
        self.path_points.clear()
        self.recent_reward_poses.clear()
        self.recent_recoveries.clear()
        self.last_reward_terms = {}
        obs, distances, backend = self._build_obs(None)
        self._update_path_memory(backend)
        self.last_backend = backend
        self.last_distances = distances
        return obs, {'distances': distances, 'backend': backend}

    def _build_obs(self, action, execution_feedback=None):
        if self.args.backend in ('pcvm', 'pcvm-m', 'pcvm-t'):
            return self.obs_builder.build_pcvm(action, execution_feedback)
        return self.obs_builder.build_predictive(action)

    def _recover_if_blocked(self, distances):
        reverse = self.safety.reverse_if_too_close(self.args.drive_pwm, distances)
        if reverse is not None:
            distances = self.safety.read_distances()
        safe, front, threshold = self.safety.is_front_safe(self.args.drive_pwm, distances)
        if safe:
            return {'reverse': reverse, 'continue_after_reverse': True} if reverse is not None else None
        turn_dir = self.safety.freer_side(distances)
        report = self.safety.turn_until_clear(turn_dir, speed_pct=self.args.turn_pwm)
        return {
            'reverse': reverse,
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
        if self.args.reward_mode == 'slow_rlxf':
            return self._slow_rlxf_reward(execution, backend, recovery)
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

    def _executed_motion_terms(self, execution, recovery):
        execution = execution or {}
        recovery = recovery or {}
        drive = execution.get('drive') or {}
        turn = execution.get('turn') or {}
        distance_cm = max(0.0, float(execution.get('clipped_distance_cm') or 0.0)) if drive.get('ok') else 0.0
        yaw_deg = abs(float(turn.get('yaw_deg') or execution.get('theta_deg') or 0.0))
        moved = distance_cm >= self.args.motion_gate_cm
        turned = yaw_deg >= self.args.yaw_gate_deg
        drive_reason = str(drive.get('reason') or '')
        front_stop = 'front_safety_stop' in drive_reason
        already_at_front = drive_reason == 'already_at_front_threshold'
        zero_distance = execution.get('reason') == 'distance_clipped_to_zero' or drive_reason == 'distance_clipped_to_zero'
        turn_failed = bool(turn) and not bool(turn.get('ok'))
        drive_failed = bool(drive) and not bool(drive.get('ok'))
        recovery_reverse = recovery.get('reverse') if isinstance(recovery, dict) else None
        recovery_turn = recovery.get('recovery') if isinstance(recovery, dict) else None
        return {
            'executed_distance_cm': distance_cm,
            'executed_yaw_deg_abs': yaw_deg,
            'motion_gate': bool(moved or turned),
            'moved_gate': bool(moved),
            'turned_gate': bool(turned),
            'front_stop': bool(front_stop),
            'zero_distance': bool(zero_distance),
            'already_at_front_threshold': bool(already_at_front),
            'turn_failed': bool(turn_failed),
            'drive_failed': bool(drive_failed),
            'recovery_reverse': bool(recovery_reverse),
            'recovery_turn': bool(recovery_turn),
        }

    def _near_obstacle_penalty(self, distances):
        if not distances:
            return 0.0
        penalty = 0.0
        front = distances.get('front')
        if front is not None:
            threshold = max(1e-6, float(self.args.front_clear_cm))
            penalty = max(penalty, max(0.0, 1.0 - float(front) / threshold))
        for side_name in ('left', 'right'):
            side = distances.get(side_name)
            if side is not None:
                penalty = max(penalty, 0.5 * max(0.0, 1.0 - float(side) / max(1e-6, self.args.side_near_cm)))
        return float(np.clip(penalty, 0.0, 1.0))

    def _slow_rlxf_reward(self, execution, backend, recovery):
        terms = self._executed_motion_terms(execution, recovery)
        novelty = float(backend.get('pcvm_mem_norm') or backend.get('predictive_novelty') or 0.0)
        path_novelty = float(backend.get('pcvm_path_mem_norm') or 0.0)
        visual_novelty = float(backend.get('pcvm_visual_mem_norm') or 0.0)
        surprise = float(backend.get('predictive_surprise') or backend.get('pcvm_surprise') or 0.0)
        path_new_cluster = bool(backend.get('pcvm_path_new_cluster'))
        visual_new_cluster = bool(backend.get('pcvm_visual_new_cluster'))
        new_cluster = bool(backend.get('pcvm_new_cluster') or path_new_cluster or visual_new_cluster)
        cluster_id = backend.get('pcvm_cluster_id')

        reward = float(self.args.base_step_cost)
        motion_gate = bool(terms['motion_gate'])
        novelty_reward = self.args.motion_novelty_weight * novelty if motion_gate else 0.0
        new_cluster_bonus = self.args.new_cluster_bonus if (motion_gate and new_cluster) else 0.0
        surprise_reward = self.args.slow_surprise_weight * surprise
        safe_motion_bonus = self.args.safe_motion_bonus if (
            motion_gate and not terms['front_stop'] and not terms['recovery_reverse'] and not terms['drive_failed'] and not terms['turn_failed']
            and terms['executed_distance_cm'] >= self.args.safe_motion_min_cm
        ) else 0.0
        distance_reward = self.args.executed_distance_weight * min(1.0, terms['executed_distance_cm'] / max(1e-6, self.args.max_distance_cm))

        recent_revisit_penalty = 0.0
        if cluster_id is not None and not new_cluster:
            recent_revisit_penalty = self.args.recent_revisit_penalty * max(0.0, 1.0 - novelty)

        near_obstacle_raw = self._near_obstacle_penalty(self.last_distances)
        near_obstacle_penalty = self.args.near_obstacle_penalty * near_obstacle_raw
        stuck_penalty = 0.0
        if terms['zero_distance'] or terms['front_stop'] or terms['drive_failed'] or terms['turn_failed']:
            stuck_penalty = self.args.stuck_penalty
        recovery_penalty = self.args.slow_recovery_penalty if (terms['recovery_reverse'] or terms['recovery_turn']) else 0.0
        self.recent_recoveries.append(1 if recovery_penalty > 0.0 else 0)
        recovery_streak_penalty = 0.0
        if recovery_penalty > 0.0:
            recovery_streak_penalty = self.args.recovery_streak_penalty * max(0, sum(self.recent_recoveries) - 1)
        zero_forward_penalty = 0.0
        if terms['already_at_front_threshold'] or (execution is not None and terms['executed_distance_cm'] < self.args.zero_forward_cm):
            zero_forward_penalty = self.args.zero_forward_penalty
        loop_penalty, loop_terms = self._loop_penalty(backend, terms)

        reward += novelty_reward + new_cluster_bonus + surprise_reward + safe_motion_bonus + distance_reward
        reward -= (
            recent_revisit_penalty
            + near_obstacle_penalty
            + stuck_penalty
            + recovery_penalty
            + recovery_streak_penalty
            + zero_forward_penalty
            + loop_penalty
        )
        self.last_reward_terms = {
            'base': self.args.base_step_cost,
            'motion_gate': motion_gate,
            'novelty_raw': novelty,
            'path_novelty_raw': path_novelty,
            'visual_novelty_raw': visual_novelty,
            'path_new_cluster': path_new_cluster,
            'visual_new_cluster': visual_new_cluster,
            'motion_gated_novelty_reward': novelty_reward,
            'new_cluster_bonus': new_cluster_bonus,
            'surprise_reward': surprise_reward,
            'safe_motion_bonus': safe_motion_bonus,
            'executed_distance_reward': distance_reward,
            'recent_revisit_penalty': recent_revisit_penalty,
            'near_obstacle_raw': near_obstacle_raw,
            'near_obstacle_penalty': near_obstacle_penalty,
            'stuck_penalty': stuck_penalty,
            'recovery_penalty': recovery_penalty,
            'recovery_streak_count': int(sum(self.recent_recoveries)),
            'recovery_streak_penalty': recovery_streak_penalty,
            'zero_forward_penalty': zero_forward_penalty,
            'loop_penalty': loop_penalty,
            **loop_terms,
            **terms,
        }
        return float(reward)

    def _loop_penalty(self, backend, motion_terms):
        pose = backend.get('pcvm_pose') if backend else None
        if not pose or len(pose) < 2:
            return 0.0, {'loop_min_dist_m': None}
        x = float(pose[0])
        y = float(pose[1])
        min_dist = None
        if self.recent_reward_poses:
            min_dist = min(float(np.hypot(x - px, y - py)) for px, py in self.recent_reward_poses)
        self.recent_reward_poses.append((x, y))
        if min_dist is None:
            return 0.0, {'loop_min_dist_m': None}
        if not (motion_terms.get('moved_gate') or motion_terms.get('turned_gate')):
            return 0.0, {'loop_min_dist_m': min_dist}
        radius = max(1e-6, self.args.loop_near_radius_m)
        pressure = max(0.0, 1.0 - min_dist / radius)
        penalty = self.args.loop_revisit_penalty * pressure
        if motion_terms.get('executed_distance_cm', 0.0) >= self.args.loop_long_move_cm:
            penalty *= self.args.loop_long_move_scale
        return penalty, {'loop_min_dist_m': min_dist, 'loop_pressure': pressure}

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)
        requested_pcvm_action = self._policy_action_to_pcvm_action(action)
        self.obs_builder.last_action = requested_pcvm_action.copy()

        recovery = self._recover_if_blocked(self.last_distances)
        execution = None
        target = None
        if recovery is None or recovery.get('continue_after_reverse'):
            if self.args.action_mode == 'theta_until_front':
                theta_norm = clamp(float(action[0]), -1.0, 1.0)
                theta_deg = theta_norm * self.args.max_theta_deg
                target = {
                    'mode': 'theta_until_front',
                    'theta_norm': theta_norm,
                    'theta_deg': theta_deg,
                    'front_stop_cm': self.args.until_front_cm,
                }
                execution = self.executor.execute_theta_until_front(theta_deg, self.args.until_front_cm)
            else:
                target = action_to_target(
                    action,
                    self.args.max_theta_deg,
                    self.args.max_distance_cm,
                    self.args.min_drive_cm,
                    self.args.min_distance_cm,
                )
                execution = self.executor.execute_local_target(target['x_cm'], target['y_cm'])
        else:
            if self.args.action_mode == 'theta_until_front':
                theta_norm = clamp(float(action[0]), -1.0, 1.0)
                target = {
                    'mode': 'theta_until_front',
                    'theta_norm': theta_norm,
                    'theta_deg': theta_norm * self.args.max_theta_deg,
                    'front_stop_cm': self.args.until_front_cm,
                }
            else:
                target = action_to_target(action, self.args.max_theta_deg, self.args.max_distance_cm, self.args.min_drive_cm, self.args.min_distance_cm)

        execution_feedback = {'execution': execution, 'recovery': recovery}
        executed_action = self._execution_feedback_to_pcvm_action(execution_feedback)
        self.obs_builder.last_action = executed_action.copy()
        if self.args.settle_seconds > 0.0:
            time.sleep(self.args.settle_seconds)
        obs, distances, backend = self._build_obs(executed_action, execution_feedback)
        reward = self._reward(execution, backend, recovery)
        self._update_path_memory(backend)
        info = {
            'step': self.step_count,
            'action': [float(x) for x in action.reshape(-1)],
            'action_mode': self.args.action_mode,
            'action_contract': '[theta_norm]' if self.args.action_mode == 'theta_until_front' else '[theta_norm, distance_norm]',
            'executed_action_for_pcvm': [float(executed_action[0]), float(executed_action[1])],
            'target': target,
            'distances': distances,
            'backend': backend,
            'execution': execution,
            'recovery': recovery,
            'reward': reward,
            'reward_terms': self.last_reward_terms,
        }
        frame_path = self._save_step_frame(self.step_count)
        if frame_path is not None:
            info['frame_path'] = frame_path
        if self.dashboard is not None:
            self.dashboard.update(info, getattr(self.obs_builder, 'last_frame', None))
        self._emit(info)
        self.last_action = action.copy()
        self.last_executed_action = executed_action.copy()
        self.last_backend = backend
        self.last_distances = distances
        time.sleep(self.args.sleep)
        return obs, reward, False, False, info

    def _policy_action_to_pcvm_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.args.action_mode == 'theta_until_front':
            return np.array([clamp(float(action[0]), -1.0, 1.0), 1.0], dtype=np.float32)
        return np.asarray(action[:2], dtype=np.float32)

    def _execution_feedback_to_pcvm_action(self, execution_feedback):
        if self.args.action_mode == 'theta_until_front':
            execution = execution_feedback.get('execution') if isinstance(execution_feedback, dict) else None
            recovery = execution_feedback.get('recovery') if isinstance(execution_feedback, dict) else None
            theta_deg = 0.0
            distance_cm = 0.0
            if recovery:
                reverse = recovery.get('reverse') or {}
                if reverse:
                    distance_cm -= max(0.0, float(reverse.get('requested_distance_cm') or 0.0))
                report = recovery.get('recovery') or {}
                theta_deg += float(report.get('yaw_deg') or 0.0)
            if execution:
                turn = execution.get('turn') or {}
                theta_deg += float(turn.get('yaw_deg') or execution.get('theta_deg') or 0.0)
                drive = execution.get('drive') or {}
                distance_cm += max(0.0, float(drive.get('estimated_distance_cm') or execution.get('clipped_distance_cm') or 0.0))
            theta_norm = clamp(theta_deg / max(1e-6, self.args.max_theta_deg), -1.0, 1.0)
            dist_norm = 1.0 if distance_cm >= self.args.motion_gate_cm else -1.0
            return np.array([theta_norm, dist_norm], dtype=np.float32)
        return execution_feedback_to_action(
            execution_feedback, self.args.max_theta_deg, self.args.max_distance_cm, self.args.min_distance_cm
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Train predictive SAC online on the real rover.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--backend', choices=['predictive', 'pcvm', 'pcvm-m', 'pcvm-t'], default='pcvm')
    parser.add_argument('--action-mode', choices=['local_target', 'theta_until_front'], default='local_target')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--log-path', default=None, help='Write per-step JSONL records for later analysis')
    parser.add_argument('--frame-dir', default=None, help='Save one post-action camera frame per step')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--sleep', type=float, default=0.2)
    parser.add_argument('--settle-seconds', type=float, default=0.0, help='Stop-and-settle time before post-action observation')
    parser.add_argument('--max-theta-deg', type=float, default=75.0)
    parser.add_argument('--max-distance-cm', type=float, default=80.0)
    parser.add_argument('--min-distance-cm', type=float, default=0.0)
    parser.add_argument('--min-drive-cm', type=float, default=10.0)
    parser.add_argument('--turn-pwm', type=float, default=65.0)
    parser.add_argument('--drive-pwm', type=float, default=75.0)
    parser.add_argument('--until-front-cm', type=float, default=40.0)
    parser.add_argument('--until-front-max-seconds', type=float, default=4.0)
    parser.add_argument('--cm-per-second', type=float, default=40.0)
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
    parser.add_argument('--reward-mode', choices=['legacy', 'slow_rlxf'], default='legacy')
    parser.add_argument('--base-step-cost', type=float, default=-0.02)
    parser.add_argument('--motion-gate-cm', type=float, default=3.0)
    parser.add_argument('--yaw-gate-deg', type=float, default=8.0)
    parser.add_argument('--motion-novelty-weight', type=float, default=0.35)
    parser.add_argument('--new-cluster-bonus', type=float, default=0.40)
    parser.add_argument('--slow-surprise-weight', type=float, default=0.07)
    parser.add_argument('--safe-motion-bonus', type=float, default=0.12)
    parser.add_argument('--safe-motion-min-cm', type=float, default=10.0)
    parser.add_argument('--executed-distance-weight', type=float, default=0.10)
    parser.add_argument('--recent-revisit-penalty', type=float, default=0.30)
    parser.add_argument('--near-obstacle-penalty', type=float, default=0.25)
    parser.add_argument('--side-near-cm', type=float, default=25.0)
    parser.add_argument('--stuck-penalty', type=float, default=0.35)
    parser.add_argument('--slow-recovery-penalty', type=float, default=0.55)
    parser.add_argument('--zero-forward-cm', type=float, default=3.0)
    parser.add_argument('--zero-forward-penalty', type=float, default=0.35)
    parser.add_argument('--loop-memory-size', type=int, default=25)
    parser.add_argument('--loop-near-radius-m', type=float, default=0.45)
    parser.add_argument('--loop-revisit-penalty', type=float, default=0.45)
    parser.add_argument('--loop-long-move-cm', type=float, default=80.0)
    parser.add_argument('--loop-long-move-scale', type=float, default=0.45)
    parser.add_argument('--recovery-streak-window', type=int, default=8)
    parser.add_argument('--recovery-streak-penalty', type=float, default=0.10)
    parser.add_argument('--viz-port', type=int, default=0)
    parser.add_argument('--viz-depth-model', default='depth-anything/Depth-Anything-V2-Small-hf')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_path is None:
        if args.backend == 'pcvm-m':
            args.save_path = 'results/pcvm_m_sac_real.zip'
        elif args.backend == 'pcvm-t':
            args.save_path = 'results/pcvm_t_sac_real.zip'
        elif args.backend == 'pcvm':
            args.save_path = 'results/pcvm_cnn_sac_real.zip'
        else:
            args.save_path = 'results/predictive_sac_real.zip'
    from stable_baselines3 import SAC

    dashboard = None
    if args.viz_port:
        from tools.rover_visual_dashboard import DashboardConfig, RoverVisualDashboard

        dashboard = RoverVisualDashboard(DashboardConfig(port=args.viz_port, depth_model=args.viz_depth_model))
        dashboard.start()
        print(json.dumps({'visual_dashboard': f'http://0.0.0.0:{args.viz_port}', 'depth_model': args.viz_depth_model}, sort_keys=True), flush=True)
    env = RealPredictiveSACEnv(args, dashboard=dashboard)
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
