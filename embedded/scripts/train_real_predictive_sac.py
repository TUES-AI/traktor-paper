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
    pcvm_obs_dim,
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
        self.policy_action_dim = int(self.action_space.shape[0])
        obs_dim = pcvm_obs_dim(self.policy_action_dim) if args.backend in ('pcvm', 'pcvm-m', 'pcvm-d', 'pcvm-d3', 'pcvm-j', 'pcvm-t') else 79
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
        if args.backend in ('pcvm', 'pcvm-m', 'pcvm-d', 'pcvm-d3', 'pcvm-j', 'pcvm-t'):
            self.obs_builder = PCVMRoverObsBuilder(
                self.rover,
                self.safety,
                mobilenet=(args.backend == 'pcvm-m'),
                dino=(args.backend == 'pcvm-d'),
                dino3=(args.backend == 'pcvm-d3'),
                jepa=(args.backend == 'pcvm-j'),
                transformer=(args.backend == 'pcvm-t'),
                action_dim=self.policy_action_dim,
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
        if self.args.backend in ('pcvm', 'pcvm-m', 'pcvm-d', 'pcvm-d3', 'pcvm-j', 'pcvm-t'):
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

    def _blocked_open_turn_bonus(self, execution):
        if not execution or self.args.blocked_open_turn_bonus <= 0.0:
            return 0.0, {}
        turn = execution.get('turn') or {}
        drive = execution.get('drive') or {}
        start_distances = execution.get('start_distances') or {}
        after_turn_distances = drive.get('start_distances') or {}
        before_front = start_distances.get('front')
        after_front = after_turn_distances.get('front')
        yaw_deg = abs(float(turn.get('yaw_deg') or execution.get('theta_deg') or 0.0))
        terms = {
            'blocked_open_before_front_cm': before_front,
            'blocked_open_after_front_cm': after_front,
            'blocked_open_yaw_deg': yaw_deg,
        }
        if before_front is None or after_front is None:
            return 0.0, terms
        before_front = float(before_front)
        after_front = float(after_front)
        if before_front > self.args.blocked_open_before_cm:
            return 0.0, terms
        if yaw_deg < self.args.blocked_open_min_theta_deg:
            return 0.0, terms
        improvement = after_front - before_front
        terms['blocked_open_improvement_cm'] = improvement
        if improvement < self.args.blocked_open_min_improvement_cm:
            return 0.0, terms
        open_score = min(1.0, after_front / max(1e-6, self.args.blocked_open_scale_cm))
        improvement_score = min(1.0, improvement / max(1e-6, self.args.blocked_open_scale_cm))
        bonus = self.args.blocked_open_turn_bonus * max(open_score, improvement_score)
        terms['blocked_open_score'] = max(open_score, improvement_score)
        return float(bonus), terms

    def _coverage_expansion_bonus(self, backend):
        pose = backend.get('pcvm_pose') if backend else None
        if not pose or len(pose) < 2 or not self.path_points:
            return 0.0, {
                'coverage_bbox_area_before_m2': 0.0,
                'coverage_bbox_area_after_m2': 0.0,
                'coverage_bbox_delta_m2': 0.0,
                'coverage_radius_delta_m': 0.0,
            }
        x = float(pose[0])
        y = float(pose[1])
        xs = [p[0] for p in self.path_points]
        ys = [p[1] for p in self.path_points]
        area_before = (max(xs) - min(xs)) * (max(ys) - min(ys)) if len(xs) > 1 else 0.0
        area_after = (max(max(xs), x) - min(min(xs), x)) * (max(max(ys), y) - min(min(ys), y))
        area_delta = max(0.0, area_after - area_before)

        start_x, start_y = self.path_points[0]
        radius_before = max(float(np.hypot(px - start_x, py - start_y)) for px, py in self.path_points)
        radius_after = max(radius_before, float(np.hypot(x - start_x, y - start_y)))
        radius_delta = max(0.0, radius_after - radius_before)
        bonus = self.args.coverage_bbox_weight * area_delta + self.args.coverage_radius_weight * radius_delta
        return float(bonus), {
            'coverage_bbox_area_before_m2': area_before,
            'coverage_bbox_area_after_m2': area_after,
            'coverage_bbox_delta_m2': area_delta,
            'coverage_radius_before_m': radius_before,
            'coverage_radius_after_m': radius_after,
            'coverage_radius_delta_m': radius_delta,
            'coverage_expansion_bonus': float(bonus),
        }

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
        blocked_open_turn_bonus, blocked_open_terms = self._blocked_open_turn_bonus(execution)
        coverage_expansion_bonus, coverage_terms = self._coverage_expansion_bonus(backend)

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

        reward += (
            novelty_reward
            + new_cluster_bonus
            + surprise_reward
            + safe_motion_bonus
            + distance_reward
            + blocked_open_turn_bonus
            + coverage_expansion_bonus
        )
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
            'blocked_open_turn_bonus': blocked_open_turn_bonus,
            'coverage_expansion_bonus': coverage_expansion_bonus,
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
            **blocked_open_terms,
            **coverage_terms,
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
            'executed_action_for_pcvm': [float(x) for x in executed_action.reshape(-1)],
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
            return np.array([clamp(float(action[0]), -1.0, 1.0)], dtype=np.float32)
        return np.asarray(action[:2], dtype=np.float32)

    def _execution_feedback_to_pcvm_action(self, execution_feedback):
        if self.args.action_mode == 'theta_until_front':
            execution = execution_feedback.get('execution') if isinstance(execution_feedback, dict) else None
            recovery = execution_feedback.get('recovery') if isinstance(execution_feedback, dict) else None
            theta_deg = 0.0
            if recovery:
                reverse = recovery.get('reverse') or {}
                report = recovery.get('recovery') or {}
                theta_deg += float(report.get('yaw_deg') or 0.0)
            if execution:
                turn = execution.get('turn') or {}
                theta_deg += float(turn.get('yaw_deg') or execution.get('theta_deg') or 0.0)
            theta_norm = clamp(theta_deg / max(1e-6, self.args.max_theta_deg), -1.0, 1.0)
            return np.array([theta_norm], dtype=np.float32)
        return execution_feedback_to_action(
            execution_feedback, self.args.max_theta_deg, self.args.max_distance_cm, self.args.min_distance_cm
        )


def default_pcvm_path(sac_path):
    path = Path(sac_path)
    if path.suffix:
        return str(path.with_name(f'{path.stem}_pcvm.pt'))
    return str(path) + '_pcvm.pt'


def default_replay_path(sac_path):
    path = Path(sac_path)
    if path.suffix:
        return str(path.with_name(f'{path.stem}_replay.pkl'))
    return str(path) + '_replay.pkl'


def get_pcvm_model(env):
    builder = getattr(env, 'obs_builder', None)
    return getattr(builder, 'model', None)


def _running_mean_state(obj):
    if obj is None:
        return None
    return {'n': int(getattr(obj, 'n', 0)), 'mean': float(getattr(obj, 'mean', 0.0))}


def _load_running_mean(obj, state):
    if obj is None or not isinstance(state, dict):
        return
    obj.n = int(state.get('n', 0))
    obj.mean = float(state.get('mean', 0.0))


def _memory_bank_state(bank):
    if bank is None:
        return None
    return {
        'bank': [x.detach().cpu() for x in getattr(bank, 'bank', [])],
        'counts': list(getattr(bank, 'counts', [])),
        'last_seen': list(getattr(bank, 'last_seen', [])),
        'maxlen': int(getattr(bank, 'maxlen', 0)),
        'known_dist': float(getattr(bank, 'known_dist', 0.0)),
        'update_rate': float(getattr(bank, 'update_rate', 0.0)),
    }


def _load_memory_bank(bank, state, device):
    if bank is None or not isinstance(state, dict):
        return
    bank.bank = [x.detach().to(device) for x in state.get('bank', [])]
    bank.counts = list(state.get('counts', []))
    bank.last_seen = list(state.get('last_seen', []))
    if 'maxlen' in state:
        bank.maxlen = int(state['maxlen'])
    if 'known_dist' in state:
        bank.known_dist = float(state['known_dist'])
    if 'update_rate' in state:
        bank.update_rate = float(state['update_rate'])


def pcvm_state_dict(model, args):
    if model is None:
        return None
    state = {
        'format': 'pcvm_state_v1',
        'backend': args.backend,
        'action_mode': args.action_mode,
        'action_dim': int(getattr(model, 'action_dim', 0)),
        'class_name': model.__class__.__name__,
        'device': str(getattr(model, 'device', 'cpu')),
        'step': int(getattr(model, 'step', 0)),
        'net': model.net.state_dict() if hasattr(model, 'net') else None,
        'opt': model.opt.state_dict() if hasattr(model, 'opt') else None,
        'rnd_opt': model.rnd_opt.state_dict() if hasattr(model, 'rnd_opt') else None,
        'memory': _memory_bank_state(getattr(model, 'memory', None)),
        'visual_memory': _memory_bank_state(getattr(model, 'visual_memory', None)),
        'rnd_norm': _running_mean_state(getattr(model, 'rnd_norm', None)),
        'surprise_norm': _running_mean_state(getattr(model, 'surprise_norm', None)),
        'hidden': getattr(model, 'hidden', None).detach().cpu() if getattr(model, 'hidden', None) is not None else None,
        'prev_hidden': getattr(model, 'prev_hidden', None).detach().cpu() if getattr(model, 'prev_hidden', None) is not None else None,
        'prev_z': getattr(model, 'prev_z', None).detach().cpu() if getattr(model, 'prev_z', None) is not None else None,
        'prev_visual': getattr(model, 'prev_visual', None).detach().cpu() if getattr(model, 'prev_visual', None) is not None else None,
        'prev_proprio': getattr(model, 'prev_proprio', None).detach().cpu() if getattr(model, 'prev_proprio', None) is not None else None,
        'tokens': [x.detach().cpu() for x in getattr(model, 'tokens', [])],
        'pose': [float(getattr(model, 'pose_x', 0.0)), float(getattr(model, 'pose_y', 0.0)), float(getattr(model, 'yaw_rad', 0.0))],
    }
    return state


def save_pcvm_state(model, path, args):
    if model is None or path is None:
        return None
    import torch

    state = pcvm_state_dict(model, args)
    if state is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return str(path)


def load_pcvm_state(model, path, args):
    if model is None or path is None:
        return False
    path = Path(path)
    if not path.exists():
        print(json.dumps({'warning': 'pcvm_resume_missing', 'path': str(path)}, sort_keys=True), flush=True)
        return False
    import torch

    state = torch.load(path, map_location=getattr(model, 'device', 'cpu'))
    device = getattr(model, 'device', 'cpu')
    saved_action_dim = state.get('action_dim')
    current_action_dim = int(getattr(model, 'action_dim', 0))
    if saved_action_dim is not None and int(saved_action_dim) != current_action_dim:
        print(json.dumps({
            'warning': 'pcvm_resume_action_dim_mismatch',
            'path': str(path),
            'saved_action_dim': int(saved_action_dim),
            'current_action_dim': current_action_dim,
        }, sort_keys=True), flush=True)
        return False
    if state.get('net') is not None and hasattr(model, 'net'):
        missing, unexpected = model.net.load_state_dict(state['net'], strict=False)
        if missing or unexpected:
            print(json.dumps({'warning': 'pcvm_net_partial_load', 'missing': missing, 'unexpected': unexpected}, sort_keys=True), flush=True)
    for attr, key in (('opt', 'opt'), ('rnd_opt', 'rnd_opt')):
        opt = getattr(model, attr, None)
        if opt is not None and state.get(key) is not None:
            try:
                opt.load_state_dict(state[key])
            except Exception as exc:
                print(json.dumps({'warning': f'pcvm_{key}_load_failed', 'error': repr(exc)}, sort_keys=True), flush=True)
    if not args.reset_pcvm_memory_on_resume:
        _load_memory_bank(getattr(model, 'memory', None), state.get('memory'), device)
        _load_memory_bank(getattr(model, 'visual_memory', None), state.get('visual_memory'), device)
    if not args.reset_pcvm_norms_on_resume:
        _load_running_mean(getattr(model, 'rnd_norm', None), state.get('rnd_norm'))
        _load_running_mean(getattr(model, 'surprise_norm', None), state.get('surprise_norm'))
    if not args.reset_pcvm_hidden_on_resume:
        for attr in ('hidden', 'prev_hidden', 'prev_z', 'prev_visual', 'prev_proprio'):
            tensor = state.get(attr)
            if tensor is not None and hasattr(model, attr):
                setattr(model, attr, tensor.detach().to(device))
        if hasattr(model, 'tokens') and state.get('tokens') is not None:
            model.tokens.clear()
            for token in state.get('tokens', []):
                model.tokens.append(token.detach().to(device))
    elif hasattr(model, 'reset'):
        model.reset()
    else:
        for attr in ('prev_hidden', 'prev_z', 'prev_visual', 'prev_proprio'):
            if hasattr(model, attr):
                setattr(model, attr, None)
        if hasattr(model, 'tokens'):
            model.tokens.clear()
    if not args.reset_pcvm_pose_on_resume and isinstance(state.get('pose'), (list, tuple)) and len(state['pose']) >= 3:
        model.pose_x = float(state['pose'][0])
        model.pose_y = float(state['pose'][1])
        model.yaw_rad = float(state['pose'][2])
    else:
        for attr in ('pose_x', 'pose_y', 'yaw_rad'):
            if hasattr(model, attr):
                setattr(model, attr, 0.0)
    if not args.reset_pcvm_step_on_resume and 'step' in state:
        model.step = int(state['step'])
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='Train predictive SAC online on the real rover.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--backend', choices=['predictive', 'pcvm', 'pcvm-m', 'pcvm-d', 'pcvm-d3', 'pcvm-j', 'pcvm-t'], default='pcvm')
    parser.add_argument('--action-mode', choices=['local_target', 'theta_until_front'], default='local_target')
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--log-path', default=None, help='Write per-step JSONL records for later analysis')
    parser.add_argument('--frame-dir', default=None, help='Save one post-action camera frame per step')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--pcvm-save-path', default=None, help='Save PCVM/PCVM-M/PCVM-T learnable state here; defaults to <save>_pcvm.pt')
    parser.add_argument('--pcvm-resume-path', default=None, help='Resume PCVM state from here; defaults to sidecar next to --resume')
    parser.add_argument('--no-auto-pcvm-resume', action='store_true', help='Do not auto-load <resume>_pcvm.pt when --resume is used')
    parser.add_argument('--reset-pcvm-memory-on-resume', action='store_true', help='Resume PCVM weights but clear path/visual memory banks')
    parser.add_argument('--reset-pcvm-hidden-on-resume', action='store_true', help='Resume PCVM weights but clear recurrent/context state')
    parser.add_argument('--reset-pcvm-pose-on-resume', action='store_true', help='Resume PCVM weights but reset dead-reckoned PCVM pose')
    parser.add_argument('--reset-pcvm-norms-on-resume', action='store_true', help='Resume PCVM weights but reset novelty/surprise running normalizers')
    parser.add_argument('--reset-pcvm-step-on-resume', action='store_true', help='Resume PCVM weights but reset its internal step counter')
    parser.add_argument('--replay-save-path', default=None, help='Save SAC replay buffer here; defaults to <save>_replay.pkl')
    parser.add_argument('--replay-resume-path', default=None, help='Resume SAC replay buffer from here; defaults to sidecar next to --resume')
    parser.add_argument('--no-auto-replay-resume', action='store_true', help='Do not auto-load <resume>_replay.pkl when --resume is used')
    parser.add_argument('--no-save-replay-buffer', action='store_true', help='Skip saving the SAC replay buffer sidecar')
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
    parser.add_argument('--blocked-open-turn-bonus', type=float, default=0.0)
    parser.add_argument('--blocked-open-before-cm', type=float, default=55.0)
    parser.add_argument('--blocked-open-min-theta-deg', type=float, default=25.0)
    parser.add_argument('--blocked-open-min-improvement-cm', type=float, default=20.0)
    parser.add_argument('--blocked-open-scale-cm', type=float, default=120.0)
    parser.add_argument('--coverage-bbox-weight', type=float, default=0.0)
    parser.add_argument('--coverage-radius-weight', type=float, default=0.0)
    parser.add_argument('--viz-port', type=int, default=0)
    parser.add_argument('--viz-depth-model', default='depth-anything/Depth-Anything-V2-Small-hf')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_path is None:
        if args.backend == 'pcvm-m':
            args.save_path = 'results/pcvm_m_sac_real.zip'
        elif args.backend == 'pcvm-d':
            args.save_path = 'results/pcvm_d_sac_real.zip'
        elif args.backend == 'pcvm-d3':
            args.save_path = 'results/pcvm_d3_sac_real.zip'
        elif args.backend == 'pcvm-j':
            args.save_path = 'results/pcvm_d_sac_real.zip'
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
            replay_resume_path = args.replay_resume_path
            if replay_resume_path is None and not args.no_auto_replay_resume:
                replay_resume_path = default_replay_path(args.resume)
            if replay_resume_path is not None and Path(replay_resume_path).exists():
                try:
                    model.load_replay_buffer(replay_resume_path)
                    print(json.dumps({'replay_resumed': replay_resume_path}, sort_keys=True), flush=True)
                except Exception as exc:
                    print(json.dumps({'warning': 'replay_resume_failed', 'path': replay_resume_path, 'error': repr(exc)}, sort_keys=True), flush=True)
            pcvm_resume_path = args.pcvm_resume_path
            if pcvm_resume_path is None and not args.no_auto_pcvm_resume:
                pcvm_resume_path = default_pcvm_path(args.resume)
            if args.backend in ('pcvm', 'pcvm-m', 'pcvm-d', 'pcvm-d3', 'pcvm-j', 'pcvm-t') and pcvm_resume_path is not None:
                resumed = load_pcvm_state(get_pcvm_model(env), pcvm_resume_path, args)
                print(json.dumps({
                    'pcvm_resumed': bool(resumed),
                    'pcvm_resume_path': pcvm_resume_path,
                    'reset_pcvm_memory': bool(args.reset_pcvm_memory_on_resume),
                    'reset_pcvm_hidden': bool(args.reset_pcvm_hidden_on_resume),
                    'reset_pcvm_pose': bool(args.reset_pcvm_pose_on_resume),
                    'reset_pcvm_norms': bool(args.reset_pcvm_norms_on_resume),
                }, sort_keys=True), flush=True)
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
        saved = {'sac_saved': args.save_path}
        if not args.no_save_replay_buffer:
            replay_save_path = args.replay_save_path or default_replay_path(args.save_path)
            try:
                Path(replay_save_path).parent.mkdir(parents=True, exist_ok=True)
                model.save_replay_buffer(replay_save_path)
                saved['replay_saved'] = replay_save_path
            except Exception as exc:
                saved['replay_save_failed'] = repr(exc)
        if args.backend in ('pcvm', 'pcvm-m', 'pcvm-d', 'pcvm-d3', 'pcvm-j', 'pcvm-t'):
            pcvm_save_path = args.pcvm_save_path or default_pcvm_path(args.save_path)
            try:
                saved['pcvm_saved'] = save_pcvm_state(get_pcvm_model(env), pcvm_save_path, args)
            except Exception as exc:
                saved['pcvm_save_failed'] = repr(exc)
        print(json.dumps(saved, sort_keys=True), flush=True)
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
