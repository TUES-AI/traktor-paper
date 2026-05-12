#!/usr/bin/env python3
"""Run a deterministic visionless baseline through the same safety executor.

This is not a learned policy. It is a simple reactive baseline for real-life
comparison against SAC:

1. If front is open, go mostly forward with a small deterministic sweep.
2. If front is narrowing, bias toward the freer side.
3. If front is blocked, rotate hard toward the freer side.

Every action goes through LocalTargetExecutor.execute_theta_until_front(), so it
uses the post-turn front gate and the front-only forward polling path. The script
logs a theoretical TWF reward for apples-to-apples comparison with SAC logs.
"""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150
from twf.reward import sensory_novelty, world_feedback_reward


def finite_or(value, fallback):
    return fallback if value is None else float(value)


def norm_range_cm(value, max_cm=200.0):
    if value is None:
        return 1.0
    return float(np.clip(float(value) / max_cm, 0.0, 1.0))


class DeterministicExplorer:
    """Tiny reactive policy with memory only for sweep phase."""

    def __init__(self, open_front_cm=95.0, narrow_front_cm=65.0, blocked_front_cm=42.0):
        self.open_front_cm = float(open_front_cm)
        self.narrow_front_cm = float(narrow_front_cm)
        self.blocked_front_cm = float(blocked_front_cm)
        self.sweep_sign = 1.0
        self.forward_count = 0

    def choose_theta(self, distances):
        front = distances.get('front')
        left = finite_or(distances.get('left'), 250.0)
        right = finite_or(distances.get('right'), 250.0)
        front_clear = finite_or(front, 250.0)
        freer_sign = 1.0 if left >= right else -1.0
        side_gap = abs(left - right)

        if front is not None and front <= self.blocked_front_cm:
            self.sweep_sign = freer_sign
            self.forward_count = 0
            theta = 90.0 * freer_sign
            mode = 'blocked_turn_to_freer_side'
        elif front_clear < self.narrow_front_cm:
            self.sweep_sign = freer_sign
            self.forward_count = 0
            theta = (65.0 if side_gap > 20.0 else 45.0) * freer_sign
            mode = 'narrow_bias_to_freer_side'
        elif front_clear >= self.open_front_cm:
            # Keep mostly forward, but inject a tiny deterministic sweep so the
            # baseline does not drive one long wall-parallel line forever.
            self.forward_count += 1
            if self.forward_count % 5 == 0:
                self.sweep_sign *= -1.0
            theta = 12.0 * self.sweep_sign
            mode = 'open_forward_sweep'
        else:
            self.forward_count += 1
            theta = 28.0 * freer_sign
            mode = 'medium_space_soft_bias'

        return float(theta), {
            'mode': mode,
            'front_cm': front,
            'left_cm': distances.get('left'),
            'right_cm': distances.get('right'),
            'freer_side': 'left' if freer_sign > 0 else 'right',
            'theta_deg': float(theta),
        }


def state_vector(distances, theta_deg, feedback):
    return np.asarray(
        [
            norm_range_cm(distances.get('left')),
            norm_range_cm(distances.get('right')),
            norm_range_cm(distances.get('front')),
            float(theta_deg) / 90.0,
            float(feedback.get('executed_distance_cm', 0.0)) / 100.0,
            float(feedback.get('zero_progress', 0.0)),
            float(feedback.get('post_turn_blocked', 0.0)),
        ],
        dtype=np.float32,
    )


def score_step(*, state, recent_states, theta_deg, distances, execution):
    drive = execution.get('drive') or {}
    post_turn = execution.get('post_turn_front_check') or {}
    executed_cm = float(execution.get('clipped_distance_cm', 0.0) or 0.0)
    novelty = sensory_novelty(state.tolist(), list(recent_states))
    revisit_score = max(0.0, 1.0 - novelty)
    post_turn_blocked = execution.get('reason') == 'post_turn_front_blocked_before_drive'
    zero_progress = executed_cm < 2.0
    near_obstacle = float(distances.get('front') or 200.0) < 25.0
    open_front_after_forward_action = bool(abs(theta_deg) <= 20.0 and post_turn.get('open_front_bonus'))
    loop_score = 1.0 if revisit_score > 0.82 and executed_cm > 5.0 else 0.0

    reward = world_feedback_reward(
        sensory_novelty_value=novelty,
        executed_distance_cm=executed_cm,
        safe_motion=executed_cm > 5.0 and not execution.get('reverse_recovery'),
        revisit_score=revisit_score,
        zero_progress=zero_progress,
        recovery=bool(execution.get('reverse_recovery') or drive.get('contact_recovery')),
        near_obstacle=float(near_obstacle),
        loop_score=loop_score,
    )
    if post_turn_blocked:
        reward -= 1.0
    if open_front_after_forward_action:
        reward += 1.0

    terms = {
        'sensory_novelty': novelty,
        'revisit_score': revisit_score,
        'executed_distance_cm': executed_cm,
        'safe_motion': executed_cm > 5.0 and not execution.get('reverse_recovery'),
        'zero_progress': zero_progress,
        'near_obstacle': near_obstacle,
        'post_turn_blocked': post_turn_blocked,
        'open_front_after_forward_action': open_front_after_forward_action,
        'loop_score': loop_score,
    }
    return float(reward), terms


def main():
    parser = argparse.ArgumentParser(description='Run deterministic TWF baseline on real rover.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--log-path', default='results/deterministic_twf_baseline.jsonl')
    parser.add_argument('--drive-pwm', type=float, default=90.0)
    parser.add_argument('--turn-pwm', type=float, default=65.0)
    parser.add_argument('--front-stop-cm', type=float, default=40.0)
    parser.add_argument('--open-front-cm', type=float, default=95.0)
    parser.add_argument('--narrow-front-cm', type=float, default=65.0)
    parser.add_argument('--blocked-front-cm', type=float, default=42.0)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    policy = DeterministicExplorer(
        open_front_cm=args.open_front_cm,
        narrow_front_cm=args.narrow_front_cm,
        blocked_front_cm=args.blocked_front_cm,
    )
    recent_states = deque(maxlen=32)
    last_feedback = {}

    rover = RoverAPI(camera_enabled=False)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(rover, imu=imu, config=SafetyConfig())
    executor = LocalTargetExecutor(
        safety,
        config=LocalTargetExecutorConfig(
            drive_pwm=args.drive_pwm,
            turn_pwm=args.turn_pwm,
            until_front_stop_cm=args.front_stop_cm,
        ),
    )

    try:
        gyro_z_bias = safety.calibrate_gyro()
        with log_path.open('w') as f:
            for step in range(1, args.steps + 1):
                distances = safety.read_distances()
                theta_deg, decision = policy.choose_theta(distances)
                execution = executor.execute_theta_until_front(theta_deg, front_stop_cm=args.front_stop_cm)
                state = state_vector(distances, theta_deg, last_feedback)
                reward, reward_terms = score_step(
                    state=state,
                    recent_states=recent_states,
                    theta_deg=theta_deg,
                    distances=distances,
                    execution=execution,
                )
                recent_states.append(state.tolist())
                last_feedback = reward_terms
                row = {
                    'step': step,
                    'time': time.time(),
                    'policy': 'deterministic_twf_baseline',
                    'gyro_z_bias': gyro_z_bias,
                    'distances_cm': distances,
                    'decision': decision,
                    'theta_deg': theta_deg,
                    'execution': execution,
                    'state': state.tolist(),
                    'reward': reward,
                    'reward_terms': reward_terms,
                }
                f.write(json.dumps(row, sort_keys=True) + '\n')
                f.flush()
    finally:
        safety.close()
        imu.close()
        rover.close()


if __name__ == '__main__':
    raise SystemExit(main())
