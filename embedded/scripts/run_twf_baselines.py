#!/usr/bin/env python3
"""Fair non-learned baselines through the SAC movement pipeline.

Both baselines output the same scalar action contract as SAC: a value in [-1, 1]
mapped to a relative heading. The executor remains responsible for rotation,
post-turn front gating, forward drive, and recovery.

Baselines:
- random: action ~ Uniform(-1, 1), no sensor-based policy logic.
- sensor_det: ultra-generic reactive policy using only current ultrasonic ranges;
  no IMU, map, memory, room knowledge, freer-side history, or environment tuning.
"""

import argparse
import json
import random
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


def norm_range_cm(value, max_cm=200.0):
    if value is None:
        return 1.0
    return float(np.clip(float(value) / max_cm, 0.0, 1.0))


def finite_or(value, fallback):
    return fallback if value is None else float(value)


def random_policy_action(_distances, rng):
    return rng.uniform(-1.0, 1.0), {'mode': 'uniform_random'}


def sensor_det_policy_action(distances, _rng):
    """Generic range-only reactive action in [-1, 1].

    This is deliberately plain: go forward when front is the most open direction;
    otherwise steer away from the closer side/front obstacle. It uses only the
    three current ultrasonic readings and has no environment-specific constants
    except range normalization.
    """
    left = finite_or(distances.get('left'), 200.0)
    right = finite_or(distances.get('right'), 200.0)
    front = finite_or(distances.get('front'), 200.0)

    side_sum = max(1e-6, left + right)
    side_bias = (left - right) / side_sum  # positive means more free space left
    front_clear = np.clip((front - 35.0) / 100.0, 0.0, 1.0)

    if front_clear > 0.65:
        action = 0.35 * side_bias
        mode = 'range_forward_bias'
    else:
        action = np.sign(side_bias) * (0.45 + 0.55 * (1.0 - front_clear))
        if action == 0.0:
            action = 1.0
        mode = 'range_avoid_front'

    return float(np.clip(action, -1.0, 1.0)), {
        'mode': mode,
        'side_bias': float(side_bias),
        'front_clear': float(front_clear),
    }


POLICIES = {
    'random': random_policy_action,
    'sensor_det': sensor_det_policy_action,
}


def state_vector(distances, action_norm, feedback):
    return np.asarray(
        [
            norm_range_cm(distances.get('left')),
            norm_range_cm(distances.get('right')),
            norm_range_cm(distances.get('front')),
            float(action_norm),
            float(feedback.get('executed_distance_cm', 0.0)) / 100.0,
            float(feedback.get('zero_progress', 0.0)),
            float(feedback.get('post_turn_blocked', 0.0)),
        ],
        dtype=np.float32,
    )


def score_step(*, state, recent_states, action_norm, distances, execution):
    drive = execution.get('drive') or {}
    post_turn = execution.get('post_turn_front_check') or {}
    executed_cm = float(execution.get('clipped_distance_cm', 0.0) or 0.0)
    novelty = sensory_novelty(state.tolist(), list(recent_states))
    revisit_score = max(0.0, 1.0 - novelty)
    post_turn_blocked = execution.get('reason') == 'post_turn_front_blocked_before_drive'
    zero_progress = executed_cm < 2.0
    near_obstacle = float(distances.get('front') or 200.0) < 25.0
    open_front_after_forward_action = bool(abs(action_norm) < 0.25 and post_turn.get('open_front_bonus'))
    loop_score = 1.0 if revisit_score > 0.82 and executed_cm > 5.0 else 0.0
    recovery = bool(execution.get('reverse_recovery') or drive.get('contact_recovery'))

    reward = world_feedback_reward(
        sensory_novelty_value=novelty,
        executed_distance_cm=executed_cm,
        safe_motion=executed_cm > 5.0 and not recovery,
        revisit_score=revisit_score,
        zero_progress=zero_progress,
        recovery=recovery,
        near_obstacle=float(near_obstacle),
        loop_score=loop_score,
    )
    if post_turn_blocked:
        reward -= 1.0
    if open_front_after_forward_action:
        reward += 1.0

    return float(reward), {
        'sensory_novelty': novelty,
        'revisit_score': revisit_score,
        'executed_distance_cm': executed_cm,
        'safe_motion': executed_cm > 5.0 and not recovery,
        'zero_progress': zero_progress,
        'near_obstacle': near_obstacle,
        'post_turn_blocked': post_turn_blocked,
        'open_front_after_forward_action': open_front_after_forward_action,
        'loop_score': loop_score,
        'recovery': recovery,
    }


def main():
    parser = argparse.ArgumentParser(description='Run fair TWF non-learned baselines.')
    parser.add_argument('--policy', choices=sorted(POLICIES), required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--log-path', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-theta-deg', type=float, default=90.0)
    parser.add_argument('--drive-pwm', type=float, default=90.0)
    parser.add_argument('--turn-pwm', type=float, default=65.0)
    parser.add_argument('--front-stop-cm', type=float, default=40.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policy_fn = POLICIES[args.policy]
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
                action_norm, decision = policy_fn(distances, rng)
                theta_deg = float(action_norm) * args.max_theta_deg
                execution = executor.execute_theta_until_front(theta_deg, front_stop_cm=args.front_stop_cm)
                state = state_vector(distances, action_norm, last_feedback)
                reward, reward_terms = score_step(
                    state=state,
                    recent_states=recent_states,
                    action_norm=action_norm,
                    distances=distances,
                    execution=execution,
                )
                recent_states.append(state.tolist())
                last_feedback = reward_terms
                row = {
                    'step': step,
                    'time': time.time(),
                    'policy': args.policy,
                    'seed': args.seed,
                    'gyro_z_bias': gyro_z_bias,
                    'distances_cm': distances,
                    'decision': decision,
                    'action_norm': action_norm,
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
