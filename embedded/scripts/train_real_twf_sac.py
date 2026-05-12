#!/usr/bin/env python3
"""Train Tiny World Feedback SAC on the real rover.

Clean replacement for the old VMM/PCVM training scripts. Runtime observations are
range/IMU/action/executor feedback only. No camera frames are read and no visual
features are passed to the GRU or SAC policy.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.local_target_executor import LocalTargetExecutor
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150
from twf.reward import sensory_novelty, world_feedback_reward


def norm_range_cm(value, max_cm=200.0):
    if value is None:
        return 1.0
    return float(np.clip(float(value) / max_cm, 0.0, 1.0))


def build_state(distances, last_action, last_feedback):
    ranges = [
        norm_range_cm(distances.get('left')),
        norm_range_cm(distances.get('right')),
        norm_range_cm(distances.get('front')),
    ]
    motion = [
        float(last_feedback.get('executed_distance_cm', 0.0)) / 100.0,
        float(last_feedback.get('yaw_delta_deg', 0.0)) / 90.0,
        float(last_feedback.get('drive_seconds', 0.0)) / 3.0,
        float(last_feedback.get('turn_seconds', 0.0)) / 3.0,
    ]
    executor = [
        float(last_feedback.get('safe_motion', 0.0)),
        float(last_feedback.get('zero_progress', 0.0)),
        float(last_feedback.get('recovery', 0.0)),
        float(last_feedback.get('near_obstacle', 0.0)),
    ]
    return np.asarray(ranges + motion + [float(last_action)] + executor, dtype=np.float32)


def heuristic_action(step, state):
    """Temporary deterministic action source until SAC wiring is rebuilt."""
    front = state[2]
    if front < 0.28:
        return -0.85 if step % 2 else 0.85
    return float(np.sin(step * 0.73) * 0.55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--log-path', default='results/twf_real_run.jsonl')
    parser.add_argument('--max-turn-deg', type=float, default=90.0)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recent = deque(maxlen=32)
    last_action = 0.0
    last_feedback = {}

    rover = RoverAPI()
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(rover, imu=imu, config=SafetyConfig())
    executor = LocalTargetExecutor(safety)
    try:
        safety.calibrate_gyro()
        with log_path.open('w') as f:
            for step in range(1, args.steps + 1):
                distances = safety.read_distances()
                state = build_state(distances, last_action, last_feedback)
                action = heuristic_action(step, state)
                result = executor.execute_theta_until_front(float(action) * args.max_turn_deg)
                drive = result.get('drive') or {}
                turn = result.get('turn') or {}
                post_turn = result.get('post_turn_front_check') or {}
                executed_cm = float(result.get('clipped_distance_cm', 0.0) or 0.0)
                novelty = sensory_novelty(state.tolist(), recent)
                recent.append(state.tolist())
                feedback = {
                    'executed_distance_cm': executed_cm,
                    'yaw_delta_deg': float(turn.get('yaw_delta_deg', 0.0) or turn.get('yaw_deg', 0.0) or 0.0),
                    'drive_seconds': float(drive.get('seconds', 0.0) or 0.0),
                    'turn_seconds': float(turn.get('seconds', 0.0) or 0.0),
                    'safe_motion': executed_cm > 5.0 and not result.get('reverse_recovery'),
                    'zero_progress': executed_cm < 2.0,
                    'recovery': bool(result.get('reverse_recovery')),
                    'near_obstacle': float(distances.get('front') or 200.0) < 25.0,
                    'post_turn_blocked': result.get('reason') == 'post_turn_front_blocked_before_drive',
                    'open_front_after_forward_action': bool(
                        abs(action) < 0.25 and post_turn.get('open_front_bonus')
                    ),
                }
                reward = world_feedback_reward(
                    sensory_novelty_value=novelty,
                    executed_distance_cm=executed_cm,
                    safe_motion=feedback['safe_motion'],
                    revisit_score=max(0.0, 1.0 - novelty),
                    zero_progress=feedback['zero_progress'],
                    recovery=feedback['recovery'],
                    near_obstacle=float(feedback['near_obstacle']),
                    loop_score=0.0,
                )
                if feedback['post_turn_blocked']:
                    reward -= 1.0
                if feedback['open_front_after_forward_action']:
                    reward += 1.0
                row = {
                    'step': step,
                    'time': time.time(),
                    'state': state.tolist(),
                    'action': action,
                    'distances_cm': distances,
                    'executor': result,
                    'feedback': feedback,
                    'reward': reward,
                    'reward_terms': {'sensory_novelty': novelty, **feedback},
                }
                f.write(json.dumps(row, sort_keys=True) + '\n')
                f.flush()
                last_action = action
                last_feedback = feedback
    finally:
        rover.stop_motors()
        imu.close()
        rover.close()


if __name__ == '__main__':
    raise SystemExit(main())
