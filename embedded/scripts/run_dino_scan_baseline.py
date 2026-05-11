#!/usr/bin/env python3
"""Deterministic active DINOv3 scan baseline.

No SAC. At each step the rover captures candidate views at [-angle, 0, +angle]
without updating memory, chooses the highest visual novelty view, turns there,
drives until front threshold, then updates visual memory with the post-action
camera frame.
"""

import argparse
import json
from pathlib import Path
import time

import _paths  # noqa: F401
from api.rover_api import RoverAPI
from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
from control.safety import SafetyConfig, SafetyController
from drivers.sensors.mpu9150 import MPU9150
from dino_scan_common import DINOScanner, make_visual_cluster_sheet, read_distances, save_frame, summarize_log


def capture_scan(rover, executor, scanner, angle_deg, frame_dir, step):
    candidates = []
    sequence = [
        ('center', 0.0, 0.0),
        ('left', +angle_deg, +angle_deg),
        ('right', -angle_deg, -2.0 * angle_deg),
    ]
    for name, candidate_theta, turn_delta in sequence:
        if abs(turn_delta) > 1e-6:
            turn = executor.turn_to(turn_delta)
        else:
            turn = {'ok': True, 'reason': 'already_center', 'yaw_deg': 0.0, 'target_deg': 0.0}
        time.sleep(0.15)
        frame = rover.get_camera_frame()
        frame_path = frame_dir / f'step_{step:04d}_scan_{name}.jpg'
        save_frame(frame, frame_path)
        q = scanner.query_frame(frame)
        candidates.append({
            'name': name,
            'theta_deg': candidate_theta,
            'turn_to_scan': turn,
            'frame_path': str(frame_path),
            'novelty': q['novelty'],
            'dist': q['dist'],
            'cluster_id': q['cluster_id'],
            'bank_size': q['bank_size'],
        })
    recenter = executor.turn_to(+angle_deg)
    return candidates, recenter


def parse_args():
    p = argparse.ArgumentParser(description='Run deterministic DINOv3 scan baseline on real rover.')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--angle-deg', type=float, default=75.0)
    p.add_argument('--save-prefix', default='results/dino_scan_baseline_50')
    p.add_argument('--front-stop-cm', type=float, default=40.0)
    p.add_argument('--front-clear-cm', type=float, default=45.0)
    p.add_argument('--turn-pwm', type=float, default=60.0)
    p.add_argument('--drive-pwm', type=float, default=65.0)
    p.add_argument('--until-front-max-seconds', type=float, default=6.0)
    p.add_argument('--cm-per-second', type=float, default=40.0)
    p.add_argument('--input-size', type=int, default=336)
    p.add_argument('--known-dist', type=float, default=1.00)
    p.add_argument('--norm-dist', type=float, default=2.20)
    p.add_argument('--memory-warmup', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    prefix = Path(args.save_prefix)
    log_path = prefix.with_suffix('.jsonl')
    out_path = prefix.with_suffix('.out.json')
    frame_dir = Path(str(prefix) + '_frames')
    frame_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rover = RoverAPI(camera_enabled=True)
    imu = MPU9150(bus=1, address=0x68)
    safety = SafetyController(
        rover,
        imu=imu,
        config=SafetyConfig(
            min_front_stop_cm=args.front_stop_cm,
            max_front_stop_cm=args.front_stop_cm,
            front_clear_to_resume_cm=args.front_clear_cm,
        ),
    )
    executor = LocalTargetExecutor(
        safety,
        config=LocalTargetExecutorConfig(
            turn_pwm=args.turn_pwm,
            drive_pwm=args.drive_pwm,
            until_front_stop_cm=args.front_stop_cm,
            max_drive_seconds=args.until_front_max_seconds,
            cm_per_second=args.cm_per_second,
        ),
        status_callback=lambda s: print(json.dumps({'status': s}, sort_keys=True), flush=True),
    )
    scanner = DINOScanner(input_size=args.input_size, known_dist=args.known_dist, norm_dist=args.norm_dist, warmup=args.memory_warmup)
    safety.calibrate_gyro()

    try:
        with log_path.open('a', buffering=1) as f:
            for step in range(1, args.steps + 1):
                before_distances = read_distances(safety)
                candidates, recenter = capture_scan(rover, executor, scanner, args.angle_deg, frame_dir, step)
                chosen = max(candidates, key=lambda c: c['novelty'])
                turn_to_choice = executor.turn_to(float(chosen['theta_deg']))
                execution = None
                if turn_to_choice.get('ok'):
                    execution = {'turn': turn_to_choice, 'drive': executor.drive_until_front(args.front_stop_cm)}
                else:
                    execution = {'turn': turn_to_choice, 'drive': None}
                time.sleep(0.2)
                post_frame = rover.get_camera_frame()
                post_path = frame_dir / f'step_{step:04d}_post.jpg'
                save_frame(post_frame, post_path)
                post = scanner.update_frame(post_frame, step)
                drive = execution.get('drive') or {}
                moved_cm = float(drive.get('estimated_distance_cm') or 0.0)
                reward = float(post['novelty'] + (1.0 if post['new_cluster'] else 0.0) + min(0.3, moved_cm / 200.0))
                if drive.get('reason') == 'contact_or_stall':
                    reward -= 1.0
                if moved_cm <= 1e-3:
                    reward -= 0.3
                record = {
                    'step': step,
                    'time': time.time(),
                    'before_distances': before_distances,
                    'candidates': candidates,
                    'recenter': recenter,
                    'chosen': chosen,
                    'turn_to_choice': turn_to_choice,
                    'execution': execution,
                    'post_frame_path': str(post_path),
                    'post_update': {k: v for k, v in post.items() if k != 'embedding'},
                    'reward': reward,
                }
                f.write(json.dumps(record, sort_keys=True) + '\n')
                print(json.dumps({
                    'step': step,
                    'chosen': chosen['name'],
                    'chosen_theta': chosen['theta_deg'],
                    'candidate_novelty': {c['name']: round(float(c['novelty']), 3) for c in candidates},
                    'post_cluster': post['cluster_id'],
                    'visual_bank': post['bank_size'],
                    'moved_cm': round(moved_cm, 1),
                    'reward': round(reward, 3),
                }, sort_keys=True), flush=True)
    finally:
        rover.stop_motors()
        safety.close()
        imu.close()
        rover.close()

    sheet = make_visual_cluster_sheet(log_path, frame_dir, Path(str(prefix) + '_visual_clusters.png'))
    summary = summarize_log(log_path)
    summary['cluster_sheet'] = sheet
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
