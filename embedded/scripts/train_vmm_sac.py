#!/usr/bin/env python3
"""Train SAC on the real rover using VMM (RND + clusters + temporal smoothing) as novelty reward.

Observation (12-dim, matches sim SAC-VMM):
    [left, right, front, sin_yaw, cos_yaw, novelty,
     novelty, novelty, novelty, yaw_rate_norm, vl_norm, vr_norm]

Reward:
    +VMM novelty * novelty_weight
    +execution quality (distance fraction)
    -collision / recovery penalty
    -small living cost

Logs every step to <log_dir>/training_<timestamp>.jsonl:
    step, reward, vmm_novelty, vmm_mem_dist, vmm_rnd_norm,
    vmm_clusters, vmm_bank_size, vmm_new_cluster, distances, action

Run:
    python embedded/scripts/train_vmm_sac.py --steps 200 --log-dir results/vmm_training
    python embedded/scripts/train_vmm_sac.py --steps 200 --resume results/vmm_sac.zip
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import _paths  # noqa: F401

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_sac_vmm_local_targets import action_to_target, clamp, norm_distance

ULTRA_MAX_CM    = 400.0
YAW_RATE_MAX    = math.radians(180.0)   # rad/s — normalisation ceiling for obs
GYRO_CLAMP      = math.radians(360.0)   # physical upper bound on raw gyro reading
DT_MAX          = 0.5                   # cap integration window to prevent huge steps
MAX_WHEEL_SPEED = 1.0


def _norm_cm(value):
    if value is None:
        return 1.0
    return clamp(float(value) / ULTRA_MAX_CM, 0.0, 1.0)


class SensorTracker:
    """Tracks ultrasonic novelty and safety pressure without using a map."""
    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def update(self, distances, front_stop_cm):
        current = np.array([
            _norm_cm(distances.get('left')),
            _norm_cm(distances.get('right')),
            _norm_cm(distances.get('front')),
        ], dtype=np.float32)
        if self.prev is None:
            delta = np.zeros_like(current)
        else:
            delta = np.abs(current - self.prev)
        self.prev = current

        front_cm = distances.get('front')
        obstacle_risk = 0.0
        front_clearance_norm = 1.0
        if front_cm is not None:
            front_clearance_norm = clamp(float(front_cm) / max(front_stop_cm, 1.0), 0.0, 1.0)
            obstacle_risk = clamp((front_stop_cm - float(front_cm)) / max(front_stop_cm, 1.0), 0.0, 1.0)

        sensor_novelty = clamp(float(delta.mean()) * 3.0, 0.0, 1.0)
        return {
            'sensor_novelty': sensor_novelty,
            'sensor_delta_mean': float(delta.mean()),
            'front_clearance_norm': front_clearance_norm,
            'obstacle_risk': obstacle_risk,
        }


class VMMRoverEnv(gym.Env):
    """Real-rover Gym environment driven by VMM novelty reward."""

    metadata = {'render_modes': []}

    def __init__(self, args):
        super().__init__()
        from api.rover_api import RoverAPI
        from control.local_target_executor import LocalTargetExecutor, LocalTargetExecutorConfig
        from control.safety import SafetyConfig, SafetyController
        from drivers.sensors.mpu9150 import MPU9150
        from VMM.vmm import VMM

        self.args = args
        self.action_space = spaces.Box(
            low =np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        self.rover   = RoverAPI(camera_enabled=True)
        self.imu     = MPU9150(bus=1, address=0x68)
        self.safety  = SafetyController(
            self.rover, imu=self.imu,
            config=SafetyConfig(
                min_front_stop_cm=args.front_stop_cm,
                max_front_stop_cm=args.front_stop_cm,
                front_clear_to_resume_cm=args.front_clear_cm,
                no_echo_is_clear=not args.no_echo_is_wall,
            ),
        )
        self.executor = LocalTargetExecutor(
            self.safety,
            config=LocalTargetExecutorConfig(turn_pwm=args.turn_pwm, drive_pwm=args.drive_pwm),
            status_callback=lambda s: print(json.dumps({'executor': s}, sort_keys=True), flush=True),
        )
        self.vmm = VMM()

        # State
        self.yaw_deg      = 0.0
        self.yaw_rate     = 0.0
        self.last_t       = time.monotonic()
        self.last_vl      = 0.0
        self.last_vr      = 0.0
        self.last_action  = np.zeros(2, dtype=np.float32)
        self.last_vmm     = {}
        self.last_distances = {}
        self.last_sensor  = {}
        self.sensor_tracker = SensorTracker()
        self.step_count   = 0
        self.episode_reward = 0.0

        # Logging
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_path = log_dir / f'training_{ts}.jsonl'
        self._log_fh   = self._log_path.open('w')
        self._score_path = log_dir / f'training_{ts}_scores.csv'
        self._score_fh = self._score_path.open('w', newline='')
        self._score_writer = csv.DictWriter(self._score_fh, fieldnames=[
            'step', 'reward', 'episode_reward', 'vmm_novelty', 'vmm_mem_dist',
            'vmm_rnd_norm', 'vmm_clusters', 'vmm_new_cluster', 'vmm_is_novel',
            'sensor_novelty', 'sensor_delta_mean', 'front_clearance_norm',
            'obstacle_risk', 'target_x_cm', 'target_y_cm', 'yaw_deg',
            'yaw_rate_dps', 'recovery',
        ])
        self._score_writer.writeheader()
        self._summary_path = log_dir / f'training_{ts}_summary.json'
        self._plot_path = log_dir / f'training_{ts}_scores.png'
        self._score_rows = []
        print(json.dumps({'log_path': str(self._log_path), 'score_path': str(self._score_path)}), flush=True)

        self.safety.calibrate_gyro()

    # ── IMU ────────────────────────────────────────────────────────────────────
    def _update_imu(self):
        now = time.monotonic()
        # Cap dt so a long recovery maneuver cannot produce one huge integration step
        dt  = clamp(now - self.last_t, 1e-3, DT_MAX)
        self.last_t = now
        reading = self.imu.read_all()
        gyro_z  = reading['gyro']['z'] - self.safety._gyro_z_bias
        # Clamp to physical gyro limits — spikes beyond ±GYRO_CLAMP are sensor noise
        gyro_z       = clamp(gyro_z, -GYRO_CLAMP, GYRO_CLAMP)
        self.yaw_rate = gyro_z
        self.yaw_deg += math.degrees(gyro_z * dt)
        # Wrap to [-180, 180) to prevent unbounded accumulation; sin/cos are unaffected
        self.yaw_deg = (self.yaw_deg + 180.0) % 360.0 - 180.0

    # ── Observation ────────────────────────────────────────────────────────────
    def _build_obs(self, distances, vmm_result):
        novelty  = float(vmm_result.get('novelty', 0.0)) if vmm_result else 0.0
        yaw_rad  = math.radians(self.yaw_deg)
        yaw_norm = clamp(self.yaw_rate / YAW_RATE_MAX, -1.0, 1.0)
        return np.array([
            norm_distance(distances.get('left')),
            norm_distance(distances.get('right')),
            norm_distance(distances.get('front')),
            math.sin(yaw_rad),
            math.cos(yaw_rad),
            clamp(novelty, 0.0, 1.0),
            clamp(novelty, 0.0, 1.0),   # FOV L (real FOV not yet implemented)
            clamp(novelty, 0.0, 1.0),   # FOV C
            clamp(novelty, 0.0, 1.0),   # FOV R
            yaw_norm,
            clamp(self.last_vl, -1.0, 1.0),
            clamp(self.last_vr, -1.0, 1.0),
        ], dtype=np.float32)

    # ── VMM ────────────────────────────────────────────────────────────────────
    def _observe_vmm(self):
        try:
            frame = self.rover.get_camera_frame()
            if frame is None:
                print(json.dumps({'warning': 'vmm_no_frame'}), flush=True)
                return {}
            return self.vmm.observe(frame, yaw_rate_rad_s=self.yaw_rate)
        except Exception as exc:
            print(json.dumps({'warning': 'vmm_failed', 'error': repr(exc)}), flush=True)
            return {}

    # ── Reward ─────────────────────────────────────────────────────────────────
    def _reward(self, vmm_result, sensor_result, execution, recovery):
        r = -0.02   # living cost

        novelty = float((vmm_result or {}).get('novelty', 0.0))
        r += self.args.novelty_weight * novelty
        r += self.args.sensor_novelty_weight * float(sensor_result.get('sensor_novelty', 0.0))
        r -= self.args.obstacle_risk_weight * float(sensor_result.get('obstacle_risk', 0.0))

        if recovery is not None:
            r -= self.args.recovery_penalty

        if execution is not None:
            clipped   = float(execution.get('clipped_distance_cm') or 0.0)
            requested = max(1.0, float(execution.get('requested_distance_cm') or 1.0))
            r += self.args.distance_weight * min(1.0, clipped / requested)
            drive = execution.get('drive') or {}
            if drive.get('ok'):
                r += self.args.success_bonus
            if 'front_safety_stop' in str(drive.get('reason', '')):
                r -= self.args.front_stop_penalty

        return float(r)

    # ── Gym interface ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.yaw_deg  = 0.0
        self.last_vl  = 0.0
        self.last_vr  = 0.0
        self.last_action    = np.zeros(2, dtype=np.float32)
        self.episode_reward = 0.0
        self.sensor_tracker.reset()
        self._update_imu()
        distances   = self.safety.read_distances()
        vmm_result  = self._observe_vmm()
        sensor_result = self.sensor_tracker.update(distances, self.args.front_stop_cm)
        self.last_distances = distances
        self.last_vmm       = vmm_result
        self.last_sensor    = sensor_result
        return self._build_obs(distances, vmm_result), {}

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # Snapshot IMU now so the next _update_imu call has a short dt
        self._update_imu()

        # Safety recovery before executing
        recovery = None
        safe, _, _ = self.safety.is_front_safe(self.args.drive_pwm, self.last_distances)
        if not safe:
            turn_dir = self.safety.freer_side(self.last_distances)
            recovery = self.safety.turn_until_clear(turn_dir, speed_pct=self.args.turn_pwm)
            # Track the rotation that happened during recovery
            self._update_imu()

        # Execute action
        target    = action_to_target(action, self.args.max_theta_deg,
                                     self.args.max_distance_cm, self.args.min_drive_cm)
        execution = None
        if recovery is None:
            execution = self.executor.execute_local_target(target['x_cm'], target['y_cm'])
            drive = (execution or {}).get('drive') or {}
            if drive.get('ok'):
                self.last_vl = clamp(action[0], -1.0, 1.0)   # fixed: was action[1]
                self.last_vr = clamp(action[1], -1.0, 1.0)

        # Sense
        self._update_imu()
        distances  = self.safety.read_distances()
        vmm_result = self._observe_vmm()
        sensor_result = self.sensor_tracker.update(distances, self.args.front_stop_cm)
        reward     = self._reward(vmm_result, sensor_result, execution, recovery)
        self.episode_reward += reward
        obs        = self._build_obs(distances, vmm_result)

        self.last_action    = action.copy()
        self.last_distances = distances
        self.last_vmm       = vmm_result
        self.last_sensor    = sensor_result

        # ── Log ──────────────────────────────────────────────────────────────
        log_entry = {
            'step':           self.step_count,
            'reward':         round(reward, 5),
            'episode_reward': round(self.episode_reward, 5),
            'vmm_novelty':    round(float((vmm_result or {}).get('novelty',    0.0)), 4),
            'vmm_mem_dist':   round(float((vmm_result or {}).get('mem_dist',   0.0)), 4),
            'vmm_rnd_norm':   round(float((vmm_result or {}).get('rnd_norm',   0.0)), 4),
            'vmm_clusters':   int(  (vmm_result or {}).get('bank_size', 0)),
            'vmm_new_cluster':int(  (vmm_result or {}).get('new_cluster', False)),
            'vmm_is_novel':   int(  (vmm_result or {}).get('is_novel',  False)),
            'vmm_mem_source':  str(  (vmm_result or {}).get('mem_source', 'none')),
            'vmm_raw_mem_dist': round(float((vmm_result or {}).get('raw_mem_dist', 0.0)), 4),
            'vmm_smooth_mem_dist': round(float((vmm_result or {}).get('smooth_mem_dist', 0.0)), 4),
            'vmm_smooth_reset': int((vmm_result or {}).get('smooth_reset', False)),
            'sensor_novelty': round(sensor_result['sensor_novelty'], 4),
            'sensor_delta_mean': round(sensor_result['sensor_delta_mean'], 4),
            'front_clearance_norm': round(sensor_result['front_clearance_norm'], 4),
            'obstacle_risk': round(sensor_result['obstacle_risk'], 4),
            'distances':      distances,
            'action':         [round(float(action[0]), 4), round(float(action[1]), 4)],
            'target_x_cm':    round(target['x_cm'], 2),
            'target_y_cm':    round(target['y_cm'], 2),
            'yaw_deg':        round(self.yaw_deg, 2),
            'yaw_rate_dps':   round(math.degrees(self.yaw_rate), 2),
            'recovery':       recovery is not None,
        }
        self._log_fh.write(json.dumps(log_entry) + '\n')
        self._log_fh.flush()
        score_row = {k: log_entry[k] for k in self._score_writer.fieldnames}
        self._score_writer.writerow(score_row)
        self._score_fh.flush()
        self._score_rows.append(score_row)
        print(json.dumps({k: log_entry[k] for k in
              ('step', 'reward', 'vmm_novelty', 'sensor_novelty', 'vmm_clusters', 'vmm_is_novel', 'recovery')},
              sort_keys=True), flush=True)

        time.sleep(self.args.sleep)
        return obs, reward, False, False, log_entry

    def close(self):
        if self._score_rows:
            rewards = np.array([float(r['reward']) for r in self._score_rows], dtype=np.float32)
            visual = np.array([float(r['vmm_novelty']) for r in self._score_rows], dtype=np.float32)
            sensors = np.array([float(r['sensor_novelty']) for r in self._score_rows], dtype=np.float32)
            risks = np.array([float(r['obstacle_risk']) for r in self._score_rows], dtype=np.float32)
            summary = {
                'steps': len(self._score_rows),
                'total_reward': round(float(rewards.sum()), 5),
                'mean_reward': round(float(rewards.mean()), 5),
                'mean_vmm_novelty': round(float(visual.mean()), 5),
                'mean_sensor_novelty': round(float(sensors.mean()), 5),
                'mean_obstacle_risk': round(float(risks.mean()), 5),
                'novel_events': int(sum(int(r['vmm_is_novel']) for r in self._score_rows)),
                'new_clusters': int(sum(int(r['vmm_new_cluster']) for r in self._score_rows)),
                'recoveries': int(sum(int(r['recovery']) for r in self._score_rows)),
                'final_clusters': int(self._score_rows[-1]['vmm_clusters']),
                'score_path': str(self._score_path),
                'plot_path': str(self._plot_path),
                'log_path': str(self._log_path),
            }
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                steps = np.array([int(r['step']) for r in self._score_rows], dtype=np.int32)
                fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                axes[0].plot(steps, rewards, label='reward')
                axes[0].plot(steps, np.cumsum(rewards), label='cumulative reward')
                axes[0].set_ylabel('reward')
                axes[0].legend(loc='best')
                axes[1].plot(steps, visual, label='MobileNet-VMM novelty')
                axes[1].plot(steps, sensors, label='sensor novelty')
                axes[1].set_ylabel('novelty')
                axes[1].set_ylim(-0.05, 1.05)
                axes[1].legend(loc='best')
                axes[2].plot(steps, risks, label='obstacle risk')
                axes[2].set_ylabel('risk')
                axes[2].set_xlabel('step')
                axes[2].set_ylim(-0.05, 1.05)
                axes[2].legend(loc='best')
                fig.tight_layout()
                fig.savefig(self._plot_path, dpi=140)
                plt.close(fig)
            except Exception as exc:
                summary['plot_error'] = repr(exc)
            self._summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
            print(json.dumps({'summary_path': str(self._summary_path), **summary}, sort_keys=True), flush=True)
        self._log_fh.close()
        self._score_fh.close()
        self.safety.close()
        self.imu.close()
        self.rover.close()


def parse_args():
    p = argparse.ArgumentParser(description='Train SAC with VMM novelty reward on the real rover.')
    p.add_argument('--steps',            type=int,   default=200)
    p.add_argument('--save-path',        default='results/vmm_sac.zip')
    p.add_argument('--resume',           default=None)
    p.add_argument('--log-dir',          default='results/vmm_training')
    p.add_argument('--sleep',            type=float, default=0.2)
    p.add_argument('--max-theta-deg',    type=float, default=75.0)
    p.add_argument('--max-distance-cm',  type=float, default=80.0)
    p.add_argument('--min-drive-cm',     type=float, default=10.0)
    p.add_argument('--turn-pwm',         type=float, default=65.0)
    p.add_argument('--drive-pwm',        type=float, default=75.0)
    p.add_argument('--front-stop-cm',    type=float, default=45.0)
    p.add_argument('--front-clear-cm',   type=float, default=55.0)
    p.add_argument('--no-echo-is-wall',  action='store_true')
    p.add_argument('--novelty-weight',   type=float, default=0.8)
    p.add_argument('--sensor-novelty-weight', type=float, default=0.15)
    p.add_argument('--obstacle-risk-weight', type=float, default=0.25)
    p.add_argument('--distance-weight',  type=float, default=0.3)
    p.add_argument('--success-bonus',    type=float, default=0.2)
    p.add_argument('--recovery-penalty', type=float, default=0.3)
    p.add_argument('--front-stop-penalty', type=float, default=0.2)
    p.add_argument('--learning-starts',  type=int,   default=20)
    p.add_argument('--batch-size',       type=int,   default=32)
    p.add_argument('--buffer-size',      type=int,   default=5000)
    p.add_argument('--checkpoint-every', type=int,   default=50)
    return p.parse_args()


def main():
    args = parse_args()
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    class CheckpointAndLogCallback(BaseCallback):
        def __init__(self, save_path, checkpoint_every):
            super().__init__()
            self.save_path        = save_path
            self.checkpoint_every = checkpoint_every

        def _on_step(self):
            if self.num_timesteps % self.checkpoint_every == 0:
                path = self.save_path.replace('.zip', f'_step{self.num_timesteps}.zip')
                self.model.save(path)
                print(json.dumps({'checkpoint': path, 'step': self.num_timesteps}), flush=True)
            return True

    env = VMMRoverEnv(args)
    try:
        if args.resume:
            model = SAC.load(args.resume, env=env)
            print(json.dumps({'resumed': args.resume}), flush=True)
        else:
            model = SAC(
                'MlpPolicy', env,
                learning_rate   = 3e-4,
                buffer_size     = args.buffer_size,
                learning_starts = args.learning_starts,
                batch_size      = args.batch_size,
                gamma           = 0.98,
                tau             = 0.02,
                train_freq      = 1,
                gradient_steps  = 1,
                policy_kwargs   = {'net_arch': [128, 128]},
                verbose         = 0,
                device          = 'cpu',
            )

        cb = CheckpointAndLogCallback(args.save_path, args.checkpoint_every)
        model.learn(total_timesteps=args.steps, callback=cb, log_interval=10, progress_bar=False)
        model.save(args.save_path)
        print(json.dumps({'saved': args.save_path, 'total_steps': args.steps}), flush=True)
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
