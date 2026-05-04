"""Deterministic continuous executor for local 2D rover targets."""

import math
import time
from dataclasses import dataclass


@dataclass
class LocalTargetExecutorConfig:
    turn_pwm: float = 65.0
    max_turn_pwm: float = 100.0
    drive_pwm: float = 90.0
    dt: float = 0.05
    turn_tolerance_deg: float = 5.0
    min_turn_progress_deg: float = 0.8
    turn_stall_seconds: float = 1.2
    max_turn_seconds: float = 12.0
    max_drive_seconds: float = 4.0
    cm_per_second: float = 40.0
    obstacle_margin_cm: float = 20.0


class LocalTargetExecutor:
    """Executes a clicked/local 2D target through SafetyController.

    Target frame is rover-local: `+x` forward, `+y` left, centimeters.
    The executor is deterministic and continuous: turn toward target, then drive
    forward for a clipped distance. SafetyController gates turns and forward
    motion.
    """

    def __init__(self, safety, config=None, status_callback=None):
        self.safety = safety
        self.config = config or LocalTargetExecutorConfig()
        self.status_callback = status_callback

    def set_status(self, status):
        if self.status_callback is not None:
            self.status_callback(status)

    def clip_distance(self, theta_deg, distance_cm, distances):
        allowed = distance_cm
        margin = self.config.obstacle_margin_cm
        if abs(theta_deg) <= 45 and distances['front'] is not None:
            allowed = min(allowed, max(20.0, distances['front'] - margin))
        if theta_deg > 25 and distances['left'] is not None:
            allowed = min(allowed, max(20.0, distances['left'] - margin))
        if theta_deg < -25 and distances['right'] is not None:
            allowed = min(allowed, max(20.0, distances['right'] - margin))
        return allowed

    def execute_local_target(self, x_cm, y_cm):
        theta = math.degrees(math.atan2(y_cm, x_cm))
        requested_distance = math.hypot(x_cm, y_cm)
        distances = self.safety.read_distances()
        distance = self.clip_distance(theta, requested_distance, distances)
        report = {
            'target_local': {'x_cm': x_cm, 'y_cm': y_cm},
            'theta_deg': theta,
            'requested_distance_cm': requested_distance,
            'clipped_distance_cm': distance,
            'start_distances': distances,
            'turn': None,
            'drive': None,
            'reason': 'started',
        }

        turn = self.turn_to(theta)
        report['turn'] = turn
        if not turn['ok']:
            report['reason'] = f'turn_failed_or_blocked {turn["reason"]}'
            self.set_status(report['reason'])
            return report

        drive = self.drive_for(distance)
        report['drive'] = drive
        report['reason'] = 'complete' if drive['ok'] else drive['reason']
        self.set_status('idle' if drive['ok'] else drive['reason'])
        return report

    def turn_to(self, theta_deg):
        cfg = self.config
        direction = 'left' if theta_deg >= 0 else 'right'
        left_cmd, right_cmd = ('backward', 'forward') if direction == 'left' else ('forward', 'backward')
        target = abs(theta_deg) if direction == 'left' else -abs(theta_deg)
        yaw = 0.0
        last = time.monotonic()
        start = last
        pwm = cfg.turn_pwm
        last_progress_time = start
        last_abs_yaw = 0.0

        self.set_status(f'turning_{direction}_{theta_deg:.1f} pwm={pwm:.0f}')
        self.safety.rover.drive(left_cmd, right_cmd, left_speed=pwm, right_speed=pwm)
        try:
            while time.monotonic() - start < cfg.max_turn_seconds:
                distances = self.safety.read_distances()
                turn_safe, side, reason = self.safety.is_turn_safe(direction, distances)
                if not turn_safe:
                    side_label = 'NO_ECHO' if side is None else f'{side:.1f}'
                    self.set_status(f'turn_safety_stop {direction}={side_label}')
                    return {'ok': False, 'reason': reason, 'yaw_deg': yaw, 'target_deg': target}

                now = time.monotonic()
                imu = self.safety.imu.read_all() if self.safety.imu is not None else {'gyro': {'z': 0.0}}
                gyro_z = imu['gyro']['z'] - self.safety._gyro_z_bias
                yaw += gyro_z * (now - last)
                abs_yaw = abs(yaw)
                if abs_yaw > last_abs_yaw + cfg.min_turn_progress_deg:
                    last_progress_time = now
                    last_abs_yaw = abs_yaw
                if now - last_progress_time > cfg.turn_stall_seconds:
                    if pwm < cfg.max_turn_pwm:
                        pwm = cfg.max_turn_pwm
                        last_progress_time = now
                        self.set_status(f'turning_{direction}_{theta_deg:.1f} pwm={pwm:.0f} full_boost')
                        self.safety.rover.drive(left_cmd, right_cmd, left_speed=pwm, right_speed=pwm)
                    elif abs_yaw < 3.0:
                        return {'ok': False, 'reason': f'stalled yaw={yaw:.1f}/{target:.1f}', 'yaw_deg': yaw, 'target_deg': target}
                last = now

                if direction == 'left' and yaw >= target - cfg.turn_tolerance_deg:
                    return {'ok': True, 'reason': 'target_reached', 'yaw_deg': yaw, 'target_deg': target}
                if direction == 'right' and yaw <= target + cfg.turn_tolerance_deg:
                    return {'ok': True, 'reason': 'target_reached', 'yaw_deg': yaw, 'target_deg': target}
                time.sleep(cfg.dt)
            return {'ok': False, 'reason': f'max_turn_time yaw={yaw:.1f}/{target:.1f}', 'yaw_deg': yaw, 'target_deg': target}
        finally:
            self.safety.rover.stop_motors()
            time.sleep(0.15)

    def drive_for(self, distance_cm):
        cfg = self.config
        seconds = max(0.45, min(cfg.max_drive_seconds, distance_cm / cfg.cm_per_second))
        start = time.monotonic()
        self.set_status(f'driving_{distance_cm:.1f}cm')
        self.safety.rover.drive('forward', 'forward', left_speed=cfg.drive_pwm, right_speed=cfg.drive_pwm)
        try:
            while time.monotonic() - start < seconds:
                distances = self.safety.read_distances()
                safe, front, threshold = self.safety.is_front_safe(cfg.drive_pwm, distances)
                if not safe:
                    reason = f'front_safety_stop front={front} threshold={threshold:.1f}'
                    self.set_status(reason)
                    return {'ok': False, 'reason': reason, 'seconds': time.monotonic() - start}
                time.sleep(cfg.dt)
            return {'ok': True, 'reason': 'duration_complete', 'seconds': time.monotonic() - start}
        finally:
            self.safety.rover.stop_motors()
            time.sleep(0.15)
