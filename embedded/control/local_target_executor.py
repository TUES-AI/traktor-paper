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
    until_front_stop_cm: float = 40.0
    until_front_emergency_cm: float = 28.0
    post_turn_front_settle_seconds: float = 0.75
    post_turn_front_sample_dt: float = 0.08
    open_front_bonus_cm: float = 90.0


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
        if abs(theta_deg) <= 45:
            front = distances['front']
            if front is not None and front < margin:
                return 0.0
            if front is not None:
                allowed = min(allowed, max(0.0, front - margin))
        if theta_deg > 25:
            left = distances['left']
            if left is not None and left < margin:
                allowed = 0.0
            elif left is not None:
                allowed = min(allowed, max(0.0, left - margin))
        if theta_deg < -25:
            right = distances['right']
            if right is not None and right < margin:
                allowed = 0.0
            elif right is not None:
                allowed = min(allowed, max(0.0, right - margin))
        return allowed

    def execute_local_target(self, x_cm, y_cm):
        theta = math.degrees(math.atan2(y_cm, x_cm))
        requested_distance = math.hypot(x_cm, y_cm)
        distances = self.safety.read_distances()
        reverse_recovery = self.safety.reverse_if_too_close(self.config.drive_pwm, distances)
        if reverse_recovery is not None:
            distances = self.safety.read_distances()
        distance = self.clip_distance(theta, requested_distance, distances)
        report = {
            'target_local': {'x_cm': x_cm, 'y_cm': y_cm},
            'theta_deg': theta,
            'requested_distance_cm': requested_distance,
            'clipped_distance_cm': distance,
            'start_distances': distances,
            'reverse_recovery': reverse_recovery,
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
        if abs(float(theta_deg)) <= cfg.turn_tolerance_deg:
            return {
                'ok': True,
                'reason': 'no_turn_needed',
                'yaw_deg': 0.0,
                'target_deg': float(theta_deg),
            }
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
        if distance_cm <= 0.0:
            return {'ok': False, 'reason': 'distance_clipped_to_zero', 'seconds': 0.0}
        distances = self.safety.read_distances()
        safe, front, threshold = self.safety.is_front_safe(cfg.drive_pwm, distances)
        if not safe:
            reason = f'front_safety_stop_before_drive front={front} threshold={threshold:.1f}'
            self.set_status(reason)
            return {'ok': False, 'reason': reason, 'seconds': 0.0}
        seconds = max(0.45, min(cfg.max_drive_seconds, distance_cm / cfg.cm_per_second))
        start = time.monotonic()
        self.set_status(f'driving_{distance_cm:.1f}cm')
        self.safety.rover.drive('forward', 'forward', left_speed=cfg.drive_pwm, right_speed=cfg.drive_pwm)
        contact_monitor = self.safety.begin_forward_contact_monitor()
        try:
            while time.monotonic() - start < seconds:
                distances = self.safety.read_distances()
                safe, front, threshold = self.safety.is_front_safe(cfg.drive_pwm, distances)
                if not safe:
                    reason = f'front_safety_stop front={front} threshold={threshold:.1f}'
                    self.set_status(reason)
                    return {'ok': False, 'reason': reason, 'seconds': time.monotonic() - start}
                contact = self.safety.update_forward_contact_monitor(contact_monitor)
                if contact and contact.get('contact_or_stall'):
                    elapsed = time.monotonic() - start
                    self.set_status('contact_or_stall reversing')
                    recovery = self.safety.forward_contact_recovery(contact)
                    return {
                        'ok': False,
                        'reason': 'contact_or_stall',
                        'seconds': elapsed,
                        'estimated_distance_cm': 0.0,
                        'contact_or_stall': True,
                        'stall_score': recovery.get('stall_score', 1.0),
                        'contact_recovery': recovery,
                    }
                time.sleep(cfg.dt)
            return {'ok': True, 'reason': 'duration_complete', 'seconds': time.monotonic() - start}
        finally:
            self.safety.rover.stop_motors()
            time.sleep(0.15)

    def execute_theta_until_front(self, theta_deg, front_stop_cm=None):
        """Turn to a relative heading, then drive until the front sensor reaches a threshold.

        This is for the scalar SAC action experiment. The heading is rover-relative,
        because the rover does not have reliable global pose/SLAM. Forward distance
        is estimated from drive duration and `cm_per_second`; the safety layer still
        gates close-front recovery before the motion starts.
        """
        front_stop_cm = self.config.until_front_stop_cm if front_stop_cm is None else float(front_stop_cm)
        distances = self.safety.read_distances()
        reverse_recovery = self.safety.reverse_if_too_close(self.config.drive_pwm, distances)
        if reverse_recovery is not None:
            distances = self.safety.read_distances()
        report = {
            'target_mode': 'theta_until_front',
            'theta_deg': float(theta_deg),
            'front_stop_cm': front_stop_cm,
            'requested_distance_cm': None,
            'clipped_distance_cm': 0.0,
            'start_distances': distances,
            'reverse_recovery': reverse_recovery,
            'turn': None,
            'drive': None,
            'reason': 'started',
        }
        turn = self.turn_to(theta_deg)
        report['turn'] = turn
        if not turn['ok']:
            report['reason'] = f'turn_failed_or_blocked {turn["reason"]}'
            self.set_status(report['reason'])
            return report
        front_check = self.wait_for_front_after_turn(front_stop_cm)
        report['post_turn_front_check'] = front_check
        report['front_space_after_turn_cm'] = front_check.get('front_cm')
        if not front_check['ok']:
            report['reason'] = front_check['reason']
            self.set_status(report['reason'])
            return report
        drive = self.drive_until_front(front_stop_cm)
        report['drive'] = drive
        report['clipped_distance_cm'] = float(drive.get('estimated_distance_cm') or 0.0)
        report['reason'] = 'complete' if drive['ok'] else drive['reason']
        self.set_status('idle' if drive['ok'] else drive['reason'])
        return report

    def wait_for_front_after_turn(self, front_stop_cm):
        cfg = self.config
        self.set_status(f'post_turn_front_settle_{front_stop_cm:.1f}cm')
        start = time.monotonic()
        readings = []
        while time.monotonic() - start < cfg.post_turn_front_settle_seconds:
            front = self.safety.read_front_distance()
            readings.append(front)
            if front is not None and front <= front_stop_cm:
                return {
                    'ok': False,
                    'reason': 'post_turn_front_blocked_before_drive',
                    'front_cm': front,
                    'threshold_cm': front_stop_cm,
                    'readings': readings,
                    'seconds': time.monotonic() - start,
                }
            time.sleep(cfg.post_turn_front_sample_dt)
        concrete = [x for x in readings if x is not None]
        front = min(concrete) if concrete else None
        return {
            'ok': True,
            'reason': 'post_turn_front_clear',
            'front_cm': front,
            'threshold_cm': front_stop_cm,
            'open_front_bonus': bool(front is not None and front >= cfg.open_front_bonus_cm),
            'readings': readings,
            'seconds': time.monotonic() - start,
        }

    def drive_until_front(self, front_stop_cm=None):
        cfg = self.config
        front_stop_cm = cfg.until_front_stop_cm if front_stop_cm is None else float(front_stop_cm)
        emergency_cm = min(float(front_stop_cm), float(cfg.until_front_emergency_cm))
        front = self.safety.read_front_distance()
        start_distances = {'front': front}
        if front is not None and front <= front_stop_cm:
            return {
                'ok': True,
                'reason': 'already_at_front_threshold',
                'seconds': 0.0,
                'estimated_distance_cm': 0.0,
                'start_distances': start_distances,
                'front_cm': front,
                'threshold_cm': front_stop_cm,
            }
        start = time.monotonic()
        self.set_status(f'driving_until_front_{front_stop_cm:.1f}cm')
        self.safety.rover.drive('forward', 'forward', left_speed=cfg.drive_pwm, right_speed=cfg.drive_pwm)
        contact_monitor = self.safety.begin_forward_contact_monitor()
        last_front = front
        try:
            while time.monotonic() - start < cfg.max_drive_seconds:
                front = self.safety.read_front_distance()
                if front is not None:
                    last_front = front
                    elapsed = time.monotonic() - start
                    est = min(cfg.max_drive_seconds, elapsed) * cfg.cm_per_second
                    if front <= front_stop_cm:
                        return {
                            'ok': True,
                            'reason': 'front_threshold_reached',
                            'seconds': elapsed,
                            'estimated_distance_cm': est,
                            'start_distances': start_distances,
                            'front_cm': front,
                            'threshold_cm': front_stop_cm,
                        }
                    if front <= emergency_cm:
                        return {
                            'ok': False,
                            'reason': 'front_emergency_stop',
                            'seconds': elapsed,
                            'estimated_distance_cm': est,
                            'start_distances': start_distances,
                            'front_cm': front,
                            'threshold_cm': emergency_cm,
                        }
                contact = self.safety.update_forward_contact_monitor(contact_monitor)
                if contact and contact.get('contact_or_stall'):
                    elapsed = time.monotonic() - start
                    self.set_status('contact_or_stall reversing')
                    recovery = self.safety.forward_contact_recovery(contact)
                    return {
                        'ok': False,
                        'reason': 'contact_or_stall',
                        'seconds': elapsed,
                        'estimated_distance_cm': 0.0,
                        'start_distances': start_distances,
                        'front_cm': last_front,
                        'threshold_cm': front_stop_cm,
                        'contact_or_stall': True,
                        'stall_score': recovery.get('stall_score', 1.0),
                        'contact_recovery': recovery,
                    }
                time.sleep(cfg.dt)
            elapsed = time.monotonic() - start
            return {
                'ok': True,
                'reason': 'max_drive_time_before_front_threshold',
                'seconds': elapsed,
                'estimated_distance_cm': elapsed * cfg.cm_per_second,
                'start_distances': start_distances,
                'front_cm': last_front,
                'threshold_cm': front_stop_cm,
            }
        finally:
            self.safety.rover.stop_motors()
            time.sleep(0.15)
