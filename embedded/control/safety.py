"""Safety-filtered motion layer between high-level policy and raw hardware."""

import math
import time
from dataclasses import dataclass


@dataclass
class SafetyConfig:
    min_front_stop_cm: float = 10.0
    max_front_stop_cm: float = 35.0
    side_turn_clear_cm: float = 20.0
    front_clear_to_resume_cm: float = 30.0
    wall_seen_cm: float = 45.0
    close_front_recovery_ratio: float = 0.8
    close_front_reverse_cm: float = 15.0
    reverse_recovery_cm_per_second: float = 20.0
    reverse_recovery_seconds: float = 0.75
    reverse_recovery_speed: float = 45.0
    recovery_turn_target_deg: float = 70.0
    stuck_check_seconds: float = 0.8
    stuck_min_yaw_change_deg: float = 2.0
    stuck_min_accel_delta_g: float = 0.025
    forward_contact_enabled: bool = True
    forward_contact_grace_seconds: float = 0.45
    forward_contact_window_seconds: float = 0.45
    forward_contact_min_accel_delta_g: float = 0.018
    forward_contact_min_abs_yaw_deg: float = 1.0
    forward_contact_baseline_alpha: float = 0.08
    forward_contact_min_baseline_score: float = 0.025
    forward_contact_stall_ratio: float = 0.45
    forward_contact_reverse_cm: float = 25.0
    ultrasonic_timeout_seconds: float = 0.03
    no_echo_is_clear: bool = True


class SafetyController:
    def __init__(self, rover, imu=None, config=None):
        self.rover = rover
        self.imu = imu
        self.config = config or SafetyConfig()
        self._gyro_z_bias = 0.0
        self._forward_motion_baseline = None

    def close(self):
        self.rover.stop_motors()

    def calibrate_gyro(self, seconds=1.2, dt=0.04):
        if self.imu is None:
            self._gyro_z_bias = 0.0
            return 0.0
        samples = []
        end = time.monotonic() + seconds
        self.rover.stop_motors()
        while time.monotonic() < end:
            samples.append(self.imu.read_all()['gyro']['z'])
            time.sleep(dt)
        self._gyro_z_bias = sum(samples) / max(1, len(samples))
        return self._gyro_z_bias

    def read_distances(self):
        data = self.rover.get_ultrasonic(timeout_seconds=self.config.ultrasonic_timeout_seconds)
        return {'left': data.get(2), 'right': data.get(1), 'front': data.get(3)}

    def read_front_distance(self):
        return self.rover.get_ultrasonic(sensor_id='front', timeout_seconds=self.config.ultrasonic_timeout_seconds)

    def front_stop_cm(self, speed_pct):
        speed_ratio = max(0.0, min(1.0, float(speed_pct) / 100.0))
        cfg = self.config
        return cfg.min_front_stop_cm + speed_ratio * (cfg.max_front_stop_cm - cfg.min_front_stop_cm)

    def is_front_safe(self, speed_pct, distances=None):
        distances = distances or self.read_distances()
        front = distances['front']
        threshold = self.front_stop_cm(speed_pct)
        if front is None:
            return self.config.no_echo_is_clear, front, threshold
        return front >= threshold, front, threshold

    def is_turn_safe(self, direction, distances=None):
        distances = distances or self.read_distances()
        side_name = 'left' if direction == 'left' else 'right'
        side_cm = distances[side_name]
        # PURE IN-PLACE TANK ROTATION IS ALLOWED REGARDLESS OF SIDE RANGE.
        # KEEP THE SIDE-BLOCKING LOGIC BELOW ONLY IF WE REINTRODUCE ARC TURNS
        # OR DOUBLE-PWM FORWARD+TURN MOVEMENTS WHERE THE ROVER TRANSLATES INTO
        # THE SIDE SENSOR DIRECTION.
        return True, side_cm, f'{side_name}_ignored_for_in_place_turn'

        # if side_cm is None:
        #     return self.config.no_echo_is_clear, side_cm, f'{side_name}_no_echo'
        # if side_cm < self.config.side_turn_clear_cm:
        #     return False, side_cm, f'{side_name}_blocked'
        # return True, side_cm, f'{side_name}_clear'

    def freer_side(self, distances=None):
        distances = distances or self.read_distances()
        left = distances['left'] if distances['left'] is not None else 999.0
        right = distances['right'] if distances['right'] is not None else 999.0
        return 'left' if left >= right else 'right'

    def drive_forward_tick(self, speed_pct, tick_seconds=0.12):
        safe, front, threshold = self.is_front_safe(speed_pct)
        if not safe:
            self.rover.stop_motors()
            return {'executed': False, 'reason': 'front_blocked', 'front_cm': front, 'threshold_cm': threshold}
        self.rover.drive('forward', 'forward', left_speed=speed_pct, right_speed=speed_pct)
        time.sleep(tick_seconds)
        self.rover.stop_motors()
        return {'executed': True, 'reason': 'forward', 'front_cm': front, 'threshold_cm': threshold}

    def reverse_recovery(self, distance_cm=None):
        distance_cm = self.config.close_front_reverse_cm if distance_cm is None else float(distance_cm)
        seconds = max(0.0, distance_cm / max(1e-6, self.config.reverse_recovery_cm_per_second))
        self.rover.drive('backward', 'backward', left_speed=self.config.reverse_recovery_speed, right_speed=self.config.reverse_recovery_speed)
        try:
            time.sleep(seconds)
        finally:
            self.rover.stop_motors()
        return {
            'reason': 'reverse_recovery',
            'requested_distance_cm': distance_cm,
            'seconds': seconds,
            'speed_pct': self.config.reverse_recovery_speed,
        }

    def begin_forward_contact_monitor(self):
        """Create state for detecting forward contact/stall during a drive.

        This stays in the safety layer. It is intentionally not a SAC input or
        auxiliary loss. The detector is conservative and uses IMU motion/vibration
        while a forward motor command is active; if it fires, the caller should
        stop forward motion and run reverse recovery before asking the planner for
        the next action.
        """
        return {
            'enabled': bool(self.config.forward_contact_enabled and self.imu is not None),
            'start': time.monotonic(),
            'last': time.monotonic(),
            'window_start': None,
            'yaw_deg': 0.0,
            'accel_min': None,
            'accel_max': None,
            'samples': 0,
        }

    def update_forward_contact_monitor(self, monitor):
        if not monitor or not monitor.get('enabled'):
            return None
        now = time.monotonic()
        elapsed = now - monitor['start']
        if elapsed < self.config.forward_contact_grace_seconds:
            monitor['last'] = now
            return None

        if monitor.get('window_start') is None:
            monitor['window_start'] = now
            monitor['yaw_deg'] = 0.0
            monitor['accel_min'] = None
            monitor['accel_max'] = None
            monitor['samples'] = 0

        d = self.imu.read_all()
        dt = max(1e-3, now - monitor['last'])
        monitor['last'] = now
        gyro_z = d['gyro']['z'] - self._gyro_z_bias
        monitor['yaw_deg'] += gyro_z * dt
        a = d['accel']
        accel_norm = math.sqrt(a['x'] ** 2 + a['y'] ** 2 + a['z'] ** 2)
        monitor['accel_min'] = accel_norm if monitor['accel_min'] is None else min(monitor['accel_min'], accel_norm)
        monitor['accel_max'] = accel_norm if monitor['accel_max'] is None else max(monitor['accel_max'], accel_norm)
        monitor['samples'] += 1

        window_elapsed = now - monitor['window_start']
        if window_elapsed < self.config.forward_contact_window_seconds:
            return None

        accel_delta = (monitor['accel_max'] or 0.0) - (monitor['accel_min'] or 0.0)
        abs_yaw = abs(float(monitor['yaw_deg']))
        motion_score = accel_delta + 0.002 * abs_yaw
        baseline = self._forward_motion_baseline
        absolute_stall = accel_delta < self.config.forward_contact_min_accel_delta_g and abs_yaw < self.config.forward_contact_min_abs_yaw_deg
        adaptive_stall = False
        if baseline is not None and baseline >= self.config.forward_contact_min_baseline_score:
            adaptive_stall = motion_score < baseline * self.config.forward_contact_stall_ratio
        stall = monitor['samples'] >= 3 and (absolute_stall or adaptive_stall)
        report = {
            'contact_or_stall': bool(stall),
            'stall_score': 1.0 if stall else 0.0,
            'motion_score': motion_score,
            'free_motion_baseline': baseline,
            'adaptive_stall_ratio': None if baseline is None else motion_score / max(1e-9, baseline),
            'accel_delta_g': accel_delta,
            'yaw_deg': monitor['yaw_deg'],
            'samples': monitor['samples'],
            'window_seconds': window_elapsed,
            'grace_seconds': self.config.forward_contact_grace_seconds,
            'reason': 'low_imu_motion_while_forward' if stall else 'imu_motion_ok',
        }
        if stall:
            return report

        # Clean forward windows update the free-motion baseline. This is a
        # safety-layer estimate only: it is not exposed to SAC. A later wall push
        # is considered suspicious when its motion score is far below this moving
        # baseline, even if the rover still jitters a little.
        if motion_score >= self.config.forward_contact_min_baseline_score:
            if self._forward_motion_baseline is None:
                self._forward_motion_baseline = motion_score
            else:
                a = self.config.forward_contact_baseline_alpha
                self._forward_motion_baseline = (1.0 - a) * self._forward_motion_baseline + a * motion_score

        monitor['window_start'] = now
        monitor['yaw_deg'] = 0.0
        monitor['accel_min'] = None
        monitor['accel_max'] = None
        monitor['samples'] = 0
        return None

    def forward_contact_recovery(self, monitor_report=None, distance_cm=None):
        self.rover.stop_motors()
        report = dict(monitor_report or {})
        reverse = self.reverse_recovery(
            self.config.forward_contact_reverse_cm if distance_cm is None else distance_cm
        )
        return {
            **report,
            'reason': 'contact_or_stall_reverse_recovery',
            'reverse': reverse,
        }

    def reverse_if_too_close(self, speed_pct, distances=None):
        distances = distances or self.read_distances()
        front = distances['front']
        threshold = self.front_stop_cm(speed_pct)
        trigger = threshold * self.config.close_front_recovery_ratio
        if front is None or front >= trigger:
            return None
        report = self.reverse_recovery(self.config.close_front_reverse_cm)
        return {
            **report,
            'front_cm': front,
            'threshold_cm': threshold,
            'trigger_cm': trigger,
        }

    def spin_tick(self, direction, speed_pct, tick_seconds=0.08):
        if direction == 'left':
            self.rover.drive('backward', 'forward', left_speed=speed_pct, right_speed=speed_pct)
        else:
            self.rover.drive('forward', 'backward', left_speed=speed_pct, right_speed=speed_pct)
        time.sleep(tick_seconds)
        self.rover.stop_motors()

    def turn_until_clear(self, direction, speed_pct=65.0, max_seconds=8.0, dt=0.08):
        start = time.monotonic()
        yaw = 0.0
        last = start
        left_cmd, right_cmd = ('backward', 'forward') if direction == 'left' else ('forward', 'backward')
        self.rover.drive(left_cmd, right_cmd, left_speed=speed_pct, right_speed=speed_pct)
        try:
            while time.monotonic() - start < max_seconds:
                distances = self.read_distances()
                front_clear = (
                    distances['front'] is not None
                    and distances['front'] >= self.config.front_clear_to_resume_cm
                )
                if front_clear:
                    return {'reason': 'clear', 'yaw_deg': yaw, **distances}
                turn_safe, _, turn_reason = self.is_turn_safe(direction, distances)
                if not turn_safe:
                    return {'reason': turn_reason, 'yaw_deg': yaw, **distances}
                if self.imu is not None:
                    now = time.monotonic()
                    gyro_z = self.imu.read_all()['gyro']['z'] - self._gyro_z_bias
                    yaw += gyro_z * (now - last)
                    last = now
                time.sleep(dt)
            return {'reason': 'timeout', 'yaw_deg': yaw, **self.read_distances()}
        finally:
            self.rover.stop_motors()

    def detect_stuck_during_forward(self, speed_pct, seconds=None, dt=0.08):
        if self.imu is None:
            return False, {'reason': 'no_imu'}
        seconds = seconds or self.config.stuck_check_seconds
        yaw = 0.0
        accel_norm_min = None
        accel_norm_max = None
        last = time.monotonic()
        start = last
        self.rover.drive('forward', 'forward', left_speed=speed_pct, right_speed=speed_pct)
        try:
            while time.monotonic() - start < seconds:
                safe, front, threshold = self.is_front_safe(speed_pct)
                if not safe:
                    return False, {'reason': 'front_blocked', 'front_cm': front, 'threshold_cm': threshold}
                now = time.monotonic()
                d = self.imu.read_all()
                gyro_z = d['gyro']['z'] - self._gyro_z_bias
                yaw += gyro_z * (now - last)
                last = now
                a = d['accel']
                accel_norm = math.sqrt(a['x'] ** 2 + a['y'] ** 2 + a['z'] ** 2)
                accel_norm_min = accel_norm if accel_norm_min is None else min(accel_norm_min, accel_norm)
                accel_norm_max = accel_norm if accel_norm_max is None else max(accel_norm_max, accel_norm)
                time.sleep(dt)
        finally:
            self.rover.stop_motors()
        accel_delta = (accel_norm_max or 0.0) - (accel_norm_min or 0.0)
        stuck = abs(yaw) < self.config.stuck_min_yaw_change_deg and accel_delta < self.config.stuck_min_accel_delta_g
        return stuck, {'yaw_deg': yaw, 'accel_delta_g': accel_delta}
