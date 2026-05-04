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
    reverse_recovery_seconds: float = 0.75
    reverse_recovery_speed: float = 45.0
    recovery_turn_target_deg: float = 70.0
    stuck_check_seconds: float = 0.8
    stuck_min_yaw_change_deg: float = 2.0
    stuck_min_accel_delta_g: float = 0.025
    ultrasonic_timeout_seconds: float = 0.03
    no_echo_is_clear: bool = True


class SafetyController:
    def __init__(self, rover, imu=None, config=None):
        self.rover = rover
        self.imu = imu
        self.config = config or SafetyConfig()
        self._gyro_z_bias = 0.0

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
        if side_cm is None:
            return self.config.no_echo_is_clear, side_cm, f'{side_name}_no_echo'
        if side_cm < self.config.side_turn_clear_cm:
            return False, side_cm, f'{side_name}_blocked'
        return True, side_cm, f'{side_name}_clear'

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

    def reverse_recovery(self):
        self.rover.drive('backward', 'backward', left_speed=self.config.reverse_recovery_speed, right_speed=self.config.reverse_recovery_speed)
        time.sleep(self.config.reverse_recovery_seconds)
        self.rover.stop_motors()

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
                front_clear = distances['front'] is not None and distances['front'] >= self.config.front_clear_to_resume_cm
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
