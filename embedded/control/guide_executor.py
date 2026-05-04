"""Execute SAC-style local guide vectors through the safety layer."""

import math
import time


class GuideExecutor:
    """Deterministic executor fallback for planner actions.

    The planner action is `[curvature, horizon, speed]`:
    - curvature: `-1.0` hard right, `0.0` straight, `+1.0` hard left
    - horizon: seconds to keep executing the guide
    - speed: motor duty cycle percent
    """

    def __init__(self, safety, tick_seconds=0.05, turn_speed=70.0):
        self.safety = safety
        self.tick_seconds = tick_seconds
        self.turn_speed = turn_speed

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, float(value)))

    def execute(self, curvature, horizon, speed):
        curvature = self._clamp(curvature, -1.0, 1.0)
        horizon = self._clamp(horizon, 0.0, 30.0)
        speed = self._clamp(speed, 0.0, 100.0)

        report = {
            'requested': {'curvature': curvature, 'horizon_s': horizon, 'speed_pct': speed},
            'ticks': 0,
            'recoveries': [],
            'start_distances': self.safety.read_distances(),
            'yaw_deg': 0.0,
            'accel_delta_g': 0.0,
            'stopped_reason': 'horizon_complete',
        }
        if horizon <= 0.0 or speed <= 0.0:
            report['stopped_reason'] = 'zero_horizon_or_speed'
            return report

        turn_direction = 'left' if curvature >= 0.0 else 'right'
        end_time = time.monotonic() + horizon
        last_time = time.monotonic()
        accel_norm_min = None
        accel_norm_max = None
        inner_acc = 0.0
        inner_ratio = max(0.0, 1.0 - abs(curvature))
        pivot_curvature = 0.65

        last_command = None
        try:
            while time.monotonic() < end_time:
                distances = self.safety.read_distances()
                front_safe, front, threshold = self.safety.is_front_safe(speed, distances)
                if not front_safe:
                    self.safety.rover.stop_motors()
                    last_command = None
                    if abs(curvature) < 0.05:
                        report['stopped_reason'] = 'front_blocked_straight_guide'
                        report['front_cm'] = front
                        report['threshold_cm'] = threshold
                        break
                    turn = self.safety.turn_until_clear(turn_direction, speed_pct=self.turn_speed)
                    report['recoveries'].append({'action': 'turn_until_clear', 'direction': turn_direction, 'result': turn})
                    if turn['reason'] != 'clear':
                        report['stopped_reason'] = 'turn_blocked'
                        break
                    continue

                left_direction, right_direction, inner_acc = self._arc_directions(
                    curvature,
                    inner_ratio,
                    inner_acc,
                    pivot_curvature,
                )
                command = (left_direction, right_direction, speed)
                if command != last_command:
                    self.safety.rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                    last_command = command
                time.sleep(self.tick_seconds)
                report['ticks'] += 1

                if self.safety.imu is not None:
                    now = time.monotonic()
                    d = self.safety.imu.read_all()
                    gyro_z = d['gyro']['z'] - self.safety._gyro_z_bias
                    report['yaw_deg'] += gyro_z * (now - last_time)
                    last_time = now
                    a = d['accel']
                    accel_norm = math.sqrt(a['x'] ** 2 + a['y'] ** 2 + a['z'] ** 2)
                    accel_norm_min = accel_norm if accel_norm_min is None else min(accel_norm_min, accel_norm)
                    accel_norm_max = accel_norm if accel_norm_max is None else max(accel_norm_max, accel_norm)
        finally:
            self.safety.rover.stop_motors()

        report['end_distances'] = self.safety.read_distances()
        if accel_norm_min is not None and accel_norm_max is not None:
            report['accel_delta_g'] = accel_norm_max - accel_norm_min
        report['executed_s'] = report['ticks'] * self.tick_seconds
        return report

    def _arc_directions(self, curvature, inner_ratio, inner_acc, pivot_curvature):
        if abs(curvature) < 0.05:
            return 'forward', 'forward', inner_acc

        if abs(curvature) >= pivot_curvature:
            if curvature > 0.0:
                return 'backward', 'forward', inner_acc
            return 'forward', 'backward', inner_acc

        inner_acc += inner_ratio
        inner_on = inner_acc >= 1.0
        if inner_on:
            inner_acc -= 1.0

        if curvature > 0.0:
            return 'forward' if inner_on else 'stop', 'forward', inner_acc
        return 'forward', 'forward' if inner_on else 'stop', inner_acc
