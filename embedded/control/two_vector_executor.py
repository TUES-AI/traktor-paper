"""Deterministic executor for two relative local vectors with data logging."""

import csv
import json
import math
import os
import time
from dataclasses import dataclass


@dataclass
class TwoVectorGuide:
    theta1_deg: float
    distance1_cm: float
    theta2_deg: float
    distance2_cm: float

    def points(self):
        h1 = math.radians(self.theta1_deg)
        x1 = math.cos(h1) * self.distance1_cm
        y1 = math.sin(h1) * self.distance1_cm
        h2 = h1 + math.radians(self.theta2_deg)
        x2 = x1 + math.cos(h2) * self.distance2_cm
        y2 = y1 + math.sin(h2) * self.distance2_cm
        return (x1, y1), (x2, y2)


class TwoVectorExecutor:
    """Executes `[theta1, d1, theta2, d2]` through SafetyController.

    This is intentionally deterministic. It chooses speed, turn speed, and timing
    locally, while the guide stays pure geometry. Logs are meant as training data
    for a future learned executor.
    """

    FIELDNAMES = [
        'run_id', 'sample_i', 't_monotonic', 'phase', 'phase_elapsed_s',
        'theta1_deg', 'distance1_cm', 'theta2_deg', 'distance2_cm',
        'target_heading_deg', 'yaw_deg', 'heading_error_deg',
        'left_cmd', 'right_cmd', 'left_speed_pct', 'right_speed_pct',
        'front_cm', 'left_cm', 'right_cm', 'front_threshold_cm', 'safety_state',
        'gyro_x_dps', 'gyro_y_dps', 'gyro_z_dps',
        'accel_x_g', 'accel_y_g', 'accel_z_g', 'accel_norm_g',
    ]

    def __init__(self, safety, guide, log_path, speed_pct=85.0, turn_speed_pct=80.0, cm_per_second=40.0, dt=0.05):
        self.safety = safety
        self.guide = guide
        self.log_path = log_path
        self.speed_pct = float(speed_pct)
        self.turn_speed_pct = float(turn_speed_pct)
        self.cm_per_second = float(cm_per_second)
        self.dt = float(dt)
        self.run_id = time.strftime('%Y%m%d_%H%M%S')
        self.sample_i = 0
        self.yaw_deg = 0.0
        self.last_imu_time = time.monotonic()
        self.accel_norm_min = None
        self.accel_norm_max = None
        self._log_file = None
        self._writer = None

    @staticmethod
    def normalize_angle(deg):
        while deg > 180.0:
            deg -= 360.0
        while deg < -180.0:
            deg += 360.0
        return deg

    def open_log(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_file = open(self.log_path, 'w', newline='')
        self._writer = csv.DictWriter(self._log_file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def close_log(self):
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def read_imu(self):
        if self.safety.imu is None:
            return None, None
        now = time.monotonic()
        reading = self.safety.imu.read_all()
        gyro = reading['gyro']
        accel = reading['accel']
        gyro_z = gyro['z'] - self.safety._gyro_z_bias
        self.yaw_deg += gyro_z * (now - self.last_imu_time)
        self.last_imu_time = now
        accel_norm = math.sqrt(accel['x'] ** 2 + accel['y'] ** 2 + accel['z'] ** 2)
        self.accel_norm_min = accel_norm if self.accel_norm_min is None else min(self.accel_norm_min, accel_norm)
        self.accel_norm_max = accel_norm if self.accel_norm_max is None else max(self.accel_norm_max, accel_norm)
        return reading, accel_norm

    def log_sample(self, phase, phase_start, target_heading, left_cmd, right_cmd, left_speed, right_speed, safety_state, front_threshold=None):
        distances = self.safety.read_distances()
        reading, accel_norm = self.read_imu()
        gyro = reading['gyro'] if reading else {'x': None, 'y': None, 'z': None}
        accel = reading['accel'] if reading else {'x': None, 'y': None, 'z': None}
        heading_error = self.normalize_angle(target_heading - self.yaw_deg) if target_heading is not None else None
        self._writer.writerow({
            'run_id': self.run_id,
            'sample_i': self.sample_i,
            't_monotonic': time.monotonic(),
            'phase': phase,
            'phase_elapsed_s': time.monotonic() - phase_start,
            'theta1_deg': self.guide.theta1_deg,
            'distance1_cm': self.guide.distance1_cm,
            'theta2_deg': self.guide.theta2_deg,
            'distance2_cm': self.guide.distance2_cm,
            'target_heading_deg': target_heading,
            'yaw_deg': self.yaw_deg,
            'heading_error_deg': heading_error,
            'left_cmd': left_cmd,
            'right_cmd': right_cmd,
            'left_speed_pct': left_speed,
            'right_speed_pct': right_speed,
            'front_cm': distances['front'],
            'left_cm': distances['left'],
            'right_cm': distances['right'],
            'front_threshold_cm': front_threshold,
            'safety_state': safety_state,
            'gyro_x_dps': gyro['x'],
            'gyro_y_dps': gyro['y'],
            'gyro_z_dps': gyro['z'],
            'accel_x_g': accel['x'],
            'accel_y_g': accel['y'],
            'accel_z_g': accel['z'],
            'accel_norm_g': accel_norm,
        })
        self._log_file.flush()
        self.sample_i += 1
        return distances

    def turn_to_heading(self, target_heading, tolerance_deg=8.0, timeout_s=4.0):
        phase_start = time.monotonic()
        reason = 'target_heading'
        self.last_imu_time = time.monotonic()
        while time.monotonic() - phase_start < timeout_s:
            error = self.normalize_angle(target_heading - self.yaw_deg)
            if abs(error) <= tolerance_deg:
                break
            direction = 'left' if error > 0.0 else 'right'
            left_cmd, right_cmd = ('backward', 'forward') if direction == 'left' else ('forward', 'backward')
            distances = self.safety.read_distances()
            side_cm = distances[direction]
            if side_cm is None:
                reason = f'{direction}_no_echo'
                break
            if side_cm < self.safety.config.side_turn_clear_cm:
                reason = f'{direction}_blocked'
                break
            self.safety.rover.drive(left_cmd, right_cmd, left_speed=self.turn_speed_pct, right_speed=self.turn_speed_pct)
            self.log_sample('turn', phase_start, target_heading, left_cmd, right_cmd, self.turn_speed_pct, self.turn_speed_pct, 'turning')
            time.sleep(self.dt)
        else:
            reason = 'timeout'
        self.safety.rover.stop_motors()
        self.log_sample('turn_stop', phase_start, target_heading, 'stop', 'stop', 0.0, 0.0, reason)
        return {'phase': 'turn', 'target_heading_deg': target_heading, 'yaw_deg': self.yaw_deg, 'reason': reason}

    def drive_distance(self, distance_cm, target_heading):
        phase_start = time.monotonic()
        duration = max(0.0, distance_cm / max(1e-6, self.cm_per_second))
        reason = 'duration_complete'
        self.last_imu_time = time.monotonic()
        self.safety.rover.drive('forward', 'forward', left_speed=self.speed_pct, right_speed=self.speed_pct)
        try:
            while time.monotonic() - phase_start < duration:
                safe, front, threshold = self.safety.is_front_safe(self.speed_pct)
                if not safe:
                    reason = 'front_blocked'
                    self.log_sample('drive', phase_start, target_heading, 'forward', 'forward', self.speed_pct, self.speed_pct, reason, threshold)
                    break
                self.log_sample('drive', phase_start, target_heading, 'forward', 'forward', self.speed_pct, self.speed_pct, 'driving', threshold)
                time.sleep(self.dt)
        finally:
            self.safety.rover.stop_motors()
        self.log_sample('drive_stop', phase_start, target_heading, 'stop', 'stop', 0.0, 0.0, reason)
        return {'phase': 'drive', 'distance_cm': distance_cm, 'duration_s': duration, 'yaw_deg': self.yaw_deg, 'reason': reason}

    def execute(self):
        self.open_log()
        try:
            (x1, y1), (x2, y2) = self.guide.points()
            heading1 = self.guide.theta1_deg
            heading2 = self.guide.theta1_deg + self.guide.theta2_deg
            report = {
                'run_id': self.run_id,
                'guide': {
                    'theta1_deg': self.guide.theta1_deg,
                    'distance1_cm': self.guide.distance1_cm,
                    'theta2_deg': self.guide.theta2_deg,
                    'distance2_cm': self.guide.distance2_cm,
                },
                'points_cm': {'p1': [x1, y1], 'p2': [x2, y2]},
                'log_path': self.log_path,
                'phases': [],
                'start_distances': self.safety.read_distances(),
            }
            for target_heading, distance in ((heading1, self.guide.distance1_cm), (heading2, self.guide.distance2_cm)):
                turn = self.turn_to_heading(target_heading)
                report['phases'].append(turn)
                if turn['reason'] != 'target_heading':
                    report['stopped_reason'] = 'turn_failed'
                    break
                drive = self.drive_distance(distance, target_heading)
                report['phases'].append(drive)
                if drive['reason'] != 'duration_complete':
                    report['stopped_reason'] = 'drive_failed'
                    break
            else:
                report['stopped_reason'] = 'complete'
            report['end_distances'] = self.safety.read_distances()
            report['yaw_deg'] = self.yaw_deg
            report['accel_delta_g'] = 0.0 if self.accel_norm_min is None else self.accel_norm_max - self.accel_norm_min
            report['samples'] = self.sample_i
            return report
        finally:
            self.safety.rover.stop_motors()
            self.close_log()

    def write_summary(self, report):
        summary_path = self.log_path[:-4] + '.json' if self.log_path.endswith('.csv') else self.log_path + '.json'
        with open(summary_path, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True)
        return summary_path
