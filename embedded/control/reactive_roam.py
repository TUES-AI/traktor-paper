"""Reactive roam policy above the safety layer."""

import time


class ReactiveRoamPolicy:
    def __init__(self, safety, slow_speed=45.0, fast_speed=65.0, turn_speed=70.0, fast_clear_cm=100.0):
        self.safety = safety
        self.slow_speed = slow_speed
        self.fast_speed = fast_speed
        self.turn_speed = turn_speed
        self.fast_clear_cm = fast_clear_cm

    def choose_forward_speed(self, distances):
        front = distances['front']
        if front is None or front >= self.fast_clear_cm:
            return self.fast_speed
        return self.slow_speed

    def step(self):
        distances = self.safety.read_distances()
        speed = self.choose_forward_speed(distances)
        front_safe, front, threshold = self.safety.is_front_safe(speed, distances)

        if front_safe:
            stuck, report = self.safety.detect_stuck_during_forward(speed)
            if stuck:
                self.safety.reverse_recovery()
                direction = self.safety.freer_side()
                turn = self.safety.turn_until_clear(direction, speed_pct=self.turn_speed)
                return {'action': 'stuck_recovery', 'direction': direction, 'stuck': report, 'turn': turn}
            return {'action': 'forward', 'speed': speed, 'front_cm': front, 'threshold_cm': threshold, 'motion': report}

        direction = self.safety.freer_side(distances)
        turn = self.safety.turn_until_clear(direction, speed_pct=self.turn_speed)
        return {'action': 'turn_until_clear', 'direction': direction, 'front_cm': front, 'threshold_cm': threshold, 'turn': turn}

    def run(self, seconds=None, sleep_seconds=0.05):
        start = time.monotonic()
        while seconds is None or time.monotonic() - start < seconds:
            yield self.step()
            time.sleep(sleep_seconds)
