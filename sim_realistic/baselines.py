from __future__ import annotations

import numpy as np


class DeterministicExplorer:
    """Reactive local-only baseline over [turn, speed]."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.turn_timer = 0
        self.turn_dir = 1.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        sensors = obs[-7:-4]
        left, right, front = [float(x) for x in sensors]
        if self.turn_timer > 0:
            self.turn_timer -= 1
            return np.array([self.turn_dir, 0.0], dtype=np.float32)
        if front < 0.20:
            self.turn_dir = 1.0 if left > right else -1.0
            self.turn_timer = int(self.rng.integers(8, 18))
            return np.array([self.turn_dir, 0.0], dtype=np.float32)
        if left < 0.10:
            return np.array([-0.65, 0.45], dtype=np.float32)
        if right < 0.10:
            return np.array([0.65, 0.45], dtype=np.float32)
        if self.rng.random() < 0.035:
            return np.array([float(self.rng.uniform(-0.7, 0.7)), 0.6], dtype=np.float32)
        return np.array([0.0, 0.75], dtype=np.float32)
