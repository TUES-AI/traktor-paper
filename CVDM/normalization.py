from __future__ import annotations

import numpy as np


class RunningMean:
    """Small running mean normalizer for positive online errors."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0

    def update(self, value: float) -> float:
        value = float(value)
        self.n += 1
        self.mean += (value - self.mean) / max(1, self.n)
        return self.normalized(value)

    def normalized(self, value: float) -> float:
        value = float(value)
        if self.n <= 0 or self.mean <= 1e-12:
            return float(np.clip(value, 0.0, 1.0))
        return float(np.clip(value / (self.mean + 1e-8), 0.0, 3.0) / 3.0)

    def state_dict(self) -> dict[str, float | int]:
        return {"n": int(self.n), "mean": float(self.mean)}

    def load_state_dict(self, state: dict) -> None:
        self.n = int(state.get("n", 0))
        self.mean = float(state.get("mean", 0.0))


def normalize_l2_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return (x / (float(np.linalg.norm(x)) + eps)).astype(np.float32)
