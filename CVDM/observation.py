from __future__ import annotations

import numpy as np
import torch

from CVDM.config import CVDMConfig


def norm_distance(cm: float | None, config: CVDMConfig) -> float:
    if cm is None:
        return float(np.clip(config.none_range_norm, 0.0, 1.0))
    return float(np.clip(float(cm) / max(1e-6, config.max_range_cm), 0.0, 1.0))


def normalized_ranges(distances: dict, config: CVDMConfig) -> np.ndarray:
    return np.array(
        [
            norm_distance(distances.get("left"), config),
            norm_distance(distances.get("right"), config),
            norm_distance(distances.get("front"), config),
        ],
        dtype=np.float32,
    )


def candidate_clearance_scores(distances: dict, config: CVDMConfig) -> np.ndarray:
    left = norm_distance(distances.get("left"), config)
    right = norm_distance(distances.get("right"), config)
    front = norm_distance(distances.get("front"), config)
    return np.array(
        [
            right,
            0.5 * (right + front),
            front,
            0.5 * (left + front),
            left,
        ],
        dtype=np.float32,
    )


def normalize_candidate_scores(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.sigmoid((scores - mean) / std)
