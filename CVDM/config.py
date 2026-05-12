from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CVDMConfig:
    """Configuration for the controllable visual dynamics model."""

    dino_dim: int = 384
    range_dim: int = 3
    action_dim: int = 1
    phi_dim: int = 128
    hidden_dim: int = 256
    rnd_out_dim: int = 128

    candidate_actions: tuple[float, ...] = (-1.0, -0.6, 0.0, 0.6, 1.0)
    max_range_cm: float = 400.0
    none_range_norm: float = 0.0

    lr: float = 2e-4
    rnd_lr: float = 5e-5
    forward_weight: float = 1.0
    inverse_weight: float = 0.2
    static_weight: float = 0.5
    rnd_weight: float = 0.1
    grad_clip_norm: float = 5.0

    static_distance_cm: float = 3.0
    memory_size: int = 2000
    memory_known_distance: float = 0.35
    memory_norm_distance: float = 1.25
    memory_update_rate: float = 0.02
    memory_min_assignment_margin: float = 0.03

    visual_min_motion_cm: float = 10.0
    visual_min_yaw_deg: float = 5.0
    visual_min_front_cm: float = 25.0
    visual_front_close_clear_cm: float = 100.0
    image_min_laplacian_var: float = 25.0
    image_min_mean: float = 18.0
    image_max_mean: float = 238.0
    image_min_std: float = 8.0
    image_max_dark_frac: float = 0.55
    image_max_bright_frac: float = 0.45

    novelty_weight: float = 1.00
    learning_progress_weight: float = 0.55
    distance_reward_weight: float = 0.30
    safe_motion_bonus: float = 0.10
    new_cluster_bonus: float = 0.35
    contact_penalty: float = 0.75
    zero_progress_penalty: float = 0.22
    recovery_penalty: float = 0.12
    near_obstacle_penalty: float = 0.25
    obstructed_forward_penalty: float = 0.45
    obstructed_forward_front_cm: float = 40.0
    obstructed_forward_theta_deg: float = 25.0
    clear_front_turn_penalty: float = 0.05
    clear_front_turn_start_cm: float = 45.0
    clear_front_turn_scale_cm: float = 80.0
    safe_motion_min_cm: float = 5.0
    safe_front_min_cm: float = 35.0
    reward_distance_scale_cm: float = 120.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
