from __future__ import annotations

import numpy as np

from CVDM.config import CVDMConfig


def reward_from_transition(
    config: CVDMConfig,
    novelty: float,
    learning_progress: float,
    executed_distance_cm: float,
    front_after_cm: float | None,
    contact_or_stall: bool,
    recovery: bool,
    zero_progress: bool,
    new_cluster: bool = False,
    visual_memory_valid: bool = True,
    pre_front_cm: float | None = None,
    requested_theta_deg: float = 0.0,
    requested_theta_norm: float | None = None,
    loop_penalty: float = 0.0,
    recovery_streak_penalty: float = 0.0,
    coverage_expansion_bonus: float = 0.0,
    path_revisit_penalty: float = 0.0,
) -> tuple[float, dict[str, float | bool | None]]:
    front_ok = front_after_cm is not None and float(front_after_cm) > config.safe_front_min_cm
    safe_motion = (
        float(executed_distance_cm) > config.safe_motion_min_cm
        and not bool(contact_or_stall)
        and front_ok
    )
    visual_reward_gate = bool(visual_memory_valid) and not bool(contact_or_stall)
    novelty_cluster_weight = 1.0 if bool(new_cluster) else float(config.novelty_existing_cluster_weight)

    novelty_reward = config.novelty_weight * novelty_cluster_weight * float(novelty) if visual_reward_gate else 0.0
    learning_progress_reward = config.learning_progress_weight * float(learning_progress) if visual_reward_gate else 0.0
    distance_reward = 0.0
    if safe_motion:
        distance_reward = config.distance_reward_weight * min(1.0, float(executed_distance_cm) / max(1e-6, config.reward_distance_scale_cm))
    safe_motion_bonus = config.safe_motion_bonus if safe_motion else 0.0
    new_cluster_bonus = config.new_cluster_bonus if (visual_reward_gate and bool(new_cluster)) else 0.0

    contact_penalty = config.contact_penalty if contact_or_stall else 0.0
    zero_progress_penalty = config.zero_progress_penalty if zero_progress else 0.0
    recovery_penalty = config.recovery_penalty if recovery else 0.0
    obstructed_forward_penalty = 0.0
    obstructed_front_pressure = 0.0
    obstructed_forward_weight = 0.0
    if pre_front_cm is not None:
        obstructed_front_pressure = float(
            np.clip((config.obstructed_forward_front_cm - float(pre_front_cm)) / max(1e-6, config.obstructed_forward_front_cm), 0.0, 1.0)
        )
        obstructed_forward_weight = float(
            np.clip(1.0 - abs(float(requested_theta_deg)) / max(1e-6, config.obstructed_forward_theta_deg), 0.0, 1.0)
        )
        obstructed_forward_penalty = config.obstructed_forward_penalty * obstructed_front_pressure * obstructed_front_pressure * obstructed_forward_weight

    clear_front_turn_penalty = 0.0
    clear_front_turn_gate = 0.0
    if pre_front_cm is not None and requested_theta_norm is not None:
        clear_front_turn_gate = float(
            np.clip((float(pre_front_cm) - config.clear_front_turn_start_cm) / max(1e-6, config.clear_front_turn_scale_cm), 0.0, 1.0)
        )
        clear_front_turn_penalty = config.clear_front_turn_penalty * clear_front_turn_gate * float(requested_theta_norm) * float(requested_theta_norm)
    near_obstacle_penalty = 0.0
    if front_after_cm is not None and float(front_after_cm) < config.safe_front_min_cm:
        near_obstacle_penalty = config.near_obstacle_penalty * ((config.safe_front_min_cm - float(front_after_cm)) / config.safe_front_min_cm) ** 2

    reward = novelty_reward + learning_progress_reward + distance_reward + safe_motion_bonus + new_cluster_bonus + float(coverage_expansion_bonus)
    reward -= (
        contact_penalty
        + zero_progress_penalty
        + recovery_penalty
        + near_obstacle_penalty
        + obstructed_forward_penalty
        + clear_front_turn_penalty
        + float(loop_penalty)
        + float(recovery_streak_penalty)
        + float(path_revisit_penalty)
    )
    terms = {
        "safe_motion": bool(safe_motion),
        "visual_reward_gate": bool(visual_reward_gate),
        "novelty_reward": float(novelty_reward),
        "novelty_cluster_weight": float(novelty_cluster_weight),
        "novelty_existing_cluster_weight": float(config.novelty_existing_cluster_weight),
        "learning_progress_reward": float(learning_progress_reward),
        "distance_reward": float(distance_reward),
        "safe_motion_bonus": float(safe_motion_bonus),
        "new_cluster_bonus": float(new_cluster_bonus),
        "contact_penalty": float(contact_penalty),
        "zero_progress_penalty": float(zero_progress_penalty),
        "recovery_penalty": float(recovery_penalty),
        "near_obstacle_penalty": float(near_obstacle_penalty),
        "obstructed_forward_penalty": float(obstructed_forward_penalty),
        "obstructed_front_pressure": float(obstructed_front_pressure),
        "obstructed_forward_weight": float(obstructed_forward_weight),
        "clear_front_turn_penalty": float(clear_front_turn_penalty),
        "clear_front_turn_gate": float(clear_front_turn_gate),
        "loop_penalty": float(loop_penalty),
        "recovery_streak_penalty": float(recovery_streak_penalty),
        "coverage_expansion_bonus": float(coverage_expansion_bonus),
        "path_revisit_penalty": float(path_revisit_penalty),
        "pre_front_cm": None if pre_front_cm is None else float(pre_front_cm),
        "requested_theta_deg": float(requested_theta_deg),
        "requested_theta_norm": None if requested_theta_norm is None else float(requested_theta_norm),
        "front_after_cm": None if front_after_cm is None else float(front_after_cm),
        "executed_distance_cm": float(executed_distance_cm),
        "novelty_phi": float(novelty),
        "learning_progress": float(learning_progress),
    }
    if not np.isfinite(reward):
        reward = 0.0
        terms["reward_was_nonfinite"] = True
    return float(reward), terms
