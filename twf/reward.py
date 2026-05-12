"""Reward terms for visionless world-feedback exploration."""


def sensory_novelty(current, recent, scale=1.0):
    """Return distance from recent sensory states.

    This intentionally uses compact sensory/motion state, not images. The reward
    should be gated by actual movement so obstacle contact or stationary jitter
    cannot become novelty.
    """
    if not recent:
        return float(scale)
    best = min(sum((float(a) - float(b)) ** 2 for a, b in zip(current, item)) ** 0.5 for item in recent)
    return float(scale) * best


def world_feedback_reward(
    *,
    sensory_novelty_value,
    executed_distance_cm,
    safe_motion,
    revisit_score,
    zero_progress,
    recovery,
    near_obstacle,
    loop_score,
):
    """Auditable scalar reward for the TWF planner."""
    positive = (
        0.45 * float(sensory_novelty_value)
        + 0.010 * max(0.0, float(executed_distance_cm))
        + 0.10 * float(bool(safe_motion))
    )
    negative = (
        0.20 * float(revisit_score)
        + 0.35 * float(bool(zero_progress))
        + 0.12 * float(bool(recovery))
        + 0.20 * float(near_obstacle)
        + 0.45 * float(loop_score)
    )
    return positive - negative
