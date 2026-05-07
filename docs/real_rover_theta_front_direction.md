# Real rover theta-front RLxF direction

This note captures the current real-rover direction after the May 2026 apartment runs.

## Current action abstraction

The planner no longer chooses raw motor commands and no longer chooses a short `(theta, distance)` waypoint as the main test mode. The current useful mode is:

```text
SAC action: [theta_norm]
theta_deg = theta_norm * 75°
executor: turn relative theta, then drive forward until front distance <= 40 cm or max drive time
```

This worked better because it strongly biases the rover toward actually moving through the apartment instead of dithering with tiny local targets.

## Why relative theta

The rover has weak pose estimates and no reliable SLAM. Relative heading is cleaner than a global target heading because it is independent of start orientation and does not pretend the dead-reckoned pose is ground truth.

## Current reward direction

The reward now tries to separate useful exploration from corridor looping:

- reward executed forward distance
- reward safe meaningful motion
- reward motion-gated novelty and new memory clusters
- penalize recoveries and recovery streaks
- penalize already-at-front / zero-forward steps
- penalize recent pose revisits and local loop pressure
- penalize near obstacles

The reward is intentionally not treated as perfect. Human visual review of the run remains the final alignment signal.

## Key empirical result

The 200-step PCVM theta-front run on 2026-05-07 was human-labeled as good: it covered all 5 apartment rooms, but it also looped many times and hit walls repeatedly. This makes it a valuable offline evaluation dataset for reward alignment and script/executor improvements.

Dataset:

```text
data/autonomous_runs/pcvm_theta_front_20260507_200step_allrooms/
```

## Not allowed as core method

This dataset is not for policy pretraining. It is allowed for:

- reward auditing
- visual-memory descriptor testing
- offline script diagnostics
- selecting better safety/executor heuristics
- producing figures and qualitative examples

The core method should remain real-world online RLxF training/adaptation.

## Next likely engineering focus

Do not change learning mechanics before reviewing the scripts/executor. The current biggest practical issue is wall/head collisions and repeated recovery behavior, not SAC itself.
