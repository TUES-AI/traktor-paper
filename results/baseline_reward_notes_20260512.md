# Baseline reward notes — 2026-05-12

All runs used the same scalar heading movement pipeline:

```text
action_norm in [-1, 1] -> theta_deg = action_norm * 90
    -> IMU turn
    -> post-turn front gate
    -> drive until front threshold / timeout / stall
```

## Random SAC-action baseline

Artifacts:

- `results/twf_random_policy_baseline_100_20260512.jsonl`
- `results/twf_random_policy_baseline_100_20260512.out`

Policy:

- `action_norm ~ Uniform(-1, 1)`
- no sensor-conditioned action logic

Metrics:

- steps: `100`
- total reward: `-24.789`
- mean reward: `-0.248`
- positive / negative steps: `44 / 56`
- executed distance: `31.49m`
- post-turn blocked before drive: `45`
- zero-progress steps: `49`
- recovery steps: `37`
- complete moves: `51`
- contact/stall: `4`

## Generic sensor-only deterministic baseline

Artifacts:

- `results/twf_sensor_det_baseline_100_20260512.jsonl`
- `results/twf_sensor_det_baseline_100_20260512.out`

Policy:

- uses only current ultrasonic `{left, right, front}`
- no IMU in policy
- no map, memory, room/environment knowledge, or learned params

Metrics:

- steps: `100`
- total reward: `21.843`
- mean reward: `0.218`
- positive / negative steps: `63 / 37`
- executed distance: `48.97m`
- post-turn blocked before drive: `16`
- zero-progress steps: `19`
- recovery steps: `33`
- complete moves: `81`
- contact/stall: `3`

## Reference numbers from nearby runs

- trained SAC-ish post-turn-gate run: mean reward `0.448`
- over-crafted deterministic free-space baseline: mean reward `0.561`

The over-crafted baseline is useful for debugging but should not be the main paper baseline because it encodes too much behavior specific to this rover setup.
