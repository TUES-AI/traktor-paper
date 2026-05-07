# PCVM theta-front autonomous run — 2026-05-07

This is a real rover autonomous RLxF run kept as an offline evaluation and reward-auditing dataset. It should not be used for supervised pretraining of the exploration policy.

## Human label

- Overall quality: good autonomous exploration run.
- Coverage: reached all 5 rooms in the apartment during 200 steps.
- Failure mode: looped in the same corridor/area many times and hit its head against walls repeatedly.
- Use this run as a positive/high-coverage example with significant local-control and safety-script failures.

## Method used

- Backend: PCVM, no MobileNet variant.
- Policy action: scalar relative heading `theta_norm`.
- Executor: turn by `theta_norm * 75°`, then drive forward until front ultrasonic distance reaches 40 cm or the max drive time ends.
- Reward mode: loop-aware slow RLxF reward with executed-distance reward, safe-motion bonus, novelty terms, recovery penalties, zero-forward penalty, recent-revisit penalty, and loop penalty.
- Training was online on the physical rover; these files are saved afterward only for analysis, reward alignment tests, visual review, and script debugging.

## Files

- `trajectory.jsonl`: per-step log with action, target/execution report, recovery report, reward terms, PCVM backend diagnostics, distance sensors, and frame path metadata.
- `frames/`: 200 post-action camera frames, one per training step.
- `contact_sheet.jpg`: every 10th frame for quick visual inspection.
- `sac_model.zip`: Stable-Baselines SAC checkpoint saved at the end of the run.

## Run metrics

- Steps: 200
- Frames: 200
- Reward sum: -11.913
- Mean reward: -0.060
- Median reward: -0.523
- Travel estimate: 63.496 m
- Net displacement estimate: 9.801 m
- Dead-reckoned bbox area: 121.292 m²
- Executions: 133
- Drive ok: 133 / 133
- Recoveries: 90
- Reverse recoveries: 50
- Path clusters: 1
- Visual clusters: 1

## Important interpretation

The run was physically useful despite the negative reward total. The reward correctly penalized many local loops/recoveries, but the rover still covered the apartment over the full 200-step horizon. The PCVM memory collapsed to one path cluster and one visual cluster, so this dataset is especially useful for testing better visual descriptors, reward-audit scripts, and wall-hit / recovery handling without changing the core online RLxF training story.
