# PCVM theta-front resume50 loopfix good run — 2026-05-09

This is a keeper autonomous run. It resumed SAC from the previous 100-step theta-front checkpoint while resetting PCVM/novelty memory by creating a fresh environment.

## Human label

- Quality: really good.
- Behavior: reached 2 rooms with only a little corridor looping.
- Interpretation: likely partly luck; do not claim the loop penalty change caused the improvement without repeated runs.

## Files

- `trajectory.jsonl`: 50-step per-step log.
- `frames/`: post-action camera frames.
- `summary_3x3.jpg`: visual summary.
- `sac_model.zip`: checkpoint after this 50-step continuation.
- `run_stdout.txt`: quiet run stdout/stderr.
- `params.json`: key run/reward parameters.

## Metrics

- Reward sum: +13.255
- Mean reward: +0.265
- Median reward: -0.248
- Positive steps: 20 / 50
- Travel estimate: 16.287 m
- Net displacement estimate: 4.045 m
- Dead-reckoned bbox area: 17.385 m²
- Executions: 36
- Drive ok: 33 / 36
- Recoveries: 17
- Reverse recoveries: 7
- Contact/stall events: 3
- Path clusters: 2
- Visual clusters: 1

## Reward mismatch note

The scalar reward under-rated this run relative to visual judgment. The previous 50-step resume run looked much worse to the human reviewer but scored +23.603, while this better run scored only +13.255. The main reason is bookkeeping, not behavior quality:

- This run used stronger loop/revisit penalties: `loop_revisit_penalty=0.75`, `recent_revisit_penalty=0.12`.
- The discarded previous run used lower penalties: `loop_revisit_penalty=0.45`, `recent_revisit_penalty=0.06`.
- This alone cost the keeper run about 7 reward points compared with the previous scoring scale.
- The previous bad run also got a larger new-cluster bonus (+5.0 vs +2.5), even though visually it looped in a corridor. This reinforces that current PCVM cluster creation is not a reliable room-level coverage signal.

If rescored with the older lower loop/revisit penalties, this keeper run would score around +20.3. It still would not fully dominate the bad run because the reward lacks a strong human-aligned room-transition/coverage signal and visual memory still collapses to one cluster.

## Keep as diagnostic data

Use this run to test reward alignment and executor/safety changes. Do not use it for policy pretraining.
