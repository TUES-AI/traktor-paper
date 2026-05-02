# Plan

Objective: learn real-world no-map exploration that covers new area/experience fast, with low overlap and safe motion.

Core interface:

```text
planner output = [curvature, horizon, speed]
executor output = execution_report
reward = novelty + safe motion - overlap - collision/stuck/executor failure
```

## Person 1: Planner / Reward

- [ ] Sim: implement toy loop: `state -> [curvature, horizon, speed] -> simulated motion -> reward`.
- [ ] Sim: test reward shaping: novelty, safe motion, overlap, collision, executor failure.
- [ ] Sim: verify policy learns obstacle behavior: box ahead should prefer left/right arc over straight.
- [ ] Real world: start with random guide policy and logging-compatible planner shell.
- [ ] Real world: implement compact RND using ultrasonic + IMU summary + last guide + execution score.
- [ ] Real world: add replay buffer and planner reward.
- [ ] Real world: replace random guide policy with SAC over `[curvature, horizon, speed]`.
- [ ] Real world: add camera embedding after camera capture is reliable.
- [ ] Real world: generate plots: novelty, repeat score, safety stops, guide distribution, executor failures.

## Person 2: Executor / Hardware

- [ ] Real world: stabilize motors, ultrasonic sensors, IMU, timestamps, and emergency stop.
- [ ] Real world: build sensor logger: IMU + distance + motor commands + timestamp.
- [ ] Real world: implement deterministic guide executor: `[curvature, horizon, speed] -> left/right motors`.
- [ ] Real world: run calibration grid over curvature, speed, and horizon.
- [ ] Real world: compute execution report: yaw achieved, yaw error, vibration, slip/stuck, min distance, safety stop.
- [ ] Sim: fit simple executor noise model from real logs: requested guide -> noisy achieved guide.
- [ ] Sim: provide noisy executor model to planner work.
- [ ] Real world: implement tiny learnable executor correction: guide + IMU/history -> motor correction.
- [ ] Real world: compare deterministic executor vs learned executor on yaw error, stuck/slip, command smoothness.

## Integration Milestones

- [ ] Real world: random guide policy + deterministic executor + logging.
- [ ] Real world: execution reports are trustworthy.
- [ ] Sim: planner learns with toy/noisy executor.
- [ ] Real world: RND novelty gives reasonable signal during manual/random driving.
- [ ] Real world: SAC replaces random guide policy.
- [ ] Real world: learned executor correction improves guide tracking.
- [ ] Real world: camera embedding is added.
- [ ] Final comparison: random guides, reactive baseline, SAC planner + deterministic executor, SAC planner + learned executor.

## Ablation Checklist

- [ ] Ablation 1: trajectory-guide policy vs raw motor policy.
  - Compare: `SAC -> [curvature, horizon, speed]` against `SAC -> left/right motor command`.
  - Shows whether forcing the action into local trajectory intent helps exploration and obstacle avoidance.
  - Metrics: novelty per minute, safety-stop rate, stuck rate, repeated-state score.

- [ ] Ablation 2: learned executor vs deterministic executor.
  - Compare: fixed arc controller against tiny learnable executor correction.
  - Shows whether IMU/world feedback helps execute the requested curvature on imperfect terrain.
  - Metrics: yaw error, motion-success score, slip/stuck score, planner reward after execution.

- [ ] Ablation 3: RND novelty vs no RND.
  - Compare: full reward with RND against reward without intrinsic novelty.
  - Shows whether novelty feedback actually drives broader exploration.
  - Metrics: distinct observation clusters, repeated-state score, guide diversity, time until behavior loops.

- [ ] Ablation 4: overlap memory vs no overlap memory.
  - Compare: RND + recent/topological memory penalty against RND alone.
  - Shows whether memory prevents the rover from revisiting visually novel-but-repeated loops.
  - Metrics: repeated-state score, loop frequency, distinct clusters per meter/second.

- [ ] Ablation 5: camera embedding vs compact sensors only.
  - Compare: ultrasonic + IMU + motor history against camera embedding + compact sensors.
  - Shows whether vision improves left/right obstacle choice and novelty estimation.
  - Metrics: obstacle bypass success, novel clusters, false novelty from vibration/lighting, inference time.

## Minimum Result Worth Writing

- [ ] Show that `[curvature, horizon, speed]` as a learned local guide explores better than raw motor SAC or random guides.
- [ ] Show that the learned executor lowers execution error on real terrain compared with deterministic control.
- [ ] Show that RND/memory reward produces more distinct real-world observation clusters without external maps.
