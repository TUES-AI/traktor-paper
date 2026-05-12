# Tiny visionless exploration from world feedback

## Working title

Tiny World Feedback Exploration (TWF): mapless real-rover exploration without cameras, SLAM, or occupancy maps.

## Hard direction

The runtime model is visionless.

- No camera input to the GRU.
- No camera input to SAC.
- No VMM, PCVM, MobileNet, visual memory bank, or visual clustering in the policy observation.
- No SLAM, occupancy reconstruction, frontier map, or LiDAR-style mapping objective.

The robot learns from onboard scalar feedback only: ultrasonic range, IMU/motion summaries, previous action, and executor outcome.

## Runtime architecture

```text
ultrasonic + IMU/motion + previous action + executor feedback
    -> tiny MLP encoder
    -> GRU state
    -> SAC local heading policy
    -> deterministic/safety executor
    -> world feedback reward
```

The policy decides where/how to try moving next. The executor turns the requested local heading into motor commands and reports what actually happened.

## Reward direction

Reward must come from grounded consequences, not images:

- positive: safe executed distance, sensory novelty, successful escape from blocked states
- negative: zero progress, repeated recent sensory state, recovery, near obstacle, loop/stall patterns

Looping must become unprofitable even if the rover keeps moving. Distance reward should be gated by novelty/progress so kitchen/corridor circles do not accumulate positive reward forever.

## Name of the new model

Use `TWF` for now: Tiny World Feedback.

Old names are retired for runtime code:

- VMM: removed
- PCVM: removed
- MobileNet/DINO runtime encoders: removed

## Offline DINO exception

The only retained vision component is offline DINO image clustering for paper analysis and possible future reward-only ablations. It lives in `tools/replay_dinov3_onnx_clusters.py` and must not be imported by the runtime policy.

Allowed use:

```text
saved frames -> frozen DINO embeddings -> image clusters -> paper figure / future reward-only ablation
```

Forbidden use:

```text
DINO/MobileNet/CNN features -> GRU/SAC observation
```

## Current code shape

- `twf/`: tiny sensory recurrent model and reward functions
- `embedded/`: rover hardware, safety, executor, and real-run scripts
- `tools/replay_dinov3_onnx_clusters.py`: offline DINO clustering only
- `paper/`: ICML-style draft being rewritten around the visionless claim

If an old visual/mapping/simulation component is needed again, recover it from git instead of keeping it in the clean tree.
