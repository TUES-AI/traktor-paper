# Realistic Local-Only Rover Simulator

Clean simulator for testing exploration methods without leaking information the real rover cannot observe.

## Contract

Agent observations include only:

- egocentric ray-camera image flattened to `16 x 32`
- three ultrasonic-like readings: left, right, front
- yaw-rate / acceleration proxy
- last action

Agent observations and rewards do **not** include:

- true `x, y`
- room id
- top-down map
- coverage grid
- synthetic lookahead from hidden map state

Every action goes through a deterministic rover-style safety filter before
physics, matching the real stack's separation between proposed and executable
motion. Pure in-place spins are allowed, while forward motion is slowed or
replaced with a spin when the front/side local ranges are too close.

Ground truth appears only in `info` and result CSVs for evaluation.

## Methods

```bash
python -m sim_realistic.train --steps 50000 --seed 42 \
  --methods deterministic sac_novmm sac_vmm predictive_sac
```

- `deterministic`: local reactive baseline over `[turn, speed]`
- `sac_novmm`: SAC on local onboard observation only
- `sac_vmm`: SAC plus online RND novelty on the onboard observation
- `predictive_sac`: SAC on an online predictive latent model with RND and transition surprise

## Metrics

The metrics are hidden from the agent and used only for evaluation:

- coverage of hidden grid
- rooms reached
- door crossings
- collisions
- safety clamps
