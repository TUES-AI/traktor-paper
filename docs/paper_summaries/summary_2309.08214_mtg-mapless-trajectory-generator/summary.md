# MTG: Mapless Trajectory Generator

MTG is the closest paper so far to our **no-map local trajectory planner** idea. It does not output one action. It outputs a **set of possible short future paths** from onboard perception.

```text
lidar history + velocity history
    -> encoder
    -> stochastic latent path candidates
    -> attention between candidates
    -> GRU decoders
    -> K egocentric waypoint trajectories
```

Each path is a sequence of robot-local deltas:

```text
[(dx1, dy1), ..., (dx16, dy16)]
```

Those deltas are accumulated into a local path. In the paper the horizon is about `15m`; for our rover it should be `1-3m`.

## What The Modules Mean

- **Encoder**: compresses raw sensing into a compact vector. MTG uses PointCNN over 3D lidar plus an MLP over velocity history. Our version would use camera, ultrasonic, IMU, and motor history.
- **CVAE-style latent**: lets the same observation generate multiple plausible futures instead of one averaged path.
- **DLOW-style transforms**: pushes those futures to be different useful options, not five copies of the same path.
- **Self-attention over paths**: candidate paths look at each other so they spread over useful traversable directions and avoid redundant/stupid duplicates.
- **GRU decoder**: draws each path step-by-step as waypoint deltas.

## The Catch

MTG is **mapless only at inference**. Training uses offline traversability maps and A* paths. That part does not match our final method.

So the architecture is useful, but the training recipe is not:

```text
MTG training: offline map + A* supervision
our training: real-world feedback only, no full map, no A*
```

## Why It Matters For Us

This paper supports the shape of our slower planner:

```text
sensors + memory
    -> K candidate local trajectories for next 1-3m
    -> score by novelty / safety / low overlap
    -> selected trajectory goes to executor
```

Then our small executor handles terrain/motor response:

```text
chosen local path + IMU/motor feedback -> wheel commands
```

## What To Steal

- Generate **multiple local futures**, not one command.
- Output egocentric waypoint deltas, not global coordinates.
- Make the candidate set cover useful free/coverable directions.
- Use attention or another coupling mechanism so candidates do not collapse.
- Keep the planner local and short-horizon for our weak sensors.

## What To Replace

- Replace lidar encoder with camera/ultrasonic/IMU/action-history encoder.
- Replace A* supervision with real-world reward: novelty, progress, low overlap, no collision, no stuck behavior.
- Replace `15m` horizon with `1-3m`.
- Replace their external low-level planner with our fast adaptive executor.

## First Experiment

Start simple: fixed candidate arcs first, learned scoring second, learned generation later.

```text
left arc / slight left / straight / slight right / right arc / backup / turn
    -> execute one for 1-3s
    -> reward outcome
    -> learn which local futures are useful
```
