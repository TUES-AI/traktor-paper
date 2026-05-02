# RECON: Rapid Exploration With Latent Goal Models

RECON is not a trajectory generator. It is useful because it shows how a real robot can learn exploration/navigation structure from **real-world experience without a metric map**.

Its core loop:

```text
current image + goal image
    -> compressed latent goal
    -> action + predicted time-to-goal
    -> store observations in topological memory
```

The robot explores a new place, stores visual nodes, predicts reachability between them, and later uses that graph to reach a visual goal faster.

## What The Modules Mean

- **Latent goal model**: compresses “where I am vs where I want to be” into a small vector. It is not a 2D coordinate.
- **Encoder**: MobileNet over current image + goal image, concatenated as 6 channels.
- **Decoder**: current image + latent goal -> velocity action and predicted temporal distance.
- **Information bottleneck**: forces the latent to ignore useless visual details like lighting/distractors.
- **Topological memory**: graph of seen images, reachability edges, and visit counts. This is not SLAM; it is a memory of places and how reachable they seem.

## Why It Matters For Us

The useful part is not goal-image navigation. Our rover has no fixed goal. The useful part is:

```text
real robot trajectories can supervise future behavior without A*
```

RECON turns real trajectory logs into training pairs:

```text
observation now
future observation
actions taken
time between them
```

For us this becomes:

```text
current sensors + future sensors + action history
    -> learn reachability / novelty / revisit likelihood
```

## How It Fits With MTG

MTG gives the local trajectory candidates. RECON gives the memory/reachability idea.

Combined rover version:

```text
camera + ultrasonic + IMU + motor history
    -> local observation embedding
    -> topological/visual memory with visit counts
    -> score whether candidate trajectories lead to new/safe places
```

## What To Steal

- Learn from real trajectory logs using self-supervised frame pairs.
- Use time between observations as a rough reachability/distance label.
- Store a lightweight graph of sensory places and visit counts.
- Use memory to penalize overlap/revisits without needing a metric map.
- Use IMU/stuck/collision events as weak labels.

## What To Replace

- Replace user goal images with intrinsic coverage/novelty objectives.
- Replace velocity-action decoder with local trajectory scoring/generation.
- Replace large MobileNet setup with a small encoder suitable for Pi or offline training.
- Treat the graph as coverage memory, not as a goal-navigation map.

## First Experiment

Log manual rover driving:

```text
camera frame / ultrasonic / IMU / motor command / timestamp
```

Then train a small model:

```text
obs_t + obs_future -> predicted time/action difficulty
```

If it can recognize “I have been here before,” it can become the overlap penalty for coverage.
