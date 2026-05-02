# RND: Random Network Distillation

RND is not a planner. It is a simple way to create **intrinsic novelty reward** when there is no teacher, no map, and no dense task reward.

It sounds dumb, but the mechanism is clear:

```text
observation
    -> frozen random target network -> target fingerprint
    -> trainable predictor network -> predicted fingerprint

novelty reward = MSE(predicted fingerprint, target fingerprint)
```

The frozen network gives every observation a stable random fingerprint. The predictor learns fingerprints for observations the agent has already visited.

```text
familiar state -> predictor is good -> low reward
novel state -> predictor is bad -> high reward
repeated novel state -> predictor learns it -> reward decays
```

## Gradient Flow

There are two learning signals, and they update different things.

```text
PPO / policy loss:
    high RND error becomes reward
    policy learns to reach states that produced novelty

RND predictor loss:
    same error trains predictor
    repeated states become less novel
```

The target network is frozen. The predictor does not choose actions. It only produces the novelty reward that gets plugged into PPO or whatever RL objective we use.

## Why The Random Network Does Not Need To Be Meaningful

It is basically a deterministic random hash / feature map:

```text
same observation -> same fingerprint
similar observations -> often somewhat similar fingerprints
new observation distribution -> predictor generalizes worse -> higher error
```

The risk is real: bad random features, raw camera noise, lighting flicker, or vibration can all look novel. That is why for the rover RND should run on normalized compact features first, not raw shaky pixels.

## Avoiding Predictor-Target Collapse

The predictor should learn visited regions, not globally copy the target. Make that harder by design:

```text
target: frozen, wider random MLP/CNN
predictor: smaller, bottlenecked trainable MLP/CNN
train predictor on only a subset of samples
normalize inputs and target scale
detach target output completely
```

If the predictor globally copied the target, novelty would die everywhere. In practice it usually only fits the visited input distribution.

## Why It Matters For Us

Our rover needs no-A* feedback for coverage. RND can be the “this seems new” part of the reward:

```text
candidate trajectory executed
    -> observe resulting camera/ultrasonic/IMU state
    -> RND novelty score
    -> reward planner if it reached unfamiliar safe experience
```

But RND cannot be the whole reward. For the rover:

```text
reward = novelty
       + motion/progress
       - overlap/revisit
       - obstacle too close
       - collision
       - stuck / commanded motion but no IMU response
       - spinning in place
       - camera noise/vibration artifacts
```

## What To Steal

- Frozen random target network.
- Trainable predictor network.
- Prediction error as novelty reward.
- Separate novelty reward/value stream from safety/task reward.
- Observation and intrinsic-reward normalization.
- Use novelty decay as an automatic pressure to move on.

## First Experiment

Before camera works reliably, run RND on compact rover state:

```text
[ultra_left, ultra_front, ultra_right,
 gyro_z, accel_norm, vibration,
 last_left_cmd, last_right_cmd]
```

During manual driving, check:

```text
new place / new motion response -> higher RND
same place / repeated behavior -> lower RND
stuck/noisy vibration -> must be suppressed by penalties
```
