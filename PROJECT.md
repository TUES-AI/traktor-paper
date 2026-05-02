# Project scratchboard document

## Name: Grounded Exploration: Decoupled Planning and Terrain-Adaptive Execution from Onboard World Feedback

---
This is our main whiteboard document about reading research papers and ideas for our paper for the RLxF (reinforcement learning from world feedback) ICML workshop.
> This is mainly a human shared and managed document
---
This is our hackathon project:
https://github.com/backprop-pray/Tracky
The tractor has: 3 distance sensors in front, 2 on the sides, front camera and a MPU.

We want to reuse the tractor for this:
https://sites.google.com/view/rlxf-icml2026
(we will just use the hardware and whatever RL code we have, we will not frame the paper as agriculture or such, we just so happen to have the needed hardware to do tests in the RLxF realm)

## Current direction

Objective - maximize exploration as area covered the fastest. Find more new places.

Current method idea: real-world no-map local exploration from world feedback.

The robot never knows the whole map. Training and adaptation for the final result happen on the real tractor, not in sim. Sim is only for hypothesis checks.

We should not learn raw motor PWM directly as the main policy output. The slow policy should output a short local movement guide:

```text
action = [curvature, horizon, speed]
```

Meaning:

```text
curvature < 0: arc right
curvature = 0: go straight
curvature > 0: arc left
horizon: how long / far to commit
speed: how aggressively to execute
```

This keeps a trajectory-planning shape without needing a full MTG waypoint generator yet.

Pipeline:

```text
camera + ultrasonic + IMU + motor/action history
    -> encoder / short memory
    -> SAC-style local guide policy: [curvature, horizon, speed]
    -> adaptive executor converts guide to left/right motor commands
    -> world feedback scores what happened
```

Executor idea: tiny learnable controller with deterministic fallback.

```text
input:
    desired guide = [curvature, horizon, speed]
    current IMU / distance sensors / last motor commands

output:
    left/right motor command for the next control tick

base fallback:
    deterministic arc model converts curvature + speed to left/right ratio

feedback:
    gyro yaw rate says whether desired curvature is happening
    accelerometer says motion/vibration/slip/stuck/terrain response
    distance sensors say emergency obstacle

learning target / reward:
    execute requested curvature/horizon as closely as possible
    avoid slip, stuck, obstacle approach, violent vibration
```

This executor is intentionally learnable. It can be very small, e.g. `10-200` neurons, because it does not decide where to explore. It only learns how to make the physical tractor execute the guide that the SAC planner requested.

So the split is:

```text
SAC planner: choose useful local guide for exploration
tiny executor: make the motors/terrain actually realize that guide
```

Example:

```text
planner outputs: curve left for 1.2s at 60% speed
executor outputs motor ticks while watching IMU
IMU says: not turning enough / slipping
executor learns: this terrain needs stronger differential drive for same curvature
```

We need many short real-world execution tests from IMU/distance sensors. Each test provides world feedback:

```text
requested curvature/horizon/speed
motor commands actually sent
gyro yaw achieved
acceleration/vibration/stuck signal
distance safety signal
execution score
```

Training should split the objectives.

Planner / SAC reward: exploration value of the chosen guide.

```text
planner_reward = new sensory experience / likely new area
               + safe successful motion
               - collision / near obstacle
               - repeated recent state / overlap
               - choosing guides the executor consistently fails to execute
```

Executor loss/reward: physical execution quality of the requested guide.

```text
executor_loss = curvature_error
              + progress_error
              + slip / stuck penalty
              + vibration penalty
              + unsafe distance penalty
              + motor jerk / violent command penalty
```

The planner should learn **where/how to explore**. The executor should learn **how to make the tractor actually move like the requested guide on the current terrain**.

RND can provide the novelty part of the reward, but it must be filtered by safety and motion signals. RECON-style memory can estimate revisits/overlap. MTG is the architecture inspiration for later replacing `[curvature, horizon, speed]` with multiple generated local trajectory candidates.
