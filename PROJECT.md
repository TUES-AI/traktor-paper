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

---

## Experiments

### Session 1 — Manual drive, ~30–45 minutes

**Purpose:** Validate VMM perceptual quality on real camera input, independent of RL.

**Protocol:**
- Drive the rover manually through every room in the apartment, entering each room at least twice — once fresh, once as a revisit.
- One pass with normal lighting, one pass with one room's light off or curtains drawn (lighting-variation false-novelty test).
- Total: 3–4 rooms × 2 passes × 2 lighting conditions ≈ 20–30 minutes of logged data.
- Place a unique visible object in each room before starting (red chair cushion, plant, poster) — makes room labels unambiguous in cluster plots and novelty spikes obvious to reviewers.

**Figures:**

**Figure A — Novelty score over time (MobileNet-VMM vs classical baseline)**
Dual time-series plot. X-axis: time in seconds. Y-axis: novelty score. Annotated with vertical lines at room transitions ("entered kitchen t=34s", "re-entered hallway t=71s"). MobileNet should spike at transitions and decay on revisits faster than the classical baseline.

**Figure B — Embedding cluster separability**
Every 5th frame from the manual drive. Compute MobileNet-VMM embeddings and classical embeddings, run UMAP or t-SNE on each, color by room label (ground truth known from manual drive). MobileNet clusters should separate by room more cleanly. Two panels side by side, four colors for four rooms.

---

### Session 2 — Autonomous RL deployment, ~1–2 hours

**Purpose:** Demonstrate that the trained system actually explores the apartment autonomously.

**Setup:** Deploy the sim-trained SAC planner with the MobileNet-VMM novelty signal and the deterministic arc controller. Zero-shot sim-to-real — do not retrain on the rover. Run 3–5 episodes of fixed duration (3–5 minutes per episode, enough to visit 2–3 rooms).

**Log:** All sensor data, embeddings, novelty scores, motor commands, and an approximate top-down trajectory reconstructed from IMU integration + motor commands (will drift, but visually interpretable for a single episode).

**Figures:**

**Figure C — Autonomous trajectory with novelty overlay**
2D approximate top-down path of the rover through the apartment, colored by VMM novelty score at each point (cool = low novelty, warm = high). Warm patches in rooms entered for the first time, cool patches on return paths. Annotate with "novel place found" markers where the metric fires.

**Figure D — Novel-places-found over time, autonomous run**
X-axis: time in seconds. Y-axis: cumulative novel-places-found count. Full system vs random-walk baseline (forward + random turns, no RL). A step-function that climbs faster than random is sufficient — does not need to beat frontier exploration.

**Practical notes:**
- Run the emergency stop on a separate thread, tested before starting. One stuck rover wastes 20 minutes.
- If zero-shot transfer performs poorly, report it honestly as a limitation: "zero-shot transfer shows reduced exploration speed relative to sim; we attribute this to the visual domain gap between rendered and real apartment frames." This is an honest negative finding that fits RLxF.
- Three successful episodes with clean logs beats one perfect episode. Run multiple short episodes rather than one long one.
