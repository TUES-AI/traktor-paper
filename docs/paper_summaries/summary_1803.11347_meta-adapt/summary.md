# Summary: Learning to Adapt in Dynamic, Real-World Environments through Meta-Reinforcement Learning

**arXiv**: 1803.11347 | **Authors**: Nagabandi, Clavera, Liu, Fearing, Abbeel, Levine, Finn (UC Berkeley) | **2018**

---

## Problem and Core Idea

Model-based RL suffers from needing a globally accurate dynamics model — impractical when dynamics change (terrain, damage, disturbances). This paper meta-trains a neural network dynamics model so it can be **adapted online** using just the last M timesteps of experience. Two instantiations: **GrBAL** (gradient-based, MAML-style) and **ReBAL** (recurrence-based, LSTM hidden state).

---

## Method Details

- **Meta-objective**: sample trajectory segments of length M+K, use past M timesteps to adapt model parameters $\theta \to \theta'$, evaluate prediction loss on next K timesteps, backprop through the adaptation step.
- **GrBAL**: inner loop = 1 gradient descent step on model params using past M datapoints. Meta-learns the initial parameters from which this single step is maximally effective.
- **ReBAL**: inner loop = LSTM ingests past M transitions, outputs adapted hidden state; meta-learns the LSTM weights.
- **Controller**: MPC with either MPPI (sim) or random shooting (real robot). After each action, reset model to $\theta_*$ and re-adapt from scratch — prevents error accumulation.
- **Network**: 3 hidden layers of 512 units, ReLU. Real robot: 24-dim state, 2-dim action (velocity setpoints).
- **Training data**: ~1.5-3 hours equivalent per domain (1000x less than model-free meta-RL).

---

## Key Results

| Metric | Result |
|---|---|
| Sample efficiency | 1000x less data than model-free meta-RL (MAML-RL) |
| HC disabled joint (F.A.) | Surpasses MB oracle; GrBAL > ReBAL > MB+DE > MB |
| HC pier | Surpasses MB oracle — adaptation to rapidly-changing block dynamics |
| Ant crippled leg (gen.) | Best among all methods; adapts to unseen crippled legs |
| Real robot (6-legged millirobot) | GrBAL substantially outperforms MB and MB+DE on missing leg, novel terrains, slopes, pose errors, payloads |
| Training terrains | Comparable to MB (adaptation doesn't hurt in-distribution) |

---

## Relevance to This Project

- **Direct connection**: Model-based RL with online adaptation of a small dynamics model — exactly the two-network idea (big network = meta-learned prior, small = fast-adapted local model).
- **Real robot validation**: Method was deployed on a real legged robot with limited compute (streaming at 10Hz), proving practicality on Pi-class hardware.
- **Terrain adaptation**: Robot adapted to carpet/styrofoam/turf (training) and novel slopes/missing leg (test) — analogous to our tractor adapting to different field conditions.
- **Network size**: 3x512 layers is modest; could run on Pi. 24-dim state + 2-dim action is comparable to our tractor setup.
- **PPO compatibility**: While they use MPC, the meta-learning framework for online adaptation is orthogonal to the planning/control method — could be combined with PPO-based policies too.

---

## Concrete Experiments to Run

1. Port GrBAL to our tractor: meta-train dynamics model on trajectory data from multiple terrain types (grass, gravel, dirt, pavement). Test adaptation when transitioning between terrains mid-run.
2. Compare our current PPO approach against GrBAL-on-MPC baseline on coverage path planning — does online adaptation help when terrain varies within a single field run?
3. Small network ablation: find the minimum model size (layers/units) that still enables effective online adaptation — target Pi inference < 10ms.
4. Multi-timescale: combine GrBAL-adapted low-level dynamics model with a slower-updating high-level planner (the "two-network" idea from project.md).
5. Test ReBAL vs GrBAL on tractor — ReBAL might capture longer temporal dependencies (terrain transitions that unfold over seconds).
6. Real-world data collection protocol: drive tractor on N terrains with random control inputs + safety guard, collect ~30min per terrain, meta-train offline.

---

## Risks / Open Questions

- The method uses MPC (not PPO) — adapting the meta-learning framework to work with policy gradient methods is non-trivial.
- Meta-training requires multi-task data collection across diverse environments; for us that means recording data on many field conditions before we see any benefit.
- Real robot experiments in the paper used motion capture (ground truth pose) — our tractor uses wheel odometry + IMU. Adaptation may need to compensate for noisier state estimates.
- The paper's adaptation horizon M=10-20 timesteps (~1-2 seconds at 10Hz). For tractor field navigation, terrain changes might be slower (meters, not centimeters) — need to verify M/K hyperparameters suit our domain.
- No coverage planning — the paper focuses on trajectory tracking, not exploration. Combining coverage path planning with online adaptation is novel but unexplored.
