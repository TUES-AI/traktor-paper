# Summary: Reinforced Imitation for Map-less Navigation

**arXiv**: 1805.07095 | **Title**: Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations | **Authors**: M. Pfeiffer, S. Shukla, M. Turchetta, C. Cadena, A. Krause, R. Siegwart, J. Nieto | **Venue**: IEEE RA-L 2018

---

## Problem and Core Idea

This paper studies target-driven mapless navigation where the robot does not use a global environment map at deployment. The policy maps local range measurements and relative target position directly to translational and rotational velocity commands.

The core idea is reinforced imitation: first pretrain the navigation policy with imitation learning from expert demonstrations, then improve and harden it with reinforcement learning. Imitation gives the agent a useful initial behavior and reduces RL sample complexity; RL lets the policy recover from states not covered by demonstrations and improves robustness.

---

## Method Details

- **Input**: 2D laser range findings and relative target position in robot-local coordinates.
- **Laser preprocessing**: 1080 lidar readings are min-pooled into 36 values; min pooling preserves nearby obstacle risk.
- **State size**: ASL architecture uses 38 inputs: 36 pooled laser values plus relative target distance/angle.
- **Output**: continuous translational and rotational velocity `(v, omega)`.
- **Policy model**: three fully connected layers with tanh activations.
- **Actor architecture from repo**: ASL hidden sizes are `(1000, 300, 100)`; alternative Tai-style architecture uses `(512, 512, 512)` with ReLU.
- **Training phase 1**: supervised imitation learning from ROS `move_base` expert demonstrations.
- **Training phase 2**: CPO/TRPO-style reinforcement learning initialized from the imitation weights.
- **Safety**: collision avoidance is encoded as a constrained optimization problem with CPO instead of only a fixed negative collision reward.
- **Rewards tested**: sparse success reward, Euclidean distance progress, and shortest-path-distance progress.
- **Training scale**: RL batches use about 60k timesteps per epoch; 1000 iterations are about 50 hours in accelerated Stage simulation.

The repo is old ROS/TensorFlow/Python 2 code, but its algorithmic structure is directly useful: `jump_start` loads imitation weights, then CPO optimizes with safety-cost constraints.

---

## Key Results

- Pretraining with imitation reduces RL training time by about **80%** to reach similar final performance.
- Even **10 expert demonstration trajectories** can significantly improve RL exploration.
- With pretraining, sparse reward can perform similarly to richer shaped rewards.
- Best reported unseen-simulation generalization is **79% success** for `c_1000 + CPO`.
- R-IL models after only **200 RL iterations** reach similar performance to pure RL trained from scratch much longer.
- CPO handles collision avoidance better than fixed collision penalty TRPO because safety is a constraint, not a tuned reward tradeoff.
- In real-world tests, `c_1000 + CPO` had **0 crashes** and **0 manual joystick distance** over five runs, with path length about **1.19x** and time about **1.04x** relative to `move_base`.
- A sparse-reward model pretrained from only 10 simple demonstrations also achieved **0 crashes** in real-world testing, though it was slower.

---

## Relevance to This Project

- This is the best match so far for the rover's real constraint: no full map, only local sensors and a target/coverage objective.
- It gives a practical training recipe: privileged planner generates demonstrations in sim, student policy only sees local observations, then RL improves robustness.
- The rover can use PPO instead of CPO initially, but the paper strongly suggests separating safety as a constraint or at least tracking collision cost separately.
- It supports the project novelty: combine a learned mapless local planner with online adaptation for terrain/control mismatch.
- It also gives a strong experimental baseline: imitation-only, RL-only, and imitation+RL.
- For Pi constraints, the feed-forward MLP is lightweight. A rover version with 3 ultrasonic sensors plus a tiny image encoder should fit.

---

## Concrete Experiments to Run Next

1. Build three baselines in the current sim: imitation-only, PPO-only, and imitation-pretrained PPO.
2. Use privileged A*/Dijkstra/RRT or scripted expert as demonstration generator, but hide the map from the student.
3. Student observation should start small: `front/left/right distance`, relative target distance, relative target angle, previous action.
4. Add a GRU after the first baseline to handle partial observability and dead ends.
5. Add camera input only after the range-only pipeline is working and measured.
6. Evaluate success, crash rate, timeout rate, path length, control smoothness, and sim-to-real behavior on the Pi.

---

## Risks / Open Questions

- CPO is not trivial to implement cleanly today; PPO with explicit safety metrics may be the pragmatic first step.
- Their lidar observation is much richer than our 3 ultrasonic readings.
- Sparse sensors may need memory, optical flow, or a learned local occupancy representation from camera frames.
- The method still needs a target signal. Coverage requires a policy for choosing local targets, not only reaching a target.
- Sim-to-real transfer depends heavily on whether our simulated camera/ultrasonic noise resembles the rover.
- Real-world RL is unsafe unless we constrain speed, add emergency stop, and keep early tests in simple arenas.
