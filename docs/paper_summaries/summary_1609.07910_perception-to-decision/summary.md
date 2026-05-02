# Summary: From Perception to Decision

**arXiv**: 1609.07910 | **Title**: From Perception to Decision: A Data-driven Approach to End-to-end Motion Planning for Autonomous Ground Robots | **Authors**: Mark Pfeiffer, Michael Schaeuble, Juan Nieto, Roland Siegwart, Cesar Cadena

---

## Problem and Core Idea

This paper targets mapless, target-oriented navigation for a differential-drive ground robot. Instead of building a full map and running a classical global planner at deployment time, the robot receives only local 2D laser range readings and a relative target position, then directly outputs steering commands.

The core idea is supervised imitation learning: use a classical ROS planner in simulation as the expert, record local sensor observations, relative goal information, and expert velocity commands, then train a neural network to imitate the expert. At test time, the learned policy uses only local perception and the relative target, not the global map.

---

## Method Details

- **Input**: raw 2D laser range vector and relative target position in robot-centric polar coordinates.
- **Output**: differential-drive steering command `(v, omega)`.
- **Model**: CNN over laser scan data, with two residual blocks for feature extraction.
- **Fusion**: laser features are concatenated with target information and processed through fully connected layers.
- **Variants**: `CNN_smallFC` uses fully connected sizes `(256, 256, 256)`; `CNN_bigFC` uses `(1024, 1024, 512)`.
- **Training data**: generated in Stage simulation using ROS navigation stack as expert.
- **Expert**: grid-based Dijkstra global planner plus DWA local planner.
- **Loss**: supervised action imitation, minimizing absolute difference between predicted steering command and expert steering command.
- **Deployment**: frame-by-frame policy; no memory, no map, no recurrent state.

Important limitation stated by the authors: the policy is still fundamentally a local planner. In complex environments it benefits from external target/waypoint selection and can fail in convex dead ends because it has no memory or global map.

---

## Key Results

- The model learns target-oriented navigation and collision avoidance from simulated expert demonstrations.
- The trained policy transfers to unseen simulated maps and real-world environments.
- CPU-only query time is about **4.3 ms** on an Intel i7-4810MQ, so inference is predictable and fast.
- Training used about **6000 trajectories** in one simulated map and **4000 trajectories** in a more complex map, producing roughly **4.3M** input/output tuples total.
- Training took about **8 hours** for **2M training steps** on a GTX 980 Ti.
- In real-world tests, the larger fully connected model reduced manual joystick interventions compared with the smaller model.
- The robot reacted to suddenly appearing obstacles using only local range observations.

---

## Relevance to This Project

- This is much closer to the rover than MPNet/PathNet because it does **not** require a full map at deployment.
- It gives a clean architecture pattern: local sensor encoder + relative goal + direct motor command output.
- It shows that a privileged classical planner can be used only during training, while the deployed student policy sees only local observations.
- The rover can replace the 1080-beam Hokuyo scan with camera features plus 3 ultrasonic distances, though this is a much harder observation space.
- The paper supports a sim-to-real story: train in simulation, deploy on a real differential-drive robot.
- It also exposes the main gap we need to solve: memory. Their feed-forward model cannot escape dead ends once inside them; our rover likely needs history via GRU/LSTM or a short local map.

---

## Concrete Experiments to Run Next

1. Implement a mapless imitation baseline in `rover_coverage_env.py`: observation = 3 range sensors + relative goal angle/distance; action = wheel command or discrete steering action.
2. Generate expert demonstrations from a privileged planner in sim, but train the student only on local observations.
3. Compare feed-forward MLP versus GRU policy to test whether memory helps with dead ends and partial observability.
4. Add camera features once the range-only student works: small CNN image encoder + ultrasonic MLP + goal vector.
5. Evaluate sim-to-real transfer on the rover with fixed local target points and emergency stop constraints.
6. Use the expert planner only for training/evaluation, not for onboard execution.

---

## Risks / Open Questions

- Their sensor is a high-resolution 270-degree lidar; our rover currently has only 3 ultrasonic sensors plus camera.
- Feed-forward policies can be myopic and get stuck in dead ends.
- Pure imitation inherits expert biases and suffers distribution mismatch when the student drifts off expert trajectories.
- The policy requires a relative target; for coverage we still need a target-generation layer.
- Sim-to-real is harder with cheap ultrasonic and camera noise than with a Hokuyo lidar.
- No explicit safety guarantee exists; physical testing needs speed limits and hard obstacle stops.
