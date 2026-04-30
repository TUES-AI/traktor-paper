# Summary: Neural Network-Based Model Predictive Trajectory Tracking Control for Dual-Motor-Driven a Tracked Unmanned Vehicle

**Sensors 2025**, 25, 6877 | **Authors**: Zhai, Yao, Yan, Wang, Liu, Qi | **Published**: Nov 2025

---

## Problem and Core Idea

Tracked unmanned vehicles (TUVs) have coupled longitudinal-lateral dynamics that are hard to model with physics-based approaches. This paper replaces the physics-based prediction model inside MPC with an **LSTM neural network** trained on simulation data (RecurDyn multi-body dynamics). The LSTM predicts next-step vehicle states from 20-step history of states + motor torques. The resulting NN-MPC computes optimal left/right motor torques for trajectory tracking.

---

## Method Details

- **LSTM dynamics model**: Input = 20 timesteps of [X, Y, heading, yaw rate, speed, TL, TR], output = next [X, Y, heading, yaw rate, speed]. LSTM layer (32 hidden units) + FC (16) + dropout + output (5).
- **Training data**: 50,000 samples from RecurDyn multi-body simulation (1000s of random torque commands). Normalized to [-1, 1].
- **NN-MPC framework**: LSTM model is linearized at each timestep around current trajectory point (local time-varying linearization). Quadratic programming solves for optimal torque sequence over prediction horizon (NP=15, NC=3). Output = first control increment applied.
- **Constraints**: lateral error (±1.5m), heading error (±0.52 rad), yaw rate (speed-dependent), torque (motor-limited), terminal constraint η(t+Np)=0.
- **Stability**: Lyapunov analysis shows monotonic decrease of cost function under nominal model; practical stability within O(√ρ) under bounded model error ρ.

---

## Key Results

| Metric | Medium-speed (36 km/h) | High-speed (72 km/h) |
|---|---|---|
| Lateral error RMS reduction | 12.1% vs Dyn-MPC | 80.0% vs Dyn-MPC |
| Heading error RMS reduction | 7.9% vs Dyn-MPC | 14.0% vs Dyn-MPC |
| Lateral error MA reduction | 13.1% vs Dyn-MPC | 80.0% vs Dyn-MPC |
| Heading error MA reduction | 7.5% vs Dyn-MPC | 15.0% vs Dyn-MPC |

- **Field experiment**: Scaled TUV (120kg, 1m track width) at 10.8 km/h double-lane-change. Max lateral error ~1.4m, heading error ~0.16 rad. Real-time: 0.05s control loop on 16-core industrial PC.
- **Parameter sensitivity**: 32 hidden units + 20-step history was optimal (diminishing returns beyond).
- Model comparison: LSTM (MSE=0.0429) > DNN (0.0995) > physics-based dynamic model (0.1002).

---

## Relevance to This Project

- **Direct hardware match**: Tracked vehicle with dual-motor differential steering — same kinematic class as our tractor. Results on coupled longitudinal-lateral dynamics directly transferable.
- **LSTM for dynamics modeling**: Shows that a very small LSTM (32 hidden units, ~5700 params) can capture complex tracked vehicle dynamics. This would run trivially on a Pi.
- **NN-MPC trajectory tracking**: Proves NN-based MPC works in field conditions on a TUV. Could replace or augment our current PPO approach for the low-level trajectory tracking part of the two-network architecture.
- **Real-world validation**: Field experiment at 10.8 km/h with acceptable tracking error demonstrates practical deployability.
- **Training on simulation + transfer**: They train on RecurDyn multi-body sim, deploy on real scaled vehicle — similar to our sim-to-real setup.

---

## Concrete Experiments to Run

1. Train an LSTM dynamics model for our tractor: collect trajectory data (position, heading, speed, motor commands), compare prediction accuracy vs our current model. Use their architecture (32-16 LSTM-FC) as starting point.
2. Implement NN-MPC trajectory tracking on tractor: replace physics-based MPC or PPO policy with LSTM-in-the-loop MPC for waypoint following. Measure tracking error on a reference path.
3. Ablate network size: test if LSTM with 8-16 hidden units (Pi-suitable) still achieves useful prediction accuracy for control.
4. Compare LSTM model + MPC vs our PPO policy on same trajectory tracking task — which gives smoother, more accurate tracking?
5. Online model update experiment: after deployment, fine-tune the LSTM on real tractor data to reduce sim-to-real gap (the paper identifies this as future work — we can do it now).

---

## Risks / Open Questions

- The paper trains exclusively on RecurDyn simulation data; real-world validation shows degraded accuracy (max lateral error 1.4m vs simulated ~0.3m at low speed). Sim-to-real gap is significant.
- No adaptation — the LSTM model is trained once and frozen. Cannot handle changing terrain/conditions mid-run. Would need to combine with online adaptation approach (see meta-RL paper summary).
- Field experiment at only 10.8 km/h (low speed); high-speed validation only in simulation.
- Real-time demonstration was on a 16-core industrial PC, not embedded hardware. Control loop at 0.05s (20Hz) should be achievable on Pi with the small LSTM, but needs benchmarking.
- No coverage planning — pure trajectory tracking. A separate higher-level planner is assumed.
- LSTM input uses 20 timesteps at 0.1s intervals = 2 second history. Tractor may need longer horizon for terrain transitions.
