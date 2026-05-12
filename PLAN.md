# Current rover plan

This is a working plan from the latest real-rover discussion. It is not the final paper method.

## Boundary we want to keep

The SAC policy should learn exploration direction, not safety.

Allowed SAC-facing signals/reward:

- camera-derived novelty / memory signal
- left, right, and front distance as simple obstacle/open-space context
- executed progress / coverage-style reward
- possibly a small penalty for choosing a direction that is obviously blocked by the distance sensors

Not allowed for SAC for now:

- contact/stall classifier as policy input
- IMU safety classifier as policy input
- motor-active safety state as policy input
- crash labels as an auxiliary SAC/model loss
- learned navigation/safety mixed into the planner

Safety must stay in a separate deterministic executor layer. A future tiny MLP/LSTM safety smoother is acceptable only inside that safety layer, not as part of SAC's policy learning.

## Next safety work

Main hardware problem: the rover sometimes keeps driving forward into a wall because distance sensing misses or is late.

Add a deterministic contact/stall event first:

```text
forward motor command active
+ IMU says body is not moving / impact-like / stalled
+ command has lasted longer than a short grace period
= contact_or_stall
```

Initial use:

- stop motors immediately
- optionally reverse / recover
- log `contact_or_stall` and a simple `stall_score`
- do not expose this to SAC yet

Later, after logs exist, train a tiny MLP/LSTM as a safety smoother/predictor that fires slightly earlier than the deterministic rule. This model still lives only in the safety layer.

## Training environment plan

The apartment is small enough that a 100--200 step run can cover it fully. Longer single episodes become less useful because novelty saturates.

Use repeated short real episodes instead of artificial random memory deletion:

```text
run 50--100 steps
save SAC checkpoint and logs
manually reposition rover
reset PCVM/novelty memory
continue training from the saved SAC checkpoint
repeat
```

First implementation can save/load the normal Stable-Baselines SAC zip. If needed later, explicitly save replay buffer and optimizer state too. Do not overbuild this until the simple episodic workflow is tested.

## Sensor/data formatting plan

Current PCVM theta-front SAC input is 143 dims:

```text
128 PCVM latent
2 novelty/surprise
6 candidate scores
7 raw tail
```

The raw tail is:

```text
left_distance_norm, right_distance_norm, front_distance_norm,
yaw_rate_norm, 0.0 dummy,
last_action_0, last_action_1
```

Distance normalization:

```text
distance_norm = clamp(cm / 400, 0, 1)
None/no echo = 1.0
```

Yaw normalization:

```text
yaw_rate_norm = clamp(gyro_z / 180 deg/s, -1, 1)
```

Accelerometer is not currently in SAC/PCVM input.

For now, do not add many derived features. If adding anything, add it to logs and safety first, not SAC. The minimal safety-side additions are:

- `contact_or_stall`
- `stall_score`
- maybe `yaw_achieved` for turn diagnostics

## Camera timing

Keep the current main observation choice:

```text
execute action -> stop/settle -> capture camera frame
```

This is defensible because moving frames are often blurry and harder to compare for novelty. Continuous/moving-frame streams can be logged later as diagnostics, but should not replace the main observation path yet.

## Paper notes to remember

Paper-worthy or maybe paper-worthy:

- action abstraction matters: scalar relative heading plus world-bounded forward execution worked better than timid short local targets
- world-bounded execution is a meaningful interface: the policy chooses direction, while onboard sensor feedback determines realized travel horizon
- executed-action memory is likely better than requested-action memory because safety clipping/recovery changes what actually happened
- dual memory is worth testing, but not yet a final architecture claim

Not paper-worthy as claims yet:

- exact speed/theta tuning
- wall hits as a result by themselves
- final reward architecture
- final visual-memory architecture

## Immediate next implementation order

1. Add deterministic contact/stall detector to safety/executor.
2. Log contact/stall details, but keep them out of SAC input.
3. Add a simple episodic training wrapper: run, save, wait for reposition, reset novelty memory, continue SAC.
4. Keep theta-front learning mechanics unchanged for the next real test.
