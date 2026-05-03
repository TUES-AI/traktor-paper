# Control Layer

Current architecture:

```text
future SAC planner or scripted behavior
    -> SafetyController
    -> RoverAPI
    -> hardware drivers
```

Higher-level code should output intent, not raw GPIO:

```text
[curvature, horizon, speed]
```

The safety layer is responsible for deciding whether that intent may be executed, interrupted, or replaced by recovery behavior.

## Safety Contract

- Forward motion is blocked if front ultrasonic is below the speed-scaled threshold.
- Front `NO_ECHO` is unsafe, not clear.
- Turning into a side is blocked if that side is below `20cm` or returns `NO_ECHO`.
- After an obstacle, turning can resume forward once front ultrasonic is at least `30cm`.
- IMU response is used to detect commanded motion that does not physically move the rover.
- Stuck recovery reverses briefly and then turns toward the freer side.

## Current Behavior

`ReactiveRoamPolicy` is a non-learning baseline:

```text
if front is safe:
    drive forward, faster if front >100cm
if front is blocked:
    turn toward side with more clearance
if stuck:
    reverse and turn toward freer side
```

Run:

```bash
PYTHONUNBUFFERED=1 python3 embedded/scripts/reactive_roam.py --seconds 30
```

`GuideExecutor` is the deterministic fallback for future SAC planner actions:

```text
SAC action [curvature, horizon, speed]
    -> GuideExecutor
    -> SafetyController
    -> RoverAPI
```

Run one left-avoidance guide:

```bash
PYTHONUNBUFFERED=1 python3 embedded/scripts/execute_sac_guide.py --preset avoid_left
```
