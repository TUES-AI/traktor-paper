# Guide Executor Calibration

Purpose: test whether the tractor physically executes local guide commands.

Guide format:

```text
[curvature, horizon, speed]
```

Meaning:

```text
curvature < 0: right arc
curvature = 0: straight
curvature > 0: left arc
horizon: seconds to execute
speed: shared PWM duty cycle
```

Current wiring shares one PWM enable for both L298N channels, so shallow curves are created by pulsing the inner track on/off while the outer track stays active.

Run one test from the rover:

```bash
cd /home/yasen/traktor-paper
python3 embedded/scripts/execute_guide.py --preset shallow_left
```

Useful presets:

```text
straight
shallow_left
shallow_right
hard_left
hard_right
reverse
spin_left
spin_right
```

Write down after each run:

```text
preset:
curvature:
horizon_s:
speed_pct:
moved forward distance estimate:
yaw direction and rough angle:
did it match intended arc:
slip/stuck/vibration:
obstacle/safety issue:
notes:
```

The learnable executor will later use the same fields, but from IMU/distance sensors instead of human notes.

Current safety defaults are enforced by `embedded/control/safety.py` and should be used by higher-level behaviors before raw motor commands.

```text
front stop threshold: speed-scaled 10cm to 35cm
front clear to resume after turning: 30cm
turn-side stop: 20cm on the side being turned toward
NO_ECHO: unsafe for front / turn-side checks
```

For full reactive behavior, use:

```bash
python3 embedded/scripts/reactive_roam.py --seconds 30
```
