# Traktor Embedded Code

This folder contains the Raspberry Pi rover stack. Current direction is **layered control**, not direct raw motor control from learning code.

```text
scripts / future SAC planner
    -> control safety layer
    -> RoverAPI
    -> raw motor / ultrasonic / IMU / camera drivers
```

## Current Hardware

Motors use an L298N with shared enable PWM:

```text
LEFT_MOTOR_PINS = (16, 1)
RIGHT_MOTOR_PINS = (20, 21)
MOTOR_PWM_PINS = (18, 18)
```

Because both enables share `GPIO18`, the rover can command independent left/right directions, but not independent left/right PWM speeds. Curves are currently approximated with direction/pulse logic or spin/arc primitives.

Ultrasonic mapping:

```text
sensor 1 = right
sensor 2 = left
sensor 3 = front
```

Important: `None` / `NO_ECHO` is treated as unsafe for forward motion because very close or angled obstacles can produce no echo.

## Main Scripts

Manual driving:

```bash
ssh -t rover 'cd /home/yasen/traktor-paper; python3 embedded/scripts/wasd_control.py'
```

Reactive safety-first roaming:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/reactive_roam.py --seconds 30'
```

Guide calibration:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/execute_guide.py --preset shallow_left'
```

SAC-style guide execution through the safety layer:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/execute_sac_guide.py --preset avoid_left'
```

Forward/spin calibration:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/forward_spin_sequence.py'
```

Live ultrasonic visualization from the Mac:

```bash
/Volumes/SSD/v/py/bin/python tools/visualize_rover_ultrasonic.py
```

## Current Safety Layer

`embedded/control/safety.py` is the central safety filter. Higher-level code should call this layer rather than commanding motors directly.

Current behavior:

```text
front stop threshold scales with speed: 10cm to 35cm
front must be >=30cm to resume after turning
turning into a side requires that side sensor to be >=20cm
NO_ECHO is unsafe for front and turn-side checks
stuck detection uses IMU response during commanded forward motion
stuck recovery reverses briefly and turns toward freer side
```

## Source Of Truth Docs

- `embedded/MOTOR_WIRING.md`: current L298N wiring.
- `embedded/TRAKTOR_DIMENSIONS.md`: measured geometry and sensor placement.
- `embedded/control/README.md`: control layering and safety contract.
- `PROJECT.md`: research direction.
- `PLAN.md`: work split and ablations.
