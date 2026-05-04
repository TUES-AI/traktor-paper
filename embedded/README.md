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

Motion note: avoid small repeated motor start/stop ticks. The tracks need continuous motion to overcome static friction, so guide executors should update motor commands while moving and stop only for safety or completion.

## Main Scripts

Manual driving:

```bash
ssh -t rover 'cd /home/yasen/traktor-paper; python3 embedded/scripts/wasd_control.py'
```

Interactive preset motion sequences:

```bash
ssh -t rover 'cd /home/yasen/traktor-paper; python3 embedded/scripts/preset_sequences.py'
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

Continuous left-side obstacle bypass shape:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/left_bypass_shape.py'
```

Two-vector SAC-style local guide with CSV logging:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/execute_two_vector_guide.py --theta1 35 --d1 35 --theta2 -55 --d2 70'
```

Draw two target points locally and execute them:

```bash
/Volumes/SSD/v/py/bin/python tools/draw_two_vector_guide.py
```

Persistent WebSocket control UI with telemetry, WASD, and guide execution:

```bash
/Volumes/SSD/v/py/bin/python tools/rover_ws_control.py
```

This starts `embedded/scripts/rover_ws_server.py` on the Pi over SSH, then uses a persistent WebSocket connection instead of launching new SSH commands for every action.

More robust plain TCP version:

```bash
/Volumes/SSD/v/py/bin/python tools/rover_tcp_control.py --restart-server
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
