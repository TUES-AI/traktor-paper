# Traktor embedded stack

This folder contains the Raspberry Pi runtime stack for the visionless TWF rover.

```text
TWF planner / scripts
    -> safety + local-target executor
    -> RoverAPI
    -> motor / ultrasonic / IMU drivers
```

No runtime camera path is kept here. Offline image clustering, if needed for paper analysis, lives in `tools/` and uses saved images only.

## Hardware

Motors use an L298N with independent enable PWM pins:

```text
LEFT_MOTOR_PINS = (16, 1)
RIGHT_MOTOR_PINS = (20, 21)
MOTOR_PWM_PINS = (19, 18)
LEFT_MOTOR_PWM_BETA = 0.85
RIGHT_MOTOR_PWM_BETA = 1.0
```

Ultrasonic mapping:

```text
sensor 1 = right
sensor 2 = left
sensor 3 = front
```

`None` / `NO_ECHO` is treated as unsafe for forward motion because close or angled obstacles can produce no echo.

## Main scripts

Manual driving:

```bash
ssh -t rover 'cd /home/yasen/traktor-paper; python3 embedded/scripts/wasd_control.py'
```

Reactive safety-first roaming:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/reactive_roam.py --seconds 30'
```

Visionless TWF real-run script:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 TWF_TRAIN_STEPS=100 bash embedded/scripts/train_real_twf_sac.sh'
```

Guide/executor calibration:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/execute_local_target.py --theta 30'
```

Forward/spin calibration:

```bash
ssh rover 'cd /home/yasen/traktor-paper; PYTHONUNBUFFERED=1 python3 embedded/scripts/forward_spin_sequence.py'
```

## Safety layer

`embedded/control/safety.py` is the central safety filter. Higher-level code should call this layer rather than commanding motors directly.

Current behavior:

```text
front stop threshold scales with speed
front must be clear before forward motion
turning into a side requires that side sensor to be clear
NO_ECHO is unsafe for front and turn-side checks
stuck detection uses IMU response during commanded forward motion
stuck recovery reverses briefly and turns toward freer side
```

## Source-of-truth docs

- `PROJECT.md`: research direction.
- `embedded/MOTOR_WIRING.md`: current L298N wiring.
- `embedded/TRAKTOR_DIMENSIONS.md`: measured geometry and sensor placement.
- `embedded/control/README.md`: control layering and safety contract.
