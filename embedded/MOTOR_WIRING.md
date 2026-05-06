# Motor Wiring

Current working setup: Raspberry Pi drives a standard L298N motor driver.

## Pin Map

```text
L298N ENA -> Raspberry Pi GPIO19
L298N ENB -> Raspberry Pi GPIO18

Left motor on L298N channel A:
L298N IN1 -> Raspberry Pi GPIO16
L298N IN2 -> Raspberry Pi GPIO1

Right motor on L298N channel B:
L298N IN3 -> Raspberry Pi GPIO20
L298N IN4 -> Raspberry Pi GPIO21
```

This is encoded in `embedded/hardware_pins.py`:

```python
LEFT_MOTOR_PINS = (16, 1)
RIGHT_MOTOR_PINS = (20, 21)
MOTOR_PWM_PINS = (19, 18)
RIGHT_MOTOR_PWM_BETA = 1.0
```

`MOTOR_PWM_PINS = (19, 18)` exposes separate left/right PWM. Current controllers still usually request `left_pwm = right_pwm = value`; the separate pins and `RIGHT_MOTOR_PWM_BETA = 1.0` are hooks for later calibration/learning.

## Direction Logic

For each L298N channel:

```text
forward:  INx1 = HIGH, INx2 = LOW
backward: INx1 = LOW,  INx2 = HIGH
stop:     INx1 = LOW,  INx2 = LOW
brake:    INx1 = HIGH, INx2 = HIGH
```

With the current wiring, logical commands are normal:

```text
left forward  -> GPIO16 HIGH, GPIO1 LOW
left backward -> GPIO16 LOW,  GPIO1 HIGH

right forward  -> GPIO20 HIGH, GPIO21 LOW
right backward -> GPIO20 LOW,  GPIO21 HIGH
```

## WASD Control

Run from a terminal with TTY forwarding:

```bash
ssh -t rover 'cd /home/yasen/traktor-paper; python3 embedded/scripts/wasd_control.py'
```

Keys:

```text
w = forward
s = backward
a = spin left
d = spin right
x = stop
q = quit
```

The script is hold-to-run over SSH. SSH does not send real key-release events, so the script stops the motors if no repeated key arrives within a short timeout.

## Notes

- Do not introduce arc/curved turns through unequal PWM until the safety implications are handled explicitly.
- If one side stops reversing, check IN wires first, not the Python direction mapping.
- `set_states()` in `drivers/motor/hbridge.py` is for raw H-bridge debugging.
