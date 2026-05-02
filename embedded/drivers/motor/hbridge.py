"""Dual H-bridge motor driver using Raspberry Pi BCM GPIO pins.

Each motor uses two direction pins. The two enable/PWM inputs may either use
separate GPIO pins or share one GPIO pin through a splitter cable:

- `forward`: IN1 high, IN2 low
- `backward`: IN1 low, IN2 high
- `stop`: IN1 low, IN2 low and PWM duty cycle 0

Speed is a PWM duty cycle from 0 to 100 percent. With shared PWM, both motors
receive the same enable signal, so only direction can differ per motor.
"""

import RPi.GPIO as GPIO


class DualHBridgeMotorDriver:
    _FORWARD = (1, 0)
    _REVERSE = (0, 1)
    _STOP = (0, 0)

    def __init__(
        self,
        left_in1=20,
        left_in2=21,
        right_in1=16,
        right_in2=12,
        left_pwm_pin=19,
        right_pwm_pin=13,
        pwm_frequency_hz=100,
    ):
        self.left_in1 = left_in1
        self.left_in2 = left_in2
        self.right_in1 = right_in1
        self.right_in2 = right_in2
        self.left_pwm_pin = left_pwm_pin
        self.right_pwm_pin = right_pwm_pin
        self.pwm_frequency_hz = pwm_frequency_hz

        self.control_pins = (left_in1, left_in2, right_in1, right_in2)
        self.pwm_pins = (left_pwm_pin, right_pwm_pin)
        self.pins = tuple(dict.fromkeys(self.control_pins + self.pwm_pins))

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        self._left_pwm = GPIO.PWM(self.left_pwm_pin, self.pwm_frequency_hz)
        self._right_pwm = self._left_pwm if self.left_pwm_pin == self.right_pwm_pin else GPIO.PWM(
            self.right_pwm_pin,
            self.pwm_frequency_hz,
        )
        self._left_pwm.start(0)
        if self._right_pwm is not self._left_pwm:
            self._right_pwm.start(0)

    def _normalize_direction(self, direction):
        value = str(direction).strip().lower()
        if value in ('forward', 'f', '1', '+1'):
            return self._FORWARD, 'forward'
        if value in ('backward', 'reverse', 'back', 'b', 'r', '-1'):
            return self._REVERSE, 'backward'
        if value in ('stop', 's', '0', 'brake'):
            return self._STOP, 'stop'
        raise ValueError(f'Invalid motor direction: {direction}')

    def _normalize_side(self, side):
        value = str(side).strip().lower()
        if value in ('left', 'l', 'm1', 'left_motor'):
            return 'left'
        if value in ('right', 'r', 'm2', 'right_motor'):
            return 'right'
        raise ValueError(f'Invalid motor side: {side}')

    def _normalize_speed(self, speed):
        duty = 100.0 if speed is None else float(speed)
        if duty < 0 or duty > 100:
            raise ValueError(f'Speed must be between 0 and 100, got {speed}')
        return duty

    def _set_left_state(self, state):
        in1, in2 = state
        GPIO.output(self.left_in1, GPIO.HIGH if in1 else GPIO.LOW)
        GPIO.output(self.left_in2, GPIO.HIGH if in2 else GPIO.LOW)

    def _set_right_state(self, state):
        in1, in2 = state
        GPIO.output(self.right_in1, GPIO.HIGH if in1 else GPIO.LOW)
        GPIO.output(self.right_in2, GPIO.HIGH if in2 else GPIO.LOW)

    def set_motor(self, side, direction, speed=100):
        """Set one motor direction and speed."""
        side_name = self._normalize_side(side)
        state, normalized_direction = self._normalize_direction(direction)
        duty = 0.0 if normalized_direction == 'stop' else self._normalize_speed(speed)

        if side_name == 'left':
            self._set_left_state(state)
            self._left_pwm.ChangeDutyCycle(duty)
        else:
            self._set_right_state(state)
            self._right_pwm.ChangeDutyCycle(duty)

        return {'side': side_name, 'direction': normalized_direction, 'speed': duty}

    def drive(self, left_direction, right_direction, left_speed=100, right_speed=100):
        """Drive both motors in one GPIO/PWM update."""
        left_state, left_norm = self._normalize_direction(left_direction)
        right_state, right_norm = self._normalize_direction(right_direction)
        left_duty = 0.0 if left_norm == 'stop' else self._normalize_speed(left_speed)
        right_duty = 0.0 if right_norm == 'stop' else self._normalize_speed(right_speed)

        self._set_left_state(left_state)
        self._set_right_state(right_state)
        if self._right_pwm is self._left_pwm:
            self._left_pwm.ChangeDutyCycle(max(left_duty, right_duty))
        else:
            self._left_pwm.ChangeDutyCycle(left_duty)
            self._right_pwm.ChangeDutyCycle(right_duty)

        return {
            'left': left_norm,
            'right': right_norm,
            'left_speed': left_duty,
            'right_speed': right_duty,
        }

    def set_speed(self, side, speed):
        """Change PWM duty cycle for one motor without changing direction pins."""
        side_name = self._normalize_side(side)
        duty = self._normalize_speed(speed)
        if side_name == 'left':
            self._left_pwm.ChangeDutyCycle(duty)
        else:
            self._right_pwm.ChangeDutyCycle(duty)
        return {'side': side_name, 'speed': duty}

    def set_speeds(self, left_speed, right_speed):
        left_duty = self._normalize_speed(left_speed)
        right_duty = self._normalize_speed(right_speed)
        if self._right_pwm is self._left_pwm:
            self._left_pwm.ChangeDutyCycle(max(left_duty, right_duty))
        else:
            self._left_pwm.ChangeDutyCycle(left_duty)
            self._right_pwm.ChangeDutyCycle(right_duty)
        return {'left_speed': left_duty, 'right_speed': right_duty}

    def set_states(self, left_in1, left_in2, right_in1, right_in2, left_speed=100, right_speed=100):
        """Set raw H-bridge states directly for low-level hardware debugging."""
        left_state = (1 if left_in1 else 0, 1 if left_in2 else 0)
        right_state = (1 if right_in1 else 0, 1 if right_in2 else 0)
        left_duty = 0.0 if left_state == self._STOP else self._normalize_speed(left_speed)
        right_duty = 0.0 if right_state == self._STOP else self._normalize_speed(right_speed)

        self._set_left_state(left_state)
        self._set_right_state(right_state)
        if self._right_pwm is self._left_pwm:
            self._left_pwm.ChangeDutyCycle(max(left_duty, right_duty))
        else:
            self._left_pwm.ChangeDutyCycle(left_duty)
            self._right_pwm.ChangeDutyCycle(right_duty)

    def set_both_forward(self, speed=100):
        return self.drive('forward', 'forward', left_speed=speed, right_speed=speed)

    def set_both_reverse(self, speed=100):
        return self.drive('backward', 'backward', left_speed=speed, right_speed=speed)

    def stop(self):
        """Stop both motors and set both PWM outputs to 0 percent."""
        return self.drive('stop', 'stop', left_speed=0, right_speed=0)

    def cleanup(self):
        """Stop motors, stop PWM, and release only the pins owned by this driver."""
        try:
            self.stop()
        finally:
            self._left_pwm.stop()
            if self._right_pwm is not self._left_pwm:
                self._right_pwm.stop()
            GPIO.cleanup(self.pins)
