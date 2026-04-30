"""High-level Raspberry Pi rover API.

`RoverAPI` owns all hardware drivers and provides one place for application code
to read sensors, capture camera frames, and command the motors.
"""

from drivers.camera.picam2 import PiCam2FrameDriver
from drivers.gps.provider import GPSProvider
from drivers.motor.hbridge import DualHBridgeMotorDriver
from drivers.sensors.ultrasonic_array import UltrasonicArray
from hardware_pins import (
    GPS_BAUD,
    GPS_FALLBACK_FILE,
    GPS_PORT,
    LEFT_MOTOR_PINS,
    MOTOR_PWM_FREQUENCY_HZ,
    MOTOR_PWM_PINS,
    RIGHT_MOTOR_PINS,
    ULTRASONIC_1_PINS,
    ULTRASONIC_2_PINS,
    ULTRASONIC_3_PINS,
)


class RoverAPI:
    def __init__(
        self,
        gps_port=GPS_PORT,
        gps_baud=GPS_BAUD,
        gps_fallback_file=GPS_FALLBACK_FILE,
        left_motor_pins=LEFT_MOTOR_PINS,
        right_motor_pins=RIGHT_MOTOR_PINS,
        motor_pwm_pins=MOTOR_PWM_PINS,
        motor_pwm_frequency_hz=MOTOR_PWM_FREQUENCY_HZ,
        ultrasonic1_pins=ULTRASONIC_1_PINS,
        ultrasonic2_pins=ULTRASONIC_2_PINS,
        ultrasonic3_pins=ULTRASONIC_3_PINS,
    ):
        self.gps = GPSProvider(port=gps_port, baud=gps_baud, fallback_file=gps_fallback_file)
        self.ultrasonic = UltrasonicArray(
            sensor1_trig=ultrasonic1_pins[0],
            sensor1_echo=ultrasonic1_pins[1],
            sensor2_trig=ultrasonic2_pins[0],
            sensor2_echo=ultrasonic2_pins[1],
            sensor3_trig=ultrasonic3_pins[0],
            sensor3_echo=ultrasonic3_pins[1],
        )
        self.motor = DualHBridgeMotorDriver(
            left_in1=left_motor_pins[0],
            left_in2=left_motor_pins[1],
            right_in1=right_motor_pins[0],
            right_in2=right_motor_pins[1],
            left_pwm_pin=motor_pwm_pins[0],
            right_pwm_pin=motor_pwm_pins[1],
            pwm_frequency_hz=motor_pwm_frequency_hz,
        )
        self.camera = PiCam2FrameDriver()

    def get_gps_values(self, timeout_seconds=2.0, allow_fallback=True):
        return self.gps.get_position(timeout_seconds=timeout_seconds, allow_fallback=allow_fallback)

    def get_ultrasonic(self, sensor_id=None, timeout_seconds=0.03):
        if sensor_id is None:
            return self.ultrasonic.read_all(timeout_seconds=timeout_seconds)
        return self.ultrasonic.read_sensor(sensor_id=sensor_id, timeout_seconds=timeout_seconds)

    def set_motor(self, side, direction, speed=100):
        return self.motor.set_motor(side=side, direction=direction, speed=speed)

    def drive(self, left_direction, right_direction, left_speed=100, right_speed=100):
        return self.motor.drive(
            left_direction=left_direction,
            right_direction=right_direction,
            left_speed=left_speed,
            right_speed=right_speed,
        )

    def set_motor_speed(self, side, speed):
        return self.motor.set_speed(side=side, speed=speed)

    def set_motor_speeds(self, left_speed, right_speed):
        return self.motor.set_speeds(left_speed=left_speed, right_speed=right_speed)

    def stop_motors(self):
        return self.motor.stop()

    def getframe(self):
        return self.camera.take_picture()

    def take_picture(self):
        return self.getframe()

    def get_camera_frame(self):
        return self.getframe()

    def close(self):
        self.gps.close()
        self.motor.cleanup()
        self.ultrasonic.cleanup()
        self.camera.close()
