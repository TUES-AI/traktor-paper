"""Default Raspberry Pi BCM pin assignments for the rover hardware."""

GPS_PORT = '/dev/ttyAMA0'
GPS_BAUD = 9600
GPS_FALLBACK_FILE = '/home/yasen/gps_fallback.env'

LEFT_MOTOR_PINS = (20, 21)
RIGHT_MOTOR_PINS = (16, 12)
MOTOR_PWM_PINS = (19, 13)
MOTOR_PWM_FREQUENCY_HZ = 100

ULTRASONIC_1_PINS = (23, 24)
ULTRASONIC_2_PINS = (27, 17)
ULTRASONIC_3_PINS = (5, 6)
