#!/usr/bin/env python3
"""Continuously print distances from the three HC-SR04 ultrasonic sensors."""

import time

import _paths  # noqa: F401
from drivers.sensors.ultrasonic_array import UltrasonicArray
from hardware_pins import ULTRASONIC_1_PINS, ULTRASONIC_2_PINS, ULTRASONIC_3_PINS

READ_INTERVAL_SECONDS = 0.02


def format_distance(label, distance):
    if distance is None:
        return f'{label}: NO_ECHO'
    return f'{label}: {distance:.2f} cm'


def main():
    sensors = UltrasonicArray(
        sensor1_trig=ULTRASONIC_1_PINS[0],
        sensor1_echo=ULTRASONIC_1_PINS[1],
        sensor2_trig=ULTRASONIC_2_PINS[0],
        sensor2_echo=ULTRASONIC_2_PINS[1],
        sensor3_trig=ULTRASONIC_3_PINS[0],
        sensor3_echo=ULTRASONIC_3_PINS[1],
    )
    print('Distance measurement in progress. Ctrl+C to stop.')

    try:
        while True:
            distances = sensors.read_all()
            print(
                ' | '.join(
                    [
                        format_distance('right', distances[1]),
                        format_distance('left', distances[2]),
                        format_distance('S3', distances[3]),
                    ]
                )
            )
            time.sleep(READ_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print('Stopped.')
    finally:
        sensors.cleanup()


if __name__ == '__main__':
    main()
