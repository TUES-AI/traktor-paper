# Traktor Embedded Code

This folder contains the Raspberry Pi hardware-control code copied and cleaned up from `patatnik/embedded`.

It intentionally keeps only the embedded pieces used to control the rover, read hardware sensors, and expose Raspberry Pi peripherals:

- DC motor control through a dual H-bridge driver.
- HC-SR04 ultrasonic distance sensors.
- GPS over UART with NMEA parsing and optional fallback coordinates.
- PiCamera2 frame capture.
- A high-level `RoverAPI` wrapper used by scripts and demos.
- Runnable scripts for manual driving, hardware checks, and camera streaming.

## Hardware Pins

All default BCM pin assignments are centralized in `hardware_pins.py`.

Motor pins:

- Left motor direction: GPIO 20, GPIO 21
- Right motor direction: GPIO 16, GPIO 12
- Left/right PWM: GPIO 19, GPIO 13
- PWM frequency: 100 Hz

Ultrasonic pins:

- Sensor 1: TRIG GPIO 23, ECHO GPIO 24
- Sensor 2: TRIG GPIO 27, ECHO GPIO 17
- Sensor 3: TRIG GPIO 5, ECHO GPIO 6

GPS defaults:

- UART port: `/dev/ttyAMA0`
- Baud rate: `9600`
- Fallback file: `/home/yasen/gps_fallback.env`

## Running On Raspberry Pi

Install hardware dependencies on the Raspberry Pi before running these scripts:

```bash
pip install RPi.GPIO picamera2 numpy opencv-python
```

Run scripts from this folder so local imports resolve correctly:

```bash
cd traktor-paper/embedded
python3 scripts/motor_demo.py
python3 scripts/wasd_control.py
python3 scripts/ultrasonic_monitor.py
python3 scripts/gps_latlon.py
python3 scripts/camera_server.py
```

## Safety Notes

- Always test the motors with the rover lifted off the ground first.
- `finally` blocks stop motors and call GPIO cleanup, but power should still be physically removable.
- Ultrasonic `None` values mean timeout/no echo, not zero distance.
- GPS fallback coordinates are only used when real GPS is unavailable or has no fix.
