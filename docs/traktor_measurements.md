# Traktor Measurement Sheet

Status: superseded by `embedded/TRAKTOR_DIMENSIONS.md` for actual current dimensions.

This file is only the original blank measurement template.

Goal: measure the real rover once, then use the same numbers in visualization, simulation, sensor projection, and map display.

Use millimeters for distance measurements. Use degrees for sensor angles.

## Coordinate Frame

Origin: center of the rover footprint on the floor.

Axes:

- `+x`: forward
- `+y`: left
- `+z`: up
- `yaw = 0`: sensor points forward
- `yaw > 0`: sensor points left
- `yaw < 0`: sensor points right

Every sensor position should be measured as:

```text
x_mm, y_mm, z_mm, yaw_deg, pitch_deg
```

## Body

```text
body_length_mm:
body_width_mm:
body_height_mm:
ground_clearance_mm:
mass_g:
```

## Drive Geometry

For differential-drive / tank-style motion model:

```text
left_track_center_y_mm:
right_track_center_y_mm:
track_width_mm:
track_contact_length_mm:
wheel_diameter_mm:
wheelbase_or_track_contact_center_length_mm:
```

Important derived value:

```text
track_separation_mm = left_track_center_y_mm - right_track_center_y_mm
```

## Distance Sensors

Sensor 1:

```text
name:
x_mm:
y_mm:
z_mm:
yaw_deg:
pitch_deg:
min_range_mm:
max_range_mm:
```

Sensor 2:

```text
name:
x_mm:
y_mm:
z_mm:
yaw_deg:
pitch_deg:
min_range_mm:
max_range_mm:
```

Sensor 3:

```text
name:
x_mm:
y_mm:
z_mm:
yaw_deg:
pitch_deg:
min_range_mm:
max_range_mm:
```

## IMU

```text
x_mm:
y_mm:
z_mm:
yaw_deg:
pitch_deg:
roll_deg:
mounting_notes:
```

## Camera

```text
x_mm:
y_mm:
z_mm:
yaw_deg:
pitch_deg:
roll_deg:
horizontal_fov_deg:
vertical_fov_deg:
resolution:
```

## First Calibration Tests

Straight-line test:

```text
surface:
motor_command:
duration_s:
measured_distance_mm:
gyro_yaw_change_deg:
notes:
```

Turn-in-place test:

```text
surface:
left_motor_command:
right_motor_command:
duration_s:
measured_yaw_change_deg:
notes:
```

Slip / stuck test:

```text
surface:
motor_command:
observed_motion:
accelerometer_pattern_notes:
gyro_pattern_notes:
```

## Measurement Priority

1. Body length/width.
2. Track separation.
3. Distance sensor positions and yaw angles.
4. IMU position and mounting orientation.
5. Camera position and pitch.
6. Straight-line and turn-in-place tests on at least two surfaces.
