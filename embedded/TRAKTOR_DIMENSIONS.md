# Traktor Dimensions

Units: centimeters.

Coordinate frame:

```text
origin: center of rover footprint on the floor
+x: forward
+y: left
+z: up
```

## Main Geometry

```text
tank_chain_length: 16.5
tank_chain_width: 4.5
tank_chain_height: 6.0
body_width_between_chains: 8.5
platform_top_height: 13.0
```

Derived:

```text
total_width = 17.5
platform_height = 7.0
```

## Sensors

Front distance sensor:

```text
x: front face of platform
y: centered
z: 7.5
yaw: 0 deg, forward
```

Left/right distance sensors:

```text
x: 14.5 from rear of tank chain
z: 10.5
left yaw: +90 deg
right yaw: -90 deg
butts touch the platform side faces
```

Pi camera:

```text
x: centered above front distance sensor
y: centered
z: 12.0
yaw: 0 deg, forward
```

MPU / IMU:

```text
x: 5.0 from rear of right tank chain
y: 4.0 inward from the rightmost edge of robot
z: 14.0
mounting: currently floating / temporary
```

## Motors

Motors sit under the platform, not inside the tank chains.

```text
left motor: rear-left, axis center 3.0 from rear of tank chain
right motor: front-right, mirrored 180 deg from left motor
motor_z: center height of tank chain = 3.0
```

## Visualization Script

Save as `/tmp/traktor_measurement_preview.py` and run with:

```bash
/Volumes/SSD/v/py/bin/python /tmp/traktor_measurement_preview.py
```

```python
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def box_faces(x0, x1, y0, y1, z0, z1):
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def add_box(ax, label, center, size, color, alpha=0.45):
    cx, cy, cz = center
    sx, sy, sz = size
    faces = box_faces(cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2)
    poly = Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=0.8, alpha=alpha)
    ax.add_collection3d(poly)
    ax.text(cx, cy, cz + sz / 2 + 0.25, label, ha='center', va='bottom', fontsize=8)


def add_sensor(ax, label, pos, yaw_deg, color):
    x, y, z = pos
    add_box(ax, label, pos, (1.2, 1.0, 0.8), color, alpha=0.9)
    length = 4.0
    import math
    yaw = math.radians(yaw_deg)
    ax.quiver(x, y, z, math.cos(yaw) * length, math.sin(yaw) * length, 0, color=color, linewidth=2, arrow_length_ratio=0.25)


def add_y_cylinder(ax, label, center, radius, length, color):
    cx, cy, cz = center
    theta = np.linspace(0, 2 * np.pi, 32)
    yy = np.linspace(cy - length / 2, cy + length / 2, 2)
    theta_grid, y_grid = np.meshgrid(theta, yy)
    x_grid = cx + radius * np.cos(theta_grid)
    z_grid = cz + radius * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=0.8, linewidth=0)
    ax.text(cx, cy, cz + radius + 0.35, label, ha='center', va='bottom', fontsize=8)


def main():
    track_length = 16.5
    track_width = 4.5
    track_height = 6.0
    body_width = 8.5
    total_width = body_width + 2 * track_width
    body_length = track_length
    body_height = 7.0

    x_back = -track_length / 2
    x_front = track_length / 2
    y_track_center = body_width / 2 + track_width / 2

    fig = plt.figure('Traktor Measurement Preview')
    ax = fig.add_subplot(111, projection='3d')

    add_box(ax, 'left track', (0, y_track_center, track_height / 2), (track_length, track_width, track_height), '#555555')
    add_box(ax, 'right track', (0, -y_track_center, track_height / 2), (track_length, track_width, track_height), '#555555')
    add_box(ax, 'body/corpus', (0, 0, track_height + body_height / 2), (body_length, body_width, body_height), '#3b82f6', alpha=0.35)

    motor_radius = 1.25
    motor_length = 3.0
    motor_z = track_height / 2
    motor_y = body_width / 2 - motor_length / 2
    add_y_cylinder(ax, 'left motor', (x_back + 3.0, motor_y, motor_z), motor_radius, motor_length, '#f97316')
    add_y_cylinder(ax, 'right motor', (x_front - 3.0, -motor_y, motor_z), motor_radius, motor_length, '#f97316')

    sensor_depth = 1.2
    add_sensor(ax, 'front sensor', (x_front + sensor_depth / 2, 0, 7.5), 0, '#ef4444')
    add_sensor(ax, 'Pi camera', (x_front + sensor_depth / 2, 0, 12.0), 0, '#a855f7')

    side_x = x_back + 14.5
    side_y = body_width / 2 + sensor_depth / 2
    add_sensor(ax, 'left sensor', (side_x, side_y, 10.5), 90, '#22c55e')
    add_sensor(ax, 'right sensor', (side_x, -side_y, 10.5), -90, '#22c55e')

    mpu_x = x_back + 5.0
    mpu_y = -total_width / 2 + 4.0
    add_box(ax, 'MPU/IMU', (mpu_x, mpu_y, 14.0), (2.0, 1.5, 0.35), '#eab308', alpha=0.9)

    ax.quiver(0, 0, 0, 5, 0, 0, color='red', linewidth=2)
    ax.text(5.4, 0, 0, '+x forward', color='red')
    ax.quiver(0, 0, 0, 0, 5, 0, color='green', linewidth=2)
    ax.text(0, 5.4, 0, '+y left', color='green')
    ax.quiver(0, 0, 0, 0, 0, 5, color='blue', linewidth=2)
    ax.text(0, 0, 5.4, '+z up', color='blue')

    ax.set_xlabel('x forward (cm)')
    ax.set_ylabel('y left (cm)')
    ax.set_zlabel('z up (cm)')
    ax.set_title('Traktor measured geometry preview')
    lim = max(total_width, track_length, 16) / 2 + 4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 16)
    ax.set_box_aspect((1, 1, 0.65))
    ax.view_init(elev=22, azim=-45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
```
