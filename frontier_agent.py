"""
Frontier-based exploration — ORACLE UPPER BOUND (not a fair sensor-only baseline).

A frontier cell is an unvisited grid cell adjacent to at least one visited cell.
The agent reads `env.visited` (the ground-truth coverage grid) directly, which is
NOT available on the real Tracky rover (no SLAM, no map).

Role in the paper:
  - Labelled "Frontier† (oracle)" in all plots and tables.
  - Provides a theoretical upper bound: the best coverage achievable if the agent
    had perfect knowledge of what it had already visited.
  - If SAC-VMM approaches this curve it demonstrates that sensor-only novelty
    (RND on ultrasonic + IMU) can approximate map-based planning.

The real rover methods (Boustrophedon, SAC-NoVMM, SAC-VMM) use only:
    HC-SR04 ultrasonic sensors (left, right, front) + IMU yaw + odometry.
"""

import numpy as np
from apartment_env import (
    ApartmentContinuousEnv, APT_W, APT_H,
    APT_CELL_W, APT_CELL_H, APT_COLS, APT_ROWS,
)
from rover_coverage_env import SENSOR_MAX, MAX_WHEEL_SPEED, DT, AXLE_LENGTH, STUCK_LIMIT


# ── Frontier finding ──────────────────────────────────────────────────────────

def find_frontiers(visited: np.ndarray):
    """Return array of (cx, cy) world-coords of frontier cell centres.
    A frontier = unvisited cell with at least one visited neighbour."""
    frontiers = []
    rows, cols = visited.shape
    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue
            # Check 4-connected neighbours
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and visited[nr, nc]:
                    cx = (c + 0.5) * APT_CELL_W
                    cy = (r + 0.5) * APT_CELL_H
                    frontiers.append((cx, cy))
                    break
    return frontiers


def nearest_frontier(visited, x, y):
    """Return world-coords of the nearest frontier to (x, y), or None."""
    frontiers = find_frontiers(visited)
    if not frontiers:
        return None
    dists = [(np.hypot(fx - x, fy - y), fx, fy) for fx, fy in frontiers]
    _, fx, fy = min(dists)
    return fx, fy


# ── Reactive navigation toward a goal ────────────────────────────────────────

FRONT_BLOCK   = 0.55
FRONT_URGENT  = 0.35
SIDE_NUDGE    = 0.30
TURN_SPEED    = 0.4
FWD_SPEED     = 0.5
GOAL_TOL      = 0.50    # metres — arrived at frontier
NAV_TIMEOUT   = 300     # steps before giving up on a frontier
ESCAPE_PROB        = 0.03   # random jitter when going straight
CORRIDOR_ESCAPE_P  = 0.25   # jitter probability when in a tight passage
CORRIDOR_WIDTH     = 0.90   # m — treat as corridor when both sides < this

omega       = (2 * MAX_WHEEL_SPEED * TURN_SPEED) / AXLE_LENGTH
TURN_TARGET = int(np.pi / omega / DT)   # steps for 180°
ESCAPE_STEPS  = TURN_TARGET * 2         # full 360° — enough to escape any corner


class FrontierAgent:
    def __init__(self, seed=0):
        self._rng         = np.random.default_rng(seed)
        self._goal        = None
        self._nav_steps   = 0
        self._escape_timer= 0
        self._turning     = False
        self._turn_steps  = 0
        self._turn_dir    = 1.0
        self._stuck_count = 0
        self._prev_cell   = (-1, -1)

    def act(self, obs, env: ApartmentContinuousEnv):
        front_m = float(obs[2]) * SENSOR_MAX
        left_m  = float(obs[0]) * SENSOR_MAX
        right_m = float(obs[1]) * SENSOR_MAX
        x, y, theta = env.x, env.y, env.theta

        # ── Stuck detection ───────────────────────────────────────────────────
        cur_cell = env._cell(x, y)
        if cur_cell != self._prev_cell:
            self._stuck_count = 0
            self._prev_cell   = cur_cell
        else:
            self._stuck_count += 1

        # ── Sensor-driven escape (highest priority) ───────────────────────────
        if self._escape_timer > 0:
            self._escape_timer -= 1
            c = 1.0 if left_m >= right_m else -1.0
            return np.array([c, 0.0, TURN_SPEED], dtype=np.float32)

        # ── Stuck → trigger escape spin, then refresh goal ────────────────────
        if self._stuck_count >= STUCK_LIMIT // 2:
            self._stuck_count  = 0
            self._escape_timer = ESCAPE_STEPS
            self._turning      = False
            self._turn_steps   = 0
            self._goal         = nearest_frontier(env.visited, x, y)
            self._nav_steps    = 0
            c = 1.0 if left_m >= right_m else -1.0
            return np.array([c, 0.0, TURN_SPEED], dtype=np.float32)

        # ── Pick / refresh frontier goal ──────────────────────────────────────
        needs_new_goal = (
            self._goal is None
            or self._nav_steps >= NAV_TIMEOUT
            or (self._goal is not None and np.hypot(self._goal[0]-x, self._goal[1]-y) < GOAL_TOL)
        )
        if needs_new_goal:
            self._goal      = nearest_frontier(env.visited, x, y)
            self._nav_steps = 0
            self._turning   = False
            if self._goal is None:
                return np.array([0.0, 0.0, FWD_SPEED], dtype=np.float32)

        self._nav_steps += 1

        # ── Cornered: all sensors close → full escape spin ───────────────────
        if front_m < FRONT_BLOCK and left_m < SIDE_NUDGE and right_m < SIDE_NUDGE:
            self._escape_timer = ESCAPE_STEPS
            self._turning      = False
            self._turn_steps   = 0
            c = 1.0 if left_m >= right_m else -1.0
            return np.array([c, 0.0, TURN_SPEED], dtype=np.float32)

        # ── Active turning (obstacle avoidance) ───────────────────────────────
        if self._turning:
            self._turn_steps += 1
            if self._turn_steps >= TURN_TARGET:
                self._turning    = False
                self._turn_steps = 0
                self._turn_dir  *= -1.0
            return np.array([self._turn_dir, 0.0, TURN_SPEED], dtype=np.float32)

        # ── Hard obstacle ─────────────────────────────────────────────────────
        if front_m < FRONT_URGENT:
            self._turning    = True
            self._turn_steps = 0
            self._turn_dir   = 1.0 if left_m >= right_m else -1.0
            return np.array([self._turn_dir, 0.0, TURN_SPEED], dtype=np.float32)

        # ── Soft obstacle — begin turn ────────────────────────────────────────
        if front_m < FRONT_BLOCK:
            self._turning    = True
            self._turn_steps = 0
            self._turn_dir   = 1.0 if left_m >= right_m else -1.0
            return np.array([self._turn_dir, 0.0, TURN_SPEED], dtype=np.float32)

        # ── Corridor / doorway: suppress heading control, push straight through ─
        in_corridor = left_m < CORRIDOR_WIDTH and right_m < CORRIDOR_WIDTH
        if in_corridor:
            if self._rng.random() < CORRIDOR_ESCAPE_P:
                c = float(self._rng.uniform(-0.3, 0.3))
                return np.array([c, 0.0, FWD_SPEED], dtype=np.float32)
            return np.array([0.0, 0.0, FWD_SPEED], dtype=np.float32)

        # ── Side nudge ────────────────────────────────────────────────────────
        if left_m < SIDE_NUDGE:
            return np.array([-0.3, 0.0, FWD_SPEED], dtype=np.float32)
        if right_m < SIDE_NUDGE:
            return np.array([ 0.3, 0.0, FWD_SPEED], dtype=np.float32)

        # ── Heading control toward frontier (only when there's room to turn) ──
        gx, gy     = self._goal
        target_ang = np.arctan2(gy - y, gx - x)
        err        = (target_ang - theta + np.pi) % (2 * np.pi) - np.pi

        if abs(np.degrees(err)) > 25:
            c = np.clip(err / np.pi, -1.0, 1.0)
            return np.array([float(c), 0.0, 0.6], dtype=np.float32)

        # Small random jitter to prevent dead-straight trapping
        if self._rng.random() < ESCAPE_PROB:
            c = float(self._rng.uniform(-0.3, 0.3))
            return np.array([c, 0.0, FWD_SPEED], dtype=np.float32)

        return np.array([0.0, 0.0, FWD_SPEED], dtype=np.float32)
