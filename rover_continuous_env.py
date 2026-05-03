"""
Continuous-action rover environment.

Action space : Box([-1, 0], [1, 1]) = [curvature, speed]
  curvature ∈ [-1, 1]  →  -1 = hard-right spin, 0 = straight, +1 = hard-left spin
  speed     ∈ [ 0, 1]  →   0 = stop,             1 = MAX_WHEEL_SPEED

Executor (deterministic arc model):
  vl = clip(1 - 2*curvature, -1, 1) * speed * MAX_WHEEL_SPEED
  vr = clip(1 + 2*curvature, -1, 1) * speed * MAX_WHEEL_SPEED

  curvature=0    → vl=vr  (straight)
  curvature=±0.5 → one wheel stopped (arc)
  curvature=±1   → vl=-vr (pure spin)

`horizon` (PROJECT.md) is left as a future extension — each step picks a
new guide, matching the discrete env's 1-action-per-step interface.
"""

import numpy as np
from gymnasium import spaces
from rover_coverage_env import (
    RoverCoverageEnv, MAX_WHEEL_SPEED, SENSOR_MAX,
    ReactiveAgent, _wheels_to_action,
)


# ── Executor ──────────────────────────────────────────────────────────────────

def guide_to_wheels(curvature: float, speed: float):
    """[curvature, speed] → (vl, vr) in m/s."""
    c  = float(np.clip(curvature, -1.0, 1.0))
    s  = float(np.clip(speed,      0.0, 1.0))
    vl = float(np.clip(1.0 - 2.0 * c, -1.0, 1.0)) * s * MAX_WHEEL_SPEED
    vr = float(np.clip(1.0 + 2.0 * c, -1.0, 1.0)) * s * MAX_WHEEL_SPEED
    return vl, vr


# ── Environment ───────────────────────────────────────────────────────────────

class ContinuousRoverEnv(RoverCoverageEnv):
    """
    Same physics and map as RoverCoverageEnv, but action = [curvature, speed].
    Inherits all rendering, sensor, coverage, and collision logic unchanged.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low =np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0], dtype=np.float32),
        )

    def step(self, action):
        curvature, speed = float(action[0]), float(action[1])
        vl, vr = guide_to_wheels(curvature, speed)

        # Safety clamp: same reflex as parent, translated to (vl, vr)
        bumper_fired = False
        if self.use_bumper:
            override = self._safety_override()
            if override is not None:
                vl, vr = self._action_to_wheels[override]
                bumper_fired = True
                self._bumper_triggers += 1

        # Physics — copied from parent step() with vl,vr already resolved
        from rover_coverage_env import diff_drive_step, in_collision, R_NEW_CELL, R_STEP, R_COLLISION
        nx, ny, ntheta = diff_drive_step(self.x, self.y, self.theta, vl, vr)
        obs_list = self.obstacles

        if not in_collision(nx, ny, ntheta, obs_list):
            self.x, self.y, self.theta = nx, ny, ntheta
            collided = False
        elif not in_collision(nx, self.y, ntheta, obs_list):
            self.x, self.theta = nx, ntheta
            collided = True
        elif not in_collision(self.x, ny, ntheta, obs_list):
            self.y, self.theta = ny, ntheta
            collided = True
        elif not in_collision(self.x, self.y, ntheta, obs_list):
            self.theta = ntheta
            collided = True
        else:
            collided = True

        if collided:
            self._collisions += 1

        r, c = self._cell(self.x, self.y)
        if (r, c) != self._prev_cell:
            self._cell_entries += 1
            self._prev_cell = (r, c)

        unique_before = int(self.visited.sum())
        self._mark_swept(self.step_count)
        new_cells = int(self.visited.sum()) - unique_before
        reward = R_NEW_CELL * new_cells + R_STEP
        if collided:
            reward += R_COLLISION

        self._trail.append((self.x, self.y))
        self._total_reward += reward
        self.step_count += 1

        info = {
            "coverage":     self._coverage(),
            "collided":     collided,
            "steps":        self.step_count,
            "collisions":   self._collisions,
            "bumper_fired": bumper_fired,
            "cross_path_pct": self.cross_path_pct(),
        }
        return self._get_obs(), reward, False, False, info


# ── Continuous reactive agent ─────────────────────────────────────────────────

class ContinuousReactiveAgent:
    """
    Same logic as ReactiveAgent but outputs (curvature, speed) instead of a
    discrete wheel-command index.  Escape routine included.
    """

    FRONT_CURVE  = 0.55
    FRONT_SPIN   = 0.30
    FRONT_CLEAR  = 0.70
    SIDE_CURVE   = 0.28
    SIDE_SPIN    = 0.22
    DRIFT_INTERVAL = 60
    STUCK_LIMIT  = 8
    BACKUP_STEPS = 20
    ESCAPE_SPINS = 60

    def __init__(self, seed: int = 0):
        self._rng           = np.random.default_rng(seed)
        self._state         = "forward"
        self._preferred_dir = "left"
        self._turn_dir      = "left"
        self._was_avoiding  = False
        self._stuck_count   = 0
        self._escape_timer  = 0
        self._escape_phase  = None
        self._forward_steps = 0
        self._drift_dir     = "left"
        self._next_drift    = int(self._rng.integers(30, self.DRIFT_INTERVAL))

    def _flip(self, d): return "right" if d == "left" else "left"

    def act(self, obs: np.ndarray, collided: bool = False):
        """Returns (curvature, speed) ∈ ([-1,1], [0,1])."""
        left_m  = float(obs[0]) * SENSOR_MAX
        right_m = float(obs[1]) * SENSOR_MAX
        front_m = float(obs[2]) * SENSOR_MAX

        # ── Escape ────────────────────────────────────────────────
        if collided:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        if self._stuck_count >= self.STUCK_LIMIT:
            self._stuck_count   = 0
            self._escape_phase  = "backup"
            self._escape_timer  = self.BACKUP_STEPS
            self._preferred_dir = self._flip(self._preferred_dir)

        if self._escape_phase == "backup":
            self._escape_timer -= 1
            if self._escape_timer <= 0:
                self._escape_phase = "spin"
                self._escape_timer = self.ESCAPE_SPINS
                self._turn_dir = self._preferred_dir
            return (0.0, -0.6)   # reverse: curvature=0, speed negative via sign

        if self._escape_phase == "spin":
            self._escape_timer -= 1
            if self._escape_timer <= 0:
                self._escape_phase = None
                self._state = "forward"
            c = 1.0 if self._turn_dir == "left" else -1.0
            return (c, 0.5)

        # ── Normal avoidance ──────────────────────────────────────
        need_spin  = False
        need_curve = False
        turn_dir   = self._preferred_dir

        if front_m < self.FRONT_SPIN:
            need_spin = True
            turn_dir  = self._preferred_dir if abs(left_m - right_m) < 0.1 \
                        else ("left" if left_m >= right_m else "right")
        elif front_m < self.FRONT_CURVE:
            need_curve = True
            turn_dir   = self._preferred_dir if abs(left_m - right_m) < 0.1 \
                         else ("left" if left_m >= right_m else "right")

        if left_m < self.SIDE_SPIN:
            need_spin = True; turn_dir = "right"
        elif left_m < self.SIDE_CURVE:
            need_curve = True; turn_dir = "right"
        if right_m < self.SIDE_SPIN:
            need_spin = True; turn_dir = "left"
        elif right_m < self.SIDE_CURVE:
            need_curve = True; turn_dir = "left"

        now_avoiding = need_spin or need_curve
        if self._was_avoiding and not now_avoiding:
            self._preferred_dir = self._flip(self._preferred_dir)
            self._drift_dir     = self._preferred_dir
            self._next_drift    = int(self._rng.integers(30, self.DRIFT_INTERVAL))
            self._forward_steps = 0
        self._was_avoiding = now_avoiding

        sign = 1.0 if turn_dir == "left" else -1.0

        if need_spin:
            return (sign * 1.0, 0.5)          # pure spin
        if need_curve:
            return (sign * 0.5, 0.7)          # arc

        # Forward + occasional drift
        self._forward_steps += 1
        if self._forward_steps >= self._next_drift:
            self._forward_steps = 0
            self._next_drift    = int(self._rng.integers(30, self.DRIFT_INTERVAL))
            drift_sign = 1.0 if self._drift_dir == "left" else -1.0
            return (drift_sign * 0.25, 1.0)   # gentle drift

        return (0.0, 1.0)                     # straight forward
