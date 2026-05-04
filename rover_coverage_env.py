"""
2D Differential Drive Rover Coverage Navigation Environment
Gymnasium environment with reactive obstacle-avoidance agent and rich visualization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
from typing import Optional
from collections import deque

# ─── Constants ───────────────────────────────────────────────────────────────
#
# These match the physical rover (see embedded/api/rover_api.py and the
# HC-SR04 / dual-H-bridge drivers). The simulator is a digital twin: the
# observation / action interface here mirrors what the real RoverAPI exposes,
# so a policy trained in sim can be dropped onto the Pi without changes.

# Field geometry — call it a 10×10 m farm patch
FIELD_W, FIELD_H = 10.0, 10.0
ROOM_W, ROOM_H   = FIELD_W, FIELD_H   # backwards-compat aliases
GRID_COLS, GRID_ROWS = 40, 40         # 25 cm cells — matches WORKING_WIDTH
CELL_W = FIELD_W / GRID_COLS
CELL_H = FIELD_H / GRID_ROWS

# Rover geometry — boxy body, axle across the short edge.
ROVER_LENGTH   = 0.30        # along heading axis (forward / back)
ROVER_WIDTH    = 0.20        # across heading axis (matches axle length)
AXLE_LENGTH    = ROVER_WIDTH
AGENT_RADIUS   = 0.5 * np.sqrt(ROVER_LENGTH ** 2 + ROVER_WIDTH ** 2)  # bounding circle
WORKING_WIDTH  = 0.30        # boustrophedon swath spacing — matches rover body

# Timing — matches embedded control loop (20 Hz)
DT = 0.05

# Motor mapping. The H-bridge accepts a direction (forward / backward / stop)
# and a PWM duty cycle in [0, 100] %. We treat the discrete action sign as
# direction and apply OPERATING_DUTY %; max physical speed is at 100 %.
OPERATING_DUTY     = 60.0    # PWM % used by demo / WASD / wall-follower
MAX_PHYSICAL_SPEED = 1.0     # m/s at 100 % duty (calibrated guess)
MAX_WHEEL_SPEED    = MAX_PHYSICAL_SPEED * OPERATING_DUTY / 100.0   # = 0.6 m/s

# HC-SR04 ultrasonic specs
SENSOR_MAX        = 3.0     # we cap usable range at 3 m (datasheet says ~4 m)
SENSOR_MIN        = 0.02    # closer than 2 cm reads as 0 (saturated)
SENSOR_NOISE_STD  = 0.005   # ±0.5 cm Gaussian noise per read

# Reflexive ultrasonic safety override. Mirrors the deterministic reflex
# in Tracky's embedded/ppo_rover.py::safety_override(): when an obstacle
# crosses a hard threshold the env unconditionally hijacks the action,
# regardless of what the policy commanded. The robot doesn't trust the
# policy to never crash. Thresholds match Tracky verbatim (cm → m).
FRONT_SAFETY_DIST = 0.25    # m (Tracky FRONT_SAFETY_CM = 25.0)
SIDE_SAFETY_DIST  = 0.15    # m (Tracky SAFETY_CM       = 15.0)

# Action indices used by the safety override (in our (vl, vr) enumeration)
ACT_BACKWARD   = 0   # (-1, -1) full reverse
ACT_SPIN_LEFT  = 2   # (-1, +1) spin CCW
ACT_SPIN_RIGHT = 6   # (+1, -1) spin CW

# Waypoint-following tolerances for the deterministic baseline
WAYPOINT_TOL      = 0.25    # arrived when within this many metres
WP_TIMEOUT_STEPS  = 300     # give up on a waypoint blocked by an obstacle

# Rewards
# Only signals available on real hardware:
#   R_MOVE     — wheel-encoder / IMU displacement (odometry)
#   R_STEP     — time pressure so the agent doesn't idle
#   R_COLLISION — bumper / full-wedge contact
#   R_STUCK    — no progress for STUCK_LIMIT steps → penalty + respawn
# R_NEW_CELL is intentionally absent: the real rover has no coverage-grid oracle.
# The VMM variant receives an additional novelty reward in train_sac.py's wrapper.
R_COLLISION   = -10.0
R_STEP        = -0.02  # mild time pressure
R_MOVE        =  2.0   # reward per metre of displacement (odometry signal)
R_STUCK       = -50.0  # strong enough to trigger escape / respawn
STUCK_LIMIT   =  200   # steps without a new cell before stuck penalty + respawn (~10 s)

# Wheel inertia — commanded speed takes a few steps to reach (first-order lag)
INERTIA_ALPHA = 0.35    # 0 = frozen, 1 = instant response; ~5 steps to converge

WHEEL_CMDS = [-1, 0, 1]

# Sensor angles relative to heading. Order is [left, right, front] so that
# obs[i] lines up with the real RoverAPI's `read_all()` dict ordering
# {1: left, 2: right, 3: front} — see drivers/sensors/ultrasonic_array.py.
SENSOR_ANGLES = [np.pi / 2, -np.pi / 2, 0.0]
SENSOR_NAMES  = ["left", "right", "front"]

# Pygame display — map area + right sidebar
PX_PER_M   = 70
MAP_W      = int(ROOM_W * PX_PER_M)
MAP_H      = int(ROOM_H * PX_PER_M)
SIDEBAR_W  = 220
WIN_W      = MAP_W + SIDEBAR_W
WIN_H      = MAP_H

# Trail length (number of past positions to draw)
TRAIL_LEN  = 300

# Colour palette
C_BG          = (18,  20,  28)
C_GRID        = (38,  40,  50)
C_BORDER      = (180, 180, 200)
C_VISITED_LO  = (20,  60,  30)   # oldest visit shade
C_VISITED_HI  = (60, 200,  90)   # newest visit shade
C_OBSTACLE    = (90,  50,  30)
C_OBSTACLE_ED = (180, 100,  55)
C_AGENT       = (255, 220,  50)
C_AGENT_ED    = (200, 160,  20)
C_HEADING     = (30,  30,  30)
C_RAY_F       = (255,  80,  80)
C_RAY_L       = ( 80, 180, 255)
C_RAY_R       = ( 80, 255, 180)
C_TRAIL_NEW   = (255, 220,  50)
C_TRAIL_OLD   = ( 60,  60,  80)
C_SIDEBAR_BG  = (22,  24,  34)
C_TEXT        = (210, 215, 230)
C_TEXT_DIM    = (110, 115, 130)
C_BAR_BG      = (40,  44,  58)
C_BAR_FG      = (60, 200,  90)
C_PATH        = (255, 150,  60)   # boustrophedon path polyline
C_PATH_DOT    = (255, 200, 110)   # waypoint markers


# ─── Obstacle generation ─────────────────────────────────────────────────────

def generate_obstacles(n: int = 5, rng: np.random.Generator = None) -> list:
    if rng is None:
        rng = np.random.default_rng()
    obstacles = []
    margin = 1.5
    for _ in range(n):
        for _ in range(50):
            w = rng.uniform(0.5, 2.0)
            h = rng.uniform(0.5, 2.0)
            x = rng.uniform(margin, ROOM_W - margin - w)
            y = rng.uniform(margin, ROOM_H - margin - h)
            rect = (x, y, w, h)
            ok = True
            for ox, oy, ow, oh in obstacles:
                if not (x + w + 0.3 < ox or ox + ow + 0.3 < x or
                        y + h + 0.3 < oy or oy + oh + 0.3 < y):
                    ok = False
                    break
            if ok:
                obstacles.append(rect)
                break
    return obstacles


# ─── Ray casting ─────────────────────────────────────────────────────────────

def ray_cast(px, py, angle, obstacles, max_dist=SENSOR_MAX):
    dx, dy = np.cos(angle), np.sin(angle)
    t_min = max_dist
    for t in _ray_wall_hits(px, py, dx, dy):
        if 1e-6 < t < t_min:
            t_min = t
    for obs in obstacles:
        t = _ray_aabb_hit(px, py, dx, dy, *obs)
        if t is not None and 1e-6 < t < t_min:
            t_min = t
    return t_min


def _ray_wall_hits(px, py, dx, dy):
    hits = []
    if abs(dx) > 1e-9:
        hits += [(0.0 - px) / dx, (ROOM_W - px) / dx]
    if abs(dy) > 1e-9:
        hits += [(0.0 - py) / dy, (ROOM_H - py) / dy]
    return hits


def _ray_aabb_hit(px, py, dx, dy, ox, oy, ow, oh):
    tx1 = (ox      - px) / dx if abs(dx) > 1e-9 else -np.inf
    tx2 = (ox + ow - px) / dx if abs(dx) > 1e-9 else  np.inf
    ty1 = (oy      - py) / dy if abs(dy) > 1e-9 else -np.inf
    ty2 = (oy + oh - py) / dy if abs(dy) > 1e-9 else  np.inf
    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))
    if tmax < 0 or tmin > tmax:
        return None
    return tmin if tmin >= 0 else tmax


# ─── Box body geometry & collision ───────────────────────────────────────────

def rover_corners(x, y, theta):
    """Return the 4 world-coords corners of the rover body in order
    [front-left, front-right, back-right, back-left]."""
    hl, hw = ROVER_LENGTH / 2.0, ROVER_WIDTH / 2.0
    c, s = np.cos(theta), np.sin(theta)
    # Local body frame: x axis = forward, y axis = left
    local = ((+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw))
    return [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in local]


def in_collision(px, py, ptheta, obstacles) -> bool:
    """Bounding-circle collision (radius = box half-diagonal).

    The rover is rendered as a rotated rectangle, but the collision shape
    is the circle that *contains* that rectangle. This keeps pure rotation
    always-collision-free (radius is invariant under rotation), so the
    deterministic agent never wedges. `ptheta` is part of the signature
    only for call-site compatibility — circle collision doesn't need it.
    The cost is a slight overestimate of the body's footprint, which only
    makes the avoidance more conservative."""
    _ = ptheta
    r = AGENT_RADIUS
    if px - r < 0.0 or px + r > ROOM_W or py - r < 0.0 or py + r > ROOM_H:
        return True
    for ox, oy, ow, oh in obstacles:
        nx = ox if px < ox else (ox + ow if px > ox + ow else px)
        ny = oy if py < oy else (oy + oh if py > oy + oh else py)
        if (px - nx) ** 2 + (py - ny) ** 2 < r * r:
            return True
    return False


# ─── Differential drive kinematics ───────────────────────────────────────────

def diff_drive_step(x, y, theta, vl, vr):
    if abs(vr - vl) < 1e-9:
        dist = vl * DT
        nx = x + dist * np.cos(theta)
        ny = y + dist * np.sin(theta)
        ntheta = theta
    else:
        R_icc = AXLE_LENGTH / 2.0 * (vl + vr) / (vr - vl)
        omega = (vr - vl) / AXLE_LENGTH
        dtheta = omega * DT
        icc_x = x - R_icc * np.sin(theta)
        icc_y = y + R_icc * np.cos(theta)
        nx = np.cos(dtheta) * (x - icc_x) - np.sin(dtheta) * (y - icc_y) + icc_x
        ny = np.sin(dtheta) * (x - icc_x) + np.cos(dtheta) * (y - icc_y) + icc_y
        ntheta = theta + dtheta
    ntheta = (ntheta + np.pi) % (2 * np.pi) - np.pi
    return nx, ny, ntheta


# ─── Gymnasium Environment ────────────────────────────────────────────────────

class RoverCoverageEnv(gym.Env):
    """
    2D differential drive rover doing continuous coverage of a single fixed farm.

    The map (obstacles) is generated once and never changes — the agent learns
    the layout on the spot. On collision the agent is respawned at a safe
    position but the visited grid and total step counter are preserved, so the
    task is truly continuous (no hard episode boundary).

    Observation: [front_sensor, left_sensor, right_sensor] in [0, 1]
    Action: Discrete(9) — all (vl, vr) combos with vl, vr in {-1, 0, 1}
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None,
                 n_obstacles: int = 5, seed: Optional[int] = None,
                 obstacles: Optional[list] = None):
        super().__init__()
        self.render_mode = render_mode
        self.n_obstacles = n_obstacles
        self._seed = seed
        self._fixed_obstacles = obstacles  # if set, skip generation and use this

        self.action_space = spaces.Discrete(9)
        self._action_to_wheels = [
            (vl * MAX_WHEEL_SPEED, vr * MAX_WHEEL_SPEED)
            for vl in WHEEL_CMDS for vr in WHEEL_CMDS
        ]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)

        self._screen = None
        self._clock  = None
        self._fonts  = {}

        # These are initialised in reset() on first call
        self.obstacles  = []
        self.x = self.y = self.theta = 0.0
        self.visited    = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        self.visit_age  = np.full((GRID_ROWS, GRID_COLS), -1, dtype=np.int32)
        self.step_count = 0
        self.done       = False
        self._trail     = deque(maxlen=TRAIL_LEN)
        self._total_reward  = 0.0
        self._collisions    = 0
        self._map_ready     = False

        # Reflexive bumper (deterministic safety layer)
        self.use_bumper        = True
        self._bumper_triggers  = 0

        # Coverage metrics — count cell *entries* (transitions), not
        # per-step occupancy, so cross-path stays meaningful at any speed.
        self._cell_entries = 0
        self._prev_cell    = (-1, -1)

        # Wheel velocity state (inertia) and stuck detector
        self._vl = 0.0
        self._vr = 0.0
        self._steps_since_new_cell = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Returns 3 normalised ultrasonic readings in [0, 1] in the same
        order the physical RoverAPI returns them: [left, right, front].

        Each sensor returns the *minimum* distance hit by a 3-ray cone
        (±15° from boresight), modelling the HC-SR04's ~30° beam width.
        A single-ray sim missed obstacles slightly off-axis from the
        sensor direction and the rover would clip them."""
        out = np.empty(3, dtype=np.float32)
        cone = (-np.pi / 6, -np.pi / 12, 0.0, np.pi / 12, np.pi / 6)  # ±30°, 5 rays
        for i, a in enumerate(SENSOR_ANGLES):
            d = min(ray_cast(self.x, self.y, self.theta + a + da, self.obstacles)
                    for da in cone)
            d += float(self._rng.normal(0.0, SENSOR_NOISE_STD))
            d  = max(SENSOR_MIN, min(SENSOR_MAX, d))
            out[i] = d / SENSOR_MAX
        return out

    def _mark_swept(self, step_idx: int):
        """Mark every grid cell whose centre lies within AGENT_RADIUS of the
        rover. Models the actual swept band — a real rover covers a body-wide
        strip, not a single point."""
        r0, c0 = self._cell(self.x, self.y)
        rad_cells = int(np.ceil(AGENT_RADIUS / min(CELL_W, CELL_H))) + 1
        r2 = AGENT_RADIUS ** 2
        for dr in range(-rad_cells, rad_cells + 1):
            r = r0 + dr
            if r < 0 or r >= GRID_ROWS:
                continue
            cy = (r + 0.5) * CELL_H
            for dc in range(-rad_cells, rad_cells + 1):
                c = c0 + dc
                if c < 0 or c >= GRID_COLS:
                    continue
                cx = (c + 0.5) * CELL_W
                if (cx - self.x) ** 2 + (cy - self.y) ** 2 <= r2:
                    if not self.visited[r, c]:
                        self.visited[r, c]   = True
                        self.visit_age[r, c] = step_idx

    def _cell(self, px, py):
        col = int(np.clip(px / CELL_W, 0, GRID_COLS - 1))
        row = int(np.clip(py / CELL_H, 0, GRID_ROWS - 1))
        return row, col

    def _coverage(self) -> float:
        return float(self.visited.sum()) / (GRID_ROWS * GRID_COLS)

    def _safe_spawn(self):
        """Find a position with comfortable clearance — not just the bare minimum
        of `not in_collision`. Prevents the agent from spawning a millimetre
        away from an obstacle and immediately hitting it on the next step."""
        clearance = AGENT_RADIUS + 0.20   # 35 cm body-edge clearance
        best = (ROOM_W / 2, ROOM_H / 2)
        for _ in range(1000):
            px = self._rng.uniform(clearance, ROOM_W - clearance)
            py = self._rng.uniform(clearance, ROOM_H - clearance)
            # Sample any heading; if the box doesn't fit, skip.
            theta_try = self._rng.uniform(-np.pi, np.pi)
            if in_collision(px, py, theta_try, self.obstacles):
                continue
            # Require the four cardinal sensor rays to all be reasonably clear
            ok = True
            for a in (0.0, np.pi / 2, np.pi, -np.pi / 2):
                if ray_cast(px, py, a, self.obstacles) < clearance:
                    ok = False
                    break
            if ok:
                return px, py
            best = (px, py)   # fall back to first non-colliding sample
        return best

    def _respawn(self):
        """Teleport agent to a safe position without touching map or visited grid.
        The new heading points toward the field centre so the rover never
        immediately drives into a wall."""
        self.x, self.y = self._safe_spawn()
        cx, cy = ROOM_W / 2, ROOM_H / 2
        # Toward centre, with a small random jitter so we don't always face the same way
        self.theta = float(np.arctan2(cy - self.y, cx - self.x)
                           + self._rng.uniform(-np.pi / 6, np.pi / 6))
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self._trail.clear()
        self.done = False
        self._vl = self._vr = 0.0
        self._steps_since_new_cell = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Build the map only once; subsequent reset() calls just respawn the agent
        if not self._map_ready:
            self.obstacles  = (list(self._fixed_obstacles) if self._fixed_obstacles is not None
                               else generate_obstacles(self.n_obstacles, self._rng))
            self.visited    = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
            self.visit_age  = np.full((GRID_ROWS, GRID_COLS), -1, dtype=np.int32)
            self.step_count = 0
            self._total_reward    = 0.0
            self._collisions      = 0
            self._bumper_triggers = 0
            self._cell_entries    = 0
            self._map_ready       = True

        self._respawn()
        self._prev_cell = (-1, -1)

        self._mark_swept(self.step_count)
        r, c = self._cell(self.x, self.y)
        self._prev_cell     = (r, c)
        self._cell_entries += 1

        return self._get_obs(), {}

    def _safety_override(self):
        """Full-circle safety reflex: sweeps 8 rays every 45° to catch diagonal
        approaches that the 3-sensor bumper misses at corners and angled walls.

        Escape logic: find the closest obstacle direction, then spin/back away
        from it.  Priority: front hemisphere → back up; left → spin right;
        right → spin left; rear → spin (pick side with most clearance)."""
        angles = np.linspace(-np.pi, np.pi, 9, endpoint=False)   # 8 rays × 45°
        dists  = [ray_cast(self.x, self.y, self.theta + a, self.obstacles)
                  for a in angles]

        min_d   = min(dists)
        min_idx = int(np.argmin(dists))
        min_ang = angles[min_idx]   # angle relative to heading of nearest obstacle

        if min_d >= FRONT_SAFETY_DIST:
            return None

        # Front hemisphere (±90°): back up
        if abs(min_ang) <= np.pi / 2:
            return ACT_BACKWARD

        # Left side: spin right to escape
        if min_ang > 0:
            return ACT_SPIN_RIGHT

        # Right side: spin left to escape
        return ACT_SPIN_LEFT

    def _physics_and_reward(self, vl_cmd: float, vr_cmd: float,
                             bumper_fired: bool):
        """Shared physics + reward kernel used by both discrete and continuous step().

        Applies first-order wheel inertia, resolves soft-contact collision,
        updates coverage, computes reward (new cells + displacement − step cost),
        runs stuck detector (R_STUCK + respawn after STUCK_LIMIT steps without
        a new cell), and returns the full gymnasium step tuple."""
        # Wheel inertia: actual speed lags commanded speed.
        # Bypass inertia when the safety bumper fires — the reflex must be instant.
        if bumper_fired:
            self._vl, self._vr = vl_cmd, vr_cmd
        else:
            self._vl += INERTIA_ALPHA * (vl_cmd - self._vl)
            self._vr += INERTIA_ALPHA * (vr_cmd - self._vr)

        x0, y0 = self.x, self.y
        nx, ny, ntheta = diff_drive_step(self.x, self.y, self.theta,
                                         self._vl, self._vr)

        obs_list = self.obstacles
        if not in_collision(nx, ny, ntheta, obs_list):
            self.x, self.y, self.theta = nx, ny, ntheta
            collided = False
        elif not in_collision(nx, self.y, ntheta, obs_list):
            self.x, self.theta = nx, ntheta
            collided = False   # sliding along wall — not a true collision
        elif not in_collision(self.x, ny, ntheta, obs_list):
            self.y, self.theta = ny, ntheta
            collided = False   # sliding along wall — not a true collision
        elif not in_collision(self.x, self.y, ntheta, obs_list):
            self.theta = ntheta
            collided = False   # rotation only — not a true collision
        else:
            collided = True    # fully wedged: rover cannot move at all

        if collided:
            self._collisions += 1

        r, c = self._cell(self.x, self.y)
        if (r, c) != self._prev_cell:
            self._cell_entries += 1
            self._prev_cell = (r, c)

        unique_before = int(self.visited.sum())
        self._mark_swept(self.step_count)
        new_cells = int(self.visited.sum()) - unique_before

        displacement = float(np.hypot(self.x - x0, self.y - y0))
        reward = R_MOVE * displacement + R_STEP
        if collided:
            reward += R_COLLISION

        # Cross-path penalty: moving over already-visited ground with no new cells.
        # Sim: checked directly from the visited grid.
        # Stuck / spinning detector: no new cell for STUCK_LIMIT steps → penalty + respawn
        if new_cells > 0:
            self._steps_since_new_cell = 0
        else:
            self._steps_since_new_cell += 1
        if self._steps_since_new_cell >= STUCK_LIMIT:
            reward += R_STUCK
            self._respawn()   # teleport out; map and visited grid are preserved

        self._trail.append((self.x, self.y))
        self._total_reward += reward
        self.step_count    += 1

        info = {
            "coverage":       self._coverage(),
            "collided":       collided,
            "steps":          self.step_count,
            "collisions":     self._collisions,
            "bumper_fired":   bumper_fired,
            "bumper_total":   self._bumper_triggers,
            "cross_path_pct": self.cross_path_pct(),
        }
        return self._get_obs(), reward, False, False, info

    def step(self, action: int):
        bumper_fired = False
        if self.use_bumper:
            override = self._safety_override()
            if override is not None:
                action = override
                bumper_fired = True
                self._bumper_triggers += 1

        vl_cmd, vr_cmd = self._action_to_wheels[action]
        return self._physics_and_reward(vl_cmd, vr_cmd, bumper_fired)

    def cross_path_pct(self) -> float:
        """Fraction of cell *entries* that revisited an already-visited cell.
        0%  = perfect single-pass coverage; 100 % = stuck looping."""
        unique = int(self.visited.sum())
        total  = int(self._cell_entries)
        if total == 0:
            return 0.0
        return 100.0 * (total - unique) / total

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode is None:
            return

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((WIN_W, WIN_H))
                pygame.display.set_caption("Rover Coverage Explorer")
            else:
                self._screen = pygame.Surface((WIN_W, WIN_H))
            self._clock = pygame.time.Clock()
            self._fonts = {
                "sm":  pygame.font.SysFont("monospace", 13),
                "md":  pygame.font.SysFont("monospace", 16),
                "lg":  pygame.font.SysFont("monospace", 20, bold=True),
            }

        self._draw()

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()
        else:
            return np.transpose(
                np.array(pygame.surfarray.array3d(self._screen)), axes=(1, 0, 2)
            )

    def _w2s(self, x, y):
        """World metres → map pixels (y-axis flipped)."""
        return int(x * PX_PER_M), int((ROOM_H - y) * PX_PER_M)

    def _lerp_color(self, c0, c1, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))

    def _draw(self):
        surf = self._screen
        surf.fill(C_BG)

        self._draw_map(surf)
        self._draw_sidebar(surf)

    def _draw_map(self, surf):
        # Visited cells with age-based colour gradient
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if not self.visited[r, c]:
                    continue
                age  = self.visit_age[r, c]
                t    = 1.0 - age / max(self.step_count, 1)   # 1 = newest
                col  = self._lerp_color(C_VISITED_LO, C_VISITED_HI, t)
                sx   = int(c * CELL_W * PX_PER_M)
                sy   = int((ROOM_H - (r + 1) * CELL_H) * PX_PER_M)
                sw   = max(1, int(CELL_W * PX_PER_M))
                sh   = max(1, int(CELL_H * PX_PER_M))
                pygame.draw.rect(surf, col, (sx, sy, sw, sh))

        # Grid lines
        for c in range(GRID_COLS + 1):
            sx = int(c * CELL_W * PX_PER_M)
            pygame.draw.line(surf, C_GRID, (sx, 0), (sx, MAP_H))
        for r in range(GRID_ROWS + 1):
            sy = int(r * CELL_H * PX_PER_M)
            pygame.draw.line(surf, C_GRID, (0, sy), (MAP_W, sy))

        # Obstacles
        for (ox, oy, ow, oh) in self.obstacles:
            sx, sy = self._w2s(ox, oy + oh)
            sw = int(ow * PX_PER_M)
            sh = int(oh * PX_PER_M)
            pygame.draw.rect(surf, C_OBSTACLE, (sx, sy, sw, sh))
            pygame.draw.rect(surf, C_OBSTACLE_ED, (sx, sy, sw, sh), 2)

        # Trail (fades from old to new)
        trail = list(self._trail)
        n = len(trail)
        for i in range(1, n):
            t   = i / max(n - 1, 1)
            col = self._lerp_color(C_TRAIL_OLD, C_TRAIL_NEW, t)
            p0  = self._w2s(*trail[i - 1])
            p1  = self._w2s(*trail[i])
            pygame.draw.line(surf, col, p0, p1, 2)

        # Sensor rays — order in SENSOR_ANGLES is [left, right, front]
        ray_colors = [C_RAY_L, C_RAY_R, C_RAY_F]
        for i, rel_angle in enumerate(SENSOR_ANGLES):
            angle = self.theta + rel_angle
            dist  = ray_cast(self.x, self.y, angle, self.obstacles)
            ex    = self.x + dist * np.cos(angle)
            ey    = self.y + dist * np.sin(angle)
            p0    = self._w2s(self.x, self.y)
            p1    = self._w2s(ex, ey)
            # Dashed ray (draw full line with alpha via surface)
            pygame.draw.line(surf, (*ray_colors[i], 160), p0, p1, 1)
            pygame.draw.circle(surf, ray_colors[i], p1, 4)

        # Agent body — rotated rectangle
        corners_px = [self._w2s(*c) for c in rover_corners(self.x, self.y, self.theta)]
        pygame.draw.polygon(surf, C_AGENT,    corners_px)
        pygame.draw.polygon(surf, C_AGENT_ED, corners_px, 2)

        # Heading arrow — from centre to a point just past the front edge
        ax, ay = self._w2s(self.x, self.y)
        hx = self.x + (ROVER_LENGTH * 0.6) * np.cos(self.theta)
        hy = self.y + (ROVER_LENGTH * 0.6) * np.sin(self.theta)
        pygame.draw.line(surf, C_HEADING, (ax, ay), self._w2s(hx, hy), 3)

        # Room border (drawn last so it sits on top)
        pygame.draw.rect(surf, C_BORDER, (0, 0, MAP_W, MAP_H), 3)

    def _draw_sidebar(self, surf):
        sx0 = MAP_W
        pygame.draw.rect(surf, C_SIDEBAR_BG, (sx0, 0, SIDEBAR_W, WIN_H))
        pygame.draw.line(surf, C_BORDER, (sx0, 0), (sx0, WIN_H), 2)

        fonts = self._fonts
        x  = sx0 + 12
        y  = 14

        def text(s, color=C_TEXT, font="sm"):
            nonlocal y
            rendered = fonts[font].render(s, True, color)
            surf.blit(rendered, (x, y))
            y += rendered.get_height() + 3

        def gap(n=6):
            nonlocal y
            y += n

        def bar(value, width=SIDEBAR_W - 24, height=14, color=C_BAR_FG):
            nonlocal y
            pygame.draw.rect(surf, C_BAR_BG, (x, y, width, height), border_radius=3)
            fill = int(np.clip(value, 0.0, 1.0) * width)
            if fill > 0:
                pygame.draw.rect(surf, color, (x, y, fill, height), border_radius=3)
            y += height + 4

        # ── Title ──────────────────────────────────────────────
        text("ROVER COVERAGE", C_TEXT, "lg")
        text("  continuous farm exploration", C_TEXT_DIM, "sm")
        gap(4)

        # ── Step counter ───────────────────────────────────────
        text(f"Step       : {self.step_count:6d}", C_TEXT_DIM)
        gap()

        # ── Coverage ───────────────────────────────────────────
        cov = self._coverage()
        text(f"Coverage   : {cov*100:5.1f}%", C_TEXT, "md")
        bar(cov, color=C_BAR_FG)
        gap()

        # ── Stats ──────────────────────────────────────────────
        text(f"Cells      : {int(self.visited.sum())} / {GRID_ROWS*GRID_COLS}", C_TEXT_DIM)
        text(f"Cross-path : {self.cross_path_pct():5.1f}%", C_TEXT_DIM)
        text(f"Collisions : {self._collisions}", C_TEXT_DIM)
        bumper_state = "on" if self.use_bumper else "OFF"
        text(f"Bumper     : {bumper_state} ({self._bumper_triggers})", C_TEXT_DIM)
        text(f"Total rew  : {self._total_reward:+.1f}", C_TEXT_DIM)
        gap()

        # ── Sensors ────────────────────────────────────────────
        text("─── Sensors ───────────", C_TEXT_DIM, "sm")
        gap(2)
        obs = self._get_obs()
        sensor_info = [
            ("Left ", obs[0], C_RAY_L),
            ("Right", obs[1], C_RAY_R),
            ("Front", obs[2], C_RAY_F),
        ]
        for label, val, col in sensor_info:
            text(f"{label}  {val*SENSOR_MAX:4.2f} m  ({val:.2f})", col, "sm")
            bar(val, color=col)

        gap(2)

        # ── Agent state ────────────────────────────────────────
        text("─── Agent state ───────", C_TEXT_DIM, "sm")
        gap(2)
        deg = np.degrees(self.theta)
        text(f"x      : {self.x:5.2f} m", C_TEXT_DIM)
        text(f"y      : {self.y:5.2f} m", C_TEXT_DIM)
        text(f"heading: {deg:+6.1f}°", C_TEXT_DIM)
        gap()

        # ── Legend ─────────────────────────────────────────────
        text("─── Legend ────────────", C_TEXT_DIM, "sm")
        gap(2)

        def legend_dot(col, label):
            nonlocal y
            pygame.draw.circle(surf, col, (x + 6, y + 7), 6)
            lbl = fonts["sm"].render(label, True, C_TEXT_DIM)
            surf.blit(lbl, (x + 18, y))
            y += lbl.get_height() + 4

        legend_dot(C_AGENT,       "Agent")
        legend_dot(C_VISITED_HI,  "Recently visited")
        legend_dot(C_VISITED_LO,  "Earlier visited")
        legend_dot(C_RAY_L,       "Left sensor")
        legend_dot(C_RAY_R,       "Right sensor")
        legend_dot(C_RAY_F,       "Front sensor")

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None


def _wheels_to_action(vl_sign: int, vr_sign: int) -> int:
    return WHEEL_CMDS.index(vl_sign) * 3 + WHEEL_CMDS.index(vr_sign)


# ─── Reactive Agent ───────────────────────────────────────────────────────────

class ReactiveAgent:
    """
    Reactive rover with alternating turn preference and random drift.

    Normal flow:
      1. Forward (with occasional random drift to break straight-line loops).
      2. Curve phase: one wheel stopped, arc toward preferred side.
      3. Spin phase: hard in-place rotation when very close.
      After each complete avoidance episode the preferred direction flips,
      so consecutive turns alternate L-R-L-R instead of always the same way.

    Escape flow (after STUCK_LIMIT consecutive collisions):
      Reverse → spin ~180° → resume.
    """

    # Front thresholds
    FRONT_CURVE  = 0.55
    FRONT_SPIN   = 0.30
    FRONT_CLEAR  = 0.70

    # Side thresholds
    SIDE_CURVE   = 0.28
    SIDE_SPIN    = 0.22

    # Escape
    STUCK_LIMIT  = 8
    BACKUP_STEPS = 20
    ESCAPE_SPINS = 60

    # Drift: every DRIFT_INTERVAL forward steps take one curved step
    DRIFT_INTERVAL = 60

    def __init__(self, seed: int = 0):
        self._rng           = np.random.default_rng(seed)
        self._state         = "forward"
        self._preferred_dir = "left"   # flips after every avoidance episode
        self._turn_dir      = "left"
        self._was_avoiding  = False
        self._stuck_count   = 0
        self._escape_timer  = 0
        self._escape_phase  = None
        self._forward_steps = 0
        self._drift_dir     = "left"
        self._next_drift    = int(self._rng.integers(30, self.DRIFT_INTERVAL))

    def _flip(self, d: str) -> str:
        return "right" if d == "left" else "left"

    def act(self, obs: np.ndarray, collided: bool = False) -> int:
        left_m  = float(obs[0]) * SENSOR_MAX
        right_m = float(obs[1]) * SENSOR_MAX
        front_m = float(obs[2]) * SENSOR_MAX

        # ── Escape ────────────────────────────────────────────────
        if collided:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        if self._stuck_count >= self.STUCK_LIMIT:
            self._stuck_count  = 0
            self._escape_phase = "backup"
            self._escape_timer = self.BACKUP_STEPS
            self._preferred_dir = self._flip(self._preferred_dir)

        if self._escape_phase == "backup":
            self._escape_timer -= 1
            if self._escape_timer <= 0:
                self._escape_phase = "spin"
                self._escape_timer = self.ESCAPE_SPINS
                self._turn_dir = self._preferred_dir
            return _wheels_to_action(-1, -1)

        if self._escape_phase == "spin":
            self._escape_timer -= 1
            if self._escape_timer <= 0:
                self._escape_phase = None
                self._state = "forward"
            return (_wheels_to_action(-1, 1) if self._turn_dir == "left"
                    else _wheels_to_action(1, -1))

        # ── Normal avoidance ──────────────────────────────────────
        need_spin  = False
        need_curve = False
        # Start from preferred direction; sensor urgency can override.
        turn_dir   = self._preferred_dir

        if front_m < self.FRONT_SPIN:
            need_spin = True
            # sensor tie → use preferred; otherwise follow clearance
            turn_dir  = self._preferred_dir if abs(left_m - right_m) < 0.1 \
                        else ("left" if left_m >= right_m else "right")
        elif front_m < self.FRONT_CURVE:
            need_curve = True
            turn_dir   = self._preferred_dir if abs(left_m - right_m) < 0.1 \
                         else ("left" if left_m >= right_m else "right")

        if left_m < self.SIDE_SPIN:
            need_spin = True
            turn_dir  = "right"
        elif left_m < self.SIDE_CURVE:
            need_curve = True
            turn_dir  = "right"

        if right_m < self.SIDE_SPIN:
            need_spin = True
            turn_dir  = "left"
        elif right_m < self.SIDE_CURVE:
            need_curve = True
            turn_dir  = "left"

        now_avoiding = need_spin or need_curve

        # Flip preferred direction when an avoidance episode ends
        if self._was_avoiding and not now_avoiding:
            self._preferred_dir = self._flip(self._preferred_dir)
            self._drift_dir     = self._preferred_dir
            self._next_drift    = int(self._rng.integers(30, self.DRIFT_INTERVAL))
            self._forward_steps = 0
        self._was_avoiding = now_avoiding

        if need_spin:
            self._state    = "spin"
            self._turn_dir = turn_dir
        elif need_curve:
            self._state    = "curve"
            self._turn_dir = turn_dir
        else:
            self._state = "forward"

        # ── Actions ───────────────────────────────────────────────
        if self._state == "forward":
            self._forward_steps += 1
            # Periodic random drift: one curved step to nudge the heading
            if self._forward_steps >= self._next_drift:
                self._forward_steps = 0
                self._next_drift = int(self._rng.integers(30, self.DRIFT_INTERVAL))
                return (_wheels_to_action(0, 1) if self._drift_dir == "left"
                        else _wheels_to_action(1, 0))
            return _wheels_to_action(1, 1)

        if self._turn_dir == "left":
            return (_wheels_to_action(0, 1) if self._state == "curve"
                    else _wheels_to_action(-1, 1))
        else:
            return (_wheels_to_action(1, 0) if self._state == "curve"
                    else _wheels_to_action(1, -1))


# ─── WASD key → action mapping ───────────────────────────────────────────────

def _wasd_action(keys) -> int:
    w = keys[pygame.K_w]
    s = keys[pygame.K_s]
    a = keys[pygame.K_a]
    d = keys[pygame.K_d]
    if w and a:
        return _wheels_to_action(0,  1)
    if w and d:
        return _wheels_to_action(1,  0)
    if w:
        return _wheels_to_action(1,  1)
    if s:
        return _wheels_to_action(-1, -1)
    if a:
        return _wheels_to_action(-1,  1)
    if d:
        return _wheels_to_action(1,  -1)
    return _wheels_to_action(0, 0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "manual"

    print("=" * 60)
    if mode == "auto":
        print("  Rover Coverage — Reactive agent (auto)")
    else:
        print("  Rover Coverage — WASD manual control")
        print("  W/S = forward/reverse   A/D = spin   W+A/W+D = curve")
    print("  R = reset   Q / close = quit")
    print("=" * 60)

    env = RoverCoverageEnv(render_mode="human")
    obs, _ = env.reset(seed=0)
    env.use_bumper = False   # reactive agent handles avoidance itself
    env.render()

    agent = ReactiveAgent() if mode == "auto" else None
    info: dict = {}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    env._map_ready = False
                    obs, _ = env.reset()
                    if agent:
                        agent.__init__()
                    print("  Reset.")

        if not running:
            break

        if agent:
            action = agent.act(obs, collided=info.get("collided", False))
        else:
            action = _wasd_action(pygame.key.get_pressed())

        obs, _, _, _, info = env.step(action)

        if info.get("collided"):
            print(f"  Collision! total={info['collisions']}  "
                  f"coverage={info['coverage']*100:.1f}%")

        env.render()

    env.close()


if __name__ == "__main__":
    main()
