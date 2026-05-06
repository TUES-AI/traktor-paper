"""
ApartmentEnv — a structured indoor map for coverage evaluation.

Layout (20 × 15 m):

  +--------+-----D-----+--------+
  |        |    r2     |   r3   |
  | room1  +--+     +--+  +--+ |
  |        |al|     |alcove  | |
  +----D---+--+-----+--------D-+
  |              hall           |
  +-D------+----------+-----D--+
  |        |          |        |
  | room4  |  room5   | room6  |
  |        |          |        |
  +--------+----D-----+--------+

Wall thickness : 0.15 m
Doorway width  : 2.0 m
Furniture      : 3 pieces per room, randomly placed

Harder for boustrophedon:
  - Top/bottom doorway positions are MISALIGNED — a straight sweep through a
    bottom door does not line up with any top door, forcing intentional turns.
  - Dead-end alcoves inside room2 and room3 require navigating around a
    partial wall — boustrophedon gets trapped in the pocket.
  - 3 furniture pieces per room increase clutter and dead-ends.
  - Vertical hall doorways are offset left/right of centre.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Field geometry ────────────────────────────────────────────────────────────
APT_W  = 20.0   # metres
APT_H  = 15.0   # metres

# Grid resolution — keep 25 cm cells (same as base env)
APT_COLS = int(APT_W / 0.25)   # 80
APT_ROWS = int(APT_H / 0.25)   # 60
APT_CELL_W = APT_W / APT_COLS
APT_CELL_H = APT_H / APT_ROWS

# Reuse physics / sensor constants from base env
from rover_coverage_env import (
    MAX_WHEEL_SPEED, SENSOR_MAX, SENSOR_MIN, SENSOR_NOISE_STD,
    SENSOR_ANGLES, WHEEL_CMDS,
    AGENT_RADIUS, ROVER_LENGTH, ROVER_WIDTH,
    FRONT_SAFETY_DIST, SIDE_SAFETY_DIST,
    ACT_BACKWARD, ACT_SPIN_LEFT, ACT_SPIN_RIGHT,
    INERTIA_ALPHA,
    R_COLLISION, R_STEP, R_MOVE, R_STUCK, STUCK_LIMIT,
    TRAIL_LEN, PX_PER_M, SIDEBAR_W,
    _ray_aabb_hit, diff_drive_step,
    rover_corners,
    C_BG, C_GRID, C_BORDER, C_OBSTACLE, C_OBSTACLE_ED,
    C_AGENT, C_AGENT_ED, C_HEADING, C_VISITED_LO, C_VISITED_HI,
    C_TRAIL_OLD, C_TRAIL_NEW, C_RAY_L, C_RAY_R, C_RAY_F,
    C_SIDEBAR_BG, C_TEXT, C_TEXT_DIM,
)


# ── Field-aware physics helpers (use APT_W/H, not ROOM_W/H=10) ───────────────

def _apt_ray_cast(px, py, angle, obstacles, max_dist=SENSOR_MAX):
    """ray_cast that uses APT_W × APT_H field boundaries."""
    dx, dy = np.cos(angle), np.sin(angle)
    t_min = max_dist
    # Field boundary hits
    for wall_x in (0.0, APT_W):
        if abs(dx) > 1e-9:
            t = (wall_x - px) / dx
            if 1e-6 < t < t_min:
                t_min = t
    for wall_y in (0.0, APT_H):
        if abs(dy) > 1e-9:
            t = (wall_y - py) / dy
            if 1e-6 < t < t_min:
                t_min = t
    for obs in obstacles:
        t = _ray_aabb_hit(px, py, dx, dy, *obs)
        if t is not None and 1e-6 < t < t_min:
            t_min = t
    return t_min


def _apt_in_collision(px, py, obstacles) -> bool:
    """in_collision that uses APT_W × APT_H field boundaries."""
    r = AGENT_RADIUS
    if px - r < 0.0 or px + r > APT_W or py - r < 0.0 or py + r > APT_H:
        return True
    for ox, oy, ow, oh in obstacles:
        if (ox - r < px < ox + ow + r) and (oy - r < py < oy + oh + r):
            # Closer AABB check
            cx = max(ox, min(px, ox + ow))
            cy = max(oy, min(py, oy + oh))
            if (px - cx) ** 2 + (py - cy) ** 2 < r * r:
                return True
    return False

WALL_T           = 0.15   # wall thickness (m)
DOOR_W           = 2.00   # doorway width (m)
MAX_HORIZON_STEPS = 30    # horizon=1.0 → 30 physics sub-steps = 1.5 s
YAW_RATE_MAX      = 6.0   # rad/s — max yaw rate at full differential spin

# ── Apartment wall layout ─────────────────────────────────────────────────────
#
#  Asymmetric 7-room layout.  4 rooms across the top, 3 across the bottom,
#  connected by a narrow 1.5m corridor.  Doors are near room corners, never
#  aligned vertically, so no straight sweep passes through two doors.
#
#  x cols (top):  0──5──10──14──20   (widths 5, 5, 4, 6)
#  x cols (bot):  0────8────14──20   (widths 8, 6, 6)
#  y rows:        0──────6.5──8.0──15
#
#  Top wall doors (y=8.0):  x=1.5, 8.0, 12.5, 18.5  (near corners)
#  Bot wall doors (y=6.5):  x=4.0, 11.0, 16.5        (misaligned)
#  Corridor fully open (no vertical dividers)
#
#  Extra walls:
#    - L-stub inside room1 (top-left) — creates dead-end pocket
#    - L-stub inside room6 (bot-right) — forces U-turn
#    - Narrow passage inside room3 (top, 4m wide) — partial divider

HALL_Y_LO = 6.5
HALL_Y_HI = 8.0

def _apt_walls():
    walls = []

    # ── Top horizontal wall (y=8.0): 4 doors near room corners ───────────────
    _hwall_with_doors(walls, y=HALL_Y_HI, x0=0, x1=APT_W,
                      doors=[(1.5, DOOR_W), (8.0, DOOR_W), (12.5, DOOR_W), (18.5, DOOR_W)])

    # ── Bottom horizontal wall (y=6.5): 3 doors, all misaligned with top ─────
    _hwall_with_doors(walls, y=HALL_Y_LO, x0=0, x1=APT_W,
                      doors=[(4.0, DOOR_W), (11.0, DOOR_W), (16.5, DOOR_W)])

    # ── Vertical dividers — top section (y=8 to 15), 4 rooms ─────────────────
    _vwall_with_doors(walls, x=5.0,  y0=HALL_Y_HI, y1=APT_H, doors=[])
    _vwall_with_doors(walls, x=10.0, y0=HALL_Y_HI, y1=APT_H, doors=[])
    _vwall_with_doors(walls, x=14.0, y0=HALL_Y_HI, y1=APT_H, doors=[])

    # ── Vertical dividers — bottom section (y=0 to 6.5), 3 rooms ─────────────
    _vwall_with_doors(walls, x=8.0,  y0=0, y1=HALL_Y_LO, doors=[])
    _vwall_with_doors(walls, x=14.0, y0=0, y1=HALL_Y_LO, doors=[])

    # ── Internal partial walls — make boustrophedon fail harder ───────────────
    # Corridor blocker at x=10: partial vertical wall, gap at top (0.8 m passage)
    walls.append((10.0 - WALL_T/2, HALL_Y_LO,       WALL_T, 0.8))   # 6.5–7.3 blocked
    # Dead-end stub in room1 (top-left): vertical wall at x=3
    walls.append((3.0,  10.0,  WALL_T, 2.5))   # x=3, y=10–12.5
    # Shelf in room4 (top-right): horizontal ledge at y=12
    walls.append((15.0, 12.0,  3.5,   WALL_T))  # x=15–18.5, y=12
    # Nook wall in room5 (bot-left): horizontal wall at y=3.5
    walls.append((0.3,  3.5,   3.5,   WALL_T))  # x=0.3–3.8, y=3.5
    # Pocket in room6 (bot-mid): vertical stub at x=11
    walls.append((11.0, 0.3,   WALL_T, 2.5))   # x=11, y=0.3–2.8

    return walls


def _hwall_with_doors(walls, y, x0, x1, doors):
    """Horizontal wall at height y from x0 to x1, with doorways (cx, width)."""
    # Sort doors by position
    cut_points = [(x0, False)] + [(cx, True) for cx, _ in doors] + [(x1, False)]
    # Build segments between doors
    cursor = x0
    for cx, dw in doors:
        seg_end = cx - dw / 2
        if seg_end > cursor + 1e-3:
            walls.append((cursor, y, seg_end - cursor, WALL_T))
        cursor = cx + dw / 2
    if x1 - cursor > 1e-3:
        walls.append((cursor, y, x1 - cursor, WALL_T))


def _vwall_with_doors(walls, x, y0, y1, doors):
    """Vertical wall at x from y0 to y1, with doorways (cy, width)."""
    cursor = y0
    for cy, dw in doors:
        seg_end = cy - dw / 2
        if seg_end > cursor + 1e-3:
            walls.append((x, cursor, WALL_T, seg_end - cursor))
        cursor = cy + dw / 2
    if y1 - cursor > 1e-3:
        walls.append((x, cursor, WALL_T, y1 - cursor))


# ── Furniture per room ────────────────────────────────────────────────────────

_ROOMS = [
    # (x0, y0, x1, y1) interior bounds — matches new 4-top / 3-bottom layout
    (0.3,  8.3,  4.7,  14.7),   # room1 top-left   (5×7)
    (5.3,  8.3,  9.7,  14.7),   # room2 top-mid-L  (5×7)
    (10.3, 8.3,  13.7, 14.7),   # room3 top-mid-R  (4×7)
    (14.3, 8.3,  19.7, 14.7),   # room4 top-right  (6×7)
    (0.3,  0.3,  7.7,  6.2),    # room5 bot-left   (8×6.5)
    (8.3,  0.3,  13.7, 6.2),    # room6 bot-mid    (6×6.5)
    (14.3, 0.3,  19.7, 6.2),    # room7 bot-right  (6×6.5)
]


def _furniture(rng, n_per_room=2):
    """Place n_per_room non-overlapping furniture pieces per room."""
    pieces = []
    for rx0, ry0, rx1, ry1 in _ROOMS:
        placed = []
        for _ in range(n_per_room * 40):
            if len(placed) >= n_per_room:
                break
            w = rng.uniform(0.4, 0.9)
            h = rng.uniform(0.4, 0.9)
            margin = AGENT_RADIUS + 0.30   # generous clearance so rover can pass
            if rx1 - rx0 - 2 * margin < w or ry1 - ry0 - 2 * margin < h:
                continue
            x = rng.uniform(rx0 + margin, rx1 - margin - w)
            y = rng.uniform(ry0 + margin, ry1 - margin - h)
            ok = all(
                x + w + 0.50 < px or px + pw + 0.50 < x or
                y + h + 0.50 < py or py + ph + 0.50 < y
                for px, py, pw, ph in placed
            )
            if ok:
                placed.append((x, y, w, h))
        pieces.extend(placed)
    return pieces


def generate_apartment(rng=None, n_furniture_per_room=3):
    """Return the full obstacle list: walls + furniture."""
    if rng is None:
        rng = np.random.default_rng()
    return _apt_walls() + _furniture(rng, n_furniture_per_room)


# ── Environment ───────────────────────────────────────────────────────────────

class ApartmentContinuousEnv(gym.Env):
    """
    Same physics and sensor model as ContinuousRoverEnv / RoverCoverageEnv,
    but on the 20×15 m apartment map.  The grid, rendering, and reward
    structure are kept identical so the same wrappers (SafetyPenaltyWrapper,
    VMMObsWrapper) work unchanged.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, seed=0, render_mode=None, obstacles=None, n_furniture=2):
        super().__init__()
        self.render_mode  = render_mode
        self._seed        = seed
        self._rng         = np.random.default_rng(seed)
        self._n_furniture = n_furniture

        # Fixed structural walls — same every episode
        self._walls = _apt_walls()
        # Furniture — randomised per episode if obstacles=None
        self._fixed_obstacles = obstacles  # None → regenerate each reset

        self.obstacles: list = []

        # Gym spaces
        # [curvature, horizon, speed]  — matches PROJECT.md guide format
        # horizon ∈ [0,1] maps to 1–MAX_HORIZON_STEPS physics sub-steps
        self.action_space = spaces.Box(
            low =np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # State — initialised in reset()
        self.x = self.y = self.theta = 0.0
        self.yaw_rate = 0.0   # rad/s — simulated gyro-z, updated each step
        self.visited    = np.zeros((APT_ROWS, APT_COLS), dtype=bool)
        self.visit_age  = np.full((APT_ROWS, APT_COLS), -1, dtype=np.int32)
        self.step_count = 0
        self._vl = self._vr = 0.0
        self._collisions   = 0
        self._bumper_triggers = 0
        self._steps_since_new_cell = 0
        self._prev_cell    = (-1, -1)
        self.use_bumper        = True
        self.use_stuck_respawn = True   # set False for deterministic baselines
        self._total_reward     = 0.0
        self._trail: list  = []

        self._screen = self._clock = None
        self._fonts: dict = {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cell(self, px, py):
        col = int(np.clip(px / APT_CELL_W, 0, APT_COLS - 1))
        row = int(np.clip(py / APT_CELL_H, 0, APT_ROWS - 1))
        return row, col

    def _coverage(self):
        return float(self.visited.sum()) / (APT_ROWS * APT_COLS)

    def _mark_swept(self, step_idx):
        r0, c0 = self._cell(self.x, self.y)
        rad_cells = int(np.ceil(AGENT_RADIUS / min(APT_CELL_W, APT_CELL_H))) + 1
        r2 = AGENT_RADIUS ** 2
        for dr in range(-rad_cells, rad_cells + 1):
            r = r0 + dr
            if r < 0 or r >= APT_ROWS:
                continue
            cy_pos = (r + 0.5) * APT_CELL_H
            for dc in range(-rad_cells, rad_cells + 1):
                c = c0 + dc
                if c < 0 or c >= APT_COLS:
                    continue
                cx_pos = (c + 0.5) * APT_CELL_W
                if (cx_pos - self.x) ** 2 + (cy_pos - self.y) ** 2 <= r2:
                    if not self.visited[r, c]:
                        self.visited[r, c]   = True
                        self.visit_age[r, c] = step_idx

    def _get_obs(self):
        out = np.empty(3, dtype=np.float32)
        cone = (-np.pi / 6, -np.pi / 12, 0.0, np.pi / 12, np.pi / 6)
        for i, a in enumerate(SENSOR_ANGLES):
            d = min(_apt_ray_cast(self.x, self.y, self.theta + a + da, self.obstacles)
                    for da in cone)
            d += float(self._rng.normal(0.0, SENSOR_NOISE_STD))
            d  = max(SENSOR_MIN, min(SENSOR_MAX, d))
            out[i] = d / SENSOR_MAX
        return out

    def _safe_spawn(self):
        clearance = AGENT_RADIUS + 0.25
        for _ in range(2000):
            room = self._rng.integers(len(_ROOMS))
            rx0, ry0, rx1, ry1 = _ROOMS[room]
            px = self._rng.uniform(rx0 + clearance, rx1 - clearance)
            py = self._rng.uniform(ry0 + clearance, ry1 - clearance)
            if not _apt_in_collision(px, py, self.obstacles):
                return px, py
        return APT_W / 2, APT_H / 2  # fallback

    def _safety_override(self):
        obs = self._get_obs()
        front_m = obs[2] * SENSOR_MAX
        left_m  = obs[0] * SENSOR_MAX
        right_m = obs[1] * SENSOR_MAX
        if front_m < FRONT_SAFETY_DIST:
            return ACT_SPIN_LEFT if left_m >= right_m else ACT_SPIN_RIGHT
        if left_m < SIDE_SAFETY_DIST:
            return ACT_SPIN_RIGHT
        if right_m < SIDE_SAFETY_DIST:
            return ACT_SPIN_LEFT
        return None

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._fixed_obstacles is not None:
            self.obstacles = list(self._walls) + list(self._fixed_obstacles)
        else:
            self.obstacles = list(self._walls) + _furniture(self._rng, self._n_furniture)

        self.visited   = np.zeros((APT_ROWS, APT_COLS), dtype=bool)
        self.visit_age = np.full((APT_ROWS, APT_COLS), -1, dtype=np.int32)
        self.step_count = 0
        self._vl = self._vr = 0.0
        self._collisions   = 0
        self._bumper_triggers = 0
        self._steps_since_new_cell = 0
        self._prev_cell    = (-1, -1)
        self._total_reward = 0.0
        self._trail        = []

        self.x, self.y = self._safe_spawn()
        self.theta = float(self._rng.uniform(-np.pi, np.pi))
        self._mark_swept(0)

        return self._get_obs(), {}

    def step(self, action):
        c = float(np.clip(action[0], -1.0, 1.0))
        h = float(np.clip(action[1],  0.0, 1.0)) if len(action) > 2 else 0.0
        s = float(np.clip(action[2] if len(action) > 2 else action[1], 0.0, 1.0))

        n_sub = max(1, round(h * MAX_HORIZON_STEPS))
        vl_cmd = float(np.clip(1.0 - 2.0 * c, -1.0, 1.0)) * s * MAX_WHEEL_SPEED
        vr_cmd = float(np.clip(1.0 + 2.0 * c, -1.0, 1.0)) * s * MAX_WHEEL_SPEED

        theta_before = self.theta
        cum_reward = 0.0
        cum_bumper  = False

        for _ in range(n_sub):
            vl, vr = vl_cmd, vr_cmd
            bumper_fired = False
            if self.use_bumper:
                override = self._safety_override()
                if override is not None:
                    _act_map = [
                        (vl_ * MAX_WHEEL_SPEED, vr_ * MAX_WHEEL_SPEED)
                        for vl_ in WHEEL_CMDS for vr_ in WHEEL_CMDS
                    ]
                    vl, vr = _act_map[override]
                    bumper_fired = True
                    cum_bumper = True
            obs, reward, term, trunc, info = self._physics_and_reward(vl, vr, bumper_fired)
            cum_reward += reward
            if term or trunc:
                break

        # Simulated gyro-z: total yaw change over sub-steps, normalised
        delta_theta = (self.theta - theta_before + np.pi) % (2 * np.pi) - np.pi
        self.yaw_rate = float(np.clip(delta_theta / (n_sub * 0.05) / YAW_RATE_MAX, -1.0, 1.0))

        if cum_bumper:
            self._bumper_triggers += 1
        info["bumper_fired"]  = cum_bumper
        info["bumper_total"]  = self._bumper_triggers
        return obs, cum_reward, term, trunc, info

    def _physics_and_reward(self, vl_cmd, vr_cmd, bumper_fired):
        # Inertia
        self._vl += INERTIA_ALPHA * (vl_cmd - self._vl)
        self._vr += INERTIA_ALPHA * (vr_cmd - self._vr)

        prev_x, prev_y = self.x, self.y
        nx, ny, ntheta = diff_drive_step(
            self.x, self.y, self.theta, self._vl, self._vr)

        collided = False
        if _apt_in_collision(nx, ny, self.obstacles):
            collided = True
            self._collisions += 1
            self._vl = self._vr = 0.0
            # Try partial moves
            if not _apt_in_collision(nx, self.y, self.obstacles):
                self.x = nx
            elif not _apt_in_collision(self.x, ny, self.obstacles):
                self.y = ny
            self.theta = ntheta
        else:
            self.x, self.y, self.theta = nx, ny, ntheta

        # Clamp to field
        r = AGENT_RADIUS
        self.x = float(np.clip(self.x, r, APT_W - r))
        self.y = float(np.clip(self.y, r, APT_H - r))

        self.step_count += 1
        self._trail.append((self.x, self.y))
        if len(self._trail) > TRAIL_LEN:
            self._trail.pop(0)

        dist = float(np.hypot(self.x - prev_x, self.y - prev_y))
        prev_cells = self.visited.sum()
        self._mark_swept(self.step_count)
        new_cells = int(self.visited.sum() - prev_cells)

        cur_cell = self._cell(self.x, self.y)
        if cur_cell != self._prev_cell:
            self._steps_since_new_cell = 0
            self._prev_cell = cur_cell
        else:
            self._steps_since_new_cell += 1

        R_NEW_CELL = 1.5   # dense bonus per newly visited grid cell
        reward = R_STEP + R_MOVE * dist + R_NEW_CELL * new_cells
        if collided:
            reward += R_COLLISION
        if self.use_stuck_respawn and self._steps_since_new_cell >= STUCK_LIMIT:
            reward += R_STUCK
            self._steps_since_new_cell = 0
            self.x, self.y = self._safe_spawn()
            self.theta = float(self._rng.uniform(-np.pi, np.pi))

        self._total_reward += reward
        obs  = self._get_obs()
        info = {
            "bumper_fired":   bumper_fired,
            "collisions":     self._collisions,
            "bumper_total":   self._bumper_triggers,
            "coverage":       self._coverage(),
            "steps":          self.step_count,
        }
        return obs, reward, False, False, info

    def render(self):
        import pygame, sys
        APT_MAP_W = int(APT_W * PX_PER_M)
        APT_MAP_H = int(APT_H * PX_PER_M)
        WIN_W     = APT_MAP_W + SIDEBAR_W
        WIN_H     = APT_MAP_H

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((WIN_W, WIN_H))
                pygame.display.set_caption("ApartmentEnv")
            else:
                self._screen = pygame.Surface((WIN_W, WIN_H))
            self._clock = pygame.time.Clock()

        def w2s(x, y):
            return int(x * PX_PER_M), int((APT_H - y) * PX_PER_M)

        def lerp(c0, c1, t):
            t = max(0.0, min(1.0, t))
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))

        surf = self._screen
        surf.fill(C_BG)

        # Visited cells
        for r in range(APT_ROWS):
            for c in range(APT_COLS):
                if not self.visited[r, c]:
                    continue
                age = self.visit_age[r, c]
                t   = 1.0 - age / max(self.step_count, 1)
                col = lerp(C_VISITED_LO, C_VISITED_HI, t)
                sx  = int(c * APT_CELL_W * PX_PER_M)
                sy  = int((APT_H - (r + 1) * APT_CELL_H) * PX_PER_M)
                sw  = max(1, int(APT_CELL_W * PX_PER_M))
                sh  = max(1, int(APT_CELL_H * PX_PER_M))
                pygame.draw.rect(surf, col, (sx, sy, sw, sh))

        # Obstacles (walls + furniture)
        for (ox, oy, ow, oh) in self.obstacles:
            sx, sy = w2s(ox, oy + oh)
            pygame.draw.rect(surf, C_OBSTACLE,    (sx, sy, int(ow * PX_PER_M), int(oh * PX_PER_M)))
            pygame.draw.rect(surf, C_OBSTACLE_ED, (sx, sy, int(ow * PX_PER_M), int(oh * PX_PER_M)), 2)

        # Trail
        trail = list(self._trail)
        n = len(trail)
        for i in range(1, n):
            t   = i / max(n - 1, 1)
            col = lerp(C_TRAIL_OLD, C_TRAIL_NEW, t)
            pygame.draw.line(surf, col, w2s(*trail[i - 1]), w2s(*trail[i]), 2)

        # Sensor rays
        for i, rel_angle in enumerate(SENSOR_ANGLES):
            angle = self.theta + rel_angle
            dist  = _apt_ray_cast(self.x, self.y, angle, self.obstacles)
            ex    = self.x + dist * np.cos(angle)
            ey    = self.y + dist * np.sin(angle)
            col   = [C_RAY_L, C_RAY_R, C_RAY_F][i]
            pygame.draw.line(surf, col, w2s(self.x, self.y), w2s(ex, ey), 1)
            pygame.draw.circle(surf, col, w2s(ex, ey), 4)

        # Agent
        corners_px = [w2s(*c) for c in rover_corners(self.x, self.y, self.theta)]
        pygame.draw.polygon(surf, C_AGENT,    corners_px)
        pygame.draw.polygon(surf, C_AGENT_ED, corners_px, 2)
        hx = self.x + ROVER_LENGTH * 0.6 * np.cos(self.theta)
        hy = self.y + ROVER_LENGTH * 0.6 * np.sin(self.theta)
        pygame.draw.line(surf, C_HEADING, w2s(self.x, self.y), w2s(hx, hy), 3)

        # Border
        pygame.draw.rect(surf, C_BORDER, (0, 0, APT_MAP_W, APT_MAP_H), 3)

        # Sidebar
        sx0 = APT_MAP_W
        pygame.draw.rect(surf, C_SIDEBAR_BG, (sx0, 0, SIDEBAR_W, WIN_H))
        if not self._fonts:
            self._fonts["sm"] = pygame.font.SysFont("monospace", 13)
        font = self._fonts["sm"]
        def text(s, col, row):
            surf.blit(font.render(s, True, col), (sx0 + 10, 10 + row * 18))
        text(f"Coverage : {self._coverage()*100:5.1f}%", C_TEXT, 0)
        text(f"Step     : {self.step_count:6d}",         C_TEXT, 1)
        text(f"Bumper   : {self._bumper_triggers:6d}",   C_TEXT, 2)

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()
        else:
            return np.transpose(np.array(pygame.surfarray.array3d(surf)), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
