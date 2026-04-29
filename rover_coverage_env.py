"""
2D Differential Drive Rover Coverage Navigation Environment
Gymnasium environment with wall-following baseline and rich visualization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
from typing import Optional
from collections import deque

# ─── Constants ───────────────────────────────────────────────────────────────

ROOM_W, ROOM_H = 10.0, 10.0          # metres
GRID_COLS, GRID_ROWS = 20, 20        # coverage grid resolution
CELL_W = ROOM_W / GRID_COLS
CELL_H = ROOM_H / GRID_ROWS

AXLE_LENGTH  = 0.20
DT           = 0.05   # 20 Hz, matching physical rover timestep
MAX_WHEEL_SPEED = 0.6

SENSOR_MAX   = 3.0
AGENT_RADIUS = 0.15

MAX_STEPS    = 2000

# Rewards
R_NEW_CELL  =  1.0
R_COLLISION = -10.0
R_STEP      = -0.01

WHEEL_CMDS = [-1, 0, 1]

# Sensor angles: front, left, right
SENSOR_ANGLES = [0.0, np.pi / 2, -np.pi / 2]

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


# ─── Obstacle generation ──────────────────────────────────────────────────────

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


# ─── Collision detection ──────────────────────────────────────────────────────

def circle_vs_aabb(cx, cy, r, ox, oy, ow, oh) -> bool:
    nx = np.clip(cx, ox, ox + ow)
    ny = np.clip(cy, oy, oy + oh)
    return (cx - nx) ** 2 + (cy - ny) ** 2 < r ** 2


def in_collision(px, py, obstacles) -> bool:
    if px - AGENT_RADIUS < 0 or px + AGENT_RADIUS > ROOM_W:
        return True
    if py - AGENT_RADIUS < 0 or py + AGENT_RADIUS > ROOM_H:
        return True
    return any(circle_vs_aabb(px, py, AGENT_RADIUS, *obs) for obs in obstacles)


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
                 n_obstacles: int = 5, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.n_obstacles = n_obstacles
        self._seed = seed

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
        self._map_ready     = False   # obstacles built yet?

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.array([
            np.float32(ray_cast(self.x, self.y, self.theta + a, self.obstacles) / SENSOR_MAX)
            for a in SENSOR_ANGLES
        ], dtype=np.float32)

    def _cell(self, px, py):
        col = int(np.clip(px / CELL_W, 0, GRID_COLS - 1))
        row = int(np.clip(py / CELL_H, 0, GRID_ROWS - 1))
        return row, col

    def _coverage(self) -> float:
        return float(self.visited.sum()) / (GRID_ROWS * GRID_COLS)

    def _safe_spawn(self):
        for _ in range(1000):
            px = self._rng.uniform(AGENT_RADIUS + 0.1, ROOM_W - AGENT_RADIUS - 0.1)
            py = self._rng.uniform(AGENT_RADIUS + 0.1, ROOM_H - AGENT_RADIUS - 0.1)
            if not in_collision(px, py, self.obstacles):
                return px, py
        return ROOM_W / 2, ROOM_H / 2

    def _respawn(self):
        """Teleport agent to a safe position without touching map or visited grid."""
        self.x, self.y = self._safe_spawn()
        self.theta = self._rng.uniform(-np.pi, np.pi)
        self._trail.clear()
        self.done = False

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Build the map only once; subsequent reset() calls just respawn the agent
        if not self._map_ready:
            self.obstacles  = generate_obstacles(self.n_obstacles, self._rng)
            self.visited    = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
            self.visit_age  = np.full((GRID_ROWS, GRID_COLS), -1, dtype=np.int32)
            self.step_count = 0
            self._total_reward = 0.0
            self._collisions   = 0
            self._map_ready    = True

        self._respawn()

        r, c = self._cell(self.x, self.y)
        self.visited[r, c]   = True
        self.visit_age[r, c] = self.step_count

        return self._get_obs(), {}

    def step(self, action: int):
        vl, vr = self._action_to_wheels[action]
        nx, ny, ntheta = diff_drive_step(self.x, self.y, self.theta, vl, vr)

        collided = in_collision(nx, ny, self.obstacles)
        if collided:
            reward = R_COLLISION
            self._collisions += 1
            self._respawn()           # stay in the same map, just move the agent
        else:
            self.x, self.y, self.theta = nx, ny, ntheta
            r, c = self._cell(self.x, self.y)
            new_cell = not self.visited[r, c]
            self.visited[r, c] = True
            if new_cell:
                self.visit_age[r, c] = self.step_count
            reward = (R_NEW_CELL if new_cell else 0.0) + R_STEP

        self._trail.append((self.x, self.y))
        self._total_reward += reward
        self.step_count    += 1

        info = {
            "coverage":    self._coverage(),
            "collided":    collided,
            "steps":       self.step_count,
            "collisions":  self._collisions,
        }
        # Never terminates — continuous task
        return self._get_obs(), reward, False, False, info

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

        # Sensor rays
        ray_colors = [C_RAY_F, C_RAY_L, C_RAY_R]
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

        # Agent body
        ax, ay = self._w2s(self.x, self.y)
        r_px   = int(AGENT_RADIUS * PX_PER_M)
        pygame.draw.circle(surf, C_AGENT, (ax, ay), r_px)
        pygame.draw.circle(surf, C_AGENT_ED, (ax, ay), r_px, 2)

        # Heading arrow
        hx = self.x + AGENT_RADIUS * 1.3 * np.cos(self.theta)
        hy = self.y + AGENT_RADIUS * 1.3 * np.sin(self.theta)
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
        text(f"Collisions : {self._collisions}", C_TEXT_DIM)
        text(f"Total rew  : {self._total_reward:+.1f}", C_TEXT_DIM)
        gap()

        # ── Sensors ────────────────────────────────────────────
        text("─── Sensors ───────────", C_TEXT_DIM, "sm")
        gap(2)
        obs = self._get_obs()
        sensor_info = [
            ("Front", obs[0], C_RAY_F),
            ("Left ", obs[1], C_RAY_L),
            ("Right", obs[2], C_RAY_R),
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
        legend_dot(C_OBSTACLE,    "Obstacle")
        legend_dot(C_RAY_F,       "Front sensor ray")
        legend_dot(C_RAY_L,       "Left sensor ray")
        legend_dot(C_RAY_R,       "Right sensor ray")

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None


# ─── Wall-following Baseline Agent ───────────────────────────────────────────

class WallFollowerAgent:
    """
    Deterministic wall-follower:
      - Drive forward while front sensor is clear.
      - On obstacle / wall ahead: pivot right in place.
      - Use right sensor P-controller to hug the right wall at target distance.
    """

    def __init__(self,
                 front_thresh: float = 0.18,
                 right_target: float = 0.14,
                 kp: float = 1.5):
        self.front_thresh = front_thresh
        self.right_target = right_target
        self.kp = kp

    def act(self, obs: np.ndarray) -> int:
        front, _, right = obs
        if front < self.front_thresh:
            return _wheels_to_action(1, -1)
        error = right - self.right_target
        correction = int(np.clip(np.round(self.kp * error * 3), -1, 1))
        if correction > 0:
            return _wheels_to_action(1, 0)
        elif correction < 0:
            return _wheels_to_action(0, 1)
        else:
            return _wheels_to_action(1, 1)


def _wheels_to_action(vl_sign: int, vr_sign: int) -> int:
    return WHEEL_CMDS.index(vl_sign) * 3 + WHEEL_CMDS.index(vr_sign)


# ─── Evaluation helper ────────────────────────────────────────────────────────

def run_episodes(env: RoverCoverageEnv, agent, n_episodes: int = 5,
                 render: bool = True, label: str = "Agent") -> list:
    stats = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7 + 13)
        done = False
        info = {}
        while not done:
            action = agent.act(obs)
            step_result = env.step(action)
            obs, done, info = step_result[0], step_result[2], step_result[4]
            if render:
                env.render()
        stats.append({
            "episode":      ep + 1,
            "coverage_pct": info.get("coverage", 0.0) * 100,
            "collided":     int(info.get("collided", False)),
            "steps":        info.get("steps", 0),
        })
        col_str = "YES" if info.get("collided") else "no"
        print(f"  [{label}] ep {ep+1:2d}  "
              f"coverage={stats[-1]['coverage_pct']:5.1f}%  "
              f"collision={col_str:<3s}  "
              f"steps={stats[-1]['steps']:4d}")
    return stats


def print_summary(stats: list, label: str):
    covs    = [s["coverage_pct"] for s in stats]
    cols    = sum(s["collided"]  for s in stats)
    lengths = [s["steps"]        for s in stats]
    print(f"\n{'─'*52}")
    print(f"  {label}  ({len(stats)} episodes)")
    print(f"  Mean coverage  : {np.mean(covs):6.1f}%  (std {np.std(covs):.1f}%)")
    print(f"  Collisions     : {cols} / {len(stats)}")
    print(f"  Mean ep length : {np.mean(lengths):.0f} steps")
    print(f"{'─'*52}\n")


# ─── WASD key → action mapping ───────────────────────────────────────────────
#
#  W        — forward straight   (+1, +1)
#  S        — reverse straight   (-1, -1)
#  A        — spin left          (-1, +1)
#  D        — spin right         (+1, -1)
#  W + A    — curve left         ( 0, +1)
#  W + D    — curve right        (+1,  0)
#  no key   — stop               ( 0,  0)
#
# Combos are checked first so W+A doesn't accidentally fire W alone.

def _wasd_action(keys) -> int:
    w = keys[pygame.K_w]
    s = keys[pygame.K_s]
    a = keys[pygame.K_a]
    d = keys[pygame.K_d]
    if w and a:
        return _wheels_to_action(0,  1)   # curve left
    if w and d:
        return _wheels_to_action(1,  0)   # curve right
    if w:
        return _wheels_to_action(1,  1)   # straight forward
    if s:
        return _wheels_to_action(-1, -1)  # straight reverse
    if a:
        return _wheels_to_action(-1,  1)  # spin left
    if d:
        return _wheels_to_action(1,  -1)  # spin right
    return _wheels_to_action(0, 0)        # stop




# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Rover Coverage — WASD manual control")
    print("  W/S = forward/reverse   A/D = spin   W+A/W+D = curve")
    print("  R = rebuild map         Q / close window = quit")
    print("  On collision the agent respawns — map stays intact.")
    print("=" * 60)

    env = RoverCoverageEnv(render_mode="human", n_obstacles=5)
    env.reset(seed=0)
    env.render()   # initialises pygame before the event loop

    running = True
    while running:
        # ── event handling ────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    # R = wipe the visited grid and rebuild the map from scratch
                    env._map_ready = False
                    env.reset()
                    print("  Map reset.")

        if not running:
            break

        # ── pick action from held keys ────────────────────────────
        action = _wasd_action(pygame.key.get_pressed())

        # ── step ──────────────────────────────────────────────────
        step_result = env.step(action)
        info = step_result[4]

        if info.get("collided"):
            print(f"  Collision — respawned  "
                  f"(total collisions: {info['collisions']}  "
                  f"coverage: {info['coverage']*100:.1f}%)")

        env.render()

    env.close()


if __name__ == "__main__":
    main()
