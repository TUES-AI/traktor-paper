from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


DT = 0.10
WORLD_W = 18.0
WORLD_H = 12.0
ROVER_RADIUS = 0.12
AXLE = 0.24
MAX_SPEED = 0.55
SENSOR_MAX = 4.0
CAM_W = 32
CAM_H = 16
CAM_FOV = math.radians(86.0)
FRONT_STOP_DIST = 0.38
FRONT_TURN_DIST = 0.55
SIDE_CLEAR_DIST = 0.25


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def _ray_aabb(px: float, py: float, dx: float, dy: float, r: Rect) -> Optional[float]:
    tx1 = (r.x - px) / dx if abs(dx) > 1e-9 else -np.inf
    tx2 = (r.x + r.w - px) / dx if abs(dx) > 1e-9 else np.inf
    ty1 = (r.y - py) / dy if abs(dy) > 1e-9 else -np.inf
    ty2 = (r.y + r.h - py) / dy if abs(dy) > 1e-9 else np.inf
    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))
    if tmax < 0.0 or tmin > tmax:
        return None
    return tmin if tmin >= 0.0 else tmax


def _ray_bounds(px: float, py: float, dx: float, dy: float) -> float:
    hits = []
    if abs(dx) > 1e-9:
        hits.extend([(0.0 - px) / dx, (WORLD_W - px) / dx])
    if abs(dy) > 1e-9:
        hits.extend([(0.0 - py) / dy, (WORLD_H - py) / dy])
    hits = [h for h in hits if h > 1e-6]
    return min(hits) if hits else SENSOR_MAX


class RealisticRoverEnv(gym.Env):
    """First-person, local-sensing rover simulator.

    The policy never receives true `x`, `y`, room id, coverage grid, or map.
    `info` contains hidden evaluation metrics so we can compare methods.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, render_mode: Optional[str] = None, max_steps: int = 2000, use_safety: bool = True):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.use_safety = use_safety
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        obs_dim = CAM_W * CAM_H + 3 + 4
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.walls: list[Rect] = []
        self.furniture: list[Rect] = []
        self.obstacles: list[Rect] = []
        self.room_boxes: list[tuple[float, float, float, float]] = []
        self.x = self.y = self.theta = 0.0
        self.vl = self.vr = 0.0
        self.last_turn = 0.0
        self.last_speed = 0.0
        self.yaw_rate = 0.0
        self.accel = 0.0
        self.steps = 0
        self.collisions = 0
        self.safety_clamps = 0
        self.visited = np.zeros((60, 90), dtype=bool)
        self.rooms_seen: set[int] = set()
        self.door_crossings = 0
        self._prev_room = -1
        self._last_valid_room = -1

    def _layout(self):
        r = []
        t = 0.12
        # Boundaries.
        r += [Rect(0, 0, WORLD_W, t), Rect(0, WORLD_H - t, WORLD_W, t), Rect(0, 0, t, WORLD_H), Rect(WORLD_W - t, 0, t, WORLD_H)]
        # Central corridor y in [5.1, 6.9], rooms above/below. Door gaps vary.
        y_lo, y_hi = 5.1, 6.9
        top_doors = [(2.1, 1.2), (7.4, 1.0), (13.2, 1.4), (16.1, 0.9)]
        bot_doors = [(3.8, 1.1), (10.5, 1.3), (15.2, 1.0)]
        self._hwall_with_gaps(r, y_hi, 0, WORLD_W, top_doors, t)
        self._hwall_with_gaps(r, y_lo, 0, WORLD_W, bot_doors, t)
        # Room dividers, no vertical gaps: corridor is the connector.
        for x in (4.5, 9.0, 13.8):
            r.append(Rect(x, y_hi, t, WORLD_H - y_hi))
        for x in (7.0, 12.8):
            r.append(Rect(x, 0, t, y_lo))
        # Door-like dead-end distractors / alcoves.
        r += [Rect(5.6, 8.9, 2.2, t), Rect(5.6, 8.9, t, 1.7), Rect(15.2, 2.4, 1.8, t), Rect(17.0, 2.4, t, 1.5)]
        self.room_boxes = [
            (0.2, y_hi + 0.2, 4.3, WORLD_H - 0.2), (4.7, y_hi + 0.2, 8.8, WORLD_H - 0.2),
            (9.2, y_hi + 0.2, 13.6, WORLD_H - 0.2), (14.0, y_hi + 0.2, WORLD_W - 0.2, WORLD_H - 0.2),
            (0.2, 0.2, 6.8, y_lo - 0.2), (7.2, 0.2, 12.6, y_lo - 0.2),
            (13.0, 0.2, WORLD_W - 0.2, y_lo - 0.2), (0.2, y_lo + 0.1, WORLD_W - 0.2, y_hi - 0.1),
        ]
        furn = []
        for bx0, by0, bx1, by1 in self.room_boxes[:-1]:
            for _ in range(2):
                for _try in range(80):
                    w = float(self._rng.uniform(0.35, 0.9)); h = float(self._rng.uniform(0.35, 0.9))
                    x = float(self._rng.uniform(bx0 + 0.5, bx1 - w - 0.5)); y = float(self._rng.uniform(by0 + 0.5, by1 - h - 0.5))
                    cand = Rect(x, y, w, h)
                    if not self._collides_point(x + w / 2, y + h / 2, list(r) + furn, radius=0.55):
                        furn.append(cand); break
        self.walls = r
        self.furniture = furn
        self.obstacles = r + furn

    @staticmethod
    def _hwall_with_gaps(out: list[Rect], y: float, x0: float, x1: float, gaps: list[tuple[float, float]], t: float):
        cur = x0
        for cx, w in sorted(gaps):
            a, b = cx - w / 2, cx + w / 2
            if a > cur:
                out.append(Rect(cur, y, a - cur, t))
            cur = max(cur, b)
        if cur < x1:
            out.append(Rect(cur, y, x1 - cur, t))

    def _room_id(self, x: float, y: float) -> int:
        for i, (x0, y0, x1, y1) in enumerate(self.room_boxes):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return i
        return -1

    def _collides_point(self, x: float, y: float, obstacles: Optional[list[Rect]] = None, radius: float = ROVER_RADIUS) -> bool:
        if x - radius < 0 or x + radius > WORLD_W or y - radius < 0 or y + radius > WORLD_H:
            return True
        for o in obstacles if obstacles is not None else self.obstacles:
            cx = max(o.x, min(x, o.x + o.w)); cy = max(o.y, min(y, o.y + o.h))
            if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                return True
        return False

    def _ray(self, angle: float, max_dist: float = SENSOR_MAX) -> float:
        dx, dy = math.cos(angle), math.sin(angle)
        d = min(_ray_bounds(self.x, self.y, dx, dy), max_dist)
        for o in self.obstacles:
            h = _ray_aabb(self.x, self.y, dx, dy, o)
            if h is not None and 1e-6 < h < d:
                d = h
        return float(np.clip(d, 0.02, max_dist))

    def _camera(self) -> np.ndarray:
        cols = np.empty(CAM_W, dtype=np.float32)
        for i in range(CAM_W):
            rel = -CAM_FOV / 2 + CAM_FOV * (i + 0.5) / CAM_W
            d = self._ray(self.theta + rel, SENSOR_MAX) / SENSOR_MAX
            cols[i] = d
        # Floor/height proxy: lower rows emphasize near obstacles, upper rows fade.
        img = np.repeat(cols[None, :], CAM_H, axis=0)
        row_gain = np.linspace(1.15, 0.75, CAM_H, dtype=np.float32)[:, None]
        img = np.clip(img * row_gain, 0, 1)
        img += self._rng.normal(0, 0.01, size=img.shape).astype(np.float32)
        return np.clip(img, 0, 1)

    def _sensors(self) -> np.ndarray:
        vals = [self._ray(self.theta + a) / SENSOR_MAX for a in (math.pi / 2, -math.pi / 2, 0.0)]
        vals = np.asarray(vals, dtype=np.float32)
        vals += self._rng.normal(0, 0.003, size=3).astype(np.float32)
        return np.clip(vals, 0, 1)

    def _obs(self) -> np.ndarray:
        cam = self._camera().reshape(-1)
        sensors = self._sensors()
        motion = np.array([
            (self.yaw_rate + 6.0) / 12.0,
            np.clip(abs(self.accel) / 2.0, 0, 1),
            (self.last_turn + 1.0) / 2.0,
            (self.last_speed + 1.0) / 2.0,
        ], dtype=np.float32)
        return np.concatenate([cam, sensors, motion]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._layout()
        for _ in range(2000):
            rid = int(self._rng.integers(len(self.room_boxes)))
            x0, y0, x1, y1 = self.room_boxes[rid]
            x = float(self._rng.uniform(x0 + 0.5, x1 - 0.5)); y = float(self._rng.uniform(y0 + 0.5, y1 - 0.5))
            if not self._collides_point(x, y, radius=0.5):
                self.x, self.y = x, y; break
        self.theta = float(self._rng.uniform(-math.pi, math.pi))
        self.vl = self.vr = self.yaw_rate = self.accel = 0.0
        self.last_turn = self.last_speed = 0.0
        self.steps = self.collisions = self.door_crossings = 0
        self.safety_clamps = 0
        self.visited[:] = False
        self.rooms_seen = {self._room_id(self.x, self.y)}
        self._prev_room = self._room_id(self.x, self.y)
        self._last_valid_room = self._prev_room
        self._mark_visited()
        return self._obs(), self._info(0.0, False)

    def _mark_visited(self):
        c = int(np.clip(self.x / WORLD_W * self.visited.shape[1], 0, self.visited.shape[1] - 1))
        r = int(np.clip(self.y / WORLD_H * self.visited.shape[0], 0, self.visited.shape[0] - 1))
        self.visited[r, c] = True

    def _safety_filter(self, turn: float, speed: float):
        proposed = np.array([turn, speed], dtype=np.float32)
        if not self.use_safety:
            return turn, speed, False, proposed, proposed.copy(), "disabled"

        left, right, front = [float(x) * SENSOR_MAX for x in self._sensors()]
        clamped = False
        reason = "clear"

        # Pure in-place tank rotation is always allowed, matching the hardware
        # safety rule. Side ranges only matter when the rover translates.
        is_pure_spin = abs(speed) < 0.05 and abs(turn) > 0.25
        if is_pure_spin:
            executed = proposed.copy()
            return turn, speed, False, proposed, executed, "pure_spin_allowed"

        if speed > 0.05 and front < FRONT_STOP_DIST:
            turn = 1.0 if left >= right else -1.0
            speed = 0.0
            clamped = True
            reason = "front_stop_spin"
        elif speed > 0.05 and front < FRONT_TURN_DIST:
            turn = max(turn, 0.75) if left >= right else min(turn, -0.75)
            speed = min(speed, 0.20)
            clamped = True
            reason = "front_slow_turn"

        if speed > 0.05 and left < SIDE_CLEAR_DIST:
            turn = min(turn, -0.65)
            speed = min(speed, 0.25)
            clamped = True
            reason = "left_side_clearance"
        if speed > 0.05 and right < SIDE_CLEAR_DIST:
            turn = max(turn, 0.65)
            speed = min(speed, 0.25)
            clamped = True
            reason = "right_side_clearance"

        executed = np.array([turn, speed], dtype=np.float32)
        return turn, speed, clamped, proposed, executed, reason

    def step(self, action):
        raw_turn = float(np.clip(action[0], -1, 1))
        raw_speed = float(np.clip(action[1], -1, 1))
        turn, speed, safety_clamped, proposed_action, executed_action, safety_reason = self._safety_filter(raw_turn, raw_speed)
        if safety_clamped:
            self.safety_clamps += 1
        self.last_turn, self.last_speed = turn, speed
        vl_cmd = np.clip(speed - turn, -1, 1) * MAX_SPEED
        vr_cmd = np.clip(speed + turn, -1, 1) * MAX_SPEED
        old_v = (self.vl + self.vr) / 2
        self.vl += 0.45 * (vl_cmd - self.vl)
        self.vr += 0.45 * (vr_cmd - self.vr)
        v = (self.vl + self.vr) / 2
        omega = (self.vr - self.vl) / AXLE
        nx = self.x + v * math.cos(self.theta) * DT
        ny = self.y + v * math.sin(self.theta) * DT
        nt = (self.theta + omega * DT + math.pi) % (2 * math.pi) - math.pi
        collided = self._collides_point(nx, ny)
        dist = 0.0
        if collided:
            self.collisions += 1
            self.vl = self.vr = 0.0
            self.theta = nt
            reward = -0.8
        else:
            dist = float(math.hypot(nx - self.x, ny - self.y))
            self.x, self.y, self.theta = nx, ny, nt
            reward = -0.01 + 0.25 * dist
        self.yaw_rate = float(np.clip(omega, -6.0, 6.0))
        self.accel = float((v - old_v) / DT)
        self.steps += 1
        room = self._room_id(self.x, self.y)
        if room >= 0:
            self.rooms_seen.add(room)
            if self._last_valid_room >= 0 and room != self._last_valid_room:
                self.door_crossings += 1
            self._last_valid_room = room
        self._prev_room = room
        self._mark_visited()
        terminated = False
        # Physical training is continuous; external scripts decide how long to run.
        truncated = False
        info = self._info(dist, collided)
        info.update({
            "safety_clamped": safety_clamped,
            "safety_clamps": self.safety_clamps,
            "safety_reason": safety_reason,
            "proposed_action": proposed_action,
            "executed_action": executed_action,
        })
        return self._obs(), float(reward), terminated, truncated, info

    def _info(self, dist: float, collided: bool):
        return {
            "coverage": float(self.visited.mean()),
            "rooms_seen": len(self.rooms_seen),
            "door_crossings": self.door_crossings,
            "collisions": self.collisions,
            "safety_clamps": self.safety_clamps,
            "distance": dist,
            "collided": collided,
            "pose": (self.x, self.y, self.theta),
        }

    def render(self):
        # Debug-only topdown RGB. Never use this as policy observation.
        scale = 50
        img = np.zeros((int(WORLD_H * scale), int(WORLD_W * scale), 3), dtype=np.uint8) + 25
        def rect(o: Rect, color):
            x0 = int(o.x * scale); x1 = int((o.x + o.w) * scale)
            y0 = int((WORLD_H - o.y - o.h) * scale); y1 = int((WORLD_H - o.y) * scale)
            img[y0:y1, x0:x1] = color
        for o in self.walls: rect(o, (120, 120, 130))
        for o in self.furniture: rect(o, (150, 90, 50))
        cx, cy = int(self.x * scale), int((WORLD_H - self.y) * scale)
        rr = max(2, int(ROVER_RADIUS * scale))
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr ** 2
        img[mask] = (245, 210, 60)
        hx = int(cx + math.cos(self.theta) * rr * 2)
        hy = int(cy - math.sin(self.theta) * rr * 2)
        if 0 <= hx < img.shape[1] and 0 <= hy < img.shape[0]:
            img[max(0, hy-2):hy+3, max(0, hx-2):hx+3] = (20, 20, 20)
        return img
