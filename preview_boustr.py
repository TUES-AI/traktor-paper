"""Boustrophedon (lawnmower) preview — clean 3-phase U-turn state machine."""
import time
import numpy as np
import pygame
from apartment_env import ApartmentContinuousEnv, generate_apartment
from rover_coverage_env import SENSOR_MAX, MAX_WHEEL_SPEED, DT, AXLE_LENGTH, STUCK_LIMIT

SEED = 42
furniture = generate_apartment(np.random.default_rng(SEED))
env = ApartmentContinuousEnv(seed=SEED, obstacles=furniture, render_mode="human")
env.use_bumper        = True
env.use_stuck_respawn = False
obs, _ = env.reset(seed=SEED)

# ── Parameters ────────────────────────────────────────────────────────────────
TURN_SPEED   = 0.45
FWD_SPEED    = 0.55
FRONT_BLOCK  = 0.50   # m — start U-turn
SIDE_NUDGE   = 0.28   # m — gentle wall avoidance while going forward

omega        = (2 * MAX_WHEEL_SPEED * TURN_SPEED) / AXLE_LENGTH
STEPS_180    = int(np.pi       / omega / DT)   # steps for 180°
STEPS_90     = int(np.pi / 2   / omega / DT)   # steps for 90°
SHIFT_DIST   = 0.55                            # metres to shift laterally
SHIFT_STEPS  = int(SHIFT_DIST / (FWD_SPEED * MAX_WHEEL_SPEED * DT))
ESCAPE_STEPS = STEPS_180 * 2                   # 360° escape when truly stuck

print(f"STEPS_90={STEPS_90}  SHIFT_STEPS={SHIFT_STEPS}  ESCAPE_STEPS={ESCAPE_STEPS}")

# ── State machine ─────────────────────────────────────────────────────────────
# FORWARD → (wall) → TURN1(90°) → SLIDE(shift) → TURN2(90°) → FORWARD
FORWARD, TURN1, SLIDE, TURN2, ESCAPE = range(5)
state        = FORWARD
state_steps  = 0
turn_dir     = 1.0      # +1 = CCW,  -1 = CW  (alternates each U-turn)
escape_count = 0
stuck_count  = 0
prev_cell    = (-1, -1)


def act(obs):
    global state, state_steps, turn_dir, escape_count, stuck_count, prev_cell

    front = float(obs[2]) * SENSOR_MAX
    left  = float(obs[0]) * SENSOR_MAX
    right = float(obs[1]) * SENSOR_MAX

    # Stuck detection
    cur = env._cell(env.x, env.y)
    if cur != prev_cell:
        stuck_count = 0; prev_cell = cur
    else:
        stuck_count += 1

    def _a(c, s): return np.array([c, 0.0, s], dtype=np.float32)

    # ── ESCAPE: spin 360° when genuinely stuck ────────────────────────────────
    if state == ESCAPE:
        state_steps += 1
        if state_steps >= ESCAPE_STEPS:
            state = FORWARD; state_steps = 0
        c = 1.0 if left >= right else -1.0
        return _a(c, TURN_SPEED)

    if stuck_count >= STUCK_LIMIT // 2:
        stuck_count = 0; state = ESCAPE; state_steps = 0
        return _a(1.0 if left >= right else -1.0, TURN_SPEED)

    # ── TURN1: first 90° of the U-turn ───────────────────────────────────────
    if state == TURN1:
        state_steps += 1
        if state_steps >= STEPS_90:
            state = SLIDE; state_steps = 0
        return _a(turn_dir, TURN_SPEED)

    # ── SLIDE: move forward one row width ────────────────────────────────────
    if state == SLIDE:
        state_steps += 1
        if front < FRONT_BLOCK:          # wall during slide → escape
            state = ESCAPE; state_steps = 0
            return _a(1.0 if left >= right else -1.0, TURN_SPEED)
        if state_steps >= SHIFT_STEPS:
            state = TURN2; state_steps = 0
        return _a(0.0, FWD_SPEED)

    # ── TURN2: second 90° of the U-turn ──────────────────────────────────────
    if state == TURN2:
        state_steps += 1
        if state_steps >= STEPS_90:
            state = FORWARD; state_steps = 0
            turn_dir *= -1.0             # alternate direction each U-turn pair
        return _a(turn_dir, TURN_SPEED)

    # ── FORWARD: go straight until wall ──────────────────────────────────────
    if front < FRONT_BLOCK:
        turn_dir = 1.0 if left >= right else -1.0
        state = TURN1; state_steps = 0
        return _a(turn_dir, TURN_SPEED)

    if left  < SIDE_NUDGE: return _a(-0.25, FWD_SPEED)
    if right < SIDE_NUDGE: return _a( 0.25, FWD_SPEED)

    return _a(0.0, FWD_SPEED)


# ── Preview loop ──────────────────────────────────────────────────────────────
print("Boustrophedon — Q or close window to quit.")
env.render()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    action = act(obs)
    obs, _, _, _, _ = env.step(action)
    env.render()
    time.sleep(0.05)

print(f"Final: coverage={env._coverage()*100:.1f}%  collisions={env._collisions}")
env.close()
