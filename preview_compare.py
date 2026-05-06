"""
10k-step preview: Boustrophedon (deterministic) then SAC-NoVMM.
If a trained model exists in results/ it is loaded; otherwise random policy is shown.
Press Q or close the window to skip to the next agent.
"""
import time
import pathlib
import numpy as np
import pygame
from apartment_env import ApartmentContinuousEnv, generate_apartment
from rover_coverage_env import SENSOR_MAX, MAX_WHEEL_SPEED, DT, AXLE_LENGTH, STUCK_LIMIT
from train_sac import make_no_vmm_env, _unwrap_inner, SEEDS

SEED      = SEEDS[0]
MAX_STEPS = 10_000
SLEEP     = 0.05   # real-time: 1 step = 0.05 s sim

# SAC actions internally run up to MAX_HORIZON_STEPS physics sub-steps.
# Run this many boustrophedon steps per render frame so both look the same speed.
from apartment_env import MAX_HORIZON_STEPS
BOUSTR_STEPS_PER_FRAME = max(1, MAX_HORIZON_STEPS // 2)  # match ~avg SAC horizon

furniture = generate_apartment(np.random.default_rng(SEED))

# ── 3-phase boustrophedon factory ────────────────────────────────────────────

def _make_boustr_agent(env_ref):
    FRONT_BLOCK  = 0.55
    FRONT_URGENT = 0.35
    SIDE_NUDGE   = 0.30
    TURN_SPEED   = 0.4
    FWD_SPEED    = 0.5
    ESCAPE_PROB  = 0.02
    omega        = (2 * MAX_WHEEL_SPEED * TURN_SPEED) / AXLE_LENGTH
    TURN_TARGET  = int(np.pi / omega / DT)
    TURN_90      = max(1, TURN_TARGET // 2)
    SHIFT_STEPS  = int(0.5 / (FWD_SPEED * MAX_WHEEL_SPEED * DT))
    ESCAPE_STEPS = TURN_TARGET * 2
    PHASE_FWD, PHASE_TURN1, PHASE_SHIFT, PHASE_TURN2 = 0, 1, 2, 3

    s = dict(phase=PHASE_FWD, phase_step=0, turn_dir=1.0,
             escape_timer=0, stuck_count=0, prev_cell=(-1,-1))
    rng = np.random.default_rng(SEED)

    def act(obs, inner):
        front_m = float(obs[2]) * SENSOR_MAX
        left_m  = float(obs[0]) * SENSOR_MAX
        right_m = float(obs[1]) * SENSOR_MAX

        cur_cell = inner._cell(inner.x, inner.y)
        if cur_cell != s['prev_cell']:
            s['stuck_count'] = 0; s['prev_cell'] = cur_cell
        else:
            s['stuck_count'] += 1

        def _a(c, sp): return np.array([c, 0.0, sp], dtype=np.float32)

        if s['escape_timer'] > 0:
            s['escape_timer'] -= 1
            return _a(1.0 if left_m >= right_m else -1.0, TURN_SPEED)

        if s['stuck_count'] >= STUCK_LIMIT // 2:
            s['stuck_count'] = 0; s['escape_timer'] = ESCAPE_STEPS
            s['phase'] = PHASE_FWD; s['phase_step'] = 0
            return _a(1.0 if left_m >= right_m else -1.0, TURN_SPEED)

        if front_m < FRONT_BLOCK and left_m < SIDE_NUDGE and right_m < SIDE_NUDGE:
            s['escape_timer'] = ESCAPE_STEPS; s['phase'] = PHASE_FWD; s['phase_step'] = 0
            return _a(1.0 if left_m >= right_m else -1.0, TURN_SPEED)

        if s['phase'] == PHASE_TURN1:
            s['phase_step'] += 1
            if s['phase_step'] >= TURN_90:
                s['phase'] = PHASE_SHIFT; s['phase_step'] = 0
            return _a(s['turn_dir'], TURN_SPEED)

        if s['phase'] == PHASE_SHIFT:
            s['phase_step'] += 1
            if front_m < FRONT_URGENT:
                s['escape_timer'] = ESCAPE_STEPS; s['phase'] = PHASE_FWD; s['phase_step'] = 0
                return _a(1.0 if left_m >= right_m else -1.0, TURN_SPEED)
            if s['phase_step'] >= SHIFT_STEPS:
                s['phase'] = PHASE_TURN2; s['phase_step'] = 0
            return _a(0.0, FWD_SPEED)

        if s['phase'] == PHASE_TURN2:
            s['phase_step'] += 1
            if s['phase_step'] >= TURN_90:
                s['phase'] = PHASE_FWD; s['phase_step'] = 0; s['turn_dir'] *= -1.0
            return _a(s['turn_dir'], TURN_SPEED)

        # PHASE_FWD
        if front_m < FRONT_URGENT or front_m < FRONT_BLOCK:
            s['turn_dir'] = 1.0 if left_m >= right_m else -1.0
            s['phase'] = PHASE_TURN1; s['phase_step'] = 0
            return _a(s['turn_dir'], TURN_SPEED)

        if left_m  < SIDE_NUDGE: return _a(-0.3, FWD_SPEED)
        if right_m < SIDE_NUDGE: return _a( 0.3, FWD_SPEED)
        if rng.random() < ESCAPE_PROB:
            return _a(float(rng.uniform(-0.3, 0.3)), FWD_SPEED)
        return _a(0.0, FWD_SPEED)

    return act


# ── Preview loops ─────────────────────────────────────────────────────────────

def run_preview_boustr(label, env, act_fn, inner_env):
    """Boustrophedon: runs BOUSTR_STEPS_PER_FRAME physics steps per render frame
    to match the visual speed of SAC (which internally does sub-steps per action)."""
    inner_env.render()
    print(f"\n[{label}]  {MAX_STEPS} steps ({BOUSTR_STEPS_PER_FRAME} steps/frame) — Q or close to skip.")
    obs, _ = env.reset(seed=SEED)
    running = True; step = 0

    while running and step < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q: running = False

        for _ in range(BOUSTR_STEPS_PER_FRAME):
            if step >= MAX_STEPS: break
            action = act_fn(obs, inner_env)
            obs, _, _, _, _ = env.step(action)
            step += 1

        inner_env.render()
        time.sleep(SLEEP)

    print(f"[{label}]  coverage={inner_env._coverage()*100:.1f}%  "
          f"collisions={inner_env._collisions}  steps={step}")
    inner_env.close()


def run_preview_sac(label, env, act_fn, inner_env):
    """SAC: 1 action per frame (action internally runs horizon sub-steps)."""
    inner_env.render()
    print(f"\n[{label}]  {MAX_STEPS} steps — Q or close to skip.")
    obs, _ = env.reset(seed=SEED)
    running = True; step = 0

    while running and step < MAX_STEPS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q: running = False

        action = act_fn(obs, inner_env)
        obs, _, _, _, _ = env.step(action)
        inner_env.render()
        time.sleep(SLEEP)
        step += 1

    print(f"[{label}]  coverage={inner_env._coverage()*100:.1f}%  "
          f"collisions={inner_env._collisions}  steps={step}")
    inner_env.close()


# ── 1. Boustrophedon ──────────────────────────────────────────────────────────
print("=" * 55)
print("  1/2  Boustrophedon — 3-phase lawnmower")
print("=" * 55)
b_env = ApartmentContinuousEnv(seed=SEED, obstacles=furniture, render_mode="human")
b_env.use_bumper = True; b_env.use_stuck_respawn = False
run_preview_boustr("Boustrophedon", b_env, _make_boustr_agent(b_env), b_env)


# ── 2. SAC-NoVMM ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
model_path = pathlib.Path(f"results/SAC-NoVMM_s{SEED}.zip")
if model_path.exists():
    print(f"  2/2  SAC-NoVMM — TRAINED model ({model_path})")
else:
    print("  2/2  SAC-NoVMM — random policy (train first to see learned behaviour)")
print("=" * 55)

from stable_baselines3 import SAC
sac_env   = make_no_vmm_env(furniture, SEED, render_mode="human")
sac_inner = _unwrap_inner(sac_env)
sac_env.reset(seed=SEED)

if model_path.exists():
    model = SAC.load(model_path, env=sac_env)
    def sac_act(obs, inner):
        action, _ = model.predict(obs, deterministic=True)
        return action
else:
    def sac_act(obs, inner):
        return sac_env.action_space.sample()

run_preview_sac("SAC-NoVMM", sac_env, sac_act, sac_inner)
print("\nDone.")
