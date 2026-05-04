"""Preview a random SAC agent (untrained) on the apartment map.
Replace model.predict with a trained model by passing --model path/to/model.zip"""
import sys
import time
import numpy as np
import pygame
from apartment_env import generate_apartment
from train_sac import make_vmm_env, make_no_vmm_env

SEED    = 42
USE_VMM = "--novmm" not in sys.argv

furniture = generate_apartment(np.random.default_rng(SEED))
env = make_vmm_env(furniture, SEED) if USE_VMM else make_no_vmm_env(furniture, SEED)

# Unwrap to inner env for rendering
inner = env
while hasattr(inner, "env"):
    inner = inner.env
inner.render_mode = "human"

obs, _ = env.reset()
inner.render()

# Load trained model if provided, otherwise random policy
model = None
for arg in sys.argv[1:]:
    if arg.endswith(".zip"):
        from stable_baselines3 import SAC
        model = SAC.load(arg, env=env)
        print(f"Loaded model: {arg}")
        break

label = "SAC-VMM" if USE_VMM else "SAC-NoVMM"
print(f"Previewing {label} ({'trained' if model else 'RANDOM — pass model.zip to load'}) — Q to quit.")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
    else:
        action = env.action_space.sample()

    obs, _, term, trunc, _ = env.step(action)
    inner.render()
    time.sleep(0.016)

    if term or trunc:
        obs, _ = env.reset()

inner.close()
