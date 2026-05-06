"""Preview the frontier-based exploration agent on the apartment map."""
import time
import numpy as np
import pygame
from apartment_env import ApartmentContinuousEnv, generate_apartment
from frontier_agent import FrontierAgent

SEED = 42

furniture = generate_apartment(np.random.default_rng(SEED))
env = ApartmentContinuousEnv(seed=SEED, obstacles=furniture, render_mode="human")
env.use_bumper        = True
env.use_stuck_respawn = False
obs, _ = env.reset(seed=SEED)

agent = FrontierAgent(seed=SEED)

print("Frontier agent — Q or close window to quit.")
env.render()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    action = agent.act(obs, env)
    obs, _, _, _, _ = env.step(action)
    env.render()
    time.sleep(0.10)   # 10 Hz render — 2× slower than real-time (DT=0.05)

env.close()
