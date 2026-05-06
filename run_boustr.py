"""Run only the boustrophedon for 200k steps on all 3 seeds."""
import numpy as np
from train_sac import _run_boustrophedon, SEEDS, TRAIN_STEPS, EVAL_EVERY
from apartment_env import generate_apartment

for seed in SEEDS:
    furniture = generate_apartment(np.random.default_rng(seed))
    print(f"\n=== Boustrophedon seed={seed} ===")
    _run_boustrophedon(furniture, seed)
