# Reinforced Imitation Architecture Notes

Reference repo: https://github.com/ethz-asl/rl-navigation

The repo was cloned to `/tmp/rl-navigation` during summary extraction.

Useful implementation details:

- `scripts/options.py` defines the main training parameters.
- `--jump_start 1 --model_init <weights>` initializes RL from imitation-learned weights.
- ASL policy input size is 38: 36 min-pooled laser values plus relative goal information.
- Action dimension is 2: translational and rotational velocity.
- ASL actor hidden sizes are `1000 -> 300 -> 100` with tanh activations.
- Tai-style comparison architecture uses `512 -> 512 -> 512` with ReLU.
- Safety threshold default is `0.4` average discounted safety cost.
- Action duration is `0.2s`, matching a 5 Hz control loop.
- Training uses `60000` timesteps per epoch and `1000` epochs by default.

Project adaptation:

```text
obs = [ultra_left, ultra_front, ultra_right,
       goal_distance, sin(goal_angle), cos(goal_angle),
       previous_left_motor, previous_right_motor]

policy(obs) -> [left_motor, right_motor]
```

Then add camera:

```text
camera -> tiny CNN -> visual latent
ultrasonic + goal + previous action -> MLP latent
concat -> GRU -> action head
```
