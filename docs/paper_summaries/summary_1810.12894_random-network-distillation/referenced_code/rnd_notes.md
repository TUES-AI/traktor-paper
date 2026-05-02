# RND Reference Code Notes

Repo cloned to `/tmp/random-network-distillation` during extraction.

Important files:

- `run_atari.py`: training entrypoint.
- `ppo_agent.py`: PPO loop with intrinsic/extrinsic returns and separate value heads.
- `policies/cnn_policy_param_matched.py`: CNN policy and RND target/predictor networks.

Core implementation pattern:

```python
target = frozen_random_network(obs)
prediction = predictor_network(obs)
intrinsic_reward = mean_squared_error(stop_gradient(target), prediction)
predictor_loss = intrinsic_reward
```

Important code details:

- Intrinsic rewards are computed after collecting each rollout.
- Intrinsic reward returns ignore episode termination by default: `nextnew = 0.0`.
- Extrinsic reward returns use environment dones.
- Advantages are combined:

```python
adv = int_coeff * adv_int + ext_coeff * adv_ext
```

- Predictor updates can use only a random fraction of rollout samples.
- Observations are whitened and clipped to `[-5, 5]` before RND target/predictor.

Project adaptation:

```text
obs = compact rover observation
target_features = frozen_random_mlp(obs)
pred_features = predictor_mlp(obs)
novelty = mse(target_features, pred_features)
```

Start with compact observations before camera:

```text
[ultra_left, ultra_front, ultra_right,
 gyro_z, accel_norm, vibration,
 last_left_cmd, last_right_cmd]
```

Then add a small camera embedding once camera capture is reliable.
