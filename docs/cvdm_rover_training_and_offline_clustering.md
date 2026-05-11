# CVDM rover training and offline clustering handoff

This project now has a CVDM package under `CVDM/` for real-rover visual-memory training.

## Model

CVDM uses frozen DINOv3 ONNX image embeddings plus ultrasonic range and last action. The trainable pipeline is:

- controllable encoder: DINO + range + last action -> normalized latent `phi`
- forward dynamics: `phi_t + action -> phi_{t+1}`
- inverse dynamics: `phi_t + phi_{t+1} -> action`
- RND head on `phi` for curiosity diagnostics
- visual memory bank for DINO/CVDM novelty
- SAC policy over CVDM observation features

The real-rover script logs every frame as JPG and writes full transition JSON with timestamps, image paths, ranges, IMU, action feedback, reward terms, CVDM metrics, and saved model params.

## Current training command

Run from repo root on the rover:

```bash
PYTHONUNBUFFERED=1 /home/yasen/.venv/bin/python CVDM/train_real_rover.py \
  --steps 100 \
  --run-name cvdm_real_100_new \
  --out-dir results/cvdm_real_100_new \
  --visual-encoder dino3 \
  --learning-starts 25 \
  --sac-batch-size 32 \
  --sac-buffer-size 3000 \
  --cvdm-batch-size 32 \
  --cvdm-gradient-steps 1 \
  --front-stop-cm 35 \
  --front-clear-cm 45 \
  --until-front-cm 40 \
  --until-front-max-seconds 3.0 \
  --turn-pwm 60 \
  --drive-pwm 65 \
  --settle-seconds 0.35 \
  --sleep 0.05
```

To continue an existing CVDM+SAC run:

```bash
PYTHONUNBUFFERED=1 /home/yasen/.venv/bin/python CVDM/train_real_rover.py \
  --steps 100 \
  --run-name cvdm_real_100_epoch2 \
  --out-dir results/cvdm_real_100_epoch2 \
  --resume-dir results/cvdm_real_100_new \
  --visual-encoder dino3
```

## Offline support-vector clustering on saved CVDM runs

Use this when the rover is off and you want to re-cluster old saved frames using the saved DINO vectors in `models/cvdm_full.pt`.

```bash
/Volumes/SSD/v/py/bin/python CVDM/offline_support_cluster.py \
  --run-dir results/cvdm_real_100_junkdrop_20260511 \
  --last-n 100 \
  --match-dist 1.00 \
  --add-support-dist 0.45 \
  --max-supports 8 \
  --open
```

Outputs:

- `offline_support_banks/support_vector_banks_overview.jpg`
- `offline_support_banks/support_vector_deleted_frames.jpg`
- `offline_support_banks/support_vector_bank_XX.jpg`
- `offline_support_banks/support_vector_summary.json`

The offline clustering is nearest-exemplar/support-vector clustering, not centroid clustering. Each bank stores real support frames; assignment distance is the nearest support vector distance. This avoids the previous catch-all centroid problem.

## Current gates

Visual memory update is allowed when:

- moved at least `10 cm` OR rotated at least `5 deg`
- no contact/stall
- image quality is not blur/dark/overexposed/low-contrast
- front is not close in a boxed-in state

Left/right close-wall rejection was removed because the camera is front-facing. Front-close rejection only fires when front is close and no range sensor sees an open distance of at least `100 cm`; close front views can still be useful distinct wall/place observations.

## Latest local run summary

Fresh initialized run:

`results/cvdm_real_100_junkdrop_20260511/`

Reward sum: `-6.75`

Mean reward: `-0.0675`

Main contributors:

- zero progress: `-7.04`
- recovery: `-5.88`
- contact: `-3.75`
- novelty: `+6.62`
- distance: `+1.72`
- safe motion: `+1.20`
- new cluster: `+1.05`

The last online run ended with 3 confirmed banks and no giant catch-all bank. Offline support-vector reclustering of the last 100 saved vectors with the updated gates produced 5 support-vector banks from 54 valid frames.
