#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/yasen/.venv/bin/python}"

exec "$PYTHON_BIN" embedded/scripts/train_real_predictive_sac.py \
  --steps "${PREDICTIVE_TRAIN_STEPS:-100}" \
  --backend "${PREDICTIVE_BACKEND:-pcvm}" \
  --save-path "${PREDICTIVE_SAVE_PATH:-results/pcvm_cnn_sac_real.zip}" \
  --max-theta-deg "${PREDICTIVE_MAX_THETA_DEG:-75}" \
  --max-distance-cm "${PREDICTIVE_MAX_DISTANCE_CM:-80}" \
  --turn-pwm "${PREDICTIVE_TURN_PWM:-65}" \
  --drive-pwm "${PREDICTIVE_DRIVE_PWM:-75}" \
  "$@"
