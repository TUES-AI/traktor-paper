#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/yasen/.venv/bin/python}"

exec "$PYTHON_BIN" embedded/scripts/train_real_predictive_sac.py \
  --steps "${PCVM_M_TRAIN_STEPS:-60}" \
  --backend pcvm-m \
  --save-path "${PCVM_M_SAVE_PATH:-results/pcvm_m_sac_real.zip}" \
  --max-theta-deg "${PCVM_M_MAX_THETA_DEG:-75}" \
  --max-distance-cm "${PCVM_M_MAX_DISTANCE_CM:-80}" \
  --turn-pwm "${PCVM_M_TURN_PWM:-65}" \
  --drive-pwm "${PCVM_M_DRIVE_PWM:-75}" \
  "$@"
