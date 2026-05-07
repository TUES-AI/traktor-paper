#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/yasen/.venv/bin/python}"

exec "$PYTHON_BIN" embedded/scripts/train_real_predictive_sac.py \
  --steps "${PCVM_T_TRAIN_STEPS:-100}" \
  --backend pcvm-t \
  --save-path "${PCVM_T_SAVE_PATH:-results/pcvm_t_sac_real.zip}" \
  --max-theta-deg "${PCVM_T_MAX_THETA_DEG:-75}" \
  --max-distance-cm "${PCVM_T_MAX_DISTANCE_CM:-80}" \
  --turn-pwm "${PCVM_T_TURN_PWM:-65}" \
  --drive-pwm "${PCVM_T_DRIVE_PWM:-75}" \
  --front-stop-cm "${PCVM_T_FRONT_STOP_CM:-40}" \
  --front-clear-cm "${PCVM_T_FRONT_CLEAR_CM:-55}" \
  --path-revisit-penalty "${PCVM_T_PATH_REVISIT_PENALTY:-0.45}" \
  --path-away-bonus "${PCVM_T_PATH_AWAY_BONUS:-0.25}" \
  --path-near-radius-m "${PCVM_T_PATH_NEAR_RADIUS_M:-0.45}" \
  --path-far-radius-m "${PCVM_T_PATH_FAR_RADIUS_M:-1.5}" \
  --viz-port "${PCVM_T_VIZ_PORT:-8765}" \
  --viz-depth-model "${PCVM_T_VIZ_DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}" \
  "$@"
