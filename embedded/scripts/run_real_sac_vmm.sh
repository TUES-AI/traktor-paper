#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/yasen/traktor-venv/bin/python}"
MODEL_PATH="${1:-results/SAC-VMM___s42.zip}"
shift || true

exec "$PYTHON_BIN" embedded/scripts/run_sac_vmm_local_targets.py \
  --model "$MODEL_PATH" \
  --steps "${SAC_STEPS:-20}" \
  --sleep "${SAC_SLEEP:-0.25}" \
  --max-theta-deg "${SAC_MAX_THETA_DEG:-75}" \
  --max-distance-cm "${SAC_MAX_DISTANCE_CM:-120}" \
  --min-drive-cm "${SAC_MIN_DRIVE_CM:-10}" \
  --turn-pwm "${SAC_TURN_PWM:-65}" \
  --drive-pwm "${SAC_DRIVE_PWM:-90}" \
  "$@"
