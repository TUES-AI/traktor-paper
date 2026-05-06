#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/yasen/.venv/bin/python}"
EXTRA_ARGS=()
if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  EXTRA_ARGS+=(--model "$1")
  shift
fi
if [[ "${SAC_MODE:-}" != "" ]]; then
  EXTRA_ARGS+=(--mode "$SAC_MODE")
fi

exec "$PYTHON_BIN" embedded/scripts/run_sac_vmm_local_targets.py \
  --steps "${SAC_STEPS:-20}" \
  --sleep "${SAC_SLEEP:-0.25}" \
  --max-theta-deg "${SAC_MAX_THETA_DEG:-75}" \
  --max-distance-cm "${SAC_MAX_DISTANCE_CM:-120}" \
  --min-drive-cm "${SAC_MIN_DRIVE_CM:-10}" \
  --turn-pwm "${SAC_TURN_PWM:-65}" \
  --drive-pwm "${SAC_DRIVE_PWM:-90}" \
  --front-stop-cm "${SAC_FRONT_STOP_CM:-45}" \
  --front-clear-cm "${SAC_FRONT_CLEAR_CM:-55}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
