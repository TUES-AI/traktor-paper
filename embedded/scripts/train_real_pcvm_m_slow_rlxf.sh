#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/yasen/.venv/bin/python}"

exec "$PYTHON_BIN" embedded/scripts/train_real_predictive_sac.py \
  --steps "${PCVM_M_SLOW_TRAIN_STEPS:-100}" \
  --backend pcvm-m \
  --reward-mode slow_rlxf \
  --save-path "${PCVM_M_SLOW_SAVE_PATH:-results/pcvm_m_slow_rlxf_sac_real.zip}" \
  --log-path "${PCVM_M_SLOW_LOG_PATH:-results/pcvm_m_slow_rlxf_train.jsonl}" \
  --max-theta-deg "${PCVM_M_SLOW_MAX_THETA_DEG:-30}" \
  --min-distance-cm "${PCVM_M_SLOW_MIN_DISTANCE_CM:-5}" \
  --max-distance-cm "${PCVM_M_SLOW_MAX_DISTANCE_CM:-20}" \
  --min-drive-cm "${PCVM_M_SLOW_MIN_DRIVE_CM:-0}" \
  --turn-pwm "${PCVM_M_SLOW_TURN_PWM:-60}" \
  --drive-pwm "${PCVM_M_SLOW_DRIVE_PWM:-65}" \
  --sleep "${PCVM_M_SLOW_SLEEP:-0.05}" \
  --settle-seconds "${PCVM_M_SLOW_SETTLE_SECONDS:-0.35}" \
  --learning-starts "${PCVM_M_SLOW_LEARNING_STARTS:-25}" \
  --batch-size "${PCVM_M_SLOW_BATCH_SIZE:-32}" \
  --buffer-size "${PCVM_M_SLOW_BUFFER_SIZE:-3000}" \
  --front-stop-cm "${PCVM_M_SLOW_FRONT_STOP_CM:-45}" \
  --front-clear-cm "${PCVM_M_SLOW_FRONT_CLEAR_CM:-55}" \
  --base-step-cost "${PCVM_M_SLOW_BASE_STEP_COST:--0.02}" \
  --motion-gate-cm "${PCVM_M_SLOW_MOTION_GATE_CM:-3}" \
  --yaw-gate-deg "${PCVM_M_SLOW_YAW_GATE_DEG:-8}" \
  --motion-novelty-weight "${PCVM_M_SLOW_MOTION_NOVELTY_WEIGHT:-0.35}" \
  --new-cluster-bonus "${PCVM_M_SLOW_NEW_CLUSTER_BONUS:-0.40}" \
  --slow-surprise-weight "${PCVM_M_SLOW_SURPRISE_WEIGHT:-0.07}" \
  --safe-motion-bonus "${PCVM_M_SLOW_SAFE_MOTION_BONUS:-0.12}" \
  --executed-distance-weight "${PCVM_M_SLOW_EXECUTED_DISTANCE_WEIGHT:-0.10}" \
  --recent-revisit-penalty "${PCVM_M_SLOW_RECENT_REVISIT_PENALTY:-0.30}" \
  --near-obstacle-penalty "${PCVM_M_SLOW_NEAR_OBSTACLE_PENALTY:-0.25}" \
  --side-near-cm "${PCVM_M_SLOW_SIDE_NEAR_CM:-25}" \
  --stuck-penalty "${PCVM_M_SLOW_STUCK_PENALTY:-0.35}" \
  --slow-recovery-penalty "${PCVM_M_SLOW_RECOVERY_PENALTY:-0.55}" \
  --viz-port "${PCVM_M_SLOW_VIZ_PORT:-0}" \
  --viz-depth-model "${PCVM_M_SLOW_VIZ_DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}" \
  "$@"
