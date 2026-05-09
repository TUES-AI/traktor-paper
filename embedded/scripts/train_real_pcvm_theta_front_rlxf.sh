#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/home/yasen/.venv/bin/python}"

exec "$PYTHON_BIN" embedded/scripts/train_real_predictive_sac.py \
  --steps "${PCVM_THETA_FRONT_TRAIN_STEPS:-100}" \
  --backend pcvm \
  --action-mode theta_until_front \
  --reward-mode slow_rlxf \
  --save-path "${PCVM_THETA_FRONT_SAVE_PATH:-results/pcvm_theta_front_rlxf_sac_real.zip}" \
  --log-path "${PCVM_THETA_FRONT_LOG_PATH:-results/pcvm_theta_front_rlxf_train.jsonl}" \
  --frame-dir "${PCVM_THETA_FRONT_FRAME_DIR:-results/pcvm_theta_front_frames}" \
  --max-theta-deg "${PCVM_THETA_FRONT_MAX_THETA_DEG:-75}" \
  --max-distance-cm "${PCVM_THETA_FRONT_REWARD_MAX_DISTANCE_CM:-120}" \
  --until-front-cm "${PCVM_THETA_FRONT_UNTIL_FRONT_CM:-40}" \
  --until-front-max-seconds "${PCVM_THETA_FRONT_MAX_SECONDS:-3.0}" \
  --cm-per-second "${PCVM_THETA_FRONT_CM_PER_SECOND:-40}" \
  --turn-pwm "${PCVM_THETA_FRONT_TURN_PWM:-60}" \
  --drive-pwm "${PCVM_THETA_FRONT_DRIVE_PWM:-65}" \
  --sleep "${PCVM_THETA_FRONT_SLEEP:-0.05}" \
  --settle-seconds "${PCVM_THETA_FRONT_SETTLE_SECONDS:-0.35}" \
  --learning-starts "${PCVM_THETA_FRONT_LEARNING_STARTS:-25}" \
  --batch-size "${PCVM_THETA_FRONT_BATCH_SIZE:-32}" \
  --buffer-size "${PCVM_THETA_FRONT_BUFFER_SIZE:-3000}" \
  --front-stop-cm "${PCVM_THETA_FRONT_FRONT_STOP_CM:-35}" \
  --front-clear-cm "${PCVM_THETA_FRONT_FRONT_CLEAR_CM:-45}" \
  --base-step-cost "${PCVM_THETA_FRONT_BASE_STEP_COST:--0.01}" \
  --motion-gate-cm "${PCVM_THETA_FRONT_MOTION_GATE_CM:-3}" \
  --yaw-gate-deg "${PCVM_THETA_FRONT_YAW_GATE_DEG:-8}" \
  --motion-novelty-weight "${PCVM_THETA_FRONT_MOTION_NOVELTY_WEIGHT:-1.50}" \
  --new-cluster-bonus "${PCVM_THETA_FRONT_NEW_CLUSTER_BONUS:-2.50}" \
  --slow-surprise-weight "${PCVM_THETA_FRONT_SURPRISE_WEIGHT:-0.05}" \
  --safe-motion-bonus "${PCVM_THETA_FRONT_SAFE_MOTION_BONUS:-0.30}" \
  --safe-motion-min-cm "${PCVM_THETA_FRONT_SAFE_MOTION_MIN_CM:-10}" \
  --executed-distance-weight "${PCVM_THETA_FRONT_EXECUTED_DISTANCE_WEIGHT:-1.00}" \
  --recent-revisit-penalty "${PCVM_THETA_FRONT_RECENT_REVISIT_PENALTY:-0.12}" \
  --near-obstacle-penalty "${PCVM_THETA_FRONT_NEAR_OBSTACLE_PENALTY:-0.10}" \
  --side-near-cm "${PCVM_THETA_FRONT_SIDE_NEAR_CM:-25}" \
  --stuck-penalty "${PCVM_THETA_FRONT_STUCK_PENALTY:-0.25}" \
  --slow-recovery-penalty "${PCVM_THETA_FRONT_RECOVERY_PENALTY:-0.12}" \
  --zero-forward-cm "${PCVM_THETA_FRONT_ZERO_FORWARD_CM:-3}" \
  --zero-forward-penalty "${PCVM_THETA_FRONT_ZERO_FORWARD_PENALTY:-0.15}" \
  --loop-memory-size "${PCVM_THETA_FRONT_LOOP_MEMORY_SIZE:-25}" \
  --loop-near-radius-m "${PCVM_THETA_FRONT_LOOP_NEAR_RADIUS_M:-0.45}" \
  --loop-revisit-penalty "${PCVM_THETA_FRONT_LOOP_REVISIT_PENALTY:-0.75}" \
  --loop-long-move-cm "${PCVM_THETA_FRONT_LOOP_LONG_MOVE_CM:-80}" \
  --loop-long-move-scale "${PCVM_THETA_FRONT_LOOP_LONG_MOVE_SCALE:-0.45}" \
  --recovery-streak-window "${PCVM_THETA_FRONT_RECOVERY_STREAK_WINDOW:-8}" \
  --recovery-streak-penalty "${PCVM_THETA_FRONT_RECOVERY_STREAK_PENALTY:-0.00}" \
  --blocked-open-turn-bonus "${PCVM_THETA_FRONT_BLOCKED_OPEN_TURN_BONUS:-1.50}" \
  --blocked-open-before-cm "${PCVM_THETA_FRONT_BLOCKED_OPEN_BEFORE_CM:-55}" \
  --blocked-open-min-theta-deg "${PCVM_THETA_FRONT_BLOCKED_OPEN_MIN_THETA_DEG:-25}" \
  --blocked-open-min-improvement-cm "${PCVM_THETA_FRONT_BLOCKED_OPEN_MIN_IMPROVEMENT_CM:-20}" \
  --blocked-open-scale-cm "${PCVM_THETA_FRONT_BLOCKED_OPEN_SCALE_CM:-120}" \
  --viz-port "${PCVM_THETA_FRONT_VIZ_PORT:-0}" \
  --viz-depth-model "${PCVM_THETA_FRONT_VIZ_DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}" \
  "$@"
