#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
TWF_TRAIN_STEPS=${TWF_TRAIN_STEPS:-100}
TWF_LOG_PATH=${TWF_LOG_PATH:-results/twf_real_run.jsonl}

"${PYTHON_BIN}" embedded/scripts/train_real_twf_sac.py \
  --steps "${TWF_TRAIN_STEPS}" \
  --log-path "${TWF_LOG_PATH}"
