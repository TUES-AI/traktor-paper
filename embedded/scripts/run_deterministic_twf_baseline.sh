#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
DETERMINISTIC_TWF_STEPS=${DETERMINISTIC_TWF_STEPS:-100}
DETERMINISTIC_TWF_LOG_PATH=${DETERMINISTIC_TWF_LOG_PATH:-results/deterministic_twf_baseline.jsonl}

"${PYTHON_BIN}" embedded/scripts/run_deterministic_twf_baseline.py \
  --steps "${DETERMINISTIC_TWF_STEPS}" \
  --log-path "${DETERMINISTIC_TWF_LOG_PATH}"
