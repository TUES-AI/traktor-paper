#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
TWF_BASELINE_POLICY=${TWF_BASELINE_POLICY:-random}
TWF_BASELINE_STEPS=${TWF_BASELINE_STEPS:-100}
TWF_BASELINE_LOG_PATH=${TWF_BASELINE_LOG_PATH:-results/twf_${TWF_BASELINE_POLICY}_baseline.jsonl}
TWF_BASELINE_SEED=${TWF_BASELINE_SEED:-0}

"${PYTHON_BIN}" embedded/scripts/run_twf_baselines.py \
  --policy "${TWF_BASELINE_POLICY}" \
  --steps "${TWF_BASELINE_STEPS}" \
  --seed "${TWF_BASELINE_SEED}" \
  --log-path "${TWF_BASELINE_LOG_PATH}"
