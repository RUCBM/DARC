#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 SOLVER_MODEL QUESTIONER_MODEL EXPERIMENT_NAME TRAIN_PARQUET" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export GROUNDING_JUDGE_BACKEND=qwen3_base
export GROUNDING_JUDGE_MODEL="${GROUNDING_JUDGE_MODEL:-Qwen3-4B-Base}"
export GROUNDING_JUDGE_BASE_URL="${GROUNDING_JUDGE_BASE_URL:-http://127.0.0.1:6100/v1}"

# Fail before starting Solver services/training if the judge is unavailable or invalid.
python3 -m examples.reward_function.grounding_judge
echo "Grounding experiment: backend=$GROUNDING_JUDGE_BACKEND model=$GROUNDING_JUDGE_MODEL"
exec bash scripts/difficulty_aware_questioner_train.sh "$@"
