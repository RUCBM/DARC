#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=<reserved GPU> bash $0 /path/to/Qwen3-4B-Base" >&2
    exit 2
fi
: "${CUDA_VISIBLE_DEVICES:?Select GPU(s) reserved for the grounding judge}"

args=(
    --model "$1"
    --served-model-name "${GROUNDING_JUDGE_MODEL:-Qwen3-4B-Base}"
    --host 127.0.0.1
    --port "${GROUNDING_JUDGE_PORT:-6100}"
    --tensor-parallel-size "${GROUNDING_JUDGE_TP_SIZE:-1}"
    --max-model-len "${GROUNDING_JUDGE_MAX_MODEL_LEN:-32768}"
    --gpu-memory-utilization "${GROUNDING_JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
    --guided-decoding-backend xgrammar
    --generation-config vllm
    --disable-log-requests
)
if [[ -n "${GROUNDING_JUDGE_API_KEY:-}" ]]; then
    args+=(--api-key "$GROUNDING_JUDGE_API_KEY")
fi
exec python3 -m vllm.entrypoints.openai.api_server "${args[@]}"
