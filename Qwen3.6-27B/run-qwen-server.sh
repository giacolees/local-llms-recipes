#!/usr/bin/env bash
# =============================================================================
# run-qwen-server.sh — Qwen3.6-27B via vLLM on Singularity
#
# Hardware profile: reads from profiles/<LLM_HW_PROFILE>/qwen.env
# Activate with:    source ../switch-profile.sh 4xa100  (or 2xa6000)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${LLM_HW_PROFILE:-4xa100}"
PROFILE_DIR="${LLM_PROFILES_DIR:-${REPO_ROOT}/profiles}/${PROFILE}"

# Source shared GPU layout
if [[ -f "${PROFILE_DIR}/config.env" ]]; then
    # shellcheck disable=SC1090
    source "${PROFILE_DIR}/config.env"
fi

# Source project-specific overrides
PROJECT_CONFIG="${PROFILE_DIR}/qwen.env"
if [[ -f "${PROJECT_CONFIG}" ]]; then
    # shellcheck disable=SC1090
    source "${PROJECT_CONFIG}"
fi

# ---------- Derived / defaults ----------
IMAGE="${IMAGE:-${REPO_ROOT}/vllm-openai.sif}"
HF_HOME="${HF_HOME:-/davinci-1/work/glisita/huggingface_cache}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/Qwen-vLLM/logs}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-${TENSOR_PARALLEL_SIZE:-4}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-true}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# ---------- Validation ----------
if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: vLLM image not found:"
    echo "  ${IMAGE}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/qwen-server-$(date +%Y%m%d-%H%M%S).log"

# ---------- Container environment ----------
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

echo "=== Hardware profile: ${PROFILE} ==="
echo "Singularity image:  ${IMAGE}"
echo "Model:              ${MODEL}"
echo "HF cache:           ${HF_HOME}"
echo "Listening on:       ${HOST}:${PORT}"
echo "Tensor parallel:    ${TP}"
echo "Max model length:   ${MAX_MODEL_LEN}"
echo "GPUs:               ${CUDA_VISIBLE_DEVICES}"
echo "Log file:           ${LOG_FILE}"
echo

echo "Available model storage:"
df -h "${HF_HOME}"

echo
echo "Initial GPU state:"
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv

echo
echo "Starting vLLM OpenAI server with ${MODEL}..."

apptainer run --nv \
    --bind "${HF_HOME}:/root/.cache/huggingface" \
    "${IMAGE}" \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --reasoning-parser "${REASONING_PARSER}" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    "$@" \
    2>&1 |
    stdbuf -oL tee -a "${LOG_FILE}"
