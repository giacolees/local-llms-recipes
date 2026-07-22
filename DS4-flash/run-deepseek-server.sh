#!/usr/bin/env bash
# =============================================================================
# run-deepseek-server.sh — DeepSeek V4 Flash via llama.cpp on Singularity
#
# Hardware profile: reads from profiles/<LLM_HW_PROFILE>/ds4-flash.env
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
PROJECT_CONFIG="${PROFILE_DIR}/ds4-flash.env"
if [[ -f "${PROJECT_CONFIG}" ]]; then
    # shellcheck disable=SC1090
    source "${PROJECT_CONFIG}"
fi

# ---------- Derived / defaults ----------
IMAGE="${IMAGE:-/davinci-1/home/glisita/Projects/local-LLMs/DS4-flash/deepseek-v4-flash-llamacpp.sif}"
MODEL_DIR="${MODEL_DIR:-$GGUF_MODELS_DIR/DeepSeek-V4-Flash-UD-IQ3_XXS/UD-IQ3_XXS}"
MODEL_FILE="${MODEL_FILE:-DeepSeek-V4-Flash-UD-IQ3_XXS-00001-of-00004.gguf}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/DS4-flash/logs}"

# ---------- Server parameters ----------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-524288}"
PARALLEL="${PARALLEL:-4}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
LOG_LEVEL="${LOG_LEVEL:-4}"

# ---------- Validation ----------
if [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: Singularity image not found:"
    echo "  ${IMAGE}"
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: model directory not found:"
    echo "  ${MODEL_DIR}"
    exit 1
fi

if [[ ! -r "${MODEL_DIR}/${MODEL_FILE}" ]]; then
    echo "ERROR: first GGUF shard is not readable:"
    echo "  ${MODEL_DIR}/${MODEL_FILE}"
    echo
    echo "Available GGUF files:"
    find "${MODEL_DIR}" -maxdepth 1 -type f -name '*.gguf' -printf '  %f\n'
    exit 1
fi

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/deepseek-server-$(date +%Y%m%d-%H%M%S).log"

# ---------- Container environment ----------
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export APPTAINERENV_GGML_CUDA_P2P="${GGML_CUDA_P2P:-1}"

echo "=== Hardware profile: ${PROFILE} ==="
echo "Singularity image: ${IMAGE}"
echo "Model directory:   ${MODEL_DIR}"
echo "First shard:       ${MODEL_FILE}"
echo "Listening on:      ${HOST}:${PORT}"
echo "Context:           ${CTX_SIZE}"
echo "Parallel slots:    ${PARALLEL}"
echo "KV cache:          K=${CACHE_TYPE_K}, V=${CACHE_TYPE_V}"
echo "GPUs:              ${CUDA_VISIBLE_DEVICES}"
echo "Tensor split:      ${TENSOR_SPLIT}"
echo "Log file:          ${LOG_FILE}"
echo

echo "Model files:"
find "${MODEL_DIR}" \
    -maxdepth 1 \
    -type f \
    -name '*.gguf' \
    -printf '  %f\n' |
sort

echo
echo "Available model storage:"
df -h "${MODEL_DIR}"

echo
echo "Initial GPU state:"
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv

echo
echo "Starting llama-server..."

numactl \
    --cpunodebind="${NUMA_CPUNODEBIND:-0,1}" \
    --interleave="${NUMA_INTERLEAVE:-0,1}" \
    stdbuf -oL -eL \
    singularity exec \
        --cleanenv \
        --nv \
        --bind "${MODEL_DIR}:/models:ro" \
        "${IMAGE}" \
        /opt/llama.cpp/build/bin/llama-server \
            --model "/models/${MODEL_FILE}" \
            --host "${HOST}" \
            --port "${PORT}" \
            --n-gpu-layers all \
            --split-mode "${SPLIT_MODE:-layer}" \
            --tensor-split "${TENSOR_SPLIT}" \
            --fit on \
            --fit-target 2048 \
            --flash-attn on \
            --cache-type-k "${CACHE_TYPE_K}" \
            --cache-type-v "${CACHE_TYPE_V}" \
            --ctx-size "${CTX_SIZE}" \
            --parallel "${PARALLEL}" \
            --batch-size "${BATCH_SIZE}" \
            --ubatch-size "${UBATCH_SIZE}" \
            --cont-batching \
            --cache-prompt \
            --cache-reuse 256 \
            --metrics \
            --slots \
            --reasoning off \
            -lv "${LOG_LEVEL}" \
    2>&1 |
    stdbuf -oL tee -a "${LOG_FILE}"
