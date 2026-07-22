#!/usr/bin/env bash
# =============================================================================
# run-laguna-server.sh — Laguna S2.1 via llama.cpp (DFlash spec. dec.) on Singularity
#
# Hardware profile: reads from profiles/<LLM_HW_PROFILE>/laguna.env
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
PROJECT_CONFIG="${PROFILE_DIR}/laguna.env"
if [[ -f "${PROJECT_CONFIG}" ]]; then
    # shellcheck disable=SC1090
    source "${PROJECT_CONFIG}"
fi

# ---------- Derived / defaults ----------
SIF="${SIF:-${REPO_ROOT}/Laguna-S2.1/laguna-s2.1-dflash.sif}"
MODEL_PATH="${MODEL_PATH:-/davinci-1/work/glisita/huggingface_cache/gguf/Laguna-S-2.1-GGUF/UD-Q6_K/Laguna-S-2.1-UD-Q6_K-00001-of-00003.gguf}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/Laguna-S2.1/logs}"

LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLAMA_PORT="${LLAMA_PORT:-8000}"
LLAMA_CTX="${LLAMA_CTX:-262144}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_BATCH="${LLAMA_BATCH:-4096}"
LLAMA_UBATCH="${LLAMA_UBATCH:-512}"
LLAMA_CACHE_K="${LLAMA_CACHE_K:-q8_0}"
LLAMA_CACHE_V="${LLAMA_CACHE_V:-q8_0}"
LLAMA_TENSOR_SPLIT="${LLAMA_TENSOR_SPLIT:-${TENSOR_SPLIT:-1,1,1,1}}"
LLAMA_CACHE_REUSE="${LLAMA_CACHE_REUSE:-256}"
LLAMA_SPEC_DRAFT_N_MAX="${LLAMA_SPEC_DRAFT_N_MAX:-15}"

# ---------- Validation ----------
if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: Singularity image not found:"
    echo "  ${SIF}"
    exit 1
fi

if [[ ! -r "${MODEL_PATH}" ]]; then
    echo "ERROR: Laguna model not readable: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ -n "${DRAFT_MODEL_PATH}" && ! -r "${DRAFT_MODEL_PATH}" ]]; then
    echo "ERROR: DFlash draft model not readable: ${DRAFT_MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/laguna-server-$(date +%Y%m%d-%H%M%S).log"

# ---------- Container environment ----------
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export APPTAINERENV_GGML_CUDA_P2P="${GGML_CUDA_P2P:-1}"

export APPTAINERENV_MODEL_PATH="${MODEL_PATH}"
export APPTAINERENV_DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH}"
export APPTAINERENV_LLAMA_HOST="${LLAMA_HOST}"
export APPTAINERENV_LLAMA_PORT="${LLAMA_PORT}"
export APPTAINERENV_LLAMA_CTX="${LLAMA_CTX}"
export APPTAINERENV_LLAMA_PARALLEL="${LLAMA_PARALLEL}"
export APPTAINERENV_LLAMA_BATCH="${LLAMA_BATCH}"
export APPTAINERENV_LLAMA_UBATCH="${LLAMA_UBATCH}"
export APPTAINERENV_LLAMA_CACHE_K="${LLAMA_CACHE_K}"
export APPTAINERENV_LLAMA_CACHE_V="${LLAMA_CACHE_V}"
export APPTAINERENV_LLAMA_TENSOR_SPLIT="${LLAMA_TENSOR_SPLIT}"
export APPTAINERENV_LLAMA_CACHE_REUSE="${LLAMA_CACHE_REUSE}"
export APPTAINERENV_LLAMA_SPEC_DRAFT_N_MAX="${LLAMA_SPEC_DRAFT_N_MAX}"

echo "=== Hardware profile: ${PROFILE} ==="
echo "Singularity image:  ${SIF}"
echo "Model:              ${MODEL_PATH}"
if [[ -n "${DRAFT_MODEL_PATH}" ]]; then
    echo "DFlash draft:       ${DRAFT_MODEL_PATH}"
fi
echo "Listening on:       ${LLAMA_HOST}:${LLAMA_PORT}"
echo "Context:            ${LLAMA_CTX}"
echo "Parallel slots:     ${LLAMA_PARALLEL}"
echo "GPUs:               ${CUDA_VISIBLE_DEVICES}"
echo "Tensor split:       ${LLAMA_TENSOR_SPLIT}"
if [[ -n "${DRAFT_MODEL_PATH}" ]]; then
    echo "Max draft tokens:   ${LLAMA_SPEC_DRAFT_N_MAX}"
fi
echo "Log file:           ${LOG_FILE}"
echo

echo "Available model storage:"
df -h "$(dirname "${MODEL_PATH}")"

echo
echo "Initial GPU state:"
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
    --format=csv

echo
echo "Starting Laguna server..."

# Bind the model directory; the container entrypoint uses MODEL_PATH from env
MODEL_BIND_DIR="$(dirname "${MODEL_PATH}")"
SIF_BINDS="--bind ${MODEL_BIND_DIR}:/models:ro"

if [[ -n "${DRAFT_MODEL_PATH}" ]]; then
    DRAFT_BIND_DIR="$(dirname "${DRAFT_MODEL_PATH}")"
    SIF_BINDS="${SIF_BINDS} --bind ${DRAFT_BIND_DIR}:/draft:ro"
fi

singularity run --nv \
    ${SIF_BINDS} \
    "${SIF}" \
    2>&1 |
    stdbuf -oL tee -a "${LOG_FILE}"
