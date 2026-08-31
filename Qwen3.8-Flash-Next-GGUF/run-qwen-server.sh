#!/usr/bin/env bash
# =============================================================================
# run-qwen-server.sh — Qwen3.8-Flash-Next GGUF via llama.cpp on Singularity
#
# Hardware profile: reads from profiles/<LLM_HW_PROFILE>/qwen3.8-flash-next.env
# Activate with:    source ../switch-profile.sh 4xa100  (or 2xa6000)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${LLM_HW_PROFILE:-4xa100}"
PROFILE_DIR="${LLM_PROFILES_DIR:-${REPO_ROOT}/profiles}/${PROFILE}"

# Source shared GPU layout
if [[ -f "${PROFILE_DIR}/config.env" ]]; then
	# shellcheck disable=SC1090,SC1091
	source "${PROFILE_DIR}/config.env"
fi

# Source project-specific overrides
PROJECT_CONFIG="${PROFILE_DIR}/qwen3.8-flash-next.env"
if [[ -f "${PROJECT_CONFIG}" ]]; then
	# shellcheck disable=SC1090,SC1091
	source "${PROJECT_CONFIG}"
fi

# ---------- Derived / defaults ----------
IMAGE="${IMAGE:-${REPO_ROOT}/Qwen3.8-Flash-Next-GGUF/qwen3.8-flash-next-llamacpp.sif}"
MODEL_DIR="${MODEL_DIR:-${GGUF_MODELS_DIR:-/davinci-1/work/glisita/huggingface_cache/gguf}/Qwen3.8-Flash-Next-GGUF/UD-IQ3_XXS}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/Qwen3.8-Flash-Next-GGUF/logs}"

# ---------- Server parameters ----------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8081}"
CTX_SIZE="${CTX_SIZE:-65536}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
# Keep Qwen's large n-gram/PLE table in host RAM instead of GPU VRAM.
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-per_layer_token_embd=CPU}"
FIT_TARGET="${FIT_TARGET:-2048}"
REASONING="${REASONING:-off}"
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

LOG_FILE="${LOG_DIR}/qwen3.8-flash-next-server-$(date +%Y%m%d-%H%M%S).log"

# ---------- Container environment ----------
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export APPTAINERENV_GGML_CUDA_P2P="${GGML_CUDA_P2P:-1}"

printf '=== Hardware profile: %s ===\n' "${PROFILE}"
printf 'Singularity image:  %s\n' "${IMAGE}"
printf 'Model directory:   %s\n' "${MODEL_DIR}"
printf 'First shard:       %s\n' "${MODEL_FILE}"
printf 'Listening on:      %s:%s\n' "${HOST}" "${PORT}"
printf 'Context:            %s\n' "${CTX_SIZE}"
printf 'Parallel slots:     %s\n' "${PARALLEL}"
printf 'KV cache:           K=%s, V=%s\n' "${CACHE_TYPE_K}" "${CACHE_TYPE_V}"
printf 'CPU tensor override: %s\n' "${OVERRIDE_TENSOR}"
printf 'Reasoning:          %s\n' "${REASONING}"
printf 'GPUs:               %s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Tensor split:       %s\n' "${TENSOR_SPLIT}"
printf 'Log file:           %s\n' "${LOG_FILE}"
printf '\nModel files:\n'
find "${MODEL_DIR}" \
	-maxdepth 1 \
	-type f \
	-name '*.gguf' \
	-printf '  %f\n' |
	sort

printf '\nAvailable model storage:\n'
df -h "${MODEL_DIR}"

printf '\nInitial GPU state:\n'
nvidia-smi \
	--query-gpu=index,name,memory.used,memory.free,utilization.gpu \
	--format=csv

printf '\nStarting llama-server...\n'

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
	--jinja \
	--n-gpu-layers all \
	--split-mode "${SPLIT_MODE:-layer}" \
	--tensor-split "${TENSOR_SPLIT}" \
	--fit on \
	--fit-target "${FIT_TARGET}" \
	--flash-attn on \
	--override-tensor "${OVERRIDE_TENSOR}" \
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
	--reasoning "${REASONING}" \
	-lv "${LOG_LEVEL}" \
	2>&1 |
	stdbuf -oL tee -a "${LOG_FILE}"
