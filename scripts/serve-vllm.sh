#!/usr/bin/env bash
# Shared vLLM command for all recipes. Extra arguments go directly to vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_RECIPE="${1:?Usage: serve-vllm.sh RECIPE [vLLM arguments]}"; shift
PROFILE="${LLM_HW_PROFILE:-4xa100}"
[[ "$MODEL_RECIPE" != */* && "$PROFILE" != */* ]] || { echo 'Invalid recipe/profile' >&2; exit 2; }
RECIPE_DIR="$ROOT/recipes/$MODEL_RECIPE"
CONFIG="${LLM_PROFILES_DIR:-$ROOT/profiles}/$PROFILE/config.env"
[[ -f "$CONFIG" && -f "$RECIPE_DIR/profiles/$PROFILE.env" ]] || { echo 'Unknown recipe or profile' >&2; exit 2; }
source "$CONFIG"
# vLLM settings deliberately do not source legacy GGUF parameters.
source "$RECIPE_DIR/vllm.env"
if [[ -f "$RECIPE_DIR/profiles/$PROFILE.vllm.env" ]]; then
    source "$RECIPE_DIR/profiles/$PROFILE.vllm.env"
fi
: "${VLLM_MODEL:?Set VLLM_MODEL to a compatible Hugging Face repo or local safetensors directory; see recipe README}"
: "${VLLM_IMAGE:?Set VLLM_IMAGE to a tested vllm/vllm-openai tag or digest}"
RUNTIME="${LLM_RUNTIME:-docker}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SIF="${VLLM_SIF:-$ROOT/images/vllm-openai.sif}"
args=(--model "$VLLM_MODEL" --served-model-name "$MODEL_RECIPE" --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
      --tensor-parallel-size "${TP:-$TENSOR_PARALLEL_SIZE}" --max-model-len "${MAX_MODEL_LEN:-8192}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}")
[[ -z "${REASONING_PARSER:-}" ]] || args+=(--reasoning-parser "$REASONING_PARSER")
[[ -z "${TOOL_CALL_PARSER:-}" ]] || args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
[[ -z "${MODEL_REVISION:-}" ]] || args+=(--revision "$MODEL_REVISION")
mounts=()
if [[ "$VLLM_MODEL" == /* ]]; then
    [[ "${DRY_RUN:-0}" == 1 || -d "$VLLM_MODEL" ]] || { echo "Missing local model: $VLLM_MODEL" >&2; exit 2; }
    mounts=("$(dirname "$VLLM_MODEL")")
fi
case "$RUNTIME" in
 docker)
    cmd=(docker run --rm --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" --ipc=host --network=host
         -v "$HF_HOME:/hf" -e HF_HOME=/hf -e "NCCL_DEBUG=$NCCL_DEBUG")
    [[ -z "${HF_TOKEN:-}" ]] || cmd+=(-e HF_TOKEN)
    for dir in "${mounts[@]}"; do cmd+=(-v "$dir:$dir:ro"); done
    cmd+=(--entrypoint python3 "$VLLM_IMAGE" -m vllm.entrypoints.openai.api_server)
    ;;
 singularity|apptainer)
    runtime="$RUNTIME"
    if ! command -v "$runtime" >/dev/null 2>&1 && [[ "${DRY_RUN:-0}" != 1 ]]; then
        if command -v apptainer >/dev/null 2>&1; then runtime=apptainer
        elif command -v singularity >/dev/null 2>&1; then runtime=singularity
        else echo 'Install Singularity or Apptainer' >&2; exit 2; fi
    fi
    [[ "${DRY_RUN:-0}" == 1 || -r "$SIF" ]] || { echo "Build SIF first: scripts/build-vllm-sif.sh ($SIF)" >&2; exit 2; }
    for prefix in APPTAINERENV SINGULARITYENV; do
        export "${prefix}_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${prefix}_HF_HOME=/hf" "${prefix}_NCCL_DEBUG=$NCCL_DEBUG"
        [[ -z "${HF_TOKEN:-}" ]] || export "${prefix}_HF_TOKEN=$HF_TOKEN"
    done
    cmd=("$runtime" exec --cleanenv --nv --bind "$HF_HOME:/hf")
    for dir in "${mounts[@]}"; do cmd+=(--bind "$dir:$dir:ro"); done
    cmd+=("$SIF" python3 -m vllm.entrypoints.openai.api_server)
    ;;
 *) echo "Unsupported LLM_RUNTIME: $RUNTIME" >&2; exit 2 ;;
esac
cmd+=("${args[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit; fi
mkdir -p "$HF_HOME" "${LOG_DIR:-$RECIPE_DIR/logs}"
"${cmd[@]}" 2>&1 | tee "${LOG_DIR:-$RECIPE_DIR/logs}/vllm-$PROFILE-$(date -u +%Y%m%dT%H%M%SZ).log"
