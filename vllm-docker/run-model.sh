#!/usr/bin/env bash
# =============================================================================
# run-model.sh — Launch a model recipe from vllm-docker/models/ as N vLLM replicas
#
# Usage:
#   ./run-model.sh <model-key> [replicas]
#   REPLICAS=1 GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=262144 ./run-model.sh gemma4-26b-a4b
#   DRY_RUN=1 ./run-model.sh qwen3.6-35b-a3b        # print docker commands only
#
# Each replica is one container pinned to one GPU (TP=1) with its own port:
#   replica i -> name vllm-<key>-<i>, GPU i % GPU_COUNT, port BASE_PORT + i
#
# Requires the active hardware profile (or defaults to 2xa6000):
#   source ../switch-profile.sh 2xa6000
# =============================================================================
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

usage() {
    echo "Usage: $0 <model-key> [replicas]"
    echo
    echo "Available model recipes:"
    for f in "${VLLM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/models/"*.env; do
        echo "  $(basename "${f}" .env)"
    done
    exit 1
}

[[ $# -ge 1 ]] || usage
MODEL_ARG="$1"
REPLICAS_ARG="${2:-}"

load_config "${MODEL_ARG}"
[[ -n "${REPLICAS_ARG}" ]] && REPLICAS="${REPLICAS_ARG}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    docker_cmd || vllm_die "cannot talk to the Docker daemon.
       First-time setup on this machine:   sudo ${VLLM_DIR}/setup-docker.sh
       (installs nvidia-container-toolkit, enables docker, adds you to the docker group)
       Then log out and back in."
    if ! "${DOCKER[@]}" image inspect "${VLLM_IMAGE}" >/dev/null 2>&1; then
        echo "Pulling ${VLLM_IMAGE} ..."
        "${DOCKER[@]}" pull "${VLLM_IMAGE}"
    fi
fi

# Refuse to double-start; make the idempotent path explicit.
EXISTING="$("${DOCKER[@]:-docker}" ps -a --filter "name=${CONTAINER_PREFIX}-" --format '{{.Names}}' 2>/dev/null || true)"
if [[ "${DRY_RUN:-0}" != "1" && -n "${EXISTING}" ]]; then
    vllm_die "containers already exist for '${MODEL_KEY}':
${EXISTING}
       Stop them first:  ./stop-model.sh ${MODEL_KEY}"
fi

echo "=== ${MODEL_KEY} — ${REPLICAS} replica(s) on profile '${PROFILE:-${LLM_HW_PROFILE:-2xa6000}}' ==="
echo "Image:            ${VLLM_IMAGE}"
echo "Model:            ${MODEL}"
echo "Quantization:     ${QUANTIZATION:-auto}"
echo "Replicas:         ${REPLICAS} (TP=${TP}, GPU_MEM_UTIL=${GPU_MEM_UTIL})"
echo "GPUs:             ${CUDA_VISIBLE_DEVICES}"
echo "Ports:            $(replica_port 0)..$((BASE_PORT + REPLICAS - 1)) (bind ${BIND_HOST})"
echo "Max model len:    ${MAX_MODEL_LEN} (KV cache: ${KV_CACHE_DTYPE})"
echo "HF cache:         ${HF_HOME}"
echo

mkdir -p "${HF_HOME}" "${VLLM_DIR}/logs"

for ((i = 0; i < REPLICAS; i++)); do
    gpu="$(replica_gpu "${i}")"
    port="$(replica_port "${i}")"
    name="$(container_name "${i}")"
    build_vllm_args "${i}"

    gpu_args=(--gpus "device=${gpu}")
    [[ -n "${gpu}" ]] || gpu_args=(--gpus all)

    cmd=(run -d
        --name "${name}"
        --label "vllm.recipe.key=${MODEL_KEY}"
        --label "vllm.recipe.replica=${i}"
        "${gpu_args[@]}"
        --ipc=host
        --shm-size 1g
        -p "${BIND_HOST}:${port}:${port}"
        -v "${HF_HOME}:/root/.cache/huggingface"
        -e HF_HOME=/root/.cache/huggingface
        ${HF_TOKEN:+-e HF_TOKEN}
        ${VLLM_LOGGING_LEVEL:+-e VLLM_LOGGING_LEVEL}
        "${VLLM_IMAGE}"
        "${VLLM_ARGS[@]}")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        run_docker "${cmd[@]}"
    else
        run_docker "${cmd[@]}" >"${VLLM_DIR}/logs/${MODEL_KEY}-${i}.container-id" 2>&1 ||
            vllm_die "failed to start replica ${i} (see ${VLLM_DIR}/logs/${MODEL_KEY}-${i}.container-id)"
    fi
    echo "  started ${name}  GPU ${gpu:-all}  port ${port}"
done

if [[ "${DRY_RUN:-0}" == "1" || "${NO_WAIT:-0}" == "1" ]]; then
    exit 0
fi

echo
echo "Waiting for /health on ${REPLICAS} replica(s) (timeout ${READY_TIMEOUT}s; first start downloads weights)..."
fail=0
for ((i = 0; i < REPLICAS; i++)); do
    url="$(replica_url "${i}")"
    name="$(container_name "${i}")"
    if wait_for_http "${url}/health" "${READY_TIMEOUT}"; then
        echo "  ${name}: ready  (${url})"
    else
        if [[ "$("${DOCKER[@]}" inspect -f '{{.State.Running}}' "${name}" 2>/dev/null)" != "true" ]]; then
            echo "  ${name}: EXITED — last log lines:" >&2
            "${DOCKER[@]}" logs --tail 20 "${name}" >&2 || true
            fail=1
            break
        fi
        echo "  ${name}: TIMEOUT after ${READY_TIMEOUT}s — check: ${DOCKER[*]} logs -f ${name}" >&2
        fail=1
    fi
done

if ((fail)); then
    vllm_die "not all replicas became ready — inspect logs with: ./status.sh ${MODEL_KEY}"
fi

echo
./status.sh "${MODEL_KEY}" 2>/dev/null || true
echo
echo "Endpoint example:"
echo "  curl $(replica_url 0)/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "       -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
