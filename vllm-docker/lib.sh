#!/usr/bin/env bash
# =============================================================================
# lib.sh — shared helpers for the vLLM-on-Docker recipes (source, don't execute)
#
# Provides:
#   load_config <model-key>   Resolve profile + model recipe into variables
#   docker_cmd                Resolve the docker invocation (docker / sudo docker)
#   run_docker <args...>      Execute (or print, with DRY_RUN=1) a docker command
#   replica_gpu <i>           Physical GPU pinned to replica i ("" = all GPUs)
#   replica_port <i>          Host port of replica i
#   replica_url <i>           Client URL of replica i
#   container_name <i>        Container name of replica i
#   build_vllm_args <i>       Fill the VLLM_ARGS array for replica i
#   wait_for_http <url> <s>   Poll a URL until it answers (returns non-zero on timeout)
#
# Configuration precedence (highest first):
#   1. Variables exported on the command line (REPLICAS=1 MAX_MODEL_LEN=... ./run-model.sh ...)
#   2. Per-model overrides in profiles/<hw>/vllm-docker.env  (e.g. GEMMA4_E4B_REPLICAS=8)
#   3. Recipe defaults in vllm-docker/models/<key>.env
# =============================================================================

vllm_die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Knobs a user may override from the environment / command line.
_VLLM_OVERRIDABLE=(
    MODEL QUANTIZATION REPLICAS TP GPU_MEM_UTIL MAX_MODEL_LEN
    KV_CACHE_DTYPE MAX_NUM_SEQS BASE_PORT EXTRA_ARGS
)

load_config() {
    local key="${1:-}"
    [[ -n "${key}" ]] || vllm_die "model key required (see vllm-docker/models/)"

    local here repo_root
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${here}/.." && pwd)"
    VLLM_DIR="${here}"
    REPO_ROOT="${repo_root}"

    local profile="${LLM_HW_PROFILE:-2xa6000}"
    PROFILE_DIR="${LLM_PROFILES_DIR:-${REPO_ROOT}/profiles}/${profile}"
    [[ -f "${PROFILE_DIR}/config.env" ]] ||
        vllm_die "unknown hardware profile '${profile}' (missing ${PROFILE_DIR}/config.env)"

    # ---- 1. capture user-supplied overrides (highest precedence) ----
    local _user_vals=() v
    for v in "${_VLLM_OVERRIDABLE[@]}"; do
        _user_vals+=("${!v-}")
    done

    # ---- 2. shared hardware layout (GPU devices, counts, NUMA) ----
    # shellcheck disable=SC1090
    source "${PROFILE_DIR}/config.env"

    # ---- 3. docker/recipe-level profile settings ----
    if [[ -f "${PROFILE_DIR}/vllm-docker.env" ]]; then
        # shellcheck disable=SC1090
        source "${PROFILE_DIR}/vllm-docker.env"
    fi

    # ---- 4. model recipe defaults (${VAR:-...} keeps anything already set) ----
    local model_file="${VLLM_MODELS_DIR:-${VLLM_DIR}/models}/${key}.env"
    [[ -f "${model_file}" ]] || vllm_die "no such model recipe '${key}' (looked in ${model_file})"
    # shellcheck disable=SC1090
    source "${model_file}"

    # ---- 5. per-model profile overrides, then user overrides win again ----
    local up i=0 ovr
    up="$(echo "${key}" | tr 'a-z.-' 'A-Z__')"
    for v in "${_VLLM_OVERRIDABLE[@]}"; do
        ovr="${up}_${v}"
        if [[ -n "${!ovr-}" ]]; then
            printf -v "${v}" '%s' "${!ovr}"
        fi
        if [[ -n "${_user_vals[$i]}" ]]; then
            printf -v "${v}" '%s' "${_user_vals[$i]}"
        fi
        i=$((i + 1))
    done

    # ---- derived / defaults for shared plumbing ----
    MODEL_KEY="${key}"
    REPLICAS="${REPLICAS:-1}"
    TP="${TP:-1}"
    BIND_HOST="${BIND_HOST:-127.0.0.1}"
    CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
    HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
    VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
    READY_TIMEOUT="${READY_TIMEOUT:-1800}"
    CONTAINER_PREFIX="vllm-${key}"

    IFS=',' read -r -a GPU_ARR <<< "${CUDA_VISIBLE_DEVICES:-0}"
    GPU_COUNT="${#GPU_ARR[@]}"
    if ((GPU_COUNT == 0)); then
        GPU_ARR=(0)
        GPU_COUNT=1
    fi

    # Sanity: replicas on one GPU must fit in the GPU's memory budget.
    if ((TP == 1)); then
        local per_gpu=$(( (REPLICAS + GPU_COUNT - 1) / GPU_COUNT ))
        if awk "BEGIN{exit !(${per_gpu} * ${GPU_MEM_UTIL} > 0.95)}"; then
            vllm_die "${per_gpu} replicas/GPU × GPU_MEM_UTIL=${GPU_MEM_UTIL} exceeds 0.95 of VRAM — lower REPLICAS or GPU_MEM_UTIL"
        fi
    else
        if awk "BEGIN{exit !(${REPLICAS} * ${GPU_MEM_UTIL} > 0.95)}"; then
            vllm_die "${REPLICAS} TP=${TP} replicas × GPU_MEM_UTIL=${GPU_MEM_UTIL} exceeds 0.95 of per-GPU VRAM"
        fi
    fi
}

# Resolve docker without assuming socket access; container setup needs sudo once.
docker_cmd() {
    if docker info >/dev/null 2>&1; then
        DOCKER=(docker)
    elif command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
    else
        return 1
    fi
}

run_docker() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'DRY-RUN:'
        printf ' %q' "${DOCKER[@]:-docker}" "$@"
        printf '\n'
        return 0
    fi
    "${DOCKER[@]}" "$@"
}

# Physical GPU for replica i (round-robin pinning). Empty string = all GPUs (TP>1).
replica_gpu() {
    local i="$1"
    if ((TP > 1)); then
        echo ""
    else
        echo "${GPU_ARR[$((i % GPU_COUNT))]}"
    fi
}

replica_port() {
    echo $((BASE_PORT + $1))
}

container_name() {
    echo "${CONTAINER_PREFIX}-$1"
}

replica_url() {
    echo "http://${CLIENT_HOST}:$(replica_port "$1")"
}

# Build the vLLM server argument list for replica i into the VLLM_ARGS array.
build_vllm_args() {
    local i="$1" port
    port="$(replica_port "${i}")"

    VLLM_ARGS=(
        --model "${MODEL}"
        --host 0.0.0.0
        --port "${port}"
        --served-model-name "${MODEL}"
        --tensor-parallel-size "${TP}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}"
        --max-model-len "${MAX_MODEL_LEN}"
        --kv-cache-dtype "${KV_CACHE_DTYPE}"
        --max-num-seqs "${MAX_NUM_SEQS}"
        --trust-remote-code
        --enable-prefix-caching
    )
    [[ -n "${QUANTIZATION:-}" ]] && VLLM_ARGS+=(--quantization "${QUANTIZATION}")
    [[ -n "${REASONING_PARSER:-}" ]] && VLLM_ARGS+=(--reasoning-parser "${REASONING_PARSER}")
    if [[ -n "${TOOL_CALL_PARSER:-}" ]]; then
        VLLM_ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER}" --enable-auto-tool-choice)
    fi
    # shellcheck disable=SC2206
    [[ -n "${EXTRA_ARGS:-}" ]] && VLLM_ARGS+=(${EXTRA_ARGS})
    return 0
}

wait_for_http() {
    local url="$1" timeout="${2:-60}" start now
    start="$(date +%s)"
    while true; do
        if curl -fsS -m 5 -o /dev/null "${url}" 2>/dev/null; then
            return 0
        fi
        now="$(date +%s)"
        if ((now - start > timeout)); then
            return 1
        fi
        sleep 2
    done
}
