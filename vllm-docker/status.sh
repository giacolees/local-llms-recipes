#!/usr/bin/env bash
# =============================================================================
# status.sh — Show replica containers and GPU memory state
#
# Usage:
#   ./status.sh [model-key]     # one recipe (default: all vllm recipes)
# =============================================================================
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

docker_cmd || vllm_die "cannot talk to the Docker daemon (try: sudo ./setup-docker.sh, or 'newgrp docker')"

FILTER_ARGS=(--filter "label=vllm.recipe.key")
NAME_ONLY=""
if [[ $# -ge 1 ]]; then
    load_config "$1"
    FILTER_ARGS=(--filter "name=${CONTAINER_PREFIX}-")
    NAME_ONLY="${CONTAINER_PREFIX}-"
fi

echo "=== vLLM replica containers ==="
"${DOCKER[@]}" ps -a "${FILTER_ARGS[@]}" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true

if [[ -n "${NAME_ONLY}" ]]; then
    echo
    echo "=== Recent errors (if any) ==="
    for ((i = 0; i < REPLICAS; i++)); do
        name="$(container_name "${i}")"
        if "${DOCKER[@]}" inspect "${name}" >/dev/null 2>&1; then
            errs="$("${DOCKER[@]}" logs --since 5m "${name}" 2>&1 | grep -Ei 'error|out of memory|traceback' | tail -3 || true)"
            [[ -z "${errs}" ]] || echo "${name}:"$'\n'"${errs}"
        fi
    done
fi

echo
echo "=== GPU state ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
