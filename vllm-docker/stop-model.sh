#!/usr/bin/env bash
# =============================================================================
# stop-model.sh — Stop and remove all replicas of a model recipe
#
# Usage:
#   ./stop-model.sh <model-key>     # stop one recipe's replicas
#   ./stop-model.sh --all           # stop every vllm-recipe container
# =============================================================================
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

docker_cmd || vllm_die "cannot talk to the Docker daemon (try: sudo ${VLLM_DIR:-.}/setup-docker.sh, or 'newgrp docker')"

if [[ "${1:-}" == "--all" ]]; then
    FILTER_ARGS=(--filter "label=vllm.recipe.key")
else
    [[ $# -ge 1 ]] || vllm_die "usage: $0 <model-key> | --all"
    load_config "$1"
    FILTER_ARGS=(--filter "name=${CONTAINER_PREFIX}-")
fi

mapfile -t IDS < <("${DOCKER[@]}" ps -a "${FILTER_ARGS[@]}" --format '{{.ID}} {{.Names}}')
if [[ ${#IDS[@]} -eq 0 ]]; then
    echo "Nothing to stop."
    exit 0
fi

for line in "${IDS[@]}"; do
    echo "  removing ${line#* }"
    run_docker rm -f "${line%% *}" >/dev/null
done
echo "Stopped ${#IDS[@]} container(s)."
