#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
source "$HERE/runtime.sh"
show_status() {
    load_config "$1"; resolve_runtime
    if [[ "$LLM_RUNTIME" == docker ]]; then
        run_command "${DOCKER[@]}" ps -a --filter "label=vllm.recipe.key=$MODEL_KEY" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    else
        run_command "$INSTANCE_RUNTIME" instance list "${CONTAINER_PREFIX}-*"
        echo "Server logs: $LOG_DIR"
    fi
}
if [[ $# -ge 1 ]]; then show_status "$1"
else while IFS= read -r key; do (show_status "$key"); done < <(recipe_keys); fi
