#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
source "$HERE/runtime.sh"
[[ $# -ge 1 ]] || vllm_die 'Usage: stop-model.sh MODEL | --all'
if [[ "$1" == --all ]]; then
    # Only stop instances/containers belonging to known repo recipes.
    while IFS= read -r key; do (load_config "$key"; resolve_runtime; stop_replicas); done < <(recipe_keys)
else
    load_config "$1"; resolve_runtime; stop_replicas
fi
