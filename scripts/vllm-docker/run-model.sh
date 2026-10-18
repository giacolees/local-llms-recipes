#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
source "$HERE/runtime.sh"
[[ $# -ge 1 ]] || { echo "Usage: $0 MODEL [replicas]"; recipe_keys; exit 2; }
[[ -z "${2:-}" ]] || export REPLICAS="$2"
load_config "$1"
resolve_runtime
if [[ "${DRY_RUN:-0}" != 1 ]]; then mkdir -p "$HF_HOME" "$LOG_DIR"; fi
for ((i=0; i<REPLICAS; i++)); do start_replica "$i"; done
if [[ "${DRY_RUN:-0}" == 1 || "${NO_WAIT:-0}" == 1 ]]; then exit 0; fi
for ((i=0; i<REPLICAS; i++)); do
    wait_for_http "$(replica_url "$i")/health" "$READY_TIMEOUT" || vllm_die "replica $i did not become healthy; inspect container or $LOG_DIR logs"
done
printf '%s: %s replicas ready on %s\n' "$MODEL_KEY" "$REPLICAS" "$LLM_RUNTIME"
