#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
if [[ "${1:-}" == --sync ]]; then
    for name in "${NODE_NAMES[@]}"; do "$SCRIPTS_DIR/provision_node.sh" "$name"; done
fi
preflight
for i in "${!NODE_NAMES[@]}"; do ensure_container "$i"; done
for i in "${!NODE_NAMES[@]}"; do ray_exec "$i" stop; done
[[ "${DRY_RUN:-0}" == 1 ]] || sleep 2
ray_exec 0 start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP" --num-gpus="${NODE_GPUS[0]}"
for ((i=1; i<NODE_COUNT; i++)); do
    ray_exec "$i" start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="${NODE_IPS[$i]}" --num-gpus="${NODE_GPUS[$i]}"
done
[[ "${DRY_RUN:-0}" == 1 ]] || sleep 4
ray_exec 0 status
echo "Cluster ready for serve.sh: $TOTAL_GPUS GPUs on $NODE_COUNT nodes ($LLM_RUNTIME)"
