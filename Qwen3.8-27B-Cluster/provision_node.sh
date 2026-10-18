#!/usr/bin/env bash
# Prepare one node from the active hardware profile: container image, model
# weights, ray wheel. Everything is copied over the local network from this
# (head) machine.
#
# Usage: ./provision_node.sh <node-name>
set -euo pipefail
source "$(dirname "$0")/lib.sh"

name="${1:-}"
if [[ -z "$name" ]]; then
  echo "usage: $0 <node-name>" >&2
  echo "nodes: ${NODE_NAMES[*]}" >&2
  exit 1
fi

if ! idx="$(find_node "$name")"; then
  echo "ERROR: unknown node '$name'. Known nodes: ${NODE_NAMES[*]}" >&2
  exit 1
fi

if [[ -z "${NODE_SSH[$idx]}" ]]; then
  echo "node '$name' is the local head ($HEAD_IP); nothing to provision."
  exit 0
fi

ssh_target="${NODE_SSH[$idx]}"
echo "== provisioning '$name' ($ssh_target) =="

# 1. container image
if run_on "$idx" docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "  image $IMAGE: present"
else
  echo "  image $IMAGE: transferring over LAN (can take ~10 min)..."
  docker save "$IMAGE" | ssh -o BatchMode=yes "$ssh_target" 'docker load'
fi

# 2. model weights
if run_on "$idx" test -f "$MODEL_HOST_PATH/model.safetensors.index.json" >/dev/null 2>&1; then
  echo "  model $MODEL_HOST_PATH: present"
else
  echo "  model $MODEL_HOST_PATH: copying over LAN..."
  run_on "$idx" mkdir -p "$MODEL_HOST_PATH"
  rsync -a --info=progress2 --exclude='.cache' \
    "$MODEL_HOST_PATH/" "$ssh_target:$MODEL_HOST_PATH/"
fi

# 3. ray wheel (installed into the container by cluster_up.sh)
if run_on "$idx" bash -c "compgen -G '$WHEELS_HOST/ray-*.whl' >/dev/null" >/dev/null 2>&1; then
  echo "  ray wheel: present"
else
  echo "  ray wheel: copying..."
  run_on "$idx" mkdir -p "$WHEELS_HOST"
  rsync -a "$WHEELS_HOST/" "$ssh_target:$WHEELS_HOST/"
fi

echo "== '$name' ready =="
