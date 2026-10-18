#!/usr/bin/env bash
# Stop the vLLM server, Ray, and (by default) remove the containers.
# Set KEEP_CONTAINERS=1 to keep the containers around.
set -uo pipefail
source "$(dirname "$0")/lib.sh"

echo "stopping vLLM server..."
container_exec 0 pkill -f "$MODEL_PATH" >/dev/null 2>&1 || true

echo "stopping ray..."
for i in "${!NODE_NAMES[@]}"; do
  ray_exec "$i" stop >/dev/null 2>&1 || true
done

if [[ "${KEEP_CONTAINERS:-0}" == "1" ]]; then
  echo "keeping containers (KEEP_CONTAINERS=1)"
else
  echo "removing containers..."
  for i in "${!NODE_NAMES[@]}"; do
    if [[ "$LLM_RUNTIME" == docker ]]; then run_on "$i" docker rm -f "$(container_of "$i")"
    else run_on "$i" "$LLM_RUNTIME" instance stop "$(container_of "$i")"; fi
  done
fi

echo "done."
