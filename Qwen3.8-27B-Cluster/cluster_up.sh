#!/usr/bin/env bash
# Bring up the containers + Ray cluster on every node listed in the active
# hardware profile (profiles/<LLM_HW_PROFILE>/qwen3.8-27b-cluster.env).
#
# Usage: ./cluster_up.sh [--sync]
#   --sync   provision each node first (copy image/weights/wheels over the LAN)
#
# WARNING: resets Ray on all nodes, which stops any running vLLM engine.
#          Run ./serve.sh afterwards.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

if [[ "${1:-}" == "--sync" ]]; then
  for name in "${NODE_NAMES[@]}"; do
    "$SCRIPTS_DIR/provision_node.sh" "$name"
  done
fi

preflight

ensure_container() {
  local idx=$1 c name ip iface
  name="${NODE_NAMES[$idx]}"; c="$(container_of "$idx")"
  ip="${NODE_IPS[$idx]}"; iface="${NODE_IFACES[$idx]}"

  if ! run_on "$idx" docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
    echo "[$name] creating container $c"
    run_on "$idx" docker run -d --name "$c" --network host --ipc host \
      --shm-size "$CLUSTER_SHM_SIZE" --gpus all \
      -v "$MODELS_DIR:$CONTAINER_MODELS_DIR" \
      -e "VLLM_HOST_IP=$ip" \
      -e "NCCL_SOCKET_IFNAME=$iface" -e "GLOO_SOCKET_IFNAME=$iface" \
      -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" \
      --entrypoint bash "$IMAGE" -c 'sleep infinity' >/dev/null
  elif ! run_on "$idx" docker ps --format '{{.Names}}' | grep -qx "$c"; then
    echo "[$name] starting existing container $c"
    run_on "$idx" docker start "$c" >/dev/null
  else
    echo "[$name] container $c already running"
  fi

  if ! run_on "$idx" docker exec "$c" which ray >/dev/null 2>&1; then
    if ! run_on "$idx" docker exec "$c" bash -c \
         "compgen -G '$WHEELS_CONTAINER/ray-${RAY_VERSION}*.whl' >/dev/null"; then
      echo "ERROR: ray wheel not found on '$name'." >&2
      echo "       Run: $SCRIPTS_DIR/provision_node.sh $name" >&2
      exit 1
    fi
    echo "[$name] installing ray==$RAY_VERSION"
    run_on "$idx" docker exec "$c" pip install --no-index \
      --find-links="$WHEELS_CONTAINER" --no-deps -q "ray==$RAY_VERSION"
  fi
}

echo "== containers =="
for i in "${!NODE_NAMES[@]}"; do ensure_container "$i"; done

echo "== resetting ray on all nodes =="
for i in "${!NODE_NAMES[@]}"; do
  run_on "$i" docker exec "$(container_of "$i")" ray stop >/dev/null 2>&1 || true
done
sleep 2

echo "== starting ray head: ${NODE_NAMES[0]} ($HEAD_IP) =="
run_on 0 docker exec "$HEAD_CONTAINER" ray start --head \
  --port="$RAY_PORT" --node-ip-address="$HEAD_IP" --num-gpus="${NODE_GPUS[0]}" \
  >/dev/null 2>&1 || true

for (( i=1; i<NODE_COUNT; i++ )); do
  echo "== joining ray worker: ${NODE_NAMES[$i]} (${NODE_IPS[$i]}) =="
  run_on "$i" docker exec "$(container_of "$i")" ray start \
    --address="${HEAD_IP}:${RAY_PORT}" --node-ip-address="${NODE_IPS[$i]}" \
    --num-gpus="${NODE_GPUS[$i]}" >/dev/null 2>&1 || true
done

sleep 4
echo
run_on 0 docker exec "$HEAD_CONTAINER" ray status | sed -n '1,20p'
echo
echo "cluster up — $TOTAL_GPUS GPUs across $NODE_COUNT node(s); next: ./serve.sh"
