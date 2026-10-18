#!/usr/bin/env bash
# Show API health, Ray resources and GPU usage for every node in the active
# hardware profile.
set -uo pipefail
source "$(dirname "$0")/lib.sh"

echo "== API (http://${HEAD_IP}:${SERVE_PORT}) =="
if curl -fsS -m 5 "http://${HEAD_IP}:${SERVE_PORT}/v1/models" -o /tmp/vllm_models.$$ 2>/dev/null; then
  python3 - /tmp/vllm_models.$$ <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for m in d.get("data", []):
    print(f"  {m['id']}  (max_model_len={m.get('max_model_len')})")
EOF
  rm -f /tmp/vllm_models.$$
else
  echo "  not responding"
  rm -f /tmp/vllm_models.$$
fi

echo
echo "== Ray cluster =="
run_on 0 docker exec "$HEAD_CONTAINER" ray status 2>/dev/null | sed -n '1,18p' || echo "  ray not reachable"

for i in "${!NODE_NAMES[@]}"; do
  echo
  echo "== GPUs on ${NODE_NAMES[$i]} (${NODE_IPS[$i]}) =="
  run_on "$i" nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "  n/a"
done
