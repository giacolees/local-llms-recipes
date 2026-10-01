#!/usr/bin/env bash
# Launch the model configured in the active hardware profile on the running
# Ray cluster (profiles/<LLM_HW_PROFILE>/qwen3.8-27b-cluster.env).
#
# Usage: ./serve.sh [--wait]
#   --wait   block until the API answers, or fail with the log tail
set -euo pipefail
source "$(dirname "$0")/lib.sh"

if (( WORLD_SIZE > TOTAL_GPUS )); then
  echo "ERROR: TP($TP_SIZE) x PP($PP_SIZE) = $WORLD_SIZE > total GPUs ($TOTAL_GPUS) in the active profile" >&2
  exit 1
fi
if (( TP_SIZE > MAX_GPUS_PER_NODE )); then
  echo "WARNING: TP=$TP_SIZE exceeds GPUs on one node ($MAX_GPUS_PER_NODE): a TP group will cross the network." >&2
fi

args=(serve "$MODEL_PATH"
  --served-model-name "$MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size "$PP_SIZE"
  --distributed-executor-backend "$BACKEND"
  --max-model-len "$MODEL_MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --host "$SERVE_HOST"
  --port "$SERVE_PORT")

if [[ -n "$REASONING_PARSER" ]]; then args+=(--reasoning-parser "$REASONING_PARSER"); fi
if [[ -n "$TOOL_PARSER" ]]; then args+=(--tool-call-parser "$TOOL_PARSER"); fi
if [[ "$AUTO_TOOL_CHOICE" == "1" ]]; then args+=(--enable-auto-tool-choice); fi
if [[ -n "$SPEC_METHOD" && "$SPEC_TOKENS" != "0" ]]; then
  spec_json="$(printf '{"method":"%s","num_speculative_tokens":%s}' "$SPEC_METHOD" "$SPEC_TOKENS")"
  args+=(--speculative-config "$spec_json")
fi
for a in "${EXTRA_ARGS[@]}"; do args+=("$a"); done

cmd="vllm"
for a in "${args[@]}"; do cmd+=" $(printf '%q' "$a")"; done
cmd+=" > $(printf '%q' "$LOG_CONTAINER") 2>&1"

echo "stopping previous server (if any)..."
run_on 0 docker exec "$HEAD_CONTAINER" pkill -f "$MODEL_PATH" >/dev/null 2>&1 || true
sleep 8

echo "launching $MODEL_NAME (TP=$TP_SIZE PP=$PP_SIZE backend=$BACKEND ctx=$MODEL_MAX_LEN)"
run_on 0 docker exec -d "$HEAD_CONTAINER" bash -c "$cmd"

if [[ "${1:-}" == "--wait" ]]; then
  printf "waiting for API"
  for _ in $(seq 1 90); do
    if curl -fsS -m 3 "http://${HEAD_IP}:${SERVE_PORT}/v1/models" >/dev/null 2>&1; then
      echo " — up."
      exit 0
    fi
    if grep -qE "Traceback \(most recent|EngineCore failed|torch.OutOfMemoryError" "$LOG_HOST" 2>/dev/null; then
      echo; echo "ERROR: server failed to start:"
      tail -n 15 "$LOG_HOST"
      exit 1
    fi
    printf "."; sleep 10
  done
  echo; echo "ERROR: timed out. Last log lines:"
  tail -n 15 "$LOG_HOST"
  exit 1
fi

echo "log:  tail -f $LOG_HOST"
echo "API:  http://${HEAD_IP}:${SERVE_PORT}/v1/models"
