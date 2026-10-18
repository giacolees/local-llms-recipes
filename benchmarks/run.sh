#!/usr/bin/env bash
# Client environment must contain the same vLLM CLI version as the server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RECIPE="${1:?Usage: benchmarks/run.sh RECIPE [extra vllm bench serve flags]}"; shift
[[ "$RECIPE" != qwen3.6-27b ]] || RECIPE=Qwen3.6-27B
[[ "$RECIPE" != */* && "${LLM_HW_PROFILE:-default}" != */* ]] || exit 2
RECIPE_DIR="$ROOT/recipes/$RECIPE"
[[ -d "$RECIPE_DIR" ]] || { echo 'Unknown recipe' >&2; exit 2; }
RUNTIME="${LLM_RUNTIME:-docker}"
case "$RUNTIME" in docker|singularity|apptainer) ;; *) echo "Unknown runtime" >&2; exit 2;; esac
if [[ "$RECIPE" == Qwen3.8-27B-Cluster ]]; then
    source "$RECIPE_DIR/lib.sh"
    VLLM_MODEL="${VLLM_MODEL:-$MODEL_REPO}"
    VLLM_IMAGE="${VLLM_IMAGE:-$IMAGE}"
    SERVED_MODEL="${SERVED_MODEL:-$MODEL_NAME}"
    BASE_URL="${BASE_URL:-http://$HEAD_IP:$SERVE_PORT}"
    TP="$TP_SIZE"; PP="$PP_SIZE"
    MAX_MODEL_LEN="$MODEL_MAX_LEN"; GPU_MEMORY_UTILIZATION="$GPU_MEM_UTIL"
    NODE_IPS_TEXT="${NODE_IPS[*]}"; NODE_GPUS_TEXT="${NODE_GPUS[*]}"
    export NODE_IPS_TEXT NODE_GPUS_TEXT NODE_COUNT PP
elif [[ -f "$RECIPE_DIR/recipe.env" ]]; then
    source "$ROOT/scripts/vllm-docker/lib.sh"
    [[ -z "${VLLM_MODEL:-}" ]] || MODEL="$VLLM_MODEL"
    load_config "$RECIPE"
    VLLM_MODEL="$MODEL"; SERVED_MODEL="${SERVED_MODEL:-$MODEL}"
    index="${REPLICA_INDEX:-0}"
    [[ "$index" =~ ^[0-9]+$ ]] && ((index < REPLICAS)) || vllm_die 'invalid REPLICA_INDEX'
    BASE_URL="${BASE_URL:-$(replica_url "$index")}"
    GPU_MEMORY_UTILIZATION="$GPU_MEM_UTIL"
    REPLICA_INDEX="$index"
    export REPLICAS QUANTIZATION KV_CACHE_DTYPE MAX_NUM_SEQS REPLICA_INDEX
else
    PROFILE="${LLM_HW_PROFILE:-4xa100}"
    [[ -f "$ROOT/profiles/$PROFILE/config.env" ]] || { echo 'Unknown profile' >&2; exit 2; }
    source "$ROOT/profiles/$PROFILE/config.env"
    source "$RECIPE_DIR/vllm.env"
    source "$RECIPE_DIR/profiles/$PROFILE.vllm.env"
    TP="${TP:-$TENSOR_PARALLEL_SIZE}"
    SERVED_MODEL="${SERVED_MODEL:-$RECIPE}"
    BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT:-8000}}"
fi
: "${VLLM_MODEL:?Set VLLM_MODEL to the server weights/tokenizer ID}"
: "${VLLM_IMAGE:?Record the exact server image tag/digest in VLLM_IMAGE}"
: "${MODEL_REVISION:?Record the server weights revision in MODEL_REVISION}"
OUT="$ROOT/benchmarks/results/$PROFILE/$RECIPE/$RUNTIME/$(date -u +%Y%m%dT%H%M%SZ)-$$"
args=(bench serve --backend vllm --base-url "$BASE_URL" --model "$SERVED_MODEL" --tokenizer "$VLLM_MODEL"
    --dataset-name random --random-input-len "${INPUT_LEN:-1024}" --random-output-len "${OUTPUT_LEN:-256}"
    --num-prompts "${NUM_PROMPTS:-100}" --max-concurrency "${CONCURRENCY:-1}" --request-rate "${REQUEST_RATE:-inf}"
    --seed 42 --save-result --save-detailed --result-dir "$OUT" --result-filename result.json "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf 'vllm '; printf '%q ' "${args[@]}"; printf '\n'; exit; fi
command -v vllm >/dev/null || { echo 'Install the matching vLLM CLI in your benchmark environment' >&2; exit 2; }
mkdir -p "$OUT"
export OUT RECIPE PROFILE RUNTIME VLLM_MODEL VLLM_IMAGE MODEL_REVISION BASE_URL SERVED_MODEL
export MAX_MODEL_LEN TP GPU_MEMORY_UTILIZATION VLLM_SIF
python3 - "${args[@]}" <<'PY'
import json, os, subprocess, platform, sys
from pathlib import Path
keys=['VLLM_MODEL','VLLM_IMAGE','MODEL_REVISION','MAX_MODEL_LEN','TP','PP',
      'GPU_MEMORY_UTILIZATION','VLLM_SIF','REPLICAS','QUANTIZATION',
      'KV_CACHE_DTYPE','MAX_NUM_SEQS','BASE_URL','SERVED_MODEL','REPLICA_INDEX']
data={k:os.environ.get(k) for k in keys}
data.update(recipe=os.environ['RECIPE'],profile=os.environ['PROFILE'],runtime=os.environ['RUNTIME'],client_host=platform.node(),benchmark_argv=sys.argv[1:])
if os.environ.get('NODE_IPS_TEXT'):
    data['topology']={'ips':os.environ['NODE_IPS_TEXT'].split(), 'gpus_per_node':os.environ['NODE_GPUS_TEXT'].split()}
data['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
try: data['client_gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)
except (FileNotFoundError,subprocess.CalledProcessError): data['client_gpus']=None
Path(os.environ['OUT'],'metadata.json').write_text(json.dumps(data,indent=2)+'\n')
PY
vllm "${args[@]}" 2>&1 | tee "$OUT/client.log"
python3 "$ROOT/benchmarks/summarize.py"
