#!/usr/bin/env bash
# Run from a Python environment containing the same vLLM version as the server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RECIPE="${1:?Usage: benchmarks/run.sh RECIPE [extra vllm bench serve flags]}"; shift
PROFILE="${LLM_HW_PROFILE:-4xa100}"
RUNTIME="${LLM_RUNTIME:-docker}"
[[ "$RECIPE" != */* && "$PROFILE" != */* ]] || exit 2
[[ -d "$ROOT/recipes/$RECIPE" && -f "$ROOT/profiles/$PROFILE/config.env" ]] || { echo 'Unknown recipe/profile' >&2; exit 2; }
: "${VLLM_MODEL:?Set VLLM_MODEL to the server weights/tokenizer ID}"
: "${VLLM_IMAGE:?Record the exact server image tag/digest in VLLM_IMAGE}"
: "${MODEL_REVISION:?Record the server weights revision in MODEL_REVISION}"
command -v vllm >/dev/null || { echo 'Install the matching vLLM CLI in your benchmark environment' >&2; exit 2; }
OUT="$ROOT/benchmarks/results/$PROFILE/$RECIPE/$RUNTIME/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
export OUT RECIPE PROFILE RUNTIME
python3 - <<'PY'
import json, os, subprocess, platform
from pathlib import Path
keys=['VLLM_MODEL','VLLM_IMAGE','MODEL_REVISION','MAX_MODEL_LEN','TP','GPU_MEMORY_UTILIZATION','VLLM_SIF']
data={k:os.environ.get(k) for k in keys}
data.update(recipe=os.environ['RECIPE'],profile=os.environ['PROFILE'],runtime=os.environ['RUNTIME'],client_host=platform.node())
data['git_commit']=subprocess.check_output(['git','-C',os.getcwd(),'rev-parse','HEAD'],text=True).strip()
try: data['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)
except (FileNotFoundError,subprocess.CalledProcessError): data['gpus']=None
Path(os.environ['OUT'],'metadata.json').write_text(json.dumps(data,indent=2)+'\n')
PY
vllm bench serve --backend vllm --base-url "${BASE_URL:-http://127.0.0.1:8000}" \
    --model "$RECIPE" --tokenizer "$VLLM_MODEL" \
    --dataset-name random --random-input-len "${INPUT_LEN:-1024}" \
    --random-output-len "${OUTPUT_LEN:-256}" --num-prompts "${NUM_PROMPTS:-100}" \
    --max-concurrency "${CONCURRENCY:-1}" --request-rate "${REQUEST_RATE:-inf}" \
    --seed 42 --save-result --save-detailed --result-dir "$OUT" --result-filename result.json \
    "$@" 2>&1 | tee "$OUT/client.log"
python3 "$ROOT/benchmarks/summarize.py"
