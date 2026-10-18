#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
source "$HERE/runtime.sh"
KEYS=()
if [[ $# -gt 0 ]]; then KEYS=("$@")
else while IFS= read -r key; do KEYS+=("$key"); done < <(recipe_keys); fi
for key in "${KEYS[@]}"; do
    (
        load_config "$key"; resolve_runtime
        [[ "${DRY_RUN:-0}" == 1 ]] || mkdir -p "$HF_HOME"
        code='import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ["DOWNLOAD_MODEL"], revision=os.environ.get("MODEL_REVISION"))'
        if [[ "$LLM_RUNTIME" == docker ]]; then
            cmd=("${DOCKER[@]}" run --rm -v "$HF_HOME:/hf" -e HF_HOME=/hf -e "DOWNLOAD_MODEL=$MODEL")
            [[ -z "${HF_TOKEN:-}" ]] || cmd+=(-e HF_TOKEN)
            [[ -z "${MODEL_REVISION:-}" ]] || cmd+=(-e MODEL_REVISION)
            run_command "${cmd[@]}" --entrypoint python3 "$VLLM_IMAGE" -c "$code"
        else
            export APPTAINERENV_HF_TOKEN="${HF_TOKEN:-}" SINGULARITYENV_HF_TOKEN="${HF_TOKEN:-}"
            export APPTAINERENV_MODEL_REVISION="${MODEL_REVISION:-}" SINGULARITYENV_MODEL_REVISION="${MODEL_REVISION:-}"
            run_command "$INSTANCE_RUNTIME" exec --cleanenv --bind "$HF_HOME:/hf" --env HF_HOME=/hf \
                --env "DOWNLOAD_MODEL=$MODEL" "$VLLM_SIF" python3 -c "$code"
        fi
    )
done
