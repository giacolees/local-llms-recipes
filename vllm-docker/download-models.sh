#!/usr/bin/env bash
# =============================================================================
# download-models.sh — Prefetch model weights into the HF cache (offline starts)
#
# Usage:
#   ./download-models.sh                 # all recipes in vllm-docker/models/
#   ./download-models.sh gemma4-e4b qwen3.6-35b-a3b
#
# Runs `huggingface_hub.snapshot_download` inside the vLLM image so the host
# needs no Python setup. Gemma models are gated: export HF_TOKEN first.
# =============================================================================
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

docker_cmd || vllm_die "cannot talk to the Docker daemon (try: sudo ./setup-docker.sh)"

if [[ $# -gt 0 ]]; then
    KEYS=("$@")
else
    mapfile -t KEYS < <(basename -s .env "${VLLM_DIR}"/models/*.env)
fi

mkdir -p "${HF_HOME}"

for key in "${KEYS[@]}"; do
    load_config "${key}"
    echo "=== ${key}: ${MODEL} ==="
    run_docker run --rm \
        --gpus all \
        -v "${HF_HOME}:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        ${HF_TOKEN:+-e HF_TOKEN} \
        --entrypoint python \
        "${VLLM_IMAGE}" \
        -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}'); print('downloaded: ${MODEL}')"
done
