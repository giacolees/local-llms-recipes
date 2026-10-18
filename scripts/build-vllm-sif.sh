#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VLLM_IMAGE:?Set VLLM_IMAGE to the same tested tag/digest used for Docker}"
SIF="${VLLM_SIF:-$ROOT/images/vllm-openai.sif}"
if command -v apptainer >/dev/null 2>&1; then runtime=apptainer
elif command -v singularity >/dev/null 2>&1; then runtime=singularity
else echo 'Install Singularity or Apptainer' >&2; exit 2; fi
mkdir -p "$(dirname "$SIF")"
exec "$runtime" pull "$SIF" "docker://$VLLM_IMAGE"
