#!/usr/bin/env bash
# Build the DeepSeek V4 Flash llama.cpp SIF image without admin privileges.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if command -v singularity >/dev/null 2>&1; then
    CONTAINER_RUNTIME=singularity
elif command -v apptainer >/dev/null 2>&1; then
    CONTAINER_RUNTIME=apptainer
else
    echo "ERROR: singularity or apptainer is required" >&2
    exit 1
fi

echo "Building deepseek-v4-flash-llamacpp.sif with ${CONTAINER_RUNTIME}..."
"${CONTAINER_RUNTIME}" build \
    --fakeroot \
    --force \
    deepseek-v4-flash-llamacpp.sif \
    deepseek-v4-flash.def
