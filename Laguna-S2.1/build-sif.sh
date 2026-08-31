#!/usr/bin/env bash
# Build the Laguna S2.1 DFlash llama.cpp SIF image without admin privileges.
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

echo "Building laguna-s-2.1-dflash.sif with ${CONTAINER_RUNTIME}..."
"${CONTAINER_RUNTIME}" build \
    --fakeroot \
    --force \
    laguna-s-2.1-dflash.sif \
    laguna-s2.1.def
