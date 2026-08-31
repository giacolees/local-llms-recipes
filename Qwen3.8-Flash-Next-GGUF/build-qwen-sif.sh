#!/usr/bin/env bash
# Build the Qwen3.8-Flash-Next llama.cpp SIF image.
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

read -rsp "GitHub PAT (press Enter if not needed): " GITHUB_PAT
printf '\n'

LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
BUILD_ENV_FILE="$(mktemp)"
cleanup() {
    rm -f "${BUILD_ENV_FILE}"
    unset GITHUB_PAT LLAMA_REPO BUILD_ENV_FILE
}
trap cleanup EXIT

chmod 600 "${BUILD_ENV_FILE}"
printf 'GITHUB_PAT=%q\nLLAMA_REPO=%q\n' \
    "${GITHUB_PAT}" "${LLAMA_REPO}" > "${BUILD_ENV_FILE}"

echo "Building qwen3.8-flash-next-llamacpp.sif with ${CONTAINER_RUNTIME}..."
sudo "${CONTAINER_RUNTIME}" build \
    --force \
    --bind "${BUILD_ENV_FILE}:/tmp/llama-build.env:ro" \
    qwen3.8-flash-next-llamacpp.sif \
    qwen3.8-flash-next.def
