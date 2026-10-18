#!/usr/bin/env bash
# Compatibility alias; serve.sh supports both container runtimes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLM_RUNTIME="${LLM_RUNTIME:-singularity}"
exec "$HERE/serve.sh" "$@"
