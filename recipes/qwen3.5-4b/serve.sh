#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../../scripts/serve-vllm.sh" "$(basename "$HERE")" "$@"
