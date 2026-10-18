#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../../scripts/vllm-docker/stop-model.sh" "$(basename "$HERE")" "$@"
