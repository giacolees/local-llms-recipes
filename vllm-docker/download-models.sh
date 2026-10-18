#!/usr/bin/env bash
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/vllm-docker/download-models.sh" "$@"
