#!/usr/bin/env bash
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/recipes/Qwen3.8-27B-Cluster/stop.sh" "$@"
