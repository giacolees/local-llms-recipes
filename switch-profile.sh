#!/usr/bin/env bash
# =============================================================================
# switch-profile.sh — Activate a hardware profile for local-LLMs
#
# Usage:
#   source ./switch-profile.sh 4xa100      # Activate 4×A100 profile
#   source ./switch-profile.sh 2xa6000     # Activate 2×A6000 profile
#   source ./switch-profile.sh             # Show current profile
#
# This sets LLM_HW_PROFILE and sources the profile's config.env so that
# all run-*.sh scripts pick up the correct GPU layout automatically.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILES_DIR="${REPO_ROOT}/profiles"

show_current() {
    if [[ -n "${LLM_HW_PROFILE:-}" ]]; then
        echo "Current hardware profile: ${LLM_HW_PROFILE}"
        echo "  Config: ${PROFILES_DIR}/${LLM_HW_PROFILE}/config.env"
    else
        echo "No hardware profile is active."
        echo "Run: source ./switch-profile.sh <profile>"
    fi
}

if [[ $# -eq 0 ]]; then
    show_current
    return 0 2>/dev/null || exit 0
fi

PROFILE="$1"
PROFILE_DIR="${PROFILES_DIR}/${PROFILE}"

if [[ ! -d "${PROFILE_DIR}" ]]; then
    echo "ERROR: Unknown hardware profile '${PROFILE}'" >&2
    echo "Available profiles:" >&2
    for d in "${PROFILES_DIR}"/*/; do
        basename "${d}" >&2
    done
    return 1 2>/dev/null || exit 1
fi

# Source the shared hardware config
CONFIG_FILE="${PROFILE_DIR}/config.env"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Missing config.env in profile '${PROFILE}'" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

export LLM_HW_PROFILE="${PROFILE}"
export LLM_PROFILES_DIR="${PROFILES_DIR}"

echo "Activated hardware profile: ${PROFILE}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  GPU_COUNT=${GPU_COUNT}"
echo "  TENSOR_SPLIT=${TENSOR_SPLIT:-${TENSOR_PARALLEL_SIZE:-N/A}}"
echo
echo "Per-project configs are at: ${PROFILE_DIR}/"
echo "Run your server scripts normally — they will source the active profile."
