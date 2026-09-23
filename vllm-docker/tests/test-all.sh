#!/usr/bin/env bash
# =============================================================================
# test-all.sh — Run the per-model test suite for every recipe in models/
#
# Usage:
#   ./test-all.sh                       # every recipe, sequentially (start→test→stop)
#   ./test-all.sh gemma4-e4b phi-4-14b  # a subset
#   KEEP=1 ./test-all.sh                # leave each recipe running after its suite
#
# Each model gets a fresh, exclusive GPU budget (previous recipes are stopped
# first), so replica counts and VRAM fit are measured honestly on this rig.
# Reports: vllm-docker/logs/test-<key>-<timestamp>.log
# =============================================================================
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${VLLM_DIR}/lib.sh"

if [[ $# -gt 0 ]]; then
    KEYS=("$@")
else
    mapfile -t KEYS < <(basename -s .env "${VLLM_DIR}"/models/*.env)
fi

LOG_DIR="${VLLM_DIR}/logs"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/test-all-$(date +%Y%m%d-%H%M%S).summary"

declare -a R_KEY=() R_RC=() R_LINE=()
overall=0

for key in "${KEYS[@]}"; do
    report="${LOG_DIR}/test-${key}-$(date +%Y%m%d-%H%M%S).log"
    echo
    echo "######################################################################"
    echo "### ${key}"
    echo "######################################################################"

    # Make sure no other recipe is competing for VRAM.
    "${VLLM_DIR}/stop-model.sh" --all >/dev/null 2>&1 || true

    rc=0
    KEEP="${KEEP:-0}" "${TESTS_DIR}/test-model.sh" "${key}" 2>&1 | tee "${report}" || rc=$?

    line="$(grep -E '^(SUITE (PASSED|FAILED))' "${report}" | tail -1)"
    R_KEY+=("${key}")
    R_RC+=("${rc}")
    R_LINE+=("${line:-no verdict (setup error?)}")
    [[ ${rc} -eq 0 ]] || overall=1
done

echo
echo "======================================================================"
echo "=== test-all summary ($(hostname), $(date -Is)) ==="
printf '%-20s %-6s %s\n' MODEL RESULT VERDICT
{
    printf '%-20s %-6s %s\n' MODEL RESULT VERDICT
    for ((i = 0; i < ${#R_KEY[@]}; i++)); do
        res="FAIL"
        [[ "${R_RC[$i]}" == "0" ]] && res="PASS"
        printf '%-20s %-6s %s\n' "${R_KEY[$i]}" "${res}" "${R_LINE[$i]}"
    done
} | tee "${SUMMARY}"

echo
echo "Summary written to ${SUMMARY}"
exit "${overall}"
