#!/usr/bin/env bash
# =============================================================================
# lib.sh — shared helpers for the multi-node Qwen3.8-27B vLLM cluster recipe.
#
# Source this file from the other cluster scripts; do not execute it directly.
#
# Configuration precedence (highest first):
#   1. Variables already exported in the environment
#   2. profiles/<LLM_HW_PROFILE>/qwen3.8-27b-cluster.env  (topology + serve)
#   3. profiles/<LLM_HW_PROFILE>/config.env              (shared GPU layout)
#   4. Defaults in this file (an A6000 2-node × 2-GPU layout)
#
# Activate a profile first, e.g.:
#   source ../switch-profile.sh 4xa6000
# =============================================================================

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${CLUSTER_DIR}"
REPO_ROOT="$(cd "${CLUSTER_DIR}/.." && pwd)"
PROFILE="${LLM_HW_PROFILE:-4xa6000}"
PROFILE_DIR="${LLM_PROFILES_DIR:-${REPO_ROOT}/profiles}/${PROFILE}"

if [[ ! -f "${PROFILE_DIR}/config.env" ]]; then
  echo "ERROR: unknown hardware profile '${PROFILE}' (missing ${PROFILE_DIR}/config.env)" >&2
  echo "       Activate one with: source ${REPO_ROOT}/switch-profile.sh <profile>" >&2
  exit 1
fi

# Scalar knobs a caller may override from the environment; captured now and
# re-applied after the profile files are sourced so `SERVE_PORT=9001 ./serve.sh`
# and friends keep working. (Node arrays are profile-only.)
_CLUSTER_OVERRIDABLE=(
  IMAGE CLUSTER_SHM_SIZE MODELS_DIR CONTAINER_MODELS_DIR RAY_PORT RAY_VERSION
  RAY_WHEELS CONTAINER_PREFIX NCCL_IB_DISABLE
  MODEL_PATH MODEL_HOST_PATH MODEL_NAME MODEL_REPO MODEL_MAX_LEN
  TP_SIZE PP_SIZE BACKEND GPU_MEM_UTIL REASONING_PARSER TOOL_PARSER
  AUTO_TOOL_CHOICE SERVE_HOST SERVE_PORT SERVE_LOG_NAME SPEC_METHOD SPEC_TOKENS
  NODE_COUNT
)
_CLUSTER_USER_VALS=()
for _v in "${_CLUSTER_OVERRIDABLE[@]}"; do _CLUSTER_USER_VALS+=("${!_v-}"); done

# ---- 1. shared GPU layout -------------------------------------------------
# shellcheck disable=SC1090
source "${PROFILE_DIR}/config.env"

# ---- 2. project-specific cluster config (topology, model, serve options) --
CLUSTER_CONFIG="${CLUSTER_CONFIG:-${PROFILE_DIR}/qwen3.8-27b-cluster.env}"
if [[ -f "${CLUSTER_CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CLUSTER_CONFIG}"
fi

# Re-apply caller-supplied overrides on top of the profile.
_i=0
for _v in "${_CLUSTER_OVERRIDABLE[@]}"; do
  if [[ -n "${_CLUSTER_USER_VALS[$_i]:-}" ]]; then
    printf -v "$_v" '%s' "${_CLUSTER_USER_VALS[$_i]}"
  fi
  _i=$(( _i + 1 ))
done
unset _CLUSTER_OVERRIDABLE _CLUSTER_USER_VALS _v _i

# ---- 3. defaults (standalone use without a populated profile) -------------
IMAGE="${IMAGE:-vllm/vllm-openai:v0.30.0}"
CLUSTER_SHM_SIZE="${CLUSTER_SHM_SIZE:-16g}"
MODELS_DIR="${MODELS_DIR:-/opt/models}"
CONTAINER_MODELS_DIR="${CONTAINER_MODELS_DIR:-/models}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_VERSION="${RAY_VERSION:-2.58.0}"
RAY_WHEELS="${RAY_WHEELS:-wheels}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-vllm-ray}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

MODEL_PATH="${MODEL_PATH:-${CONTAINER_MODELS_DIR}/Qwen3.8-27B}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-${MODELS_DIR}/Qwen3.8-27B}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B}"
MODEL_MAX_LEN="${MODEL_MAX_LEN:-262144}"

TP_SIZE="${TP_SIZE:-${TENSOR_PARALLEL_SIZE:-2}}"
PP_SIZE="${PP_SIZE:-${PIPELINE_PARALLEL_SIZE:-2}}"
BACKEND="${BACKEND:-ray}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
REASONING_PARSER="${REASONING_PARSER:-}"
TOOL_PARSER="${TOOL_PARSER:-}"
AUTO_TOOL_CHOICE="${AUTO_TOOL_CHOICE:-0}"
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-8000}"
SERVE_LOG_NAME="${SERVE_LOG_NAME:-vllm.log}"
SPEC_METHOD="${SPEC_METHOD:-}"
SPEC_TOKENS="${SPEC_TOKENS:-0}"

# Node topology: first entry is the Ray head (scripts run from it).
if [[ -z "${NODE_NAMES[*]:-}" ]]; then
  NODE_NAMES=(head worker1)
  NODE_SSH=("" "root@10.0.0.3")
  NODE_IPS=(10.0.0.2 10.0.0.3)
  NODE_IFACES=(enp153s0f1np1 enp179s0f0np0)
  NODE_GPUS=(2 2)
fi
if [[ -z "${EXTRA_ARGS[*]:-}" ]]; then
  EXTRA_ARGS=()
fi

NODE_COUNT="${NODE_COUNT:-${#NODE_NAMES[@]}}"

if (( NODE_COUNT == 0 )); then
  echo "ERROR: no cluster nodes configured for profile '${PROFILE}'" >&2
  exit 1
fi

# ---- 4. derived paths -----------------------------------------------------
HEAD_IP="${NODE_IPS[0]}"
HEAD_CONTAINER="${CONTAINER_PREFIX}-${NODE_NAMES[0]}"
WHEELS_HOST="${MODELS_DIR}/${RAY_WHEELS}"
WHEELS_CONTAINER="${CONTAINER_MODELS_DIR}/${RAY_WHEELS}"
LOG_HOST="${MODELS_DIR}/logs/${SERVE_LOG_NAME}"
LOG_CONTAINER="${CONTAINER_MODELS_DIR}/logs/${SERVE_LOG_NAME}"
WORLD_SIZE=$(( TP_SIZE * PP_SIZE ))

TOTAL_GPUS=0
for g in "${NODE_GPUS[@]}"; do TOTAL_GPUS=$(( TOTAL_GPUS + g )); done
MAX_GPUS_PER_NODE=0
for g in "${NODE_GPUS[@]}"; do (( g > MAX_GPUS_PER_NODE )) && MAX_GPUS_PER_NODE=$g; done

# run_on <index> <command...> — execute locally for ssh: null, else over ssh.
run_on() {
  local idx=$1; shift
  local ssh_target="${NODE_SSH[$idx]}"
  if [[ -z "$ssh_target" ]]; then
    "$@"
  else
    local cmd="" a
    for a in "$@"; do cmd+=" $(printf '%q' "$a")"; done
    ssh -o BatchMode=yes "$ssh_target" "$cmd"
  fi
}

container_of() { echo "${CONTAINER_PREFIX}-${NODE_NAMES[$1]}"; }

find_node() {
  local wanted=$1 i
  for i in "${!NODE_NAMES[@]}"; do
    if [[ "${NODE_NAMES[$i]}" == "$wanted" ]]; then echo "$i"; return 0; fi
  done
  return 1
}

preflight() {
  local i
  for i in "${!NODE_NAMES[@]}"; do
    if ! run_on "$i" true >/dev/null 2>&1; then
      echo "ERROR: cannot reach node '${NODE_NAMES[$i]}' (${NODE_SSH[$i]:-local})" >&2
      return 1
    fi
    if ! run_on "$i" docker info >/dev/null 2>&1; then
      echo "ERROR: docker not usable on node '${NODE_NAMES[$i]}'" >&2
      return 1
    fi
  done
}
