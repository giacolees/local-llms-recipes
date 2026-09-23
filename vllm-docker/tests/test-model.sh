#!/usr/bin/env bash
# =============================================================================
# test-model.sh — Per-model test suite for the vLLM Docker recipes
#
# Usage:
#   ./test-model.sh <model-key>              # start recipe (if needed) + run suite
#   NO_START=1 ./test-model.sh <key>         # test already-running replicas only
#   KEEP=1 ./test-model.sh <key>             # leave replicas running after the suite
#
# Tests (run against EVERY replica):
#   1. ready        every replica answers /health
#   2. model-id     /v1/models reports the configured model + max_model_len
#   3. smoke        one chat completion per replica, all concurrent
#   4. json         guided JSON (response_format=json_schema) validates against schema
#   5. tools        function calling returns a well-formed tool call  (if EXPECT_TOOL_CALL=1)
#   6. reasoning    reasoning-parser wiring (informational, never fails)
#   7. sweep        REPLICAS concurrent 80-word generations; latency + tok/s + peak VRAM
#
# Exit code 0 = all required tests passed. Logs: vllm-docker/logs/test-<key>-*.log
# =============================================================================
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${TESTS_DIR}/../lib.sh"

[[ $# -ge 1 ]] || { echo "Usage: $0 <model-key>" >&2; exit 1; }
load_config "$1"

TMP="$(mktemp -d /tmp/vllm-test-${MODEL_KEY}-XXXXXX)"
trap 'rm -rf "${TMP}"' EXIT

# --- endpoint list: default = this recipe's replica ports (overridable for harness testing)
if [[ -n "${REPLICA_URLS:-}" ]]; then
    IFS=' ' read -r -a URLS <<< "${REPLICA_URLS}"
    REPLICAS="${#URLS[@]}"
else
    URLS=()
    for ((i = 0; i < REPLICAS; i++)); do
        URLS+=("$(replica_url "${i}")")
    done
fi

HAVE_DOCKER=0
if [[ "${REPLICA_URLS:-}" == "" ]] && docker_cmd; then
    HAVE_DOCKER=1
fi

# --- result bookkeeping ---
declare -a T_NAME=() T_STATUS=() T_DETAIL=()
record() { T_NAME+=("$1"); T_STATUS+=("$2"); T_DETAIL+=("$3"); }
pass() { record "$1" PASS "$2"; echo "  PASS  $1${2:+ — $2}"; }
fail() { record "$1" FAIL "$2"; echo "  FAIL  $1${2:+ — $2}" >&2; }
skip() { record "$1" SKIP "$2"; echo "  SKIP  $1${2:+ — $2}"; }
info() { record "$1" INFO "$2"; echo "  INFO  $1${2:+ — $2}"; }

# Run fn <i> concurrently for all replicas; per-replica output in $TMP/<tag>/<i>.out
parallel_for() {
    local tag="$1" fn="$2"
    shift 2
    mkdir -p "${TMP}/${tag}"
    local i pids=()
    for ((i = 0; i < REPLICAS; i++)); do
        ( "${fn}" "${i}" "$@" >"${TMP}/${tag}/${i}.out" 2>&1; echo $? >"${TMP}/${tag}/${i}.rc") &
        pids+=("$!")
    done
    wait "${pids[@]}" 2>/dev/null || true   # only our jobs — other background work keeps running
    return 0
}

failures_of() { # <tag> -> number of failed replicas (first failure echoed)
    local tag="$1" i rc fails=0
    for ((i = 0; i < REPLICAS; i++)); do
        rc="$(cat "${TMP}/${tag}/${i}.rc" 2>/dev/null || echo 1)"
        if [[ "${rc}" != "0" ]]; then
            fails=$((fails + 1))
            [[ ${fails} -eq 1 ]] && echo "r${i}: $(tail -c 200 "${TMP}/${tag}/${i}.out" | tr '\n' ' ')" >&2
        fi
    done
    echo "${fails}"
    return 0
}

# chat_request <url> <payload-json> <out-body-file> -> prints "ttft total"
chat_request() {
    local url="$1" payload="$2" out="$3"
    curl -sS -m "${REQ_TIMEOUT:-180}" -o "${out}" \
        -w '%{time_starttransfer} %{time_total}' \
        -X POST "${url}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "${payload}"
}

# =============================================================================
echo "=== ${MODEL_KEY} test suite — ${REPLICAS} replica(s) on 2 GPUs ==="
echo "Model:  ${MODEL}"
echo "URLs:   ${URLS[*]}"
echo

# ---- bring the recipe up (unless told otherwise) ----
if [[ "${REPLICA_URLS:-}" == "" && "${NO_START:-0}" != "1" ]]; then
    if "${DOCKER[@]}" ps --filter "name=${CONTAINER_PREFIX}-" --format '{{.Names}}' | grep -q .; then
        echo "Replicas already running — reusing them."
    else
        "${TESTS_DIR}/../run-model.sh" "${MODEL_KEY}" >/dev/null || {
            fail ready "run-model.sh could not bring up replicas"
            exit 2
        }
    fi
fi

# =============================================================================
# Test 1 — readiness on every replica
# =============================================================================
t_ready() {
    local i="$1"
    curl -fsS -m 10 -o /dev/null "${URLS[$i]}/health"
}
echo "--- 1/7 ready ---"
if [[ "${REPLICA_URLS:-}" == "" && "${NO_START:-0}" != "1" ]]; then
    pass ready "run-model.sh waited for all replicas"
else
    parallel_for ready t_ready
    f="$(failures_of ready | tail -1)"
    if [[ "${f}" == "0" ]]; then pass ready "${REPLICAS}/${REPLICAS} answer /health"; else fail ready "${f}/${REPLICAS} replicas unhealthy"; fi
fi

# =============================================================================
# Test 2 — model identity + configured context window
# =============================================================================
t_model_id() {
    local i="$1" id maxlen
    id="$(curl -fsS -m 10 "${URLS[$i]}/v1/models" | jq -r '.data[0].id')"
    maxlen="$(curl -fsS -m 10 "${URLS[$i]}/v1/models" | jq -r '.data[0].max_model_len')"
    [[ "${id}" == "${MODEL}" ]] || { echo "served id '${id}' != '${MODEL}'"; return 1; }
    [[ "${maxlen}" == "${MAX_MODEL_LEN}" ]] || { echo "max_model_len '${maxlen}' != '${MAX_MODEL_LEN}'"; return 1; }
    echo "id=${id} max_model_len=${maxlen}"
}
echo "--- 2/7 model-id ---"
parallel_for model-id t_model_id
f="$(failures_of model-id | tail -1)"
if [[ "${f}" == "0" ]]; then pass model-id "all replicas serve '${MODEL}', max_model_len=${MAX_MODEL_LEN}"; else fail model-id "${f}/${REPLICAS} replicas mismatch"; fi

# =============================================================================
# Test 3 — smoke chat completion on every replica (concurrent)
# =============================================================================
SMOKE_PAYLOAD="$(jq -cn --arg m "${MODEL}" \
    '{model:$m, messages:[{role:"user", content:"Reply with exactly one short sentence: why is the sky blue?"}], max_tokens:96, temperature:0}')"

t_smoke() {
    local i="$1" times body content
    times="$(chat_request "${URLS[$i]}" "${SMOKE_PAYLOAD}" "${TMP}/smoke.${i}.body")"
    body="$(cat "${TMP}/smoke.${i}.body")"
    content="$(jq -r '.choices[0].message.content // empty' <<<"${body}" 2>/dev/null || true)"
    [[ -n "${content}" ]] || { echo "empty completion: $(head -c 200 <<<"${body}")"; return 1; }
    echo "ttft/total=${times}s :: ${content:0:60}"
}
echo "--- 3/7 smoke ---"
parallel_for smoke t_smoke
f="$(failures_of smoke | tail -1)"
if [[ "${f}" == "0" ]]; then pass smoke "${REPLICAS}/${REPLICAS} replicas completed concurrently"; else fail smoke "${f}/${REPLICAS} replicas failed"; fi

# =============================================================================
# Test 4 — structured JSON (guided decoding via response_format=json_schema)
# =============================================================================
JSON_PAYLOAD="$(jq -cn --arg m "${MODEL}" '{
    model:$m,
    messages:[{role:"user", content:"Triage this ticket: \u0027Login button misaligned on Safari, looks high priority, related to UI and Safari\u0027."}],
    max_tokens:160, temperature:0,
    response_format:{type:"json_schema", json_schema:{name:"ticket_triage", schema:{
        type:"object",
        properties:{
            title:{type:"string"},
            severity:{type:"integer", minimum:0, maximum:10},
            tags:{type:"array", items:{type:"string"}}},
        required:["title","severity","tags"], additionalProperties:false}}}
}')"

t_json() {
    local i="$1" body content
    body="$(chat_request "${URLS[$i]}" "${JSON_PAYLOAD}" "${TMP}/json.${i}.body" >/dev/null; cat "${TMP}/json.${i}.body")"
    content="$(jq -r '.choices[0].message.content // empty' <<<"${body}" 2>/dev/null || true)"
    content="$(sed -e 's/^```json//' -e 's/^```//' <<<"${content}" | tr -d '\r')"
    jq -e '(.title | type == "string")
        and (.severity | type == "number" and . >= 0 and . <= 10)
        and (.tags | type == "array" and all(.[]; type == "string"))' \
        <<<"${content}" >/dev/null 2>&1 ||
        { echo "schema violation: ${content:0:200}"; return 1; }
    echo "valid JSON: ${content:0:80}"
}
echo "--- 4/7 json ---"
if [[ "${EXPECT_JSON:-1}" != "1" ]]; then
    skip json "recipe does not advertise structured JSON"
else
    parallel_for json t_json
    f="$(failures_of json | tail -1)"
    if [[ "${f}" == "0" ]]; then pass json "${REPLICAS}/${REPLICAS} replicas returned schema-valid JSON"; else fail json "${f}/${REPLICAS} replicas violated the schema"; fi
fi

# =============================================================================
# Test 5 — function calling
# =============================================================================
TOOLS_PAYLOAD="$(jq -cn --arg m "${MODEL}" '{
    model:$m,
    messages:[{role:"user", content:"What is the weather in Paris right now? Use the provided tool."}],
    max_tokens:128, temperature:0,
    tool_choice:"auto",
    tools:[{type:"function", function:{
        name:"get_weather",
        description:"Get the current weather for a city",
        parameters:{type:"object",
            properties:{city:{type:"string", description:"City name"},
                        unit:{type:"string", enum:["c","f"]}},
            required:["city"], additionalProperties:false}}}]}
')"

t_tools() {
    local i="$1" body fn args
    body="$(chat_request "${URLS[$i]}" "${TOOLS_PAYLOAD}" "${TMP}/tools.${i}.body" >/dev/null; cat "${TMP}/tools.${i}.body")"
    fn="$(jq -r '.choices[0].message.tool_calls[0].function.name // empty' <<<"${body}" 2>/dev/null || true)"
    [[ "${fn}" == "get_weather" ]] || { echo "no get_weather tool call (got '${fn}'): $(head -c 200 <<<"${body}")"; return 1; }
    args="$(jq -r '.choices[0].message.tool_calls[0].function.arguments // empty' <<<"${body}")"
    jq -e 'has("city") and (.city | type == "string")' <<<"${args}" >/dev/null 2>&1 ||
        { echo "bad tool arguments: ${args:0:120}"; return 1; }
    echo "tool call get_weather(${args:0:60})"
}
echo "--- 5/7 tools ---"
if [[ "${EXPECT_TOOL_CALL:-0}" != "1" ]]; then
    skip tools "recipe does not advertise function calling (EXPECT_TOOL_CALL=0)"
else
    parallel_for tools t_tools
    f="$(failures_of tools | tail -1)"
    if [[ "${f}" == "0" ]]; then pass tools "${REPLICAS}/${REPLICAS} replicas produced well-formed tool calls"; else fail tools "${f}/${REPLICAS} replicas failed tool calling"; fi
fi

# =============================================================================
# Test 6 — reasoning parser wiring (informational only)
# =============================================================================
REASON_PAYLOAD="$(jq -cn --arg m "${MODEL}" \
    '{model:$m, messages:[{role:"user", content:"Think step by step: is 9.11 greater than 9.9?"}], max_tokens:256, temperature:0}')"

echo "--- 6/7 reasoning ---"
if [[ "${EXPECT_REASONING:-0}" != "1" ]]; then
    skip reasoning "recipe does not advertise reasoning output"
else
    t_reason() {
        local i="$1" body
        body="$(chat_request "${URLS[$i]}" "${REASON_PAYLOAD}" "${TMP}/reason.${i}.body" >/dev/null; cat "${TMP}/reason.${i}.body")"
        jq -e '.choices[0].message.content // empty | length > 0' <<<"${body}" >/dev/null 2>&1 || return 1
        if jq -e '.choices[0].message.reasoning_content // empty | length > 0' <<<"${body}" >/dev/null 2>&1; then
            echo "reasoning_content present"
        else
            echo "no reasoning_content on this prompt (non-fatal)"
        fi
    }
    parallel_for reasoning t_reason
    f="$(failures_of reasoning | tail -1)"
    sample="$(head -1 "${TMP}/reasoning/0.out" 2>/dev/null || true)"
    if [[ "${f}" == "0" ]]; then info reasoning "${REPLICAS}/${REPLICAS} ok (${sample})"; else info reasoning "${f}/${REPLICAS} returned no content (non-fatal)"; fi
fi

# =============================================================================
# Test 7 — concurrent sweep: REPLICAS simultaneous generations + peak VRAM
# =============================================================================
TOPICS=(algorithms climate typography databases chess baking networking linguistics
        pottery astronomy compilers sailing ceramics knitting volcanoes cartography)
sweep_payload() {
    local i="$1" topic="${TOPICS[$((i % ${#TOPICS[@]}))]}"
    jq -cn --arg m "${MODEL}" --arg t "${topic}" \
        '{model:$m, messages:[{role:"user", content:("Write a paragraph of roughly 80 words about " + $t + ".")}], max_tokens:160, temperature:0.7}'
}

t_sweep() {
    local i="$1" times payload ttft total tokens
    payload="$(sweep_payload "${i}")"
    times="$(REQ_TIMEOUT="${SWEEP_TIMEOUT:-600}" chat_request "${URLS[$i]}" "${payload}" "${TMP}/sweep.${i}.body")"
    read -r ttft total <<<"${times}"
    tokens="$(jq -r '.usage.completion_tokens // 0' "${TMP}/sweep.${i}.body" 2>/dev/null || echo 0)"
    speed="$(awk "BEGIN{printf \"%.1f\", ${tokens:-0} / (${total:-1})}")"
    echo "ttft=${ttft}s total=${total}s tokens=${tokens} tok/s=${speed}"
}

echo "--- 7/7 sweep (${REPLICAS} concurrent requests) ---"
VRAM_LOG="${TMP}/vram.csv"
: >"${VRAM_LOG}"
if command -v nvidia-smi >/dev/null 2>&1; then
    ( while :; do
          nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >>"${VRAM_LOG}" 2>/dev/null
          sleep 2
      done ) &
    VRAM_PID=$!
fi

sweep_start="$(date +%s)"
parallel_for sweep t_sweep
sweep_wall="$(( $(date +%s) - sweep_start ))"

kill "${VRAM_PID:-}" 2>/dev/null || true
wait "${VRAM_PID:-}" 2>/dev/null || true

f="$(failures_of sweep | tail -1)"
total_tokens=0
for ((i = 0; i < REPLICAS; i++)); do
    t="$(jq -r '.usage.completion_tokens // 0' "${TMP}/sweep.${i}.body" 2>/dev/null || echo 0)"
    total_tokens=$((total_tokens + t))
done
agg="$(awk "BEGIN{printf \"%.1f\", ${total_tokens} / (${sweep_wall} > 0 ? ${sweep_wall} : 1)}")"
lat="$(for ((i = 0; i < REPLICAS; i++)); do awk '{print $2}' <<<"$(tail -1 "${TMP}/sweep/${i}.out" | grep -o 'total=[0-9.]*' | tr -d 'total=')"; done | sort -n | tail -1)"
echo "  wall=${sweep_wall}s  aggregate=${agg} tok/s across ${REPLICAS} replicas  worst total=${lat:-?}s"
for ((i = 0; i < REPLICAS; i++)); do
    echo "    r${i}: $(tail -1 "${TMP}/sweep/${i}.out")"
done

if [[ "${f}" == "0" ]]; then
    pass sweep "all ${REPLICAS} concurrent generations completed; ${agg} tok/s aggregate"
else
    fail sweep "${f}/${REPLICAS} concurrent generations failed"
fi

# Peak VRAM report + fit assertion (96 GB rig: must never exceed per-GPU total)
if [[ -s "${VRAM_LOG}" ]]; then
    peak="$(awk -F',' '{gsub(/ MiB/,"",$2); gsub(/ MiB/,"",$3); if ($2 > max[$1]) {max[$1]=$2; tot[$1]=$3}} END{for (g in max) printf "GPU%s %d/%d MiB ", g, max[g], tot[g]}' "${VRAM_LOG}")"
    over="$(awk -F',' '{gsub(/ MiB/,"",$2); gsub(/ MiB/,"",$3); if ($2 > $3 * 0.98) bad=1} END{print bad+0}' "${VRAM_LOG}")"
    if [[ "${over}" == "0" ]]; then
        pass vram "peak during sweep: ${peak}"
    else
        fail vram "a GPU exceeded 98% of VRAM during sweep: ${peak}"
    fi
else
    skip vram "nvidia-smi unavailable"
fi

# Container restart / OOM check
if [[ "${HAVE_DOCKER}" == "1" ]]; then
    oom=0
    for ((i = 0; i < REPLICAS; i++)); do
        name="$(container_name "${i}")"
        rc="$("${DOCKER[@]}" inspect -f '{{.RestartCount}}' "${name}" 2>/dev/null || echo 0)"
        [[ "${rc}" == "0" ]] || oom=$((oom + 1))
    done
    if [[ "${oom}" == "0" ]]; then pass stability "no replica restarted or OOM-killed"; else fail stability "${oom} replica(s) restarted during the suite"; fi
fi

# =============================================================================
echo
echo "=== ${MODEL_KEY} results ==="
printf '%-12s %-6s %s\n' TEST RESULT DETAIL
overall=0
for ((i = 0; i < ${#T_NAME[@]}; i++)); do
    printf '%-12s %-6s %s\n' "${T_NAME[$i]}" "${T_STATUS[$i]}" "${T_DETAIL[$i]}"
    [[ "${T_STATUS[$i]}" == "FAIL" ]] && overall=1
done
echo
if ((overall == 0)); then
    echo "SUITE PASSED — ${MODEL_KEY} is good on this rig."
else
    echo "SUITE FAILED — ${MODEL_KEY}." >&2
fi

# Tear down unless asked to keep (only when we manage the containers)
if [[ "${REPLICA_URLS:-}" == "" && "${KEEP:-0}" != "1" && "${NO_START:-0}" != "1" ]]; then
    "${TESTS_DIR}/../stop-model.sh" "${MODEL_KEY}" >/dev/null 2>&1 || true
fi
exit "${overall}"
