#!/usr/bin/env bash
# =============================================================================
# with-hf-token.sh — resolve HF_TOKEN (Infisical or ambient) and exec a command
#
# The Gemma recipes are gated on Hugging Face, so download-models.sh and
# run-model.sh need HF_TOKEN set. This wrapper resolves it and hands it to the
# command you pass, so the token never has to live in a shell profile:
#
#   ./with-hf-token.sh ./download-models.sh gemma4-e4b
#   ./with-hf-token.sh ./run-model.sh gemma4-e4b
#   HF_TOKEN="$(./with-hf-token.sh --print)"
#
# Resolution precedence (first hit wins):
#   1. an already-exported HF_TOKEN          (used as-is, never re-fetched)
#   2. the `infisical` CLI                    (infisical secrets get <name> ...)
#   3. the Infisical REST API (machine identity): Universal Auth or a service token
#
# Configuration (env vars):
#   HF_SECRET_NAME            secret name in Infisical      (default: HF_TOKEN)
#   INFISICAL_ENV             Infisical environment         (default: dev)
#   INFISICAL_SECRET_PATH     Infisical folder path         (default: /)
#   INFISICAL_PROJECT_ID      Infisical project id          (required for REST)
#   INFISICAL_SITE_URL        Infisical base URL            (default: https://app.infisical.com)
#   INFISICAL_UNIVERSAL_AUTH_CLIENT_ID / _SECRET   Universal Auth machine identity
#   INFISICAL_TOKEN           a service token (alternative to Universal Auth)
#
# This is the shell counterpart of
# experiments/self_hosted_models/secrets.py in the cognitive-radar repo; kept
# dependency-free (bash + curl + python3 for JSON) so the recipes stay standalone.
# =============================================================================
set -euo pipefail

SECRET_NAME="${HF_SECRET_NAME:-HF_TOKEN}"
INFISICAL_ENV="${INFISICAL_ENV:-dev}"
INFISICAL_SECRET_PATH="${INFISICAL_SECRET_PATH:-/}"
BASE_URL="${INFISICAL_SITE_URL:-https://app.infisical.com}"

json_get() {
    # json_get <json> <python-expr over `d`>; prints the value or nothing.
    local payload="$1" expr="$2"
    python3 -c "import json,sys
try:
    d = json.loads(sys.argv[1])
    v = ($expr)
    print('' if v is None else v)
except Exception:
    pass" "$payload" 2>/dev/null || true
}

from_cli() {
    command -v infisical >/dev/null 2>&1 || return 1
    # ``--plain --silent`` prints just the value (no box-drawing table), which the
    # parser below expects.
    local args=(secrets get "$SECRET_NAME" --env "$INFISICAL_ENV" --path "$INFISICAL_SECRET_PATH" --plain --silent)
    [[ -n "${INFISICAL_PROJECT_ID:-}" ]] && args+=(--projectId "$INFISICAL_PROJECT_ID")  # camelCase: CLI rejects --project-id
    local out
    out="$(infisical "${args[@]}" 2>/dev/null | head -1)" || return 1
    out="${out#*=}"            # strip a leading NAME=
    out="${out%\"}"; out="${out#\"}"   # trim quotes
    [[ -n "$out" ]] || return 1
    printf '%s' "$out"
}

from_rest() {
    command -v curl >/dev/null 2>&1 || return 1
    [[ -n "${INFISICAL_PROJECT_ID:-}" ]] || return 1
    local token="${INFISICAL_TOKEN:-}"
    # Universal Auth machine identity: require BOTH parts before building a body,
    # so `set -u` never aborts on an unset secret half.
    if [[ -z "$token" \
        && -n "${INFISICAL_UNIVERSAL_AUTH_CLIENT_ID:-}" \
        && -n "${INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET:-}" ]]; then
        local body login
        body="$(printf '{"clientId":"%s","clientSecret":"%s"}' \
            "$INFISICAL_UNIVERSAL_AUTH_CLIENT_ID" "$INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET")"
        # Body (with the client secret) travels over stdin via --data @-, never
        # argv, so it is not exposed in /proc/*/cmdline to same-user processes.
        login="$(printf '%s' "$body" | curl -fsS -X POST \
            "$BASE_URL/api/v1/auth/universal-auth/login" \
            -H 'Content-Type: application/json' --data @-)" || return 1
        token="$(json_get "$login" "d.get('accessToken')")"
    fi
    [[ -n "$token" ]] || return 1
    local reply
    # Bearer token travels over stdin via --config -, never argv.
    reply="$(printf 'header = "Authorization: Bearer %s"\n' "$token" \
        | curl -fsS --config - \
        "$BASE_URL/api/v3/secrets/raw/$SECRET_NAME?environment=$INFISICAL_ENV&secretPath=$INFISICAL_SECRET_PATH&projectId=$INFISICAL_PROJECT_ID")" || return 1
    json_get "$reply" "(d.get('secret') or {}).get('secretValue')"
}

resolve() {
    [[ -n "${HF_TOKEN:-}" ]] && { printf '%s' "$HF_TOKEN"; return 0; }
    local value
    value="$(from_cli)" && { printf '%s' "$value"; return 0; }
    value="$(from_rest)" && { printf '%s' "$value"; return 0; }
    return 1
}

if [[ "${1:-}" == "--print" ]]; then
    resolve || { echo "could not resolve $SECRET_NAME from Infisical or the environment" >&2; exit 1; }
    echo
    exit 0
fi

[[ $# -ge 1 ]] || { echo "usage: $0 [--print] <command> [args...]" >&2; exit 1; }

if ! HF_TOKEN="$(resolve)"; then
    echo "warning: could not resolve $SECRET_NAME; proceeding without it (ungated models only)" >&2
    exec "$@"
fi
export HF_TOKEN
exec "$@"
