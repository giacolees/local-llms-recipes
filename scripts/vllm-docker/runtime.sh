#!/usr/bin/env bash
# Shared Docker / Singularity instance lifecycle for replica recipes.
resolve_runtime() {
    case "${LLM_RUNTIME:-docker}" in
        docker)
            if [[ "${DRY_RUN:-0}" == 1 ]]; then DOCKER=(docker)
            else docker_cmd || vllm_die 'cannot access Docker daemon'; fi ;;
        singularity|apptainer)
            INSTANCE_RUNTIME="$LLM_RUNTIME"
            if [[ "${DRY_RUN:-0}" != 1 ]]; then
                command -v "$INSTANCE_RUNTIME" >/dev/null || vllm_die "install $INSTANCE_RUNTIME"
                [[ -r "$VLLM_SIF" ]] || vllm_die "missing SIF: $VLLM_SIF; run scripts/build-vllm-sif.sh"
            fi ;;
        *) vllm_die "unsupported runtime: $LLM_RUNTIME" ;;
    esac
}

run_command() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then printf 'DRY-RUN:'; printf ' %q' "$@"; printf '\n'
    else "$@"; fi
}

start_replica() {
    local i="$1" gpu port name launch arg
    gpu="$(replica_gpu "$i")"; port="$(replica_port "$i")"; name="$(container_name "$i")"
    build_vllm_args "$i"
    local binds=()
    if [[ "$MODEL" == /* ]]; then binds=("$(dirname "$MODEL")"); fi
    if [[ "$LLM_RUNTIME" == docker ]]; then
        local cmd=("${DOCKER[@]}" run -d --name "$name"
            --label "vllm.recipe.key=$MODEL_KEY" --label "vllm.recipe.replica=$i"
            --gpus "\"device=$gpu\"" --ipc=host -p "$BIND_HOST:$port:$port"
            -v "$HF_HOME:/hf" -e HF_HOME=/hf -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}")
        [[ -z "${HF_TOKEN:-}" ]] || cmd+=(-e HF_TOKEN)
        [[ -z "${VLLM_LOGGING_LEVEL:-}" ]] || cmd+=(-e VLLM_LOGGING_LEVEL)
        for arg in ${binds[@]+"${binds[@]}"}; do cmd+=(-v "$arg:$arg:ro"); done
        run_command "${cmd[@]}" --entrypoint python3 "$VLLM_IMAGE" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
    else
        local cmd=(env "APPTAINERENV_CUDA_VISIBLE_DEVICES=$gpu" "SINGULARITYENV_CUDA_VISIBLE_DEVICES=$gpu" "$INSTANCE_RUNTIME" instance start --cleanenv --nv
            --bind "$HF_HOME:/hf" --bind "$LOG_DIR:/recipe-logs"
            --env HF_HOME=/hf)
        # A newline-delimited env file is unnecessary: inherit tokens by name,
        # so dry-run output never includes credentials.
        for prefix in APPTAINERENV SINGULARITYENV; do
            [[ -z "${HF_TOKEN:-}" ]] || export "${prefix}_HF_TOKEN=$HF_TOKEN"
        done
        for arg in ${binds[@]+"${binds[@]}"}; do cmd+=(--bind "$arg:$arg:ro"); done
        run_command "${cmd[@]}" "$VLLM_SIF" "$name"
        # Singularity uses host networking, so bind directly to the requested host.
        VLLM_ARGS+=(--host "$BIND_HOST")
        launch='nohup python3 -m vllm.entrypoints.openai.api_server'
        for arg in "${VLLM_ARGS[@]}"; do launch+=" $(printf '%q' "$arg")"; done
        launch+=" > /recipe-logs/$MODEL_KEY-$i.log 2>&1 < /dev/null &"
        run_command "$INSTANCE_RUNTIME" exec "instance://$name" bash -c "$launch"
    fi
}

stop_replicas() {
    if [[ "$LLM_RUNTIME" == docker ]]; then
        local ids=() id
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
            run_command "${DOCKER[@]}" ps -aq --filter "label=vllm.recipe.key=$MODEL_KEY"
            return
        fi
        while IFS= read -r id; do [[ -z "$id" ]] || ids+=("$id"); done < <("${DOCKER[@]}" ps -aq --filter "label=vllm.recipe.key=$MODEL_KEY")
        ((${#ids[@]} == 0)) || run_command "${DOCKER[@]}" rm -f "${ids[@]}"
    else
        run_command "$INSTANCE_RUNTIME" instance stop "${CONTAINER_PREFIX}-*"
    fi
}
