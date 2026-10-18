#!/usr/bin/env bash
LLM_RUNTIME="${LLM_RUNTIME:-docker}"
case "$LLM_RUNTIME" in docker|singularity|apptainer) ;; *) echo 'Unsupported LLM_RUNTIME' >&2; exit 2;; esac
VLLM_SIF="${VLLM_SIF:-$MODELS_DIR/vllm-openai.sif}"

# All arrays must describe the same set of nodes.
for values in NODE_SSH NODE_IPS NODE_IFACES NODE_GPUS; do
    eval 'count=${#'"$values"'[@]}'
    [[ "$count" == "$NODE_COUNT" ]] || { echo "Invalid topology array: $values" >&2; exit 2; }
done
[[ "${#NODE_NAMES[@]}" == "$NODE_COUNT" ]] || { echo 'Invalid NODE_COUNT' >&2; exit 2; }

container_exec() {
    local i="$1"; shift
    local name; name="$(container_of "$i")"
    if [[ "$LLM_RUNTIME" == docker ]]; then
        run_on "$i" docker exec "$name" "$@"
    else
        run_on "$i" "$LLM_RUNTIME" exec "instance://$name" "$@"
    fi
}

container_launch() {
    # Launch command as a background child that stays in the instance/container.
    local i="$1" command="$2"
    if [[ "$LLM_RUNTIME" == docker ]]; then
        run_on "$i" docker exec -d "$(container_of "$i")" bash -c "$command"
    else
        container_exec "$i" bash -c "nohup bash -c $(printf '%q' "$command") < /dev/null > /dev/null 2>&1 &"
    fi
}

ensure_container() {
    local i="$1" c ip iface
    c="$(container_of "$i")"; ip="${NODE_IPS[$i]}"; iface="${NODE_IFACES[$i]}"
    run_on "$i" mkdir -p "$MODELS_DIR/logs"
    if [[ "$LLM_RUNTIME" == docker ]]; then
        if [[ "${DRY_RUN:-0}" == 1 ]] || ! run_on "$i" docker inspect "$c" >/dev/null 2>&1; then
            run_on "$i" docker run -d --name "$c" --network host --ipc host \
                --shm-size "$CLUSTER_SHM_SIZE" --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" \
                -v "$MODELS_DIR:$CONTAINER_MODELS_DIR" \
                -e "VLLM_HOST_IP=$ip" -e "NCCL_SOCKET_IFNAME=$iface" -e "GLOO_SOCKET_IFNAME=$iface" \
                -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" --entrypoint bash "$IMAGE" -c 'sleep infinity'
        else
            run_on "$i" docker start "$c" >/dev/null
        fi
    else
        if [[ "${DRY_RUN:-0}" == 1 ]] || ! container_exec "$i" true >/dev/null 2>&1; then
            run_on "$i" env "APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$LLM_RUNTIME" instance start --cleanenv --nv --writable-tmpfs \
                --bind "$MODELS_DIR:$CONTAINER_MODELS_DIR" \
                --env "VLLM_HOST_IP=$ip" \
                --env "NCCL_SOCKET_IFNAME=$iface" --env "GLOO_SOCKET_IFNAME=$iface" \
                --env "NCCL_IB_DISABLE=$NCCL_IB_DISABLE" "$VLLM_SIF" "$c"
        fi
    fi
    # SIFs are immutable: install Ray into the shared host-mounted dependency
    # directory, rather than requiring root to write site-packages.
    local deps="$CONTAINER_MODELS_DIR/ray-runtime-$LLM_RUNTIME"
    if [[ "$LLM_RUNTIME" != docker ]]; then
        container_exec "$i" bash -c "test -d $(printf '%q' "$deps/ray") || python3 -m pip install --no-index --find-links=$(printf '%q' "$WHEELS_CONTAINER") --target=$(printf '%q' "$deps") ray==$(printf '%q' "$RAY_VERSION")"
    elif [[ "${DRY_RUN:-0}" == 1 ]] || ! container_exec "$i" which ray >/dev/null 2>&1; then
        container_exec "$i" python3 -m pip install --no-index --find-links="$WHEELS_CONTAINER" "ray==$RAY_VERSION"
    fi
}

ray_exec() {
    local i="$1"; shift
    if [[ "$LLM_RUNTIME" == docker ]]; then container_exec "$i" ray "$@"
    else
        container_exec "$i" env "PYTHONPATH=$CONTAINER_MODELS_DIR/ray-runtime-$LLM_RUNTIME" \
            python3 -c 'from ray.scripts.scripts import main; main()' "$@"
    fi
}
