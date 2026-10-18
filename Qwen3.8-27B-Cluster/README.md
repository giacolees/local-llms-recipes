# Qwen3.8-27B multi-node vLLM cluster (2 workstations × 2× A6000)

Distributed vLLM deployment of `Qwen/Qwen3.8-27B` (dense 27B BF16,
hybrid-attention VLM) across two workstations, each with 2× RTX A6000 48 GB,
connected by a 10 GbE link.

| Node | Host | GPUs | Role |
|---|---|---|---|
| `head` | 10.0.0.2 (this machine) | 2× RTX A6000 | Ray head, vLLM API server |
| `worker1` | 10.0.0.3 (root@, via SSH) | 2× RTX A6000 | Ray worker |

Parallelism: **TP=2 inside each node** (PCIe), **PP=2 across nodes** (10 GbE).
Current serving config: 262,144 ctx, MTP speculative decoding, CUDA graphs, BF16.

This is the multi-node counterpart to the single-node `Qwen3.6-27B` vLLM recipe.
Everything runs in Docker containers, one per node, joined into a Ray cluster.

## Hardware profile

The recipe is driven by the **`4xa6000`** hardware profile. All topology, model
and serve settings live in `profiles/4xa6000/`:

```
profiles/4xa6000/
├── config.env                  # 2 nodes × 2 GPUs; TP=2 / PP=2; 10 GbE, no IB
└── qwen3.8-27b-cluster.env     # cluster topology + serve options (was config.yaml)
```

Activate it before running any of the scripts:

```bash
source ../switch-profile.sh 4xa6000
```

`lib.sh` sources `config.env` first and then `qwen3.8-27b-cluster.env`, so the
profile files are the single source of truth. Environment variables exported
before launching still win for scalar knobs (e.g. `SERVE_PORT=9001 ./serve.sh`).
Missing files fall back to the defaults baked into `lib.sh`.

## Directory layout

```
Qwen3.8-27B-Cluster/
├── lib.sh                 # profile loader + node/container helpers (sourced)
├── provision_node.sh      # copy image/weights/wheels to a new node over the LAN
├── cluster_up.sh          # containers + Ray cluster on all nodes
├── serve.sh               # launch/restart the model
├── status.sh              # API + Ray + GPU status
└── stop.sh                # stop model, Ray and containers
```

vLLM writes its log inside the container to `/models/logs/vllm.log`, which is
`/opt/models/logs/vllm.log` on the host (the `MODELS_DIR` mount).

The host-side data layout expected on **every** node is the one in the profile:

```
/opt/models/
├── Qwen3.8-27B/           # HF BF16 checkpoint, 18 shards, 55.6 GB (both nodes)
├── Qwen3-0.6B/            # tiny model used for smoke tests
├── wheels/                # ray-2.58.0 wheel (installed into the containers)
└── logs/                  # vllm.log, etc.
```

## Quick start

```bash
cd Qwen3.8-27B-Cluster
source ../switch-profile.sh 4xa6000

./cluster_up.sh          # containers + Ray on all nodes (resets Ray!)
./serve.sh --wait        # launch the model and block until the API is up
./status.sh              # check API, Ray resources, GPU usage

# when done
./stop.sh                # or: KEEP_CONTAINERS=1 ./stop.sh
```

## Scripts

| Script | What it does | Notes |
|---|---|---|
| `provision_node.sh <name>` | Copies the docker image (`docker save \| ssh docker load`), model weights (`rsync`) and the Ray wheel to a node | Only needed for nodes other than the head; idempotent |
| `cluster_up.sh [--sync]` | Ensures containers exist/running, installs Ray into them if missing, **resets Ray**, starts head + workers, prints `ray status` | `--sync` runs `provision_node.sh` for every node first. **Kills a running engine** |
| `serve.sh [--wait]` | Kills any previous server, builds the `vllm serve` command from the profile, launches it detached, prints log path | `--wait` polls the API and dumps the log tail on failure |
| `status.sh` | API model list, `ray status`, per-node `nvidia-smi` | Read-only |
| `stop.sh` | Kills vLLM, `ray stop` on every node, removes containers | `KEEP_CONTAINERS=1` keeps containers |

Typical flows:

```bash
# after a reboot
source ../switch-profile.sh 4xa6000
./cluster_up.sh && ./serve.sh --wait

# restart the server after editing profiles/4xa6000/qwen3.8-27b-cluster.env
./serve.sh --wait

# add a third workstation (10.0.0.4): append it to every NODE_* array in the
# profile's qwen3.8-27b-cluster.env, then adjust TP/PP, then:
./provision_node.sh worker2
./cluster_up.sh --sync
./serve.sh --wait
```

## Profile reference (`qwen3.8-27b-cluster.env`)

| Group | Variables |
|---|---|
| Container | `IMAGE`, `CLUSTER_SHM_SIZE`, `MODELS_DIR`, `CONTAINER_MODELS_DIR`, `CONTAINER_PREFIX` |
| Ray | `RAY_PORT`, `RAY_VERSION`, `RAY_WHEELS` |
| Topology | `NODE_NAMES[]`, `NODE_SSH[]`, `NODE_IPS[]`, `NODE_IFACES[]`, `NODE_GPUS[]`, `NODE_COUNT` |
| Model | `MODEL_NAME`, `MODEL_REPO`, `MODEL_HOST_PATH`, `MODEL_PATH`, `MODEL_MAX_LEN` |
| Serve | `TP_SIZE`, `PP_SIZE`, `BACKEND`, `GPU_MEM_UTIL`, `REASONING_PARSER`, `TOOL_PARSER`, `AUTO_TOOL_CHOICE`, `SERVE_HOST`, `SERVE_PORT`, `SERVE_LOG_NAME`, `EXTRA_ARGS[]` |
| Speculative | `SPEC_METHOD`, `SPEC_TOKENS` |

`NODE_NAMES`/`NODE_SSH`/… are bash arrays; the first node is the Ray head.
All paths in the topology point at the fast 10 GbE network — keep
management/Internet traffic off it where possible.

## Using the API

Endpoint: `http://10.0.0.2:8000/v1` — model name `qwen3.8-27b`.

Thinking mode is **on by default**; disable per request with
`chat_template_kwargs: {"enable_thinking": false}`.

```bash
# text
curl http://10.0.0.2:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "What is 17*23?"}],
  "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}'

# vision (native VLM)
curl http://10.0.0.2:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
  ]}],
  "max_tokens": 200
}'
```

Any OpenAI-compatible client works (openai-python, LangChain, etc.).

## Measured performance

On the current config (BF16, TP2×PP2, MTP, CUDA graphs, 2×A6000/node):

| Workload | Throughput |
|---|---|
| Single stream | ~15 tok/s |
| 8 concurrent requests | ~77 tok/s aggregate |
| Eager mode (no CUDA graphs) | ~12 tok/s single, ~74 tok/s at 8 |

- Model weights: 13.1 GiB/GPU (PP stage 0) and 14.3 GiB/GPU (PP stage 1)
- KV cache: ~1.25M tokens (~4.8× concurrency at full 262K context)
- MTP mean acceptance length: ~2.1–2.3

## Troubleshooting / notes

- **`ray` is not in the vLLM image.** `cluster_up.sh` installs it from
  `/opt/models/wheels` into each container. If the wheel is missing on a node,
  run `./provision_node.sh <node>`.
- **A6000 (Ampere, sm_86) cannot use FP8-scaled-MM or NVFP4 checkpoints.**
  Do not switch to `Qwen/Qwen3.8-27B-FP8` — kernels are unsupported on this
  hardware. BF16 is the correct choice; it fits comfortably with TP2×PP2.
- **The worker's Internet routes through the head** (`default via 10.0.0.2`).
  Large downloads therefore share the head's uplink; always transfer images/
  weights over the 10 GbE link (`provision_node.sh` does) instead of pulling
  twice.
- **Container log paths:** the host dir `/opt/models` is mounted at `/models`
  inside containers. Container-side logs go to `/models/logs/...`, host-side to
  `/opt/models/logs/...`.
- **Ray duplicates:** if `ray status` shows more nodes than the profile, run
  `./cluster_up.sh` (it always resets both sides before starting).
- **Startup time** is ~2.5–4 min (weight load + Triton kernel warmup + CUDA
  graph capture). `serve.sh --wait` handles this.
- **Ports:** API `8000`, Ray GCS `6379`. All workers must reach each other on
  the 10 GbE subnet (currently `10.0.0.0/29`, 6 usable addresses, point-to-point
  or switched).
- **`10.0.0.0/29` has room for 6 hosts**, so up to 6 nodes can share this link
  before you need a different interconnect/subnet.
- Stale HuggingFace partial files may remain in
  `/opt/models/Qwen3.8-27B/.cache` — safe to delete.
