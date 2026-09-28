# vLLM on Docker — model recipes for 2× RTX A6000 (96 GB)

OpenAI-compatible vLLM servers, one Docker container per replica; models that
fit one card are pinned one-GPU-per-replica (TP=1) so replicas scale linearly
and never pay PCIe all-reduce costs, and models larger than one card use a
single TP=2 replica. Recipes are sized for the `2xa6000` profile (96 GB total);
run them on the 2xa6000 machine.

## Recipes

| Recipe (`models/`) | Weights | Replicas | /GPU | `GPU_MEM_UTIL` | Per replica | Ports | Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma4-e4b` | 4B dense, BF16 ~16 GB | **2** (1/GPU) | 1 | 0.45 | 21.6 GiB | 8100-8101 | Small agentic workhorse; native structured JSON + function calling |
| `qwen3.5-4b` | 4B dense, BF16 ~9.3 GB | **4** (2/GPU) | 2 | 0.30 | 14.4 GiB | 8800-8803 | Small tier; latency/quality read |
| `qwen3.5-2b` | 2B dense, BF16 ~4.5 GB | **4** (2/GPU) | 2 | 0.25 | 12.0 GiB | 8900-8903 | Smaller tier; latency-floor probe |
| `qwen3.5-0.8b` | 0.8B dense, BF16 ~1.7 GB | **8** (4/GPU) | 4 | 0.15 | 7.2 GiB | 9000-9007 | Smallest; absolute latency floor |
| `qwen3.5-9b` | 9B dense, BF16 ~19.3 GB | **2** (1/GPU) | 1 | 0.50 | 24.0 GiB | 8200-8201 | Cheap workhorse for single-turn sweeps |
| `phi-4-14b` | 14B dense, BF16 ~28 GB | **1** | 1 | 0.70 | 33.6 GiB | 8300 | Middle ground if the 4-9B tier fails the gate (JSON only, no tools) |
| `gemma4-26b-a4b` | 26B/3.8B MoE, BF16 ~51.6 GB | **1** (TP=2) | 2 | 0.70 | 33.6 GiB/GPU | 8400 | Structured JSON + function calling, 256K ctx (see profile below) |
| `qwen3.6-35b-a3b` | 35B/3B MoE, BF16 ~71.9 GB | **1** (TP=2) | 2 | 0.80 | 38.4 GiB/GPU | 8500 | Best capability-per-token on this rig |
| `qwen3.6-27b` | 27B dense, BF16 ~55.6 GB | **1** (TP=2) | 2 | 0.70 | 33.6 GiB/GPU | 8600 | **Don't** - A/B control only (see below) |
| `qwen3.5-35b-a3b` | 35B/3B MoE (VL), **FP8** ~37.5 GB | **1** (TP=2) | 2 | 0.90 | 43.2 GiB/GPU | 8700 | Previous-gen 35B/3B reasoning MoE; no official INT4, so one TP=2 replica |

**Why not Qwen3.6-27B:** a dense 27B reads ~27B weights per generated token,
≈ 9× the per-token weight traffic of the 3B-active Qwen3.6-35B-A3B MoE, at
lower capability. It is kept as the dense-vs-MoE control for gate tests.

**Why `qwen3.5-35b-a3b` is one TP=2 replica:** there is no official AWQ/INT4
build for it (the base `Qwen/Qwen3.5-35B-A3B` is 71.9 GB BF16 and does not fit),
so the recipe uses the official **FP8** build (~37.5 GB). Those weights do not fit
one 48 GiB card alongside KV, hence a single replica split across both GPUs
(`TP=2`). It is also a reasoning model: left on, it spends the whole token budget
in `reasoning_content` and returns `content: null` (measured: 600/600 tokens), so
the recipe disables thinking by default via `--default-chat-template-kwargs`.
Per-request opt-in: `"chat_template_kwargs": {"enable_thinking": true}`.

### Memory math (per 48 GiB A6000)

```text
replicas_per_gpu × GPU_MEM_UTIL ≤ 0.95        # enforced by run-model.sh
per replica  = GPU_MEM_UTIL × 48 GiB
             = BF16 weights + runtime/CUDA ctx + KV cache (fp8 @ MAX_MODEL_LEN)
per GPU      = Σ replicas + ≤ 5 GiB headroom for CUDA contexts (outside gmu)
```

All recipes use `--kv-cache-dtype fp8` so the KV cache fits alongside the BF16
weights. Everything is overridable — e.g. a single-endpoint benchmark launch on
the E4B tier:

```bash
REPLICAS=1 GPU_MEM_UTIL=0.60 ./run-model.sh gemma4-e4b
```

Gemma 4 26B-A4B's headline 256K context as a dedicated one-replica profile:

```bash
REPLICAS=1 GPU_MEM_UTIL=0.90 MAX_MODEL_LEN=262144 ./run-model.sh gemma4-26b-a4b
```

## One-time host setup (Docker + NVIDIA)

`docker run --gpus` needs the NVIDIA container toolkit; the recipes need your
user in the `docker` group:

```bash
sudo ./setup-docker.sh     # installs nvidia-container-toolkit, enables docker, groups you
newgrp docker              # or log out and back in
```

`setup-docker.sh` ends with a GPU-in-Docker self-test.

## Usage

```bash
cd vllm-docker

export HF_TOKEN=hf_...        # Gemma models are gated
./download-models.sh          # optional: prefetch weights before serving

./run-model.sh gemma4-e4b                     # start 2 replicas, wait for /health
./run-model.sh qwen3.6-35b-a3b                # 1 TP=2 replica
REPLICAS=1 ./run-model.sh phi-4-14b           # single-instance smoke
DRY_RUN=1 ./run-model.sh gemma4-26b-a4b       # print the docker commands only

./status.sh [model-key]        # replica states + GPU memory
docker logs -f vllm-gemma4-e4b-0   # one replica's vLLM log
./stop-model.sh <model-key>    # or: ./stop-model.sh --all
```

Calling a replica (each is a full OpenAI API server):

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E4B-it","messages":[{"role":"user","content":"hi"}]}'
```

### Config precedence

1. Command-line env (`REPLICAS=1 MAX_MODEL_LEN=262144 ./run-model.sh ...`)
2. Per-model overrides in `profiles/2xa6000/vllm-docker.env` (`GEMMA4_E4B_REPLICAS=10`)
3. Recipe defaults in `models/<key>.env`

## Tests (per model, on this hardware)

```bash
./tests/test-all.sh                          # all 7 recipes, sequentially
./tests/test-model.sh qwen3.6-35b-a3b        # one recipe
KEEP=1 ./tests/test-model.sh gemma4-e4b      # leave replicas up afterwards
```

Each suite starts the recipe at its full replica count and runs **against every
replica**:

| # | Test | Asserts |
|---|------|---------|
| 1 | `ready` | every replica answers `/health` at the full configured replica count |
| 2 | `model-id` | `/v1/models` serves the configured model and `max_model_len` |
| 3 | `smoke` | one chat completion per replica, all concurrent |
| 4 | `json` | guided JSON (`response_format=json_schema`) validates against the schema (skip for recipes with `EXPECT_JSON=0`) |
| 5 | `tools` | function calling returns a well-formed `get_weather` tool call (skip for `EXPECT_TOOL_CALL=0`, i.e. phi-4-14b) |
| 6 | `reasoning` | reasoning-parser wiring — informational, never fails the suite |
| 7 | `sweep` | `REPLICAS` simultaneous 80-word generations: wall time, aggregate tok/s, per-replica latency, peak VRAM per GPU (≤ 98 %), zero container restarts/OOMs |

`tests/test-all.sh` stops competing recipes between models so each replica
budget is measured against the full 96 GB, writes per-model logs plus a summary
under `vllm-docker/logs/`, and exits non-zero if any suite fails.

## Files

```
vllm-docker/
├── models/<key>.env        # per-model recipe: model id, replica count, VRAM budget,
│                           # context, parsers, capability flags for the tests
├── lib.sh                  # config resolution, replica/port/GPU mapping, docker helpers
├── run-model.sh            # start N replicas (one container per replica)
├── stop-model.sh           # stop one recipe or --all
├── status.sh               # replica + GPU state
├── download-models.sh      # prefetch weights via the vLLM image
├── setup-docker.sh         # one-time host setup (sudo)
└── tests/
    ├── test-model.sh       # the per-model suite described above
    └── test-all.sh         # every recipe, sequentially + summary
```

Recipes assume `vllm/vllm-openai:latest`; pin a version with
`VLLM_IMAGE=vllm/vllm-openai:<tag>` in `profiles/2xa6000/vllm-docker.env`.
Model IDs in `models/*.env` are the upstream (BF16) or FP8 checkpoints this rig is
sized for — override `MODEL=` at launch if you pull a different quant repo.

## Harness self-test (no GPUs needed)

`tests/mock-server.py` is a stub OpenAI server used to validate the test
harness itself (CI or when editing `test-model.sh`):

```bash
MOCK_MODEL=google/gemma-4-E4B-it MOCK_MAX_LEN=32768 \
  python3 tests/mock-server.py &
REPLICA_URLS=http://127.0.0.1:9999 NO_START=1 ./tests/test-model.sh gemma4-e4b
```
