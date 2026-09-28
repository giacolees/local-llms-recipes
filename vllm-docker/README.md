# vLLM on Docker — model recipes for 2× RTX A6000 (96 GB)

OpenAI-compatible vLLM servers, one Docker container per replica, pinned
one-GPU-per-replica (TP=1) so replicas scale linearly and never pay PCIe
all-reduce costs. Recipes are sized for the `2xa6000` profile (96 GB total);
run them on the 2xa6000 machine.

## Recipes

| Recipe (`models/`) | Weights (INT4) | Replicas | /GPU | `GPU_MEM_UTIL` | Per replica | Ports | Role |
|---|---|---|---|---|---|---|---|
| `gemma4-e4b` | 4B dense, ~2.5 GB | **12** (10–12) | 6 | 0.15 | 7.2 GiB | 8100–8111 | Maximum parallelism; native structured JSON + function calling |
| `qwen3.5-9b-awq` | 9B dense, ~5.5 GB | **8** | 4 | 0.22 | 10.6 GiB | 8200–8207 | Cheap workhorse for single-turn sweeps |
| `phi-4-14b` | 14B dense, ~8.5 GB | **6** | 3 | 0.30 | 14.4 GiB | 8300–8305 | Middle ground if the 4–9B tier fails the gate (JSON only, no tools) |
| `gemma4-26b-a4b` | 26B/3.8B MoE, ~16 GB | **4** | 2 | 0.45 | 21.6 GiB | 8400–8403 | Structured JSON + function calling, 256K ctx (see profile below) |
| `qwen3.6-35b-a3b` | 35B/3B MoE, ~21 GB | **2** | 1 | 0.60 | 28.8 GiB | 8500–8501 | Best capability-per-token on this rig |
| `qwen3.6-27b` | 27B dense, ~16 GB | 4 | 2 | 0.45 | 21.6 GiB | 8600–8603 | **Don't** — A/B control only (see below) |
| `qwen3.5-35b-a3b` | 35B/3B MoE, ~21 GB | **2** | 1 | 0.60 | 28.8 GiB | 8700–8701 | Previous-gen 35B/3B MoE — A/B control vs 3.6 |

**Why not Qwen3.6-27B:** a dense 27B reads ~27B weights per generated token,
≈ 9× the per-token weight traffic of the 3B-active Qwen3.6-35B-A3B MoE, at
lower capability. It is kept as the dense-vs-MoE control for gate tests.

### Memory math (per 48 GiB A6000)

```
replicas_per_gpu × GPU_MEM_UTIL ≤ 0.95        # enforced by run-model.sh
per replica  = GPU_MEM_UTIL × 48 GiB
             = INT4 weights + runtime/CUDA ctx + KV cache (fp8 @ MAX_MODEL_LEN)
per GPU      = Σ replicas + ≤ 5 GiB headroom for CUDA contexts (outside gmu)
```

All recipes use `--kv-cache-dtype fp8` so the KV cache fits alongside INT4
weights. Everything is overridable — e.g. more KV headroom on the E4B tier:

```bash
REPLICAS=10 GPU_MEM_UTIL=0.17 ./run-model.sh gemma4-e4b
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

./run-model.sh gemma4-e4b                     # start 12 replicas, wait for /health
./run-model.sh qwen3.6-35b-a3b                # 2 replicas
REPLICAS=1 ./run-model.sh phi-4-14b           # quick single-instance smoke
DRY_RUN=1 ./run-model.sh gemma4-26b-a4b       # print the docker commands only

./status.sh [model-key]        # replica states + GPU memory
docker logs -f vllm-gemma4-e4b-0   # one replica's vLLM log
./stop-model.sh <model-key>    # or: ./stop-model.sh --all
```

Calling a replica (each is a full OpenAI API server):

```bash
curl http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-e4b-it-awq","messages":[{"role":"user","content":"hi"}]}'
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
Model IDs in `models/*.env` are the INT4/AWQ checkpoints this rig is sized
for — override `MODEL=` at launch if you pull a different quant repo.

## Harness self-test (no GPUs needed)

`tests/mock-server.py` is a stub OpenAI server used to validate the test
harness itself (CI or when editing `test-model.sh`):

```bash
MOCK_MODEL=google/gemma-4-e4b-it-awq MOCK_MAX_LEN=32768 \
  python3 tests/mock-server.py &
REPLICA_URLS=http://127.0.0.1:9999 NO_START=1 ./tests/test-model.sh gemma4-e4b
```
