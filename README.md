# Local LLM recipes

One folder per model under `recipes/`, with vLLM launch paths for Docker,
Singularity and Apptainer. Hardware profiles cover 4× A100 40 GB, 2× RTX A6000
48 GB, and two nodes with 2× A6000 each. The multi-node Qwen recipe uses Ray.
Original llama.cpp/GGUF recipes remain available alongside the vLLM settings.

```text
recipes/<model>/
  recipe.env                  # vLLM weights, sizing, parser and capability flags
  profiles/<hardware>.vllm.env # model-specific hardware overrides
  serve.sh                    # one foreground vLLM endpoint
  run.sh / status.sh / stop.sh # detached replicas, for imported vLLM recipes
recipes/Qwen3.8-27B-Cluster/
  profiles/4xa6000.env         # model + Ray topology + node configuration
  cluster_up.sh / serve.sh / provision_node.sh / status.sh / stop.sh
profiles/<hardware>/          # shared GPU layout and runtime/cache/network config
scripts/                      # shared launch, image conversion and test tools
benchmarks/                   # client runner, raw records and aggregate CSV
```

Model files under `recipe.env` came from the former `vllm-docker/models/`
catalog. The original GGUF recipes instead use `vllm.env` for the vLLM weight
identity and `<hardware>.env` for their original engine settings.

## Model catalog

The replica and TP columns show inherited **2×A6000 configuration defaults**,
not measured performance. Each model README links to its full configuration.

| Recipe | vLLM weights | Replicas | TP |
| --- | --- | ---: | ---: |
| [gemma4-e4b](recipes/gemma4-e4b/README.md) | google/gemma-4-E4B-it | 2 | 1 |
| [gemma4-26b-a4b](recipes/gemma4-26b-a4b/README.md) | google/gemma-4-26B-A4B-it | 1 | 2 |
| [phi-4-14b](recipes/phi-4-14b/README.md) | microsoft/Phi-4 | 1 | 1 |
| [qwen3.5-0.8b](recipes/qwen3.5-0.8b/README.md) | Qwen/Qwen3.5-0.8B | 8 | 1 |
| [qwen3.5-2b](recipes/qwen3.5-2b/README.md) | Qwen/Qwen3.5-2B | 4 | 1 |
| [qwen3.5-4b](recipes/qwen3.5-4b/README.md) | Qwen/Qwen3.5-4B | 4 | 1 |
| [qwen3.5-9b](recipes/qwen3.5-9b/README.md) | Qwen/Qwen3.5-9B | 2 | 1 |
| [qwen3.5-35b-a3b](recipes/qwen3.5-35b-a3b/README.md) | Qwen/Qwen3.5-35B-A3B-FP8 | 1 | 2 |
| [Qwen3.6-27B](recipes/Qwen3.6-27B/README.md) | Qwen/Qwen3.6-27B | 1 | 2 |
| [qwen3.6-35b-a3b](recipes/qwen3.6-35b-a3b/README.md) | Qwen/Qwen3.6-35B-A3B | 1 | 2 |
| [Qwen3.8-27B-Cluster](recipes/Qwen3.8-27B-Cluster/README.md) | Qwen/Qwen3.8-27B | 1 across two nodes | 2 × PP=2 |
| [DS4-flash](recipes/DS4-flash/README.md) | Set compatible VLLM_MODEL; original GGUF retained | — | profile |
| [Laguna-S2.1](recipes/Laguna-S2.1/README.md) | Set compatible VLLM_MODEL; original GGUF retained | — | profile |
| [Qwen3.8-Flash-Next-GGUF](recipes/Qwen3.8-Flash-Next-GGUF/README.md) | Set compatible VLLM_MODEL; original GGUF retained | — | profile |

## Single-node serving

Requires Linux and NVIDIA drivers, plus Docker with NVIDIA Container Toolkit
or Singularity/Apptainer with GPU support. Choose a tested image tag/digest that
supports the model architecture, quantization and parser flags. Gemma requires
accepted model terms and an authorized `HF_TOKEN`.

```bash
source ./switch-profile.sh 2xa6000
export VLLM_IMAGE=vllm/vllm-openai:<tested-version>
export LLM_RUNTIME=docker
./recipes/qwen3.5-2b/serve.sh   # replica 0, foreground, port 8900

# Alternatively start all four configured replicas on ports 8900–8903.
./recipes/qwen3.5-2b/run.sh
./recipes/qwen3.5-2b/status.sh
./recipes/qwen3.5-2b/stop.sh
```

Replace `<tested-version>` with a real version or digest. Run the foreground
and detached examples separately. Imported recipes default to `2xa6000` when
no profile is selected. Select `4xa100` for its model-specific starting settings.
Single-node recipes on `4xa6000` use only the two local GPUs.

For Singularity or Apptainer, import the same OCI image once and switch runtime:

```bash
./scripts/build-vllm-sif.sh
LLM_RUNTIME=singularity ./recipes/qwen3.5-2b/serve.sh
LLM_RUNTIME=singularity ./recipes/qwen3.5-2b/run.sh
LLM_RUNTIME=singularity ./recipes/qwen3.5-2b/stop.sh
```

The SIF defaults to `images/vllm-openai.sif`; set `VLLM_SIF` for another path.
`run.sh` starts an instance per replica on Singularity/Apptainer. Use the same
runtime and user to inspect and stop them. `HF_HOME` defaults to
`~/.cache/huggingface`, and token values are forwarded without printing them.

`MODEL` (or `VLLM_MODEL` for foreground serving), `MODEL_REVISION`, `REPLICAS`,
`TP`, `MAX_MODEL_LEN` and `GPU_MEM_UTIL` override imported recipe defaults.
`REPLICA_INDEX` chooses a foreground endpoint; `serve.sh` forwards extra vLLM
flags. `run.sh [replicas]` changes count before memory-budget validation.
TP=1 replicas rotate over the visible GPUs; TP>1 uses the visible GPU set.
Detached replicas bind to `BIND_HOST` (default 127.0.0.1). Foreground servers
listen on `HOST` (default 0.0.0.0), with `PORT` overriding the recipe port.
Remote code requires explicit `TRUST_REMOTE_CODE=1` or a vLLM flag.

```bash
DRY_RUN=1 ./recipes/qwen3.5-2b/run.sh
REPLICAS=1 GPU_MEM_UTIL=0.50 ./recipes/qwen3.5-2b/run.sh
./scripts/vllm-docker/download-models.sh qwen3.5-2b
./scripts/vllm-docker/tests/test-model.sh qwen3.5-2b
```

See [shared runtime documentation](scripts/vllm-docker/README.md) for precedence,
capability gates, download helpers and the retained Infisical token helper.

## Qwen cluster

The [cluster recipe](recipes/Qwen3.8-27B-Cluster/README.md) owns its model and
node settings in `profiles/4xa6000.env`. Edit the SSH targets, IPs and network
interfaces there before running on your machines. Prepare the weights and
matching Ray wheelhouse on the head. Run from that head machine:

```bash
source ./switch-profile.sh 4xa6000
export LLM_RUNTIME=docker
./recipes/Qwen3.8-27B-Cluster/cluster_up.sh --sync
./recipes/Qwen3.8-27B-Cluster/serve.sh --wait
./recipes/Qwen3.8-27B-Cluster/status.sh
./recipes/Qwen3.8-27B-Cluster/stop.sh
```

For Singularity/Apptainer, convert the cluster's configured `IMAGE` using
`scripts/build-vllm-sif.sh`, set `VLLM_SIF` to the same absolute path on each
node, and use `LLM_RUNTIME=singularity` or `apptainer` throughout. Provisioning
copies the SIF, weights and wheelhouse. Ray runs inside one instance per node;
the API remains on the head. `DRY_RUN=1` prints local and remote commands without
SSH or container operations. Bringing up the cluster resets Ray on its nodes.

## Benchmarks

The [benchmark runner](benchmarks/README.md) resolves the weights, served model
name and endpoint from the recipe. The cluster also supplies TP/PP and topology.

```bash
source ./switch-profile.sh 2xa6000
export LLM_RUNTIME=docker
export MODEL_REVISION=<actual-server-weights-commit>
# Keep VLLM_IMAGE identical to the running server's image/digest.
CONCURRENCY=1 NUM_PROMPTS=100 ./benchmarks/run.sh qwen3.5-2b
CONCURRENCY=8 NUM_PROMPTS=100 ./benchmarks/run.sh qwen3.5-2b

LLM_HW_PROFILE=4xa6000 ./benchmarks/run.sh Qwen3.8-27B-Cluster
```

Raw JSON, client logs and metadata go to
`benchmarks/results/<profile>/<recipe>/<runtime>/<UTC-run>/`.
`benchmarks/summary.csv` reports output TPS, total token throughput, requests/s,
TTFT, TPOT, ITL and latency. No GPU measurements have been collected in this
refactor. Benchmark one replica at a time unless a documented load balancer
provides an aggregate endpoint; these numbers do not imply fleet throughput.

## Migration and validation

All model definitions now live under `recipes/`. Qwen3.6's Docker settings were
merged into `recipes/Qwen3.6-27B/`; the old key `qwen3.6-27b` is an alias.
Model-specific `4xa6000` overrides and the cluster topology moved beside their
models. Shared hardware and runtime settings remain under `profiles/`.

Old `vllm-docker/*.sh`, its test entry points, and
`Qwen3.8-27B-Cluster/*.sh` forward to the new locations. Stop existing replicas
before switching to the migrated launchers, since canonical model keys also
identify managed containers. Original `run-*.sh` llama.cpp launchers and SIF
build helpers are retained; set `GGUF_MODELS_DIR` for local weights.

```bash
python3 -m unittest discover -s tests -v
```

Local checks cover all single-node model/profile/runtime launch combinations,
replica placement, overrides, budget rejection, compatibility paths, cluster
command generation and benchmark resolution. They do not prove GPU inference,
quantization compatibility or multi-node networking. Validate those on each GPU
profile before calling the recipes known-good.

## Suggested improvements

* Pin tested image digests and weights revisions after GPU smoke tests.
* Add GPU CI for health, one completion and capability gates on each runtime.
* Tune context, concurrency and KV cache from measured memory capacity.
* Record topology, GPU memory, power and utilization alongside benchmarks.
* Add Slurm templates and optional load balancing for detached replicas.
* Pin llama.cpp fork commits and CUDA base image digests.

References: [vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker/),
[vLLM benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/), and
[Singularity instances](https://docs.sylabs.io/guides/latest/user-guide/running_services.html).
