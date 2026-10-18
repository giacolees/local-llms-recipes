# qwen3.5-0.8b

vLLM recipe for `Qwen/Qwen3.5-0.8B`. Weights, parser flags, capability gates and serving
defaults live in `recipe.env`; hardware overrides are in `profiles/`.

The inherited `2xa6000` defaults are 8 replica(s), TP=1,
GPU memory utilization 0.15, context 32768, and ports starting at 9000.
These are configuration values; measure performance and memory fit on your host.
`4xa100` has model-local starting overrides; `4xa6000` runs on the local two GPUs.

```bash
# From the repository root:
source ./switch-profile.sh 2xa6000
export VLLM_IMAGE=vllm/vllm-openai:<tested-version>
./recipes/qwen3.5-0.8b/serve.sh       # one foreground endpoint
./recipes/qwen3.5-0.8b/run.sh         # configured detached replicas
./recipes/qwen3.5-0.8b/status.sh
./recipes/qwen3.5-0.8b/stop.sh

# Import the same image, then choose Singularity (or apptainer).
./scripts/build-vllm-sif.sh
LLM_RUNTIME=singularity ./recipes/qwen3.5-0.8b/run.sh
LLM_RUNTIME=singularity ./recipes/qwen3.5-0.8b/stop.sh
```

`run.sh [replicas]` overrides replica count; `serve.sh [vLLM flags...]` forwards
extra server flags. Environment settings override profile and model defaults.
The served model name is the configured weights ID. `MODEL`/`VLLM_MODEL` changes
the weights; `REPLICA_INDEX` selects a foreground or benchmark endpoint.
`DRY_RUN=1` prints launch commands without accessing container daemons or GPUs.

```bash
MODEL_REVISION=<weights-commit> ./benchmarks/run.sh qwen3.5-0.8b
```

See [shared runtime instructions](../../scripts/vllm-docker/README.md) and
[benchmark instructions](../../benchmarks/README.md). GPU inference must be
validated on the target hardware. Existing original-engine scripts, where
present, are retained alongside this vLLM recipe.
