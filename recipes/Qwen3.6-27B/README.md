# Qwen3.6-27B

vLLM recipe for `Qwen/Qwen3.6-27B`. Weights, parser flags, capability gates and serving
defaults live in `recipe.env`; hardware overrides are in `profiles/`.

The inherited `2xa6000` defaults are 1 replica(s), TP=2,
GPU memory utilization 0.70, context 32768, and ports starting at 8600.
These are configuration values; measure performance and memory fit on your host.
`4xa100` has model-local starting overrides; `4xa6000` runs on the local two GPUs.

```bash
# From the repository root:
source ./switch-profile.sh 2xa6000
export VLLM_IMAGE=vllm/vllm-openai:<tested-version>
./recipes/Qwen3.6-27B/serve.sh       # one foreground endpoint
./recipes/Qwen3.6-27B/run.sh         # configured detached replicas
./recipes/Qwen3.6-27B/status.sh
./recipes/Qwen3.6-27B/stop.sh

# Import the same image, then choose Singularity (or apptainer).
./scripts/build-vllm-sif.sh
LLM_RUNTIME=singularity ./recipes/Qwen3.6-27B/run.sh
LLM_RUNTIME=singularity ./recipes/Qwen3.6-27B/stop.sh
```

`run.sh [replicas]` overrides replica count; `serve.sh [vLLM flags...]` forwards
extra server flags. Environment settings override profile and model defaults.
The served model name is the configured weights ID. `MODEL`/`VLLM_MODEL` changes
the weights; `REPLICA_INDEX` selects a foreground or benchmark endpoint.
`DRY_RUN=1` prints launch commands without accessing container daemons or GPUs.

```bash
MODEL_REVISION=<weights-commit> ./benchmarks/run.sh Qwen3.6-27B
```

See [shared runtime instructions](../../scripts/vllm-docker/README.md) and
[benchmark instructions](../../benchmarks/README.md). GPU inference must be
validated on the target hardware. Existing original-engine scripts, where
present, are retained alongside this vLLM recipe.
