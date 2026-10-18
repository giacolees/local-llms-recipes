# gemma4-26b-a4b

vLLM recipe for `google/gemma-4-26B-A4B-it`. Weights, parser flags, capability gates and serving
defaults live in `recipe.env`; hardware overrides are in `profiles/`.

The inherited `2xa6000` defaults are 1 replica(s), TP=2,
GPU memory utilization 0.70, context 32768, and ports starting at 8400.
These are configuration values; measure performance and memory fit on your host.
`4xa100` has model-local starting overrides; `4xa6000` runs on the local two GPUs.

```bash
# From the repository root:
source ./switch-profile.sh 2xa6000
export VLLM_IMAGE=vllm/vllm-openai:<tested-version>
./recipes/gemma4-26b-a4b/serve.sh       # one foreground endpoint
./recipes/gemma4-26b-a4b/run.sh         # configured detached replicas
./recipes/gemma4-26b-a4b/status.sh
./recipes/gemma4-26b-a4b/stop.sh

# Import the same image, then choose Singularity (or apptainer).
./scripts/build-vllm-sif.sh
LLM_RUNTIME=singularity ./recipes/gemma4-26b-a4b/run.sh
LLM_RUNTIME=singularity ./recipes/gemma4-26b-a4b/stop.sh
```

`run.sh [replicas]` overrides replica count; `serve.sh [vLLM flags...]` forwards
extra server flags. Environment settings override profile and model defaults.
The served model name is the configured weights ID. `MODEL`/`VLLM_MODEL` changes
the weights; `REPLICA_INDEX` selects a foreground or benchmark endpoint.
`DRY_RUN=1` prints launch commands without accessing container daemons or GPUs.

```bash
MODEL_REVISION=<weights-commit> ./benchmarks/run.sh gemma4-26b-a4b
```

See [shared runtime instructions](../../scripts/vllm-docker/README.md) and
[benchmark instructions](../../benchmarks/README.md). GPU inference must be
validated on the target hardware. Existing original-engine scripts, where
present, are retained alongside this vLLM recipe.
