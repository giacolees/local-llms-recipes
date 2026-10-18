# Shared vLLM replica tools

Model definitions now live in `../../recipes/<model>/recipe.env`; model-specific
hardware overrides live beside them in `profiles/<hardware>.vllm.env`.
`lib.sh` discovers recipes from those files, with no duplicate model list.

From the repository root:

```bash
source ./switch-profile.sh 2xa6000
export VLLM_IMAGE=vllm/vllm-openai:<tested-version>
./recipes/qwen3.5-2b/serve.sh       # one foreground endpoint (replica 0)
./recipes/qwen3.5-2b/run.sh         # all configured replicas, detached
./recipes/qwen3.5-2b/status.sh
./recipes/qwen3.5-2b/stop.sh
```

Set `LLM_RUNTIME=singularity` or `apptainer` for SIF execution. First import the
same OCI image with `scripts/build-vllm-sif.sh`. `VLLM_SIF` selects its path.
`run.sh` uses an instance per replica for these runtimes; `serve.sh` uses a
foreground exec. Instances use host networking and `BIND_HOST` directly, whereas
Docker detached replicas publish their ports on `BIND_HOST`. Foreground servers
use host networking and listen on `HOST` (default 0.0.0.0).

Replica GPU assignment is round-robin for TP=1. TP>1 uses the profile's visible
GPU set. Replica i serves on BASE_PORT+i; REPLICA_INDEX selects a foreground
endpoint or the benchmark endpoint. Use run.sh on one machine at a time; the
`4xa6000` profile gives these single-node recipes only its two local GPUs.
Use the separate cluster recipe for distributed serving.

Environment overrides win over model-specific profile values, which win over
recipe defaults. Shared runtime/cache/network settings are in
`../../profiles/<hardware>/vllm-docker.env`. Original 2×A6000 replica budgets,
KV-cache dtype, parser settings and compact EXTRA_ARGS JSON are preserved.
A100 overrides use four-way TP for the largest BF16 models and a shorter
starting context. Hardware fit must be measured; these fractions are not
memory guarantees. Budgets over 95% per GPU are rejected before launch.

`run.sh [replicas]` overrides replica count before budget validation.
`serve.sh [vLLM flags...]` appends arguments to the foreground server.
`MODEL` or `VLLM_MODEL` overrides its weights; `MODEL_REVISION` pins weights.
`HF_TOKEN` is passed by environment name. Gemma access requires accepted model
terms and an authorized token. Remote code is enabled only with
`TRUST_REMOTE_CODE=1` or an explicit `serve.sh --trust-remote-code` flag.

Other shared tools:

* `download-models.sh [recipe...]`: prefetch weights using Docker or SIF.
* `with-hf-token.sh COMMAND...`: retained ambient/Infisical token helper.
* `setup-docker.sh`: retained host setup helper; run explicitly when needed.
* `tests/test-model.sh RECIPE`: health, model identity, completion, JSON, tools
  and stability gates on a live endpoint; requires curl, jq and Python.
* `tests/test-all.sh [recipe...]`: run model gates sequentially, stopping
  registered recipe replicas between models. `KEEP=1` retains the final model.

These capability gates include an informal throughput sweep. Use
`../../benchmarks/run.sh` for reproducible performance records and TTFT/ITL.
Legacy `../../vllm-docker/*.sh` commands forward here, and the old key
`qwen3.6-27b` resolves to the canonical `Qwen3.6-27B` folder.
