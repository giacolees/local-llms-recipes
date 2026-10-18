# Local LLM recipes

Model recipes for 4× A100 40 GB (`4xa100`) and 2× RTX A6000 48 GB
(`2xa6000`). Every model lives under one `recipes/<model>/` directory.

```
recipes/<model>/
  serve.sh                    # shared vLLM launcher: Docker or Singularity
  vllm.env                    # model identity
  profiles/<profile>.vllm.env # vLLM tuning
  profiles/<profile>.env      # original engine tuning
  run-*.sh, *.def             # original launch/build recipes, where available
profiles/<profile>/config.env # shared hardware layout
scripts/                     # shared vLLM launch and SIF conversion
benchmarks/                  # runner, raw results by profile, summary CSV
```

Models: `DS4-flash`, `Laguna-S2.1`, `Qwen3.6-27B`, and
`Qwen3.8-Flash-Next-GGUF`. The GGUF suffix identifies the original recipe;
vLLM uses separately configured compatible weights.

## vLLM quick start

Requires Linux, NVIDIA drivers, and either Docker with NVIDIA Container Toolkit
or Singularity/Apptainer with NVIDIA support. Select an image version supporting
your model architecture and quantization; pin its digest for reproducible runs.

```bash
source ./switch-profile.sh 4xa100
export VLLM_IMAGE=vllm/vllm-openai:<tested-version>
export LLM_RUNTIME=docker
./recipes/Qwen3.6-27B/serve.sh

# Import exactly the same OCI image into a SIF, without a custom CUDA build.
./scripts/build-vllm-sif.sh
LLM_RUNTIME=singularity ./recipes/Qwen3.6-27B/serve.sh
# LLM_RUNTIME=apptainer is also supported.
```

Replace `<tested-version>` with a real tag or digest. For other models, set
`VLLM_MODEL` to a supported HF repository or absolute local safetensors directory;
see each recipe's README. No compatible vLLM IDs are guessed for the three
original GGUF recipes. vLLM support, quantization fit and actual GPU inference
still need validation on the target machines. The shared launcher does not
convert GGUF weights or make unsupported architectures loadable.

`HF_HOME` defaults to `~/.cache/huggingface`; `HF_TOKEN` is forwarded when set.
`VLLM_SIF` defaults to `images/vllm-openai.sif`. Host networking exposes the
server on `PORT` (default 8000). `HOST`, `PORT`, `TP`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, and `MODEL_REVISION` override defaults. Extra arguments
are forwarded to vLLM. Hardware settings are loaded fresh per invocation.

```bash
DRY_RUN=1 LLM_RUNTIME=docker ./recipes/Qwen3.6-27B/serve.sh
MAX_MODEL_LEN=16384 ./recipes/Qwen3.6-27B/serve.sh --max-num-seqs 4
curl http://127.0.0.1:8000/health
```

The served model name is the recipe directory name. Only enable remote model
code explicitly with `--trust-remote-code` when your model requires it.

## Original llama.cpp recipes and migration

Existing model directories moved under `recipes/`; model overrides moved from
`profiles/<hardware>/<model>.env` to `recipes/<model>/profiles/<hardware>.env`.
Update job scripts and local SIF locations accordingly. Existing uncommitted
build-script edits were retained. SIF files and model weights are not versioned.

Run build helpers and `run-*.sh` inside their new model directories. Set
`GGUF_MODELS_DIR` (default `~/models/gguf`) for local weights. The CUDA build
architectures include both SM80 and SM86. Original launchers retain their engine
settings; use `serve.sh` for the shared Docker/Singularity vLLM path.

## Benchmarks and validation

See [benchmarks/README.md](benchmarks/README.md) for TPS, TTFT, TPOT, ITL and
latency recording by model, hardware profile and runtime. No performance results
are invented. Run local checks with `python3 -m unittest discover -s tests -v`.

## Suggested next improvements

* Validate each model/quantization on both GPU profiles, then commit exact image
  digests and weight revisions with known-good settings.
* Add GPU CI smoke runs checking `/health` and one completion for each runtime.
* Tune context and concurrency from measured KV-cache capacity; original large
  context values should be confirmed under realistic load.
* Add Slurm job templates and record NUMA/NVLink topology rather than assuming
  the same topology on every machine with the same GPU count.
* Pin llama.cpp fork commits and CUDA base image digests for repeatable builds.
* Collect GPU memory, power and utilization alongside benchmark latency.

References: [vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker/)
and [benchmark CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/).
