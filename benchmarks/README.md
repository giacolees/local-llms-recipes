# Performance records

No GPU measurements have been collected yet. Empty profile directories are
intentional. `summary.csv` contains a header until real runs are recorded.

Start a server with `recipes/<model>/serve.sh`, wait for `/health` to succeed,
and run the matching vLLM CLI on the GPU host (or a documented client host):

```bash
export LLM_HW_PROFILE=4xa100 LLM_RUNTIME=docker
export VLLM_MODEL=Qwen/Qwen3.6-27B
export VLLM_IMAGE=vllm/vllm-openai:<tested-version>
export MODEL_REVISION=<weights-commit>
# Run from the repo root; set BASE_URL if using a different server address.
CONCURRENCY=1 NUM_PROMPTS=100 ./benchmarks/run.sh Qwen3.6-27B
CONCURRENCY=8 NUM_PROMPTS=100 ./benchmarks/run.sh Qwen3.6-27B
```

Results live in `results/<profile>/<recipe>/<runtime>/<UTC-run>/`:
raw `result.json`, environment metadata, and CLI log. The CLI records workload
parameters, request errors, TTFT, TPOT, ITL, end-to-end latency and throughput.
`python3 benchmarks/summarize.py` regenerates the aggregate CSV. Missing fields
are blank, never zero. Failed runs without a result file stay out of the CSV;
check `failed` even when a result file exists.

* Output throughput is aggregate generated tokens/s over the whole run.
* TTFT is time to first token in ms (sometimes called TFT).
* TPOT is time per output token after the first token; ITL is inter-token latency.
* Request throughput is requests/s; total token throughput includes input tokens.

Use identical input/output lengths, seed, concurrency and request rate when
comparing runtimes. Run a warmup separately, then collect at least three runs.
Keep cold-start results separate. Record actual server context, TP, quantization,
GPU selection, image digest, weight revision, driver, and launch flags alongside
metadata.json; these must describe the server, not just the client environment.
Copy the server log into the run directory for reproducibility. The runner cannot
verify a remote server's hardware or configuration. Inspect raw JSON before
claiming a result, and avoid comparing different quantizations as equivalent.
