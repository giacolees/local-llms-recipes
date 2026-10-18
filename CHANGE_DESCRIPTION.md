# Consolidate all model recipes and the Qwen cluster

The Docker model catalog and distributed Qwen deployment previously lived
outside the model recipe layout. Every model now has one home under `recipes/`,
with its serving settings and hardware overrides beside it. The ten imported
vLLM configurations include Gemma, Phi and Qwen; the duplicate Qwen3.6-27B
configuration is consolidated into its existing folder.

Single-node models support the same vLLM settings through Docker,
Singularity and Apptainer. Foreground serving selects one endpoint; detached
launching preserves the original replica counts, round-robin GPU assignment,
TP settings, ports, parser flags and memory budgets. Replica-count overrides
are checked against the budget before launch. Model downloads, capability
gates and token helpers use the relocated catalog. Legacy command paths and
the lowercase Qwen3.6 key remain available through forwarding entry points.

The Qwen3.8-27B cluster now owns its topology and deployment tools inside its
recipe folder. Its TP=2/PP=2 Ray deployment supports Docker containers or
Singularity/Apptainer instances on each node. Provisioning transfers the correct
image format, model weights and Ray wheelhouse; status and stop operations
follow the chosen runtime. Dry runs show commands without contacting hosts.

Benchmarking resolves each model's served name, tokenizer and endpoint from the
same configuration. Distributed runs record TP/PP and node topology. Raw JSON,
client logs and metadata stay grouped by hardware profile, model and runtime;
the existing summary reports output TPS, request throughput, TTFT, TPOT, ITL
and end-to-end latency. No model performance measurements were invented.

The root README, model READMEs, cluster guide and benchmark guide describe the
final layout, migration, runtime selection and commands. Shared GPU settings
remain under `profiles/`; model-specific `4xa6000` overrides moved into their
model folders.

All 11 local tests pass. Validation covers 117 single-node model/profile/runtime command combinations,
replica placement, tensor-parallel GPU selection, environment overrides,
invalid budgets, aliases, cluster command generation, benchmark endpoint
resolution, summary handling, instrumented runtime lifecycle checks and shell syntax. The retained capability harness
also passes against a local mock endpoint for health, model identity, chat,
JSON and tool calls. Mock timings are not model performance measurements.

GPU inference, quantization support, memory fit and multi-node networking still
require validation on the actual hosts. The inherited image/model references
and cluster addresses need to match the deployment; pin tested image digests
and weight revisions after those checks.
