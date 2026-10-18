# Qwen3.8-27B cluster

vLLM on two workstations, each with 2× RTX A6000 48 GB. The preserved profile
uses TP=2 inside each node and PP=2 across nodes over Ethernet. Ray coordinates
the four GPUs. Docker, Singularity and Apptainer use the same topology and
server arguments. These launch paths need validation on the actual cluster.

Shared hardware settings are in `../../profiles/4xa6000/config.env`; model,
node addresses, SSH targets, network interfaces, image, Ray version and serving
settings are in this recipe's `profiles/4xa6000.env`. Edit those for your hosts.
The checked-in node addresses and interfaces describe the original deployment.
Scripts run on the head; remote operations use BatchMode SSH.

From the repository root, for Docker:

```bash
source ./switch-profile.sh 4xa6000
export LLM_RUNTIME=docker
# Override IMAGE with your tested tag/digest if needed.
./recipes/Qwen3.8-27B-Cluster/cluster_up.sh --sync
./recipes/Qwen3.8-27B-Cluster/serve.sh --wait
./recipes/Qwen3.8-27B-Cluster/status.sh
./recipes/Qwen3.8-27B-Cluster/stop.sh
```

For Singularity (or set `LLM_RUNTIME=apptainer` throughout):

```bash
source ./switch-profile.sh 4xa6000
export LLM_RUNTIME=singularity
# Use exactly the cluster IMAGE tag/digest for conversion.
export VLLM_IMAGE=vllm/vllm-openai:v0.30.0
export VLLM_SIF=/opt/models/vllm-openai.sif
./scripts/build-vllm-sif.sh
./recipes/Qwen3.8-27B-Cluster/cluster_up.sh --sync
./recipes/Qwen3.8-27B-Cluster/serve.sh --wait
```

The v0.30.0 image reference is inherited from the original recipe; verify it
supports the requested architecture and MTP method. Use the same image and Ray
version on every node. Prepare the model checkpoint in
`/opt/models/Qwen3.8-27B` on the head and a Ray wheelhouse in `/opt/models/wheels`.
The wheelhouse must contain the configured Ray version and its dependencies
for the image's Python ABI. `provision_node.sh` copies the SIF (or Docker image),
weights, and wheelhouse to workers. Use the same absolute SIF path on each node.

Docker uses one persistent container per node. Singularity/Apptainer uses one
instance per node, with models mounted at `/models`, host networking and NVIDIA
GPU passthrough. Ray is installed from the wheelhouse into the host-mounted
`/models/ray-runtime-<runtime>` directory so it can be imported without writing
to the immutable SIF. Remove that runtime directory when changing Ray/Python
versions. Start and stop instances as the same user; that user must be able to
write the mounted model/log directories. Host `/dev/shm` must have sufficient
space for Ray and vLLM.

`cluster_up.sh` resets Ray on the configured nodes. `serve.sh` restarts this
model's API server. `stop.sh` stops the model and Ray, then removes containers
or stops instances; `KEEP_CONTAINERS=1` retains them. `status.sh` shows API,
Ray resources and GPUs. Logs are `/opt/models/logs/vllm.log` by default.
Scalar environment overrides take precedence over the profile (e.g.
`SERVE_PORT=9001`, `TP_SIZE=2`, `PP_SIZE=2`, `MODEL_MAX_LEN=32768`). Topology
arrays belong in the profile; missing topology files fail before remote calls.

```bash
DRY_RUN=1 LLM_RUNTIME=singularity ./recipes/Qwen3.8-27B-Cluster/cluster_up.sh
DRY_RUN=1 ./recipes/Qwen3.8-27B-Cluster/serve.sh
MODEL_REVISION=<weights-commit> ./benchmarks/run.sh Qwen3.8-27B-Cluster
```

Benchmarks resolve the head address, served model name and TP/PP from the same
profile and save results under `benchmarks/results/4xa6000/Qwen3.8-27B-Cluster/`.
Record the actual weights revision and image digest used on both nodes. No GPU
measurements are included. See the root README for shared benchmark controls.

The old root `Qwen3.8-27B-Cluster/*.sh` paths forward to this directory.
