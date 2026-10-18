# Qwen3.6-27B

`./serve.sh` launches vLLM using `LLM_RUNTIME=docker|singularity|apptainer`.
Hardware settings come from `../../profiles/$LLM_HW_PROFILE/config.env`;
vLLM overrides live in `profiles/<profile>.vllm.env`.

Set `VLLM_IMAGE` to a tested tag/digest and `VLLM_MODEL` to a compatible
Hugging Face repository or absolute local safetensors directory.
The default model ID is Qwen/Qwen3.6-27B.
Architecture support and memory fit must be verified on the target GPU machine.
Quantized weights may be needed to fit; use the quantization supported by your
GPU and vLLM image. Append vLLM flags to `serve.sh` as needed.

Existing `run-*.sh`, `.def`, and build helpers retain the original inference
setup; their GGUF settings are in `profiles/<profile>.env`.
