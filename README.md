# local-LLMs — Multi-Hardware LLM Inference Configs

A repository of LLM inference configurations and launchers for different GPU
hardware profiles. Currently supports:

| Profile | GPUs | Total VRAM | Inference Engines |
|---------|------|-----------|-------------------|
| `4xa100`  | 4× NVIDIA A100-SXM4-40GB | 160 GB | DwarfStar (ds4), llama.cpp (DS4-flash, Qwen3.8 GGUF, Laguna S2.1), vLLM (Qwen) |
| `2xa6000` | 2× NVIDIA RTX A6000 48GB  | 96 GB  | Same engines, adjusted params |

## Directory Layout

```
local-LLMs/
├── profiles/                        # Hardware profiles
│   ├── 4xa100/                      # 4×A100 configs
│   │   ├── config.env               # Shared GPU layout (devices, split, NUMA)
│   │   ├── ds4-flash.env            # llama.cpp DeepSeek V4 Flash overrides
│   │   ├── laguna.env               # Laguna S2.1 overrides
│   │   ├── qwen.env                 # vLLM Qwen overrides
│   │   └── qwen3.8-flash-next.env   # llama.cpp Qwen3.8 GGUF overrides
│   └── 2xa6000/                     # 2×A6000 configs (same file set)
│       ├── config.env
│       ├── ds4-flash.env
│       ├── laguna.env
│       ├── qwen.env
│       └── qwen3.8-flash-next.env
│
├── switch-profile.sh                # Source this to activate a profile
│
├── DS4-flash/                       # llama.cpp DeepSeek V4 Flash server
│   ├── run-deepseek-server.sh       # Launcher (sources active profile)
│   ├── deepseek-v4-flash.def        # Singularity definition file
│   ├── deepseek-v4-flash-llamacpp.sif  # Pre-built container
│   └── logs/
│
├── Laguna-S2.1/                     # Laguna S2.1 with DFlash speculative decoding
│   ├── run-laguna-server.sh         # Launcher (sources active profile)
│   └── laguna-s2.1.def              # Singularity definition file
│
├── Qwen3.6-27B/                     # vLLM Qwen3.6-27B server
│   ├── run-qwen-server.sh           # Launcher (sources active profile)
│   └── logs/
│
├── Qwen3.8-Flash-Next-GGUF/         # llama.cpp Qwen3.8-Flash-Next GGUF server
│   ├── run-qwen-server.sh           # Launcher (sources active profile)
│   ├── qwen3.8-flash-next.def       # Singularity definition
│   └── logs/
│
├── ds4/                             # DwarfStar native inference engine
│   └── (source code, Makefile, etc.)
│
└── vllm-openai.sif                  # Pre-built vLLM Singularity image
```

## Quick Start

```bash
# 1. Activate a hardware profile (sets CUDA_VISIBLE_DEVICES, tensor split, etc.)
source ./switch-profile.sh 4xa100    # for the 4×A100 machine
# or
source ./switch-profile.sh 2xa6000   # for the 2×A6000 machine

# 2. Run any server — it picks up the active profile automatically
cd DS4-flash && ./run-deepseek-server.sh
cd Qwen3.6-27B && ./run-qwen-server.sh
cd Qwen3.8-Flash-Next-GGUF && ./run-qwen-server.sh
cd Laguna-S2.1 && ./run-laguna-server.sh
```

## Adding a New Profile

```bash
mkdir -p profiles/<your-profile>
cp profiles/4xa100/config.env profiles/<your-profile>/config.env
# Edit config.env for your GPU layout, then add per-project overrides as needed.
```

## Adding a New Project

1. Create the project directory with its run script.
2. Have the run script source `config.env` and a project-specific `.env` from
   the active profile directory (see existing scripts for the pattern).
3. Add the corresponding `.env` file in each hardware profile.

## How It Works

1. `switch-profile.sh` sources `profiles/<profile>/config.env`, which sets
   `CUDA_VISIBLE_DEVICES`, `TENSOR_SPLIT`, `GPU_COUNT`, NUMA bindings, etc.
2. Each `run-*.sh` script reads `LLM_HW_PROFILE` (or defaults to `4xa100`),
   sources `config.env` for shared GPU layout, then sources its project-specific
   `.env` for model paths and server parameters. The Qwen3.8 GGUF launcher uses
   UD-IQ3_XXS on 2×A6000 and UD-Q4_K_XL on 4×A100.
3. The Qwen3.8 GGUF launcher passes
   `--override-tensor per_layer_token_embd=CPU` so its large n-gram/PLE table is
   kept in host RAM instead of GPU VRAM on both hardware profiles.
4. To change hardware, re-source `switch-profile.sh` with a different profile.
   No script editing needed.
