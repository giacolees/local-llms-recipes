#!/usr/bin/env bash
# =============================================================================
# setup-docker.sh — One-time host setup for the vLLM Docker recipes (2xa6000)
#
# Usage:
#   sudo ./setup-docker.sh
#
# Installs nvidia-container-toolkit (needed for `docker run --gpus`), enables
# the Docker daemon, and adds your user to the `docker` group. Log out and back
# in (or `newgrp docker`) afterwards — then run the model recipes without sudo.
# =============================================================================
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "ERROR: run this with sudo:  sudo $0" >&2
    exit 1
fi
TARGET_USER="${SUDO_USER:-${USER}}"

echo "--- 1/4 nvidia-container-toolkit (Docker GPU access) ---"
if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "already installed: $(nvidia-ctk --version 2>/dev/null | head -1)"
else
    pacman -S --needed --noconfirm nvidia-container-toolkit ||
        { echo "pacman failed — try your AUR helper: paru -S nvidia-container-toolkit" >&2; exit 1; }
fi

echo "--- 2/4 register the NVIDIA container runtime with Docker ---"
nvidia-ctk runtime configure --runtime=docker

echo "--- 3/4 enable and start Docker ---"
systemctl enable --now docker

echo "--- 4/4 grant '${TARGET_USER}' passwordless docker access ---"
getent group docker >/dev/null || groupadd docker
usermod -aG docker "${TARGET_USER}"

echo
echo "=== self-test: GPU inside Docker ==="
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L; then
    echo "OK — GPU passthrough works."
else
    echo "GPU self-test FAILED — check: nvidia-smi on host, then journalctl -u docker" >&2
    exit 1
fi

echo
echo "Done. Log out and back in (or run 'newgrp docker'), then:"
echo "  cd vllm-docker && ./tests/test-all.sh    # full per-model test suite"
