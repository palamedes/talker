#!/usr/bin/env bash
# One-time setup for talker's OPTIONAL second engine: Ditto (Ant Group),
# a motion-space talking-head specialist. Fully isolated from the LongCat
# engine: own venv (.venv-ditto), own vendor checkout, own weights dir.
# Running this never touches the LongCat installation.
#
# Requirements: NVIDIA GPU + driver, ffmpeg, git, and python3.10 or uv.
# Ditto is light: ~2-6 GB VRAM at runtime, ~4 GB of downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENDOR="$ROOT/vendor/ditto-talkinghead"
WEIGHTS="$ROOT/weights/ditto"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

step "Preflight checks"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
command -v ffmpeg     >/dev/null || fail "ffmpeg not found"
command -v git        >/dev/null || fail "git not found"

step "Creating venv (.venv-ditto, python 3.10: upstream's tested version)"
if [[ ! -d "$ROOT/.venv-ditto" ]]; then
    if command -v python3.10 >/dev/null; then
        python3.10 -m venv "$ROOT/.venv-ditto"
    elif command -v uv >/dev/null; then
        uv venv --python 3.10 --seed "$ROOT/.venv-ditto"
    else
        fail "no python3.10 and no uv on PATH (uv can fetch a private 3.10)"
    fi
fi
# shellcheck disable=SC1091
source "$ROOT/.venv-ditto/bin/activate"
echo "venv python: $(python --version)"
pip install -q -U pip wheel setuptools

step "Installing PyTorch (cu128: works Ampere through Blackwell)"
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

step "Installing Ditto dependencies (PyTorch backend; no TensorRT)"
pip install \
    librosa==0.10.2.post1 soundfile==0.13.0 soxr==0.5.0.post1 audioread==3.0.1 \
    opencv-python-headless==4.10.0.84 imageio==2.36.1 imageio-ffmpeg==0.5.1 \
    scikit-image==0.25.0 scikit-learn==1.6.0 scipy==1.15.0 numba==0.60.0 \
    filetype==1.2.0 tqdm cython pooch onnxruntime mediapipe einops \
    "huggingface_hub[cli]"

step "Cloning ditto-talkinghead"
if [[ ! -d "$VENDOR" ]]; then
    git clone --depth 1 https://github.com/antgroup/ditto-talkinghead "$VENDOR"
else
    echo "already cloned: $VENDOR"
fi

step "Downloading Ditto checkpoints (PyTorch + cfg only; skipping TensorRT engines)"
if [[ ! -e "$WEIGHTS/.download-complete" ]]; then
    # one pattern per call: huggingface_hub >= 1.0 misparses a second
    # pattern after --include as a positional filename
    hf download digital-avatar/ditto-talkinghead --include "ditto_cfg/*" --local-dir "$WEIGHTS"
    hf download digital-avatar/ditto-talkinghead --include "ditto_pytorch/*" --local-dir "$WEIGHTS"
    touch "$WEIGHTS/.download-complete"
else
    echo "already downloaded: $WEIGHTS"
fi

step "Done"
echo "try:  $ROOT/talker mp4 face.png speech.wav --engine ditto"
