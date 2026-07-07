#!/usr/bin/env bash
# One-time setup for talker: venv + LongCat-Video + Avatar-1.5 weights.
#
# Requirements before running:
#   - NVIDIA GPU + driver (CUDA 12.4-capable), nvidia-smi on PATH
#   - python3.10 or python3.11 (LongCat targets 3.10; torch 2.6 supports both)
#   - ffmpeg / ffprobe on PATH
#   - git, ~30 GB free disk (weights + deps)
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENDOR="$ROOT/vendor/LongCat-Video"
WEIGHTS="$ROOT/weights/LongCat-Video-Avatar-1.5"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

step "Preflight checks"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found — GPU required" >&2; exit 1; }
command -v ffmpeg     >/dev/null || { echo "ffmpeg not found — install it (e.g. apt install ffmpeg)" >&2; exit 1; }
command -v git        >/dev/null || { echo "git not found" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Prefer 3.10 (upstream target), fall back to 3.11.
PY=""
for cand in python3.10 python3.11; do
    command -v "$cand" >/dev/null && { PY="$cand"; break; }
done
[[ -n "$PY" ]] || { echo "need python3.10 or python3.11 on PATH" >&2; exit 1; }
echo "using $($PY --version)"

step "Creating venv"
[[ -d "$ROOT/.venv" ]] || "$PY" -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install -q -U pip wheel setuptools packaging ninja psutil

step "Installing PyTorch 2.6 (cu124)"
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

step "Installing flash-attn"
pip install flash_attn==2.7.4.post1 --no-build-isolation

step "Cloning LongCat-Video"
if [[ ! -d "$VENDOR" ]]; then
    git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video "$VENDOR"
else
    echo "already cloned: $VENDOR"
fi

step "Installing LongCat requirements"
pip install -r "$VENDOR/requirements.txt"
pip install -r "$VENDOR/requirements_avatar.txt"
pip install librosa "huggingface_hub[cli]"

step "Downloading Avatar-1.5 weights (~large; resumes if interrupted)"
if [[ ! -e "$WEIGHTS/.download-complete" ]]; then
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir "$WEIGHTS"
    touch "$WEIGHTS/.download-complete"
else
    echo "already downloaded: $WEIGHTS"
fi

step "Done"
echo "try:  $ROOT/talker mp4 face.png speech.wav"
