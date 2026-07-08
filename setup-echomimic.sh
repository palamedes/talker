#!/usr/bin/env bash
# One-time setup for talker's echomimic engine: EchoMimicV3-Flash
# (Ant Group, AAAI 2026). 1.3B diffusion talking-head model built on
# Wan2.1-Fun: paints real mouth shapes (unlike warping engines), runs in
# ~12 GB VRAM via sequential offload, and exposes genuine control knobs
# (audio guidance, working negative prompts).
#
# Fully isolated: own venv (.venv-emv3), own vendor checkout, own weights.
# Never touches the longcat or ditto installations.
#
# Requirements: NVIDIA GPU + driver, ffmpeg, git, python3.10/3.11 or uv.
# Downloads ~12 GB of weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENDOR="$ROOT/vendor/echomimic_v3"
WEIGHTS="$ROOT/weights/echomimic"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

step "Preflight checks"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
command -v ffmpeg     >/dev/null || fail "ffmpeg not found"
command -v git        >/dev/null || fail "git not found"

step "Creating venv (.venv-emv3, python 3.10)"
if [[ ! -d "$ROOT/.venv-emv3" ]]; then
    if command -v python3.10 >/dev/null; then
        python3.10 -m venv "$ROOT/.venv-emv3"
    elif command -v uv >/dev/null; then
        uv venv --python 3.10 --seed "$ROOT/.venv-emv3"
    else
        fail "no python3.10 and no uv on PATH (uv can fetch a private 3.10)"
    fi
fi
# shellcheck disable=SC1091
source "$ROOT/.venv-emv3/bin/activate"
echo "venv python: $(python --version)"
pip install -q -U pip wheel setuptools

step "Installing PyTorch (cu128: works Ampere through Blackwell)"
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

step "Installing EchoMimicV3 dependencies"
# From upstream requirements.txt, minus gradio/tensorboard/datasets (demo
# and training only). tensorflow is CPU-only here (retina-face needs it).
pip install \
    Pillow einops safetensors timm tomesd torchdiffeq torchsde decord \
    numpy scikit-image opencv-python omegaconf SentencePiece albumentations \
    "imageio[ffmpeg]" "imageio[pyav]" beautifulsoup4 ftfy func_timeout \
    onnxruntime "accelerate>=0.25.0" "diffusers>=0.30.1" \
    "transformers>=4.46.2" moviepy==2.2.1 tensorflow-cpu==2.15.0 \
    retina-face==0.0.17 librosa mmgp pyloudnorm "huggingface_hub[cli]"

step "Cloning echomimic_v3"
if [[ ! -d "$VENDOR" ]]; then
    git clone --depth 1 https://github.com/antgroup/echomimic_v3 "$VENDOR"
else
    echo "already cloned: $VENDOR"
fi

step "Downloading weights (~12 GB total; resumes if interrupted)"
# one pattern per call: huggingface_hub >= 1.0 misparses multiple patterns
if [[ ! -e "$WEIGHTS/.download-complete" ]]; then
    hf download alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP \
        --local-dir "$WEIGHTS/Wan2.1-Fun-V1.1-1.3B-InP"
    hf download BadToBest/EchoMimicV3 --include "echomimicv3-flash-pro/*" \
        --local-dir "$WEIGHTS"
    hf download TencentGameMate/chinese-wav2vec2-base \
        --local-dir "$WEIGHTS/chinese-wav2vec2-base"
    touch "$WEIGHTS/.download-complete"
else
    echo "already downloaded: $WEIGHTS"
fi

step "Done"
echo "try:  $ROOT/talker mp4 face.png speech.wav --engine echomimic"
