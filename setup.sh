#!/usr/bin/env bash
# One-time setup for talker: venv + LongCat-Video + Avatar-1.5 weights.
#
# Requirements before running:
#   - NVIDIA GPU + driver, nvidia-smi on PATH
#       * RTX 50-series (Blackwell, sm_120) is handled: torch cu128 + a
#         runtime-verified flash-attn (source-built for sm_120 if needed)
#   - python3.10/3.11 on PATH, OR `uv` (which will provision a private 3.11
#     for the venv — your system python version doesn't matter then)
#   - ffmpeg / ffprobe on PATH
#   - git, ~30 GB free disk (weights + deps)
#
# Safe to re-run: every step skips work it has already done.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENDOR="$ROOT/vendor/LongCat-Video"
WEIGHTS="$ROOT/weights/LongCat-Video-Avatar-1.5"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

step "Preflight checks"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — NVIDIA GPU + driver required"
command -v ffmpeg     >/dev/null || fail "ffmpeg not found — install it (e.g. pacman -S ffmpeg)"
command -v git        >/dev/null || fail "git not found"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Blackwell (RTX 50-series, compute capability 12.x) needs CUDA 12.8 kernels:
# torch 2.6/cu124 does NOT run on it. Pick the torch build accordingly.
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
CAP_MAJOR="${CAP%%.*}"
if (( CAP_MAJOR >= 12 )); then
    BLACKWELL=1
    echo "detected Blackwell GPU (sm_${CAP/./})"
    # sm_120 needs a source-built flash-attn, and the nvcc that builds it must
    # match torch's CUDA major version — so locate nvcc FIRST and pick the
    # torch build to match it.
    if [[ -z "${CUDA_HOME:-}" ]]; then
        for d in /opt/cuda /usr/local/cuda; do
            [[ -x "$d/bin/nvcc" ]] && { CUDA_HOME="$d"; break; }
        done
    fi
    if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null; then
        CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    fi
    [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] || fail "CUDA toolkit (nvcc) not found — required to compile flash-attn for sm_120.
  Install it:   sudo pacman -S cuda      (Arch/CachyOS; lands in /opt/cuda)
  then re-run ./setup.sh — it will auto-detect it there."
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    NVCC_VER="$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    NVCC_MAJOR="${NVCC_VER%%.*}"
    NVCC_MINOR="${NVCC_VER#*.}"
    echo "found nvcc $NVCC_VER at $CUDA_HOME"
    if (( NVCC_MAJOR == 12 )); then
        (( NVCC_MINOR >= 8 )) || fail "nvcc $NVCC_VER cannot target sm_120 — need CUDA >= 12.8"
        TORCH_SPEC="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        echo "-> torch 2.7.1 / cu128 (matches nvcc 12.x)"
    elif (( NVCC_MAJOR >= 13 )); then
        TORCH_SPEC="torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1"
        TORCH_INDEX="https://download.pytorch.org/whl/cu130"
        echo "-> torch 2.9.1 / cu130 (matches nvcc ${NVCC_MAJOR}.x)"
    else
        fail "unsupported nvcc version $NVCC_VER"
    fi
    FLASH_ATTN_VER="2.8.3"
else
    BLACKWELL=0
    TORCH_SPEC="torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    FLASH_ATTN_VER="2.7.4.post1"
    echo "detected sm_${CAP/./} GPU -> torch 2.6.0 / cu124 (upstream-pinned)"
fi

step "Creating venv (python 3.11 — LongCat's pinned deps don't support 3.13+)"
if [[ ! -d "$ROOT/.venv" ]]; then
    PY=""
    for cand in python3.11 python3.10; do
        command -v "$cand" >/dev/null && { PY="$cand"; break; }
    done
    if [[ -n "$PY" ]]; then
        "$PY" -m venv "$ROOT/.venv"
    elif command -v uv >/dev/null; then
        # uv downloads a standalone CPython 3.11 just for this venv;
        # the system python (whatever version) is not involved.
        uv venv --python 3.11 --seed "$ROOT/.venv"
    else
        fail "no python3.10/3.11 and no uv on PATH.
  Easiest fix: install uv (pacman -S uv  |  curl -LsSf https://astral.sh/uv/install.sh | sh)
  and re-run — it will fetch a private python 3.11 for the venv."
    fi
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
echo "venv python: $(python --version)"
pip install -q -U pip wheel setuptools packaging ninja psutil

step "Installing PyTorch ($TORCH_SPEC)"
# shellcheck disable=SC2086
pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch sees no CUDA device"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | OK")
PY

step "Installing flash-attn $FLASH_ATTN_VER"
flash_attn_works() {
    python - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.float16)
out = flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("flash-attn runtime check: OK", tuple(out.shape))
PY
}
if flash_attn_works 2>/dev/null; then
    echo "flash-attn already installed and working"
elif (( BLACKWELL )); then
    # Official wheels don't ship sm_120 kernels — build from source for
    # exactly this arch. One time; 10-30 min depending on CPU.
    echo "building flash-attn from source for sm_120 (normal for RTX 50-series; 10-30 min)"
    pip uninstall -y flash-attn flash_attn 2>/dev/null || true
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTN_CUDA_ARCHS="120" \
    MAX_JOBS="$(( $(nproc) < 8 ? $(nproc) : 8 ))" \
        pip install -v "flash_attn==$FLASH_ATTN_VER" --no-build-isolation
    flash_attn_works || fail "flash-attn still fails its runtime check after source build"
else
    # Non-Blackwell: the prebuilt wheel path works (still needs nvcc present
    # for pip's metadata step — that's a flash-attn packaging quirk).
    pip install "flash_attn==$FLASH_ATTN_VER" --no-build-isolation
    flash_attn_works || fail "flash-attn failed its runtime check"
fi

step "Cloning LongCat-Video"
if [[ ! -d "$VENDOR" ]]; then
    git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video "$VENDOR"
else
    echo "already cloned: $VENDOR"
fi

step "Installing LongCat requirements (torch/flash-attn pins filtered out)"
# Upstream pins torch==2.6.0 and flash-attn — installing those verbatim would
# clobber the GPU-appropriate builds we just verified. Strip them.
filter_reqs() { grep -vE '^\s*(torch|torchvision|torchaudio|flash[-_]attn)\b' "$1"; }
filter_reqs "$VENDOR/requirements.txt"        | pip install -r /dev/stdin
filter_reqs "$VENDOR/requirements_avatar.txt" | pip install -r /dev/stdin
pip install librosa "huggingface_hub[cli]"

step "Downloading Avatar-1.5 weights (large; resumes if interrupted)"
if [[ ! -e "$WEIGHTS/.download-complete" ]]; then
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir "$WEIGHTS"
    touch "$WEIGHTS/.download-complete"
else
    echo "already downloaded: $WEIGHTS"
fi

step "Done"
echo "try:  $ROOT/talker mp4 face.png speech.wav"
if (( BLACKWELL )); then
    echo "note: 16 GB VRAM is borderline for the 13.6B DiT — talker defaults to"
    echo "      int8 + 480p and sets expandable CUDA segments; if you still OOM,"
    echo "      report back and we'll wire up offloading."
fi
