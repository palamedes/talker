# talker

Animate a single portrait image to speak a single audio file, using
[LongCat-Video-Avatar 1.5](https://github.com/meituan-longcat/LongCat-Video)
(Meituan, MIT-licensed). One command in, one lip-synced clip out:

```sh
./talker mp4 me.png voice.wav            # video with the audio muxed in
./talker gif me.png voice.wav            # video-only gif (for editors that
                                         # already carry the audio track)
```

The last line printed on stdout is the output path (default
`<audio-stem>.mp4` / `.gif` in the current directory).

## Why the output stays in sync

- Avatar 1.5 generates at **25 fps**. 25 fps = 40 ms/frame = **4 centiseconds**,
  and GIF frame delays are stored in centiseconds — so the gif timing is
  *exact*, with zero cumulative drift against your audio track.
- Both formats are trimmed to the input audio's exact duration; mp4 gets the
  **original** audio muxed back in (video re-encoded to editor-friendly
  h264/yuv420p, `-crf 16`).
- Every run ends with a verification line comparing output duration to audio
  duration; anything within one frame (±40 ms) is reported as in sync.

## Setup (one time)

Needs an NVIDIA GPU + driver, ffmpeg, git, ~30 GB of disk, and either
python 3.10/3.11 on PATH **or [`uv`](https://docs.astral.sh/uv/)** (setup will
then provision a private python 3.11 for the venv — your system python
version doesn't matter). Python 3.13+ cannot work: LongCat pins
`numpy==1.26.4` / `transformers==4.41.0`, which top out at 3.12.

```sh
./setup.sh
```

This creates `.venv/`, clones LongCat-Video into `vendor/`, installs the
GPU-appropriate torch + a **runtime-verified** flash-attn, the repo
requirements (with upstream's stale torch/flash-attn pins filtered out), and
downloads the
[Avatar-1.5 weights](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5)
into `weights/`. Everything is gitignored, downloads resume, and re-running
is safe — finished steps are skipped.

GPU notes:

- **RTX 50-series (Blackwell, sm_120)** is auto-detected: setup installs
  torch 2.7.1/cu128 instead of the upstream 2.6/cu124 (which has no Blackwell
  kernels), and if the prebuilt flash-attn wheel fails its runtime check it
  compiles flash-attn from source for sm_120 (needs `nvcc` from the CUDA
  toolkit; 10–30 min, one time). LongCat has no sdpa fallback — flash-attn
  must actually work, which is why setup executes a real attention op on
  your GPU to verify it.
- **VRAM**: talker defaults to int8 + 480p + expandable CUDA segments.
  Comfortable on 24 GB; 16 GB (5070 Ti-class) is borderline — try it, and if
  it OOMs, open an issue/report back (block-swap offloading is the next lever).
  720p and `--no-int8` want more than 16 GB.

## Usage

```
talker {gif|mp4} <image> <audio> [options]

  -o, --output PATH     output file (default: <audio-stem>.<format>)
  --prompt TEXT         scene/motion prompt (default: neutral talking-head,
                        static background)
  --resolution {480p,720p}   default 480p
  --no-int8             full-precision DiT (more VRAM, marginally better)
  --steps N             override inference steps (default: distilled 8-step)
  --gif-width W         downscale the gif to width W (default: native)
  --keep-workdir        keep the temp dir with the raw model output
```

Examples:

```sh
./talker gif me.png line42.wav -o clips/line42.gif --gif-width 480
./talker mp4 me.png intro.wav --resolution 720p --prompt \
  "A man speaks warmly to camera in a sunlit office, subtle gestures."
```

Notes:

- The audio doesn't have to be wav — anything ffmpeg/librosa can read works.
- The prompt matters: it steers framing, gesture and background. The default
  keeps things neutral and static, which composites best in an editor.
- GIF is 256-color; for maximum quality in the editor prefer mp4 and mute it
  on the timeline. GIF output uses per-clip palettegen + sierra dithering,
  which is about as good as gif gets.
- First run is slower (model load + CUDA warmup); subsequent runs reuse the
  OS page cache but still reload the model each invocation.

## Layout

```
talker        bash launcher (activates .venv, execs talker.py)
talker.py     driver: builds LongCat input json, runs inference, ffmpeg post
setup.sh      one-time env + vendor clone + weight download
vendor/       LongCat-Video checkout        (gitignored)
weights/      Avatar-1.5 checkpoints        (gitignored)
.venv/        python env                    (gitignored)
```
