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

Needs an NVIDIA GPU (CUDA 12.4 driver), python 3.10/3.11, ffmpeg, git, and
roughly 30 GB of disk. INT8 + 480p should fit on a 24 GB card (4090-class);
720p and/or `--no-int8` want more.

```sh
./setup.sh
```

This creates `.venv/`, clones LongCat-Video into `vendor/`, installs
torch 2.6 (cu124) + flash-attn + the repo requirements, and downloads the
[Avatar-1.5 weights](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5)
into `weights/`. All three dirs are gitignored. The weight download resumes
if interrupted; re-running `setup.sh` is safe and skips finished steps.

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
