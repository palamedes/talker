# talker

Give it one photo of a face and one audio file of speech. It gives you back a
video of that face speaking those words, lips synced, ready to drop on a
video editor timeline.

```sh
./talker mp4 me.png voice.wav
```

Built on [LongCat-Video-Avatar 1.5](https://github.com/meituan-longcat/LongCat-Video)
by Meituan's LongCat team, with a lot of extra plumbing so it runs on a
normal gaming PC instead of a datacenter.

## What is this, in plain terms?

You know those AI "talking head" videos where a still photo appears to speak?
This is that, running entirely on your own computer. Nothing is uploaded
anywhere. You need three things:

1. **A photo.** One clear image of the person (or character, or animal; the
   model is surprisingly game). Front-facing portraits work best.
2. **Speech audio.** A wav or mp3 of someone talking. Recorded voice, TTS,
   AI-generated voice, all fine.
3. **A beefy-ish NVIDIA graphics card.** See the hardware section below.
   The short version: 16 GB of VRAM is the practical floor, and this tool
   exists largely because 16 GB normally isn't enough for this model.

What you get back is a video (or gif) where the face speaks your audio with
accurate lip movement, natural head motion, and blinking. The length of the
output always matches the length of your audio, to the frame. That last part
is treated as sacred around here, because the whole point is dropping the
result onto an editor timeline next to the audio you already have.

Two honest caveats. First, it is slow: a talking-head model of this quality
is doing an enormous amount of math, and on a 16 GB card you should budget
roughly 90 seconds of compute per second of video. Kick off a clip, get
coffee. Second, the first run downloads about 60 GB of model weights, so
setup is an evening project with a decent connection.

## Quick start

```sh
git clone git@github.com:palamedes/talker.git && cd talker
./setup.sh                                   # one time; big download
./talker mp4 me.png voice.wav                # your first clip
```

`setup.sh` is safe to re-run. It skips anything already done, and downloads
resume where they left off if interrupted.

## Two engines

talker can drive two very different models behind the same CLI:

- **longcat** (default): LongCat-Video-Avatar 1.5, a 13.6B video diffusion
  model. Re-renders every pixel: photoreal, handles bodies and scenes,
  slow (~90s of compute per second of video on a 16 GB card). Set up with
  `./setup.sh`.
- **echomimic**: [EchoMimicV3-Flash](https://github.com/antgroup/echomimic_v3)
  (Ant Group, AAAI 2026), a 1.3B diffusion talking-head model on the Wan2.1
  base. Paints real mouth shapes like longcat (puckers, corner movement),
  runs in ~12 GB VRAM via built-in offload with no surgery needed, and its
  control knobs genuinely work: text prompts and negative prompts are live
  (CFG is active), and `--lip-scale` maps to the model's supported
  audio-intensity parameter. `--steps` defaults to 8 (flash distilled).
  Long clips are generated in 81-frame windows cross-faded over 8 frames.
  `./setup-echomimic.sh` to install (~12 GB of weights, isolated).

```sh
./setup-echomimic.sh
./talker mp4 me.png voice.wav --engine echomimic
```

- **ditto**: [Ditto](https://github.com/antgroup/ditto-talkinghead)
  (Ant Group, ACM MM 2025), a motion-space talking-head specialist that
  warps YOUR ACTUAL PHOTO: perfect identity, background untouched, hands
  impossible, near-realtime, real control knobs (see `--emo`). **Honest
  caveat before you install it:** the mouth is driven by six implicit
  keypoints, so lips open and close but the mouth corners never pucker,
  spread, or tense. Speech reads as uncanny at close inspection. Good for
  drafts, timing previews, and stylized/low-scrutiny shots; it does not
  pass a photorealism bar. `./setup-ditto.sh` to install (fully isolated;
  ~10 GB; safe to add or remove at any time).

Both engines share the same output pipeline, so the frame-exact sync
guarantees below apply to either.

## Examples

```sh
# The basics: video with the audio muxed in
./talker mp4 me.png voice.wav

# A gif instead (no audio track; gifs can't hold one). Handy when your
# editor already has the audio and you just want the picture.
./talker gif me.png voice.wav

# Match your editor project's frame rate exactly. "ntsc" is the alias for
# 29.97 (precisely 30000/1001, which is what editors mean by 29.97).
./talker mp4 me.png voice.wav --fps ntsc
./talker mp4 me.png voice.wav --fps 30
./talker gif me.png voice.wav --fps 50 --gif-width 480

# Your audio is clean TTS or studio voiceover? Skip the vocal-separation
# preprocessing pass, it has nothing to separate and just wastes time.
./talker mp4 me.png voice.wav --fps ntsc --no-vocal-sep

# Acting style presets and custom prompts exist, but read the note below
# about what actually controls the performance before leaning on them.
./talker mp4 me.png voice.wav --style lively
./talker mp4 me.png intro.wav --prompt \
  "A man speaks warmly to camera in a sunlit office."

# Tone down exaggerated mouth movement. The response is nonlinear: values
# from 1.0 down to about 0.7 look identical, 0.3 is a nearly still mouth,
# and the useful range is roughly 0.4 to 0.6.
./talker mp4 me.png voice.wav --lip-scale 0.5

# Higher resolution, if your card has the memory for it (24 GB+)
./talker mp4 me.png voice.wav --resolution 720p

# Smoother motion when resampling 25 fps up to 60
./talker mp4 me.png voice.wav --fps 60 --smooth

# Name the output yourself
./talker mp4 me.png line42.wav -o clips/scene3_line42.mp4
```

**What actually controls the performance** (learned the hard way): in this
image-anchored mode the model's job is to animate YOUR PHOTO, and the photo
outranks everything else. Scene, framing, posture, and whether hands exist
all come from the image; a prompt cannot restyle them (we tested with an
underwater scene prompt; the model politely ignored it). So direct with the
photo: want no hand-waving, crop to chest-up with arms out of frame; want a
calm read, pick a relaxed, neutral, closed-mouth portrait. Motion energy
follows the audio; `--lip-scale` blends the audio embedding toward silence
to relax exaggerated mouth movement. The text prompt is best treated as a
caption of your photo plus a gentle motion hint, not a control surface.

Since the voice's delivery is what actually sets the animation energy,
`prosody_check.py` scores candidate TTS takes on the qualities the animator
responds to (pitch swings, emphasis, attack, rate) so you can pick the
calmest read in seconds instead of after a long render:

```sh
.venv/bin/python prosody_check.py take1.wav take2.wav take3.wav
```

A workflow tip for editing projects: generate one clip per line or scene
rather than one long take. A 15 second clip takes about 20 minutes; if the
avatar does something odd you regenerate 15 seconds, not 5 minutes. Identity
stays consistent across clips because every clip is anchored to the same
photo.

## Hardware requirements (estimated)

Measured on real hardware where noted; the rest is arithmetic. All numbers
are for the int8 model at 480p, the default.

| Tier | VRAM | System RAM | What to expect |
|---|---|---|---|
| Floor (tested) | 16 GB (RTX 5070 Ti, 4080, etc.) | 32 GB | Verified end to end. Works via automatic low-VRAM mode; about half the model streams from RAM each step. 36 to 43 s per denoise step, 5 to 6 min per 3.2 s segment. |
| Comfortable | 24 GB (RTX 3090, 4090) | 32 GB | Whole model stays on the GPU, low-VRAM mode switches itself off. Roughly 1.5 to 2x faster per step on a 4090. |
| Roomy | 32 to 48 GB | 64 GB | 720p and/or the bf16 model (`--no-int8`) become realistic. This is the hardware the upstream authors appear to have assumed. |

Other requirements:

- **NVIDIA GPU only.** CUDA is load-bearing. RTX 50-series (Blackwell) is
  fully handled; setup detects it and installs the right torch and compiles
  flash-attention for it (a one-time 10 to 30 minute build that needs the
  CUDA toolkit installed, e.g. `pacman -S cuda`).
- **Disk:** budget ~100 GB. The Avatar 1.5 weights, the base model
  components it borrows, and the python environment add up.
- **Python 3.10 or 3.11**, or just have [uv](https://docs.astral.sh/uv/) on
  PATH and setup will provision a private 3.11 for the project. Newer
  pythons (3.13+) cannot work; upstream pins libraries that don't exist for
  them.
- **ffmpeg and git** on PATH.

For what it's worth: upstream never states hardware requirements, and their
example commands assume two datacenter GPUs. The 16 GB floor here exists
because this project rebuilt the loading and inference path around small
cards. On stock upstream code the same machine fails at six different
points, starting with needing ~54 GB of system RAM just to load the model.

## Why the output stays in sync (the sacred part)

- Avatar 1.5 generates at 25 fps. 25 fps means 40 ms per frame, which is
  exactly 4 centiseconds, and gif frame delays are stored in centiseconds.
  So gif timing is exact, with zero cumulative drift against your audio.
- `--fps` resampling keeps the guarantee. ffmpeg rounds absolute timestamps
  rather than per-frame deltas, so even an awkward rate like 29.97 gets
  alternating frame delays that never accumulate error. Rates that divide
  100 evenly (10, 20, 25, 50) are exact per frame. Avoid gifs above 50 fps;
  many players clamp 1-centisecond delays into slideshow territory. Use mp4
  there.
- Both formats are trimmed to the audio's exact duration. For mp4 the
  original audio is muxed back in untouched, and the video re-encodes to
  editor-friendly h264/yuv420p.
- Every run ends with a verification line comparing output duration to
  audio duration. Within one frame (40 ms) is reported as in sync.

## Options reference

```
talker {gif|mp4} <image> <audio> [options]

  -o, --output PATH     output file (default: output/<audio-stem>.<format>)
  --style PRESET        acting intensity: calm (default), natural, lively
  --prompt TEXT         custom scene and motion description; overrides
                        --style entirely
  --no-hands            append a strong "hands never appear" clause to the
                        prompt (stacks with --style or --prompt)
  --lip-scale S         lip-motion intensity. Nonlinear response: 1.0 to
                        ~0.7 look identical, 0.3 is a nearly still mouth;
                        useful range is roughly 0.4-0.6.
  --resolution {480p,720p}   default 480p
  --fps RATE            resample to your timeline rate: 30, 60, a fraction
                        like 30000/1001, or an alias: ntsc (29.97),
                        ntsc-film (23.976), ntsc60 (59.94), film (24),
                        pal (25). Default: native 25.
  --smooth              motion-interpolate the --fps resample instead of
                        duplicating frames (smoother, slower)
  --gif-width W         downscale the gif to width W (default: native)
  --no-vocal-sep        skip vocal separation; use for clean TTS/VO audio
  --steps N             override inference steps (default: distilled 8)
  --no-int8             full-precision DiT (needs 32 GB+ VRAM)
  --keep-workdir        keep the temp dir with raw model output
```

Environment knobs (mostly for small-card tuning):

```
TALKER_LOWVRAM=1|0          force low-VRAM mode on/off (auto: on below 20 GB)
TALKER_VRAM_RESERVE_GB=8.5  VRAM held back for activations; lower keeps more
                            model resident (faster) but risks OOM, higher is
                            safer but streams more. 8.5 is the measured safe
                            default for 16 GB cards.
TALKER_SKIP_VOCAL_SEP=1     same as --no-vocal-sep
```

## How it runs on 16 GB (the technical part)

The model is a 13.6B-parameter video diffusion transformer. Even quantized
to int8 the weights are ~14 GB, which does not leave room on a 16 GB card
for the text encoder, the audio encoder, the VAE, the distillation LoRA,
and the actual working memory of inference. Upstream's code assumes it all
fits. Ours doesn't assume.

All adaptations live in `talker_infer.py` as runtime monkeypatches; the
upstream code in `vendor/` is never modified. Each patch has a comment with
the measurement that motivated it. The short list:

1. **Low-RAM int8 loading.** Upstream materializes the full model in fp32
   (~54 GB of system RAM) before swapping in int8 weights. We build on the
   meta device and assign-load shards instead: peak RAM roughly equals
   checkpoint size.
2. **Lazy text encoder.** The 12 GB UMT5 encoder loads on CPU only for the
   first prompt encoding, the embeddings are cached (every segment reuses
   the same prompt), and it is freed immediately. It never touches VRAM.
3. **CPU audio encoding.** Whisper runs once up front; it runs on CPU and
   costs VRAM nothing.
4. **Partial block residency.** As many transformer blocks as fit stay on
   the GPU; the rest live in RAM and stream through per step (accelerate
   dispatch hooks). On a 5070 Ti that's 22 of 53 modules streaming, costing
   under a second per step.
5. **In-place LoRA accumulation.** Upstream's LoRA forward materialized two
   extra full-size output tensors (~2 GB at the qkv projection).
6. **Chunked 3D RoPE.** Rotary embeddings upcast q/k to fp32 with several
   full-size temporaries, a 4 to 5 GB spike. Applied 8 heads at a time it's
   under 1 GB, bit-identical.
7. **Chunked SwiGLU FFN.** Same idea for the feed-forward expansion,
   processed 16k tokens at a time.
8. **DiT eviction around the VAE.** The VAE's conv3d spikes ~4 GB while the
   transformer sits idle; its resident weights round-trip to RAM for those
   seconds.
9. **KV cache in RAM** for the segment-continuation pass, and **checkpoint
   saves** every 10th segment instead of upstream's O(n²) re-encode of the
   whole video after every segment.

Long clips are generated in chained segments (93 frames, then 80 new frames
per segment with a 13-frame conditioning overlap, re-anchored to your photo
each time), so clip length costs time but never memory. There is no length
limit beyond your patience.

## Troubleshooting

- **CUDA out of memory during generation:** raise the reserve, e.g.
  `TALKER_VRAM_RESERVE_GB=7.5 ./talker ...`. Also close anything using the
  GPU; a hardware-accelerated browser can hold 1 to 2 GB.
- **Process killed with no traceback (SIGKILL):** that's system RAM, not
  VRAM. Close things or add swap.
- **flash-attn fails its runtime check after install:** your card needs a
  source build; make sure `nvcc` is installed and re-run setup. The nvcc
  CUDA major version must match torch's (setup picks torch to match).
- **A run dies mid-way on a long clip:** the working directory is kept on
  failure and contains a checkpoint video every 10 segments. The path is
  printed at exit.
- **`onnxruntime` "cannot enable executable stack":** you have an old
  onnxruntime; `.venv/bin/pip install -U onnxruntime` (setup now does this).

## Layout

```
talker           launcher script (activates .venv, runs talker.py)
talker.py        CLI: engine dispatch, ffmpeg finalize, sync verify
talker_infer.py  low-memory longcat inference driver (all the patches above)
prosody_check.py score TTS takes for animation energy before rendering
setup.sh         one-time longcat env + vendor + weights
setup-ditto.sh   one-time ditto engine (optional, isolated)
vendor/          unmodified upstream checkouts        (gitignored)
weights/         model weights                        (gitignored)
.venv/ .venv-ditto/  python environments              (gitignored)
```

## Acknowledgements

talker is a CLI and a memory-surgery layer; the model doing the actual work
is [LongCat-Video-Avatar 1.5](https://github.com/meituan-longcat/LongCat-Video)
by the Meituan LongCat Team, released openly under MIT (code and weights).
Thank you for that. The
[Avatar 1.5 weights](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5)
and base-model components download from their Hugging Face repos at setup
time; `vendor/LongCat-Video` is an unmodified checkout of their code.

Also standing on: [flash-attention](https://github.com/Dao-AILab/flash-attention)
(Tri Dao et al.), [Whisper](https://github.com/openai/whisper) large-v3 as
the audio encoder, the Wan-family VAE, Hugging Face
accelerate/transformers/diffusers, and
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
with the Kim_Vocal_2 model for vocal isolation.

## Citation

If you use output from this tool in published work, cite the underlying
models:

```bibtex
@misc{meituanlongcatteam2025longcatvideotechnicalreport,
      title={LongCat-Video Technical Report},
      author={Meituan LongCat Team and Xunliang Cai and Qilong Huang and Zhuoliang Kang and Hongyu Li and Shijun Liang and Liya Ma and Siyu Ren and Xiaoming Wei and Rixu Xie and Tong Zhang},
      year={2025},
      eprint={2510.22200},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.22200},
}

@misc{meituanlongcatteam2026longcatvideoavatar15technicalreport,
      title={LongCat-Video-Avatar 1.5 Technical Report},
      author={Meituan LongCat Team and Xunliang Cai and Meng Cheng and Feng Gao and Zhe Kong and Jiamu Li and Le Li and Weiheng Li and Hongyu Liu and Shuai Tan and Xiaoming Wei and Tianyu Yang and Yong Zhang},
      year={2026},
      eprint={2605.26486},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.26486},
}
```
