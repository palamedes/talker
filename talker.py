#!/usr/bin/env python3
"""talker — animate a portrait to speak an audio file, via LongCat-Video-Avatar 1.5.

Usage:  talker {gif|mp4} <image> <audio.wav> [options]

Pipeline:
  1. Build a temp input JSON (prompt + cond_image + cond_audio) for LongCat.
  2. Run vendor/LongCat-Video/run_demo_avatar_single_audio_to_video.py (ai2v,
     distilled 8-step, avatar-v1.5) on the GPU.
  3. Post-process with ffmpeg so the output duration matches the input audio
     exactly:
       mp4 -> re-encode h264/yuv420p, trimmed to audio duration, original
              audio muxed back in losslessly-in-sync.
       gif -> palette-based conversion at the model's native 25 fps
              (40 ms/frame = 4 centiseconds: exactly representable in GIF
              timing, so there is zero cumulative drift against the audio).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "LongCat-Video"
WEIGHTS = ROOT / "weights" / "LongCat-Video-Avatar-1.5"
# The avatar pipeline loads tokenizer/text_encoder/vae from <WEIGHTS>/../LongCat-Video
WEIGHTS_BASE = ROOT / "weights" / "LongCat-Video"
INFER_SCRIPT = "run_demo_avatar_single_audio_to_video.py"

# Editors that say "29.97" / "23.976" / "59.94" mean these exact ratios.
FPS_ALIASES = {
    "ntsc": Fraction(30000, 1001),       # 29.97
    "ntsc-film": Fraction(24000, 1001),  # 23.976
    "ntsc60": Fraction(60000, 1001),     # 59.94
    "film": Fraction(24),
    "pal": Fraction(25),
}

# Literal decimals users type that are almost certainly meant as NTSC rates.
FPS_LOOKALIKES = {
    Fraction("29.97"): "ntsc",
    Fraction("23.976"): "ntsc-film",
    Fraction("59.94"): "ntsc60",
}


def parse_fps(s: str) -> Fraction:
    return FPS_ALIASES.get(s.lower()) or Fraction(s)


DEFAULT_PROMPT = (
    "A person looks directly at the camera and speaks naturally, with subtle "
    "head movement and natural facial expressions. The background stays static."
)


def die(msg: str, code: int = 1):
    print(f"talker: error: {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str):
    print(f"talker: {msg}", file=sys.stderr)


def need(binary: str):
    if shutil.which(binary) is None:
        die(f"'{binary}' not found on PATH — install it and retry")


def ffprobe_value(path: Path, *entries: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", *entries, "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return out


def audio_duration(path: Path) -> float:
    val = ffprobe_value(path, "-show_entries", "format=duration")
    try:
        return float(val)
    except ValueError:
        die(f"could not determine audio duration of {path}")


def video_fps(path: Path) -> Fraction:
    val = ffprobe_value(
        path, "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate"
    )
    return Fraction(val)


def run_inference(image: Path, audio: Path, prompt: str, resolution: str,
                  use_int8: bool, steps: int | None, workdir: Path) -> Path:
    input_json = workdir / "input.json"
    outdir = workdir / "out"
    outdir.mkdir()
    input_json.write_text(json.dumps({
        "prompt": prompt,
        "cond_image": str(image),
        "cond_audio": {"person1": str(audio)},
    }, indent=2))

    cmd = [
        "torchrun", "--nproc_per_node=1", INFER_SCRIPT,
        "--context_parallel_size=1",
        f"--checkpoint_dir={WEIGHTS}",
        "--stage_1=ai2v",
        f"--input_json={input_json}",
        f"--output_dir={outdir}",
        "--use_distill",
        "--model_type", "avatar-v1.5",
        "--resolution", resolution,
    ]
    if use_int8:
        cmd.append("--use_int8")
    if steps is not None:
        cmd += ["--num_inference_steps", str(steps)]

    env = os.environ.copy()
    # Reduces fragmentation OOMs on VRAM-tight cards (e.g. 16 GB).
    # (torch >= 2.9 renamed the variable; set both, old name wins if user set it)
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    info(f"running LongCat inference ({resolution}, "
         f"{'int8' if use_int8 else 'bf16'}, distilled)...")
    info("  " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=VENDOR, env=env)
    if proc.returncode != 0:
        die(f"inference failed (exit {proc.returncode})")

    videos = sorted(outdir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        die(f"inference produced no .mp4 in {outdir}")
    return videos[-1]


def fps_filter(fps: Fraction, smooth: bool) -> str:
    if smooth:
        # Motion-compensated interpolation: synthesizes in-between frames
        # instead of duplicating — much smoother for e.g. 25 -> 30/60,
        # noticeably slower to encode.
        return (f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:"
                f"me_mode=bidir:vsbmc=1")
    return f"fps={fps}"


def finalize_mp4(gen: Path, audio: Path, dur: float, out: Path,
                 fps: Fraction | None, smooth: bool):
    # Re-encode so we can trim to the exact audio duration (stream-copy cuts
    # only on keyframes), mux the ORIGINAL audio back in, and end up with an
    # editor-friendly h264/yuv420p file.
    vf = ["-vf", fps_filter(fps, smooth)] if fps else []
    subprocess.run([
        "ffmpeg", "-y", "-v", "warning", "-stats",
        "-i", str(gen), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        *vf,
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{dur:.6f}",
        "-movflags", "+faststart",
        str(out),
    ], check=True)


def finalize_gif(gen: Path, dur: float, out: Path, width: int | None,
                 fps_arg: Fraction | None, smooth: bool) -> Fraction:
    fps = fps_arg or video_fps(gen)
    # GIF frame delays are stored in whole centiseconds. fps that divides 100
    # (10, 20, 25, 50) is exact per frame; anything else gets alternating
    # delays (e.g. 30 fps -> 3cs/4cs) — ffmpeg rounds absolute timestamps,
    # not deltas, so there is still ZERO cumulative drift, just per-frame
    # jitter of up to half a centisecond.
    frame_cs = Fraction(100) / fps
    if frame_cs.denominator != 1:
        if fps_arg:
            info(f"note: {fps} fps is not centisecond-exact in GIF timing — "
                 f"frame delays will alternate around {float(frame_cs):.2f}cs "
                 f"(no cumulative drift; 10/20/25/50 fps are exact)")
        else:
            info(f"warning: source fps {fps} is not centisecond-exact in GIF "
                 f"timing; resampling to 25 fps to guarantee sync")
            fps = Fraction(25)
    if fps > 50:
        info("warning: >50 fps means GIF frame delays of 1cs, which many "
             "players clamp to 10cs (slideshow!) — mp4 is safer here")
    scale = f"scale={width}:-2:flags=lanczos," if width else ""
    vf = (f"{fps_filter(fps, smooth)},{scale}"
          f"split[a][b];[a]palettegen=stat_mode=diff[p];"
          f"[b][p]paletteuse=dither=sierra2_4a")
    subprocess.run([
        "ffmpeg", "-y", "-v", "warning", "-stats",
        "-i", str(gen),
        "-t", f"{dur:.6f}",
        "-vf", vf,
        "-loop", "0",
        str(out),
    ], check=True)
    return fps


def verify(out: Path, dur: float, gif_fps: Fraction | None = None):
    if out.suffix == ".gif":
        frames = int(ffprobe_value(
            out, "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames"))
        got = frames / float(gif_fps or 25)
    else:
        got = float(ffprobe_value(
            out, "-select_streams", "v:0",
            "-show_entries", "stream=duration"))
    delta_ms = (got - dur) * 1000
    info(f"audio {dur:.3f}s | output {got:.3f}s | delta {delta_ms:+.1f}ms "
         f"({'< 1 frame, in sync' if abs(delta_ms) <= 40 else 'CHECK SYNC'})")


def main():
    ap = argparse.ArgumentParser(
        prog="talker",
        description="Animate a portrait image to speak an audio file "
                    "(LongCat-Video-Avatar 1.5).")
    ap.add_argument("format", choices=["gif", "mp4"],
                    help="output container (gif = video only, mp4 = with audio)")
    ap.add_argument("image", type=Path, help="portrait image (png/jpg)")
    ap.add_argument("audio", type=Path, help="speech audio (wav/mp3/...)")
    ap.add_argument("-o", "--output", type=Path,
                    help="output path (default: <audio-stem>.<format>)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="scene/motion prompt fed to the model")
    ap.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    ap.add_argument("--no-int8", action="store_true",
                    help="run full-precision DiT (needs more VRAM)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override inference steps (default: distilled 8)")
    ap.add_argument("--fps", type=parse_fps, default=None, metavar="RATE",
                    help="resample output to this frame rate to match your "
                         "editor timeline: 30, 60, 30000/1001, or an alias "
                         "(ntsc=29.97, ntsc-film=23.976, ntsc60=59.94, "
                         "film=24, pal=25); default: model-native 25")
    ap.add_argument("--smooth", action="store_true",
                    help="motion-interpolate when resampling --fps instead "
                         "of duplicating frames (smoother, slower)")
    ap.add_argument("--gif-width", type=int, default=None,
                    help="downscale gif to this width (default: native)")
    ap.add_argument("--keep-workdir", action="store_true",
                    help="keep the temp working dir (raw model output)")
    args = ap.parse_args()

    for binary in ("ffmpeg", "ffprobe", "torchrun"):
        need(binary)
    if not (VENDOR / INFER_SCRIPT).exists():
        die(f"LongCat-Video not found at {VENDOR} — run ./setup.sh first")
    if not WEIGHTS.exists():
        die(f"weights not found at {WEIGHTS} — run ./setup.sh first")
    if not (WEIGHTS_BASE / "text_encoder").exists():
        die(f"base model components not found at {WEIGHTS_BASE} "
            f"(tokenizer/text_encoder/vae) — re-run ./setup.sh to fetch them")
    if not args.image.is_file():
        die(f"image not found: {args.image}")
    if not args.audio.is_file():
        die(f"audio not found: {args.audio}")

    if args.fps is not None and args.fps <= 0:
        die("--fps must be positive")
    if args.fps in FPS_LOOKALIKES:
        alias = FPS_LOOKALIKES[args.fps]
        info(f"note: taking --fps {float(args.fps)} literally; if your editor "
             f"means NTSC {FPS_ALIASES[alias]}, use --fps {alias} "
             f"(drift is ~1ms per 17min either way)")
    if args.smooth and not args.fps:
        info("--smooth has no effect without --fps; ignoring")
        args.smooth = False

    image = args.image.resolve()
    audio = args.audio.resolve()
    out = (args.output or Path(f"{args.audio.stem}.{args.format}")).resolve()
    dur = audio_duration(audio)
    info(f"audio duration: {dur:.3f}s")

    workdir = Path(tempfile.mkdtemp(prefix="talker-"))
    try:
        gen = run_inference(image, audio, args.prompt, args.resolution,
                            not args.no_int8, args.steps, workdir)
        info(f"raw model output: {gen}")
        if args.format == "mp4":
            finalize_mp4(gen, audio, dur, out, args.fps, args.smooth)
            verify(out, dur)
        else:
            gif_fps = finalize_gif(gen, dur, out, args.gif_width,
                                   args.fps, args.smooth)
            verify(out, dur, gif_fps)
    finally:
        if args.keep_workdir:
            info(f"workdir kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(out)


if __name__ == "__main__":
    main()
