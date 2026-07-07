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
INFER_SCRIPT = "run_demo_avatar_single_audio_to_video.py"

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

    info(f"running LongCat inference ({resolution}, "
         f"{'int8' if use_int8 else 'bf16'}, distilled)...")
    info("  " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=VENDOR)
    if proc.returncode != 0:
        die(f"inference failed (exit {proc.returncode})")

    videos = sorted(outdir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        die(f"inference produced no .mp4 in {outdir}")
    return videos[-1]


def finalize_mp4(gen: Path, audio: Path, dur: float, out: Path):
    # Re-encode so we can trim to the exact audio duration (stream-copy cuts
    # only on keyframes), mux the ORIGINAL audio back in, and end up with an
    # editor-friendly h264/yuv420p file.
    subprocess.run([
        "ffmpeg", "-y", "-v", "warning", "-stats",
        "-i", str(gen), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{dur:.6f}",
        "-movflags", "+faststart",
        str(out),
    ], check=True)


def finalize_gif(gen: Path, dur: float, out: Path, width: int | None) -> Fraction:
    fps = video_fps(gen)
    # GIF frame delays are in centiseconds. Warn if the model fps doesn't
    # land on an exact centisecond boundary (25 fps -> 4 cs, always exact).
    frame_cs = Fraction(100) / fps
    if frame_cs.denominator != 1:
        info(f"warning: source fps {fps} is not centisecond-exact in GIF "
             f"timing; resampling to 25 fps to guarantee sync")
        fps = Fraction(25)
    scale = f"scale={width}:-2:flags=lanczos," if width else ""
    vf = (f"fps={fps},{scale}"
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
    if not args.image.is_file():
        die(f"image not found: {args.image}")
    if not args.audio.is_file():
        die(f"audio not found: {args.audio}")

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
            finalize_mp4(gen, audio, dur, out)
            verify(out, dur)
        else:
            gif_fps = finalize_gif(gen, dur, out, args.gif_width)
            verify(out, dur, gif_fps)
    finally:
        if args.keep_workdir:
            info(f"workdir kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(out)


if __name__ == "__main__":
    main()
