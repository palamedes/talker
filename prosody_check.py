#!/usr/bin/env python3
"""prosody_check — score speech wavs on the qualities that drive avatar
mouth/face animation energy, so you can pre-screen TTS takes in seconds
instead of discovering the hot ones after a 7-minute render.

Usage (from the talker repo, uses the project venv's librosa):

    .venv/bin/python prosody_check.py take1.wav take2.wav take3.wav

What it measures (all of these survive loudness normalization, which is
why absolute volume does NOT appear here — the animator can't hear it):

  pitch_var   pitch variability in semitones. Big intonation swings read
              as "excited"; flat delivery reads calm.
  emphasis    dynamic contrast (dB std over speech frames). Punched words
              vs. even delivery.
  attacks/s   onset event rate — roughly syllables per second. Faster
              speech animates harder.
  sharpness   how percussive the onsets are (peak/median onset strength).
              Hard consonant attacks open the mouth wide.

With multiple files it also prints a relative "energy" score (mean z-score
across metrics within the batch) and sorts calmest first. Scores are only
comparable within one invocation.
"""

import sys

import numpy as np
import librosa

SR = 16000


def analyze(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=35)
    if len(y) < SR // 2:
        raise ValueError("less than half a second of audio after trimming")
    dur = len(y) / SR

    # --- pitch variability (semitones over voiced frames) ---
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=SR, frame_length=1024)
    f0 = f0[np.asarray(voiced, dtype=bool)]
    f0 = f0[np.isfinite(f0)]
    pitch_var = float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 10 else float("nan")

    # --- emphasis: dB contrast across speech-active frames ---
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    db = 20 * np.log10(rms + 1e-9)
    active = db > (db.max() - 40)  # gate out silence between phrases
    emphasis = float(np.std(db[active])) if active.sum() > 10 else float("nan")

    # --- onsets: rate (~syllables/s) and sharpness ---
    onset_env = librosa.onset.onset_strength(y=y, sr=SR)
    peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, units="frames")
    attacks_per_s = len(peaks) / dur
    med = np.median(onset_env[onset_env > 0]) if (onset_env > 0).any() else 1.0
    sharpness = float(np.mean(onset_env[peaks]) / med) if len(peaks) else float("nan")

    return {
        "file": path,
        "dur": dur,
        "pitch_var": pitch_var,
        "emphasis": emphasis,
        "attacks/s": attacks_per_s,
        "sharpness": sharpness,
    }


def main(paths):
    results = []
    for p in paths:
        try:
            results.append(analyze(p))
        except Exception as e:  # noqa: BLE001 — report and continue
            print(f"skipping {p}: {e}", file=sys.stderr)
    if not results:
        sys.exit(1)

    metrics = ["pitch_var", "emphasis", "attacks/s", "sharpness"]

    if len(results) > 1:
        # relative energy: mean z-score across metrics, within this batch
        for m in metrics:
            vals = np.array([r[m] for r in results], dtype=float)
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            for r in results:
                r.setdefault("_z", []).append(
                    0.0 if sd == 0 or np.isnan(r[m]) else (r[m] - mu) / sd)
        for r in results:
            r["energy"] = float(np.mean(r["_z"]))
        results.sort(key=lambda r: r["energy"])

    header = f"{'file':<28} {'dur':>6} {'pitch_var':>9} {'emphasis':>8} {'attacks/s':>9} {'sharpness':>9}"
    if len(results) > 1:
        header += f" {'energy':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        line = (f"{r['file']:<28} {r['dur']:>5.1f}s {r['pitch_var']:>9.2f} "
                f"{r['emphasis']:>8.2f} {r['attacks/s']:>9.2f} {r['sharpness']:>9.2f}")
        if len(results) > 1:
            line += f" {r['energy']:>+7.2f}"
        print(line)

    if len(results) > 1:
        print("\ncalmest read is on top; feed that one to talker.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
