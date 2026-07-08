#!/usr/bin/env python3
"""Ditto engine driver for talker.

Upstream's inference.py supports setup kwargs (emotion index, sampling
steps, crop options) in its run() helper but doesn't expose them on the
CLI. This thin wrapper does. Run from vendor/ditto-talkinghead as cwd,
inside .venv-ditto.

Emotion indices (8-way, alphabetical affect ordering):
  0 anger, 1 contempt, 2 disgust, 3 fear, 4 happiness (upstream default),
  5 neutral, 6 sadness, 7 surprise
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # vendor/ditto-talkinghead

from inference import run, seed_everything  # noqa: E402
from stream_pipeline_offline import StreamSDK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cfg_pkl", required=True)
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--emo", type=int, default=None,
                    help="emotion index 0-7 (upstream default: 4/happiness)")
    ap.add_argument("--sampling_timesteps", type=int, default=None,
                    help="motion diffusion steps (upstream default: 50)")
    ap.add_argument("--seed", type=int, default=1024)
    args = ap.parse_args()

    setup_kwargs = {}
    if args.emo is not None:
        setup_kwargs["emo"] = args.emo
    if args.sampling_timesteps is not None:
        setup_kwargs["sampling_timesteps"] = args.sampling_timesteps

    seed_everything(args.seed)
    sdk = StreamSDK(args.cfg_pkl, args.data_root)
    run(sdk, args.audio_path, args.source_path, args.output_path,
        more_kwargs={"setup_kwargs": setup_kwargs, "run_kwargs": {}})


if __name__ == "__main__":
    main()
