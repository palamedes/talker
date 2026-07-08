#!/usr/bin/env python3
"""EchoMimicV3-Flash engine driver for talker.

Fuses the two halves upstream ships separately:
  - infer_flash.py loads the Flash models correctly but generates a single
    window, TRUNCATING audio longer than --video_length (~3.2 s).
  - app_mm.py (their gradio app) has the long-video windowed loop with
    8-frame cross-fade chaining, plus the mmgp offload call that makes the
    whole pipeline fit in ~12 GB VRAM, but targets the non-flash models.

This driver = flash loading + mmgp offload + the windowed loop, with the
flash-style per-frame audio embedding sliced per window. Run with
cwd = vendor/echomimic_v3, inside .venv-emv3.

Outputs a video-only 25 fps mp4; talker's shared finalize step muxes the
original audio and enforces the sync guarantees.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.getcwd())  # vendor/echomimic_v3

import numpy as np
import torch

# Upstream's custom from_pretrained imports load_model_dict_into_meta from
# diffusers.models.modeling_utils; newer diffusers moved it to
# model_loading_utils. Without this shim their loader silently falls back
# to low_cpu_mem_usage=False and the umt5-xxl text encoder gets the
# process OOM-killed on ordinary machines.
import diffusers.models.modeling_utils as _dmu
if not hasattr(_dmu, "load_model_dict_into_meta"):
    try:
        from diffusers.models.model_loading_utils import load_model_dict_into_meta as _lmdim
        _dmu.load_model_dict_into_meta = _lmdim
        print("[talker] shimmed load_model_dict_into_meta for new diffusers")
    except ImportError:
        print("[talker] warning: no load_model_dict_into_meta anywhere; "
              "loading will use more RAM (pin diffusers==0.30.1 if OOM-killed)")

from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
import librosa
import pyloudnorm as pyln
from einops import rearrange
from mmgp import offload, profile_type

from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
from src.utils import filter_kwargs, save_videos_grid
try:
    from src.utils import get_image_to_video_latent3 as get_i2v_latent
except ImportError:
    from src.utils import get_image_to_video_latent2 as get_i2v_latent
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from src.cache_utils import get_teacache_coefficients
from src.wav2vec2 import Wav2Vec2Model

OVERLAP = 8  # frames cross-faded between windows (app_mm default)

DEFAULT_NEG = ("Gesture is bad. Gesture is unclear. Strange and twisted "
               "hands. Bad hands. Bad fingers. Unclear and blurry hands. "
               "Unclear gestures, broken hands, fused fingers.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_root", required=True)
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--audio_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--prompt", default="A person is speaking calmly.")
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--num_inference_steps", type=int, default=8)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--audio_guidance_scale", type=float, default=3.0)
    ap.add_argument("--audio_scale", type=float, default=1.0)
    ap.add_argument("--partial_video_length", type=int, default=81)
    ap.add_argument("--sample_size", type=int, nargs=2, default=[768, 768])
    ap.add_argument("--sampler_name", default="Flow_Unipc",
                    choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max_vram_frac", type=float, default=0.75)
    ap.add_argument("--teacache_threshold", type=float, default=0.1)
    return ap.parse_args()


def get_sample_size(pil_img, sample_size):
    # from infer_flash.py: fit the model resolution to the image aspect
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])
        w = w / ratio_a // 16 * 16
        h = h / ratio_a // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16
    return [int(h), int(w)]


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def get_audio_embed(mel_input, extractor, encoder, video_length, sr=16000):
    # from infer_flash.py: wav2vec features for the whole clip at once
    feat = np.squeeze(extractor(mel_input, sampling_rate=sr).input_values)
    feat = torch.from_numpy(feat).float().unsqueeze(0)
    with torch.no_grad():
        emb = encoder(feat, seq_len=int(video_length), output_hidden_states=True)
    emb = torch.stack(emb.hidden_states[1:], dim=1).squeeze(0)
    return rearrange(emb, "b s d -> s b d").cpu().detach()


def main():
    args = parse_args()
    device = "cuda"
    dtype = torch.bfloat16
    wroot = args.weights_root
    model_name = os.path.join(wroot, "Wan2.1-Fun-V1.1-1.3B-InP")
    wav2vec_dir = os.path.join(wroot, "chinese-wav2vec2-base")

    # locate the flash transformer safetensors
    flash_dir = os.path.join(wroot, "echomimicv3-flash-pro")
    transformer_path = None
    for root, _, files in os.walk(flash_dir):
        for f in files:
            if f.endswith(".safetensors"):
                transformer_path = os.path.join(root, f)
    if transformer_path is None:
        sys.exit(f"no flash transformer .safetensors under {flash_dir}")

    config = OmegaConf.load("config/config.yaml")

    print("[talker] loading EchoMimicV3-Flash models...")
    transformer = WanTransformer.from_pretrained(
        os.path.join(model_name, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=dtype)
    from safetensors.torch import load_file
    state_dict = load_file(transformer_path)
    state_dict = state_dict.get("state_dict", state_dict)
    # assign=True: with meta-based loading the audio-injection layers (absent
    # from the base checkpoint) are meta tensors, and a copying load is a
    # silent no-op onto meta. The flash checkpoint is complete, so assign
    # materializes every parameter.
    m, u = transformer.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[talker] flash transformer loaded (missing {len(m)}, unexpected {len(u)})")
    leftover = [n for n, p in transformer.named_parameters() if p.is_meta]
    if leftover:
        sys.exit(f"[talker] still-meta parameters after flash load: {leftover[:8]}")

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=dtype).eval()
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    ).to(dtype).eval()

    scheduler_cls = {"Flow": FlowMatchEulerDiscreteScheduler,
                     "Flow_Unipc": FlowUniPCMultistepScheduler,
                     "Flow_DPM++": FlowDPMSolverMultistepScheduler}[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, OmegaConf.to_container(config["scheduler_kwargs"])))

    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer, vae=vae, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=scheduler,
        clip_image_encoder=clip_image_encoder)

    # mmgp: fit the whole pipeline (incl. the umt5-xxl text encoder) into a
    # VRAM budget: this is upstream's documented 12 GB path (app_mm.py)
    budget_mb = int(torch.cuda.get_device_properties(0).total_memory / 1048576 * args.max_vram_frac)
    print(f"[talker] mmgp offload, VRAM budget {budget_mb} MB")
    offload.profile(pipeline, profile_type.LowRAM_HighVRAM, budgets={"*": budget_mb})

    coefficients = get_teacache_coefficients(model_name)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ---- audio: full-clip wav2vec embedding, flash-style ----
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to("cpu")
    audio_encoder.feature_extractor._freeze_parameters()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)

    fps = 25
    mel_input, sr = librosa.load(args.audio_path, sr=16000)
    mel_input = loudness_norm(mel_input, sr)
    total_frames = int(len(mel_input) / sr * fps)
    tcr = vae.config.temporal_compression_ratio
    total_frames = int((total_frames - 1) // tcr * tcr) + 1
    print(f"[talker] {total_frames} frames total @ {fps} fps")

    emb = get_audio_embed(mel_input, extractor, audio_encoder, total_frames, sr)
    # explicit cpu: mmgp sets a global default device of cuda, which would
    # otherwise put these index tensors on the GPU while emb is on cpu
    idx = (torch.arange(5, device="cpu") - 2)
    centers = torch.arange(0, total_frames, 1, device="cpu").unsqueeze(1) + idx.unsqueeze(0)
    centers = torch.clamp(centers, min=0, max=emb.shape[0] - 1)
    audio_embeds_full = emb[centers]  # [F, 5, 12, 768]

    # ---- windowed generation with cross-fade chaining (app_mm-style) ----
    image = Image.open(args.image_path).convert("RGB")
    h, w = get_sample_size(image, args.sample_size)
    print(f"[talker] sample size {w}x{h}")

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        args.prompt, args.negative_prompt, dtype=dtype)

    _, _, clip_image = get_i2v_latent(image, None, video_length=args.partial_video_length,
                                      sample_size=[h, w])

    ref_img = image
    init_frames = 0
    partial = args.partial_video_length
    new_sample = None
    window = 0
    while init_frames < total_frames:
        if init_frames + partial >= total_frames:
            partial = total_frames - init_frames
            partial = int((partial - 1) // tcr * tcr) + 1 if partial != 1 else 1
            if partial <= 0:
                break

        window += 1
        n_windows = math.ceil((total_frames - OVERLAP) / (args.partial_video_length - OVERLAP))
        print(f"[talker] window {window}/{n_windows} "
              f"(frames {init_frames}..{init_frames + partial})")

        input_video, input_video_mask, _ = get_i2v_latent(
            ref_img, None, video_length=partial, sample_size=[h, w])

        a = audio_embeds_full[init_frames:init_frames + partial]
        a = a.unsqueeze(0).to(device=device, dtype=dtype)

        pipeline.transformer.enable_teacache(
            coefficients, args.num_inference_steps, args.teacache_threshold,
            num_skip_start_steps=5, offload=False)

        with torch.no_grad():
            sample = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_frames=partial,
                audio_embeds=a,
                audio_scale=args.audio_scale,
                ip_mask=None,
                use_un_ip_mask=False,
                height=h, width=w,
                generator=generator,
                neg_scale=1.0, neg_steps=0,
                use_dynamic_cfg=False, use_dynamic_acfg=False,
                guidance_scale=args.guidance_scale,
                audio_guidance_scale=args.audio_guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=input_video, mask_video=input_video_mask,
                clip_image=clip_image,
                cfg_skip_ratio=0.0, shift=args.shift,
            ).videos

        if init_frames != 0:
            mix = torch.from_numpy(
                np.array([i / OVERLAP for i in range(OVERLAP)], np.float32)
            ).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            new_sample[:, :, -OVERLAP:] = (
                new_sample[:, :, -OVERLAP:] * (1 - mix) + sample[:, :, :OVERLAP] * mix)
            new_sample = torch.cat([new_sample, sample[:, :, OVERLAP:]], dim=2)
        else:
            new_sample = sample

        if init_frames + partial >= total_frames:
            break

        ref_img = [
            Image.fromarray(
                (new_sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).numpy().astype(np.uint8))
            for i in range(-OVERLAP, 0)
        ]
        init_frames += partial - OVERLAP

    print(f"[talker] saving {new_sample.shape[2]} frames")
    save_videos_grid(new_sample, args.output_path, fps=fps)
    print(args.output_path)


if __name__ == "__main__":
    main()
