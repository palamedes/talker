#!/usr/bin/env python3
"""Low-memory inference driver for LongCat-Video-Avatar 1.5.

Wraps the upstream demo (run_demo_avatar_single_audio_to_video.py) with
monkeypatches that make it survive consumer hardware (tested target:
16 GB VRAM / 32 GB RAM). Run via torchrun with cwd = vendor/LongCat-Video,
same CLI as the upstream demo.

Upstream problems fixed here, without editing vendor code:

1. load_quantized_dit materializes the FULL 13.6B DiT in fp32 on CPU
   (~54 GB RAM) before swapping in int8 buffers -> OOM-killed on any
   normal machine. Replaced with a meta-device build + shard-wise
   assign-loading: peak RAM ~ int8 checkpoint size (~14 GB).

2. The UMT5 text encoder (~12 GB RAM) is loaded eagerly at startup and
   pipe.to() then moves it onto the GPU (it alone would eat most of a
   16 GB card). Replaced with a lazy stub: the real encoder is loaded on
   CPU only for the first prompt encoding, embeddings are cached (every
   segment reuses the same prompt), and the encoder is freed immediately.

3. The Whisper audio encoder is moved to the GPU alongside the DiT.
   Audio embedding is computed exactly once up front, so we keep Whisper
   on CPU: slower for that one step, zero VRAM for the rest of the run.

4. pipe.to() therefore only moves the DiT (+ LoRAs) and the VAE to the
   GPU. The VAE decodes with feat_cache streaming, so steady-state VRAM
   is DiT weights + activations.
"""

import gc
import os
import sys
import types

# Running as a script from another directory: put the vendor checkout
# (our cwd) on sys.path so longcat_video / the demo module import.
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn

import run_demo_avatar_single_audio_to_video as demo
from longcat_video import pipeline_longcat_video_avatar as pl
from longcat_video.modules.quantization import QuantizedLinear, DEFAULT_SKIP_PATTERNS


# --------------------------------------------------------------------------
# 1. Low-RAM int8 DiT loader (replaces quantization.load_quantized_dit)
# --------------------------------------------------------------------------

def load_quantized_dit_lowmem(checkpoint_dir, subfolder="base_model_int8", **kwargs):
    import json
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from longcat_video.modules.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )

    quantized_dir = os.path.join(checkpoint_dir, subfolder)
    with open(os.path.join(quantized_dir, "config.json")) as f:
        config = json.load(f)
    for key in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(key, None)
    config.update(kwargs)

    # Parameters on meta (no RAM); module-level buffers (rope caches etc.)
    # stay real since they may be non-persistent and absent from the shards.
    with init_empty_weights(include_buffers=False):
        model = LongCatVideoAvatarTransformer3DModel(**config)

    # Swap Linears for QuantizedLinear with META buffers — they are all
    # persistent and present in the shards, so assign-loading fills them.
    to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(p in name for p in DEFAULT_SKIP_PATTERNS):
            with init_empty_weights(include_buffers=True):
                to_replace[name] = QuantizedLinear(
                    module.in_features, module.out_features, bias=module.bias is not None)
    for name, ql in to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], ql)

    # Shard-wise assign-load: peak RAM = one shard + tensors assigned so far.
    index_path = os.path.join(quantized_dir, "quantized_model.safetensors.index.json")
    if os.path.exists(index_path):
        import json as _json
        with open(index_path) as f:
            shard_files = sorted(set(_json.load(f)["weight_map"].values()))
    else:
        shard_files = sorted(f for f in os.listdir(quantized_dir)
                             if f.endswith(".safetensors") and "index" not in f)
    for shard_file in shard_files:
        shard = load_file(os.path.join(quantized_dir, shard_file), device="cpu")
        model.load_state_dict(shard, strict=False, assign=True)
        del shard
        gc.collect()

    leftover = [n for n, p in list(model.named_parameters()) + list(model.named_buffers())
                if p.is_meta]
    if leftover:
        raise RuntimeError(f"int8 checkpoint did not cover: {leftover[:10]} ...")

    model.eval()
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            continue
        for _, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
    return model


demo.load_quantized_dit = load_quantized_dit_lowmem


# --------------------------------------------------------------------------
# 2. Lazy text encoder: load on CPU at first use, cache embeds, free
# --------------------------------------------------------------------------

class _LazyTextEncoder:
    dtype = torch.bfloat16

    def __init__(self, path, subfolder, torch_dtype):
        self.path, self.subfolder = path, subfolder
        self.torch_dtype = torch_dtype or torch.bfloat16

    def to(self, *args, **kwargs):  # pipe.to() no-op
        return self

    def load(self):
        from transformers import UMT5EncoderModel
        print("[talker] loading UMT5 text encoder on CPU (one-time)...")
        te = UMT5EncoderModel.from_pretrained(
            self.path, subfolder=self.subfolder, torch_dtype=self.torch_dtype)
        te.eval()
        return te


class _LazyUMT5Factory:
    @staticmethod
    def from_pretrained(path, subfolder=None, torch_dtype=None):
        return _LazyTextEncoder(path, subfolder, torch_dtype)


demo.UMT5EncoderModel = _LazyUMT5Factory

_PROMPT_CACHE = {}


def _patched_get_t5_prompt_embeds(self, prompt=None, num_videos_per_prompt=1,
                                  max_sequence_length=512, device=None, dtype=None):
    dtype = dtype or torch.bfloat16
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    prompts = [pl.prompt_clean(u) for u in prompts]
    key = (tuple(prompts), max_sequence_length)

    if key not in _PROMPT_CACHE:
        te = (self.text_encoder.load()
              if isinstance(self.text_encoder, _LazyTextEncoder) else self.text_encoder)
        inputs = self.tokenizer(
            prompts, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            emb = te(inputs.input_ids, inputs.attention_mask).last_hidden_state
        _PROMPT_CACHE[key] = (emb.to(torch.bfloat16), inputs.attention_mask)
        if isinstance(self.text_encoder, _LazyTextEncoder):
            del te
            gc.collect()

    emb, mask = _PROMPT_CACHE[key]
    batch_size = len(prompts)
    prompt_embeds = emb.to(dtype=dtype, device=device)
    mask = mask.to(device=device)
    mask = torch.cat([mask] * num_videos_per_prompt, dim=0)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)
    return prompt_embeds, mask


pl.LongCatVideoAvatarPipeline._get_t5_prompt_embeds = _patched_get_t5_prompt_embeds


# --------------------------------------------------------------------------
# 3. Whisper stays on CPU; audio embedding computed on CPU (once)
# --------------------------------------------------------------------------

_orig_get_audio_encoder = demo.get_audio_encoder


def _cpu_audio_encoder(path, model_type):
    enc = _orig_get_audio_encoder(path, model_type)
    enc.eval()
    enc.to = types.MethodType(lambda self, *a, **k: self, enc)  # neuter .to(gpu)
    return enc


demo.get_audio_encoder = _cpu_audio_encoder

_orig_get_audio_embedding = pl.LongCatVideoAvatarPipeline.get_audio_embedding


def _patched_get_audio_embedding(self, speech_array, fps=32, device="cpu",
                                 sample_rate=16000, model_type="avatar-v1.0"):
    print("[talker] computing audio embedding on CPU (Whisper, one-time)...")
    return _orig_get_audio_embedding(
        self, speech_array, fps=fps, device="cpu",
        sample_rate=sample_rate, model_type=model_type)


pl.LongCatVideoAvatarPipeline.get_audio_embedding = _patched_get_audio_embedding


# --------------------------------------------------------------------------
# 4. pipe.to(): move only DiT (+ LoRAs) and VAE to the GPU
# --------------------------------------------------------------------------

def _patched_to(self, device):
    self.device = device
    if self.dit is not None:
        self.dit = self.dit.to(device, non_blocking=True)
        if hasattr(self.dit, "lora_dict") and self.dit.lora_dict:
            for lora_network in self.dit.lora_dict.values():
                for lora in lora_network.loras:
                    lora.to(device, non_blocking=True)
    if self.vae is not None:
        self.vae = self.vae.to(device, non_blocking=True)
    # text_encoder: lazy CPU stub (patch 2); audio_encoder: CPU (patch 3)
    return self


pl.LongCatVideoAvatarPipeline.to = _patched_to


if __name__ == "__main__":
    args = demo._parse_args()
    demo.generate(args)
