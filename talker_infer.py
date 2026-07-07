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
import itertools
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
        self._config = None

    @property
    def config(self):
        # The pipeline reads text_encoder.config.d_model (KV-cache setup)
        # without needing the weights — serve it from the on-disk config.
        if self._config is None:
            import json
            with open(os.path.join(self.path, self.subfolder or "",
                                   "config.json")) as f:
                self._config = types.SimpleNamespace(**json.load(f))
        return self._config

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

def _lowvram() -> bool:
    return os.environ.get("TALKER_LOWVRAM") == "1"


def _patched_to(self, device):
    self.device = device
    exec_device = (torch.device(f"cuda:{device}") if isinstance(device, int)
                   else torch.device(device))
    if self.dit is not None:
        if _lowvram():
            # The int8 DiT (~14.3 GB with LoRA) does not fit a 16 GB card
            # once activations/VAE/desktop are accounted for. Keep as many
            # blocks resident as the budget allows; the rest stay in RAM
            # and stream through the GPU each forward pass (a few GB over
            # PCIe per step — cheap next to the step itself).
            from accelerate import dispatch_model, infer_auto_device_map
            # Reserve covers what lives outside the device-map budget:
            # LoRA (~1.3 GB), VAE weights + its conv activations (~2.5 GB
            # peak while encoding/decoding), DiT step activations (~1.5 GB),
            # and whatever the desktop holds.
            reserve_gb = float(os.environ.get("TALKER_VRAM_RESERVE_GB", "6.5"))
            total_gb = torch.cuda.get_device_properties(exec_device).total_memory / 2**30
            budget_gb = max(2.0, total_gb - reserve_gb)
            no_split = sorted({
                type(mod[0]).__name__
                for _, mod in self.dit.named_children()
                if isinstance(mod, nn.ModuleList) and len(mod) > 1
            })
            device_map = infer_auto_device_map(
                self.dit,
                max_memory={exec_device.index or 0: f"{budget_gb:.1f}GiB",
                            "cpu": "1000GiB"},
                no_split_module_classes=no_split or None)
            n_cpu = sum(1 for v in device_map.values() if str(v) == "cpu")
            print(f"[talker] low-VRAM mode: {n_cpu}/{len(device_map)} DiT "
                  f"submodules stream from RAM "
                  f"(GPU budget {budget_gb:.1f} GiB, blocks: {no_split})")
            self.dit = dispatch_model(self.dit, device_map=device_map,
                                      offload_buffers=True)
            # The root hook defaults to io_same_device=True, which force-
            # moves ALL outputs back to the GPU — including the ~8 GB KV
            # cache dict that the pipeline deliberately keeps on CPU
            # (offload_kv_cache). Compute already happens on the GPU, so
            # output relocation buys nothing; disable it.
            if hasattr(self.dit, "_hf_hook"):
                self.dit._hf_hook.io_same_device = False
        else:
            self.dit = self.dit.to(device, non_blocking=True)
        if hasattr(self.dit, "lora_dict") and self.dit.lora_dict:
            for lora_network in self.dit.lora_dict.values():
                for lora in lora_network.loras:
                    lora.to(exec_device, non_blocking=True)
        _ACTIVE["dit"], _ACTIVE["device"] = self.dit, exec_device
    if self.vae is not None:
        self.vae = self.vae.to(exec_device, non_blocking=True)
    # text_encoder: lazy CPU stub (patch 2); audio_encoder: CPU (patch 3)
    return self


pl.LongCatVideoAvatarPipeline.to = _patched_to


# --------------------------------------------------------------------------
# 4b. Memory-lean LoRA forward: upstream materializes total_lora_output and
#     then `org_output + total_lora_output` — two extra full-size copies of
#     the projection output (~1 GB each for qkv over a 93-frame latent).
#     Accumulate into org_output in-place instead (safe: fresh tensor,
#     inference is under @torch.no_grad()).
# --------------------------------------------------------------------------

from longcat_video.modules.avatar.longcat_video_dit_avatar import (
    LongCatVideoAvatarTransformer3DModel as _DiTClass,
)


def _lean_create_multi_lora_forward(self, module, loras):
    def multi_lora_forward(x, *args, **kwargs):
        weight_dtype = x.dtype
        org_output = module.org_forward(x, *args, **kwargs)
        for lora in loras:
            if lora.use_lora:
                lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                lx = lora.lora_up(lx)
                org_output += lx.to(weight_dtype) * (lora.multiplier * lora.alpha_scale)
        return org_output
    return multi_lora_forward


_DiTClass._create_multi_lora_forward = _lean_create_multi_lora_forward


# --------------------------------------------------------------------------
# 4e. Evict the DiT to RAM around VAE decode. The decoder spikes ~4 GB in
#     single conv3d calls even with feat_cache streaming, and the DiT is
#     idle during decode — swap its resident weights out for those seconds
#     (~10 GB each way over PCIe, a blip next to a multi-minute segment).
#     The dispatch hooks are untouched: offloaded modules keep their CPU
#     weights_map; we only round-trip the .data of cuda-resident tensors.
# --------------------------------------------------------------------------

from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan as _VAEClass

_ACTIVE = {"dit": None, "device": None}
_gpu_stash = []


def _dit_modules(dit):
    mods = [dit]
    if hasattr(dit, "lora_dict") and dit.lora_dict:
        for lora_network in dit.lora_dict.values():
            mods.extend(lora_network.loras)
    return mods


def _evict_dit():
    dit = _ACTIVE["dit"]
    if dit is None or _gpu_stash:
        return
    seen = set()
    for mod in _dit_modules(dit):
        for t in itertools.chain(mod.parameters(), mod.buffers()):
            if id(t) in seen:
                continue
            seen.add(id(t))
            if t.device.type == "cuda":
                _gpu_stash.append(t)
                t.data = t.data.to("cpu")
    torch.cuda.empty_cache()


def _restore_dit():
    device = _ACTIVE["device"]
    for t in _gpu_stash:
        t.data = t.data.to(device)
    _gpu_stash.clear()


def _wrap_vae_op(orig):
    def wrapped(self, x, *args, **kwargs):
        if not _lowvram():
            return orig(self, x, *args, **kwargs)
        _evict_dit()
        try:
            return orig(self, x, *args, **kwargs)
        finally:
            _restore_dit()
    return wrapped


# Both directions spike the same way: encode() runs per continuation
# segment on the 13 conditioning frames, decode() on every segment.
_VAEClass.decode = _wrap_vae_op(_VAEClass.decode)
_VAEClass.encode = _wrap_vae_op(_VAEClass.encode)


# --------------------------------------------------------------------------
# 5. Low-VRAM: keep the segment-continuation KV cache in RAM, not VRAM
#    (upstream hardcodes offload_kv_cache=False)
# --------------------------------------------------------------------------

_orig_generate_avc = pl.LongCatVideoAvatarPipeline.generate_avc


def _patched_generate_avc(self, *args, **kwargs):
    if _lowvram():
        kwargs["offload_kv_cache"] = True
    return _orig_generate_avc(self, *args, **kwargs)


pl.LongCatVideoAvatarPipeline.generate_avc = _patched_generate_avc


# --------------------------------------------------------------------------
# 4c. Chunked 3D-RoPE: upstream upcasts full q/k (40 heads x ~37k tokens)
#     to fp32 and rotate_half builds several more full-size temporaries —
#     a ~4-5 GB spike right before flash-attention. RoPE is independent
#     per head, so apply it 8 heads at a time: same numbers, ~1/5 the
#     transient memory. (Layout is [B, heads, seq, head_dim]; the freqs
#     broadcast over the head dim, so slicing it is safe.)
# --------------------------------------------------------------------------

from longcat_video.modules.avatar import rope_3d as _rope_mod

_orig_rope3d_forward = _rope_mod.RotaryPositionalEmbedding.forward


def _chunked_rope3d_forward(self, q, k, grid_size, frame_index=None,
                            num_ref_latents=None):
    num_heads = q.shape[1]
    chunk = 8
    if not _lowvram() or num_heads <= chunk:
        return _orig_rope3d_forward(self, q, k, grid_size, frame_index,
                                    num_ref_latents)
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    for h in range(0, num_heads, chunk):
        q_out[:, h:h + chunk], k_out[:, h:h + chunk] = _orig_rope3d_forward(
            self, q[:, h:h + chunk], k[:, h:h + chunk],
            grid_size, frame_index, num_ref_latents)
    return q_out, k_out


_rope_mod.RotaryPositionalEmbedding.forward = _chunked_rope3d_forward


# --------------------------------------------------------------------------
# 4d. Chunked SwiGLU FFN: w1/w3 expand each token to ~2x hidden width, so
#     the one-liner w2(silu(w1(x)) * w3(x)) holds ~3.5 GB of intermediates
#     for a 37k-token sequence (incl. the LoRA temp that OOMed). The FFN is
#     pointwise over tokens — process 8k tokens at a time, same numbers.
# --------------------------------------------------------------------------

import torch.nn.functional as F
from longcat_video.modules import blocks as _blocks_mod

_orig_ffn_forward = _blocks_mod.FeedForwardSwiGLU.forward

_FFN_CHUNK = 16384


def _chunked_ffn_forward(self, x):
    if not _lowvram() or x.dim() != 3 or x.shape[1] <= _FFN_CHUNK:
        return _orig_ffn_forward(self, x)
    out = torch.empty_like(x)
    for t in range(0, x.shape[1], _FFN_CHUNK):
        xs = x[:, t:t + _FFN_CHUNK]
        # self.w1/w2/w3 go through __call__, so LoRA wrappers and the
        # int8 dequant still apply per chunk.
        out[:, t:t + _FFN_CHUNK] = self.w2(F.silu(self.w1(xs)) * self.w3(xs))
    return out


_blocks_mod.FeedForwardSwiGLU.forward = _chunked_ffn_forward


# --------------------------------------------------------------------------
# 5b. Optional: skip vocal separation (TALKER_SKIP_VOCAL_SEP=1). For clean
#     speech (TTS / studio VO) there is nothing to separate — Kim_Vocal_2
#     costs ~15-20s per run for a no-op. NOTE: we copy to the temp target
#     rather than returning the source, because the demo deletes the
#     returned path after computing the audio embedding.
# --------------------------------------------------------------------------

_orig_extract_vocal = demo.extract_vocal_from_speech


def _patched_extract_vocal(source_path, target_path, vocal_separator,
                           audio_output_dir_temp):
    if os.environ.get("TALKER_SKIP_VOCAL_SEP") == "1":
        import shutil
        print("[talker] skipping vocal separation (clean-speech mode)")
        shutil.copyfile(source_path, target_path)
        return target_path
    return _orig_extract_vocal(source_path, target_path, vocal_separator,
                               audio_output_dir_temp)


demo.extract_vocal_from_speech = _patched_extract_vocal


# --------------------------------------------------------------------------
# 6. Don't re-encode the whole accumulated video after every segment
#    (upstream does — O(n^2) for long runs). Save every 10th as a
#    crash checkpoint, plus always the final one.
# --------------------------------------------------------------------------

_NUM_SEGMENTS = 1
_orig_save_video = demo.save_video_ffmpeg


def _patched_save_video(tensor, path, audio_path, **kwargs):
    base = os.path.basename(str(path))
    if _NUM_SEGMENTS > 1:
        if base.startswith(("ai2v_demo", "at2v_demo")):
            return None  # segment 1 of many: skip, continuation saves cover it
        if base.startswith("video_continue_"):
            idx = int(base.rsplit("_", 1)[1])
            if idx != _NUM_SEGMENTS and idx % 10 != 0:
                return None
    return _orig_save_video(tensor, path, audio_path, **kwargs)


demo.save_video_ffmpeg = _patched_save_video


if __name__ == "__main__":
    args = demo._parse_args()
    _NUM_SEGMENTS = max(1, args.num_segments)
    demo.generate(args)
