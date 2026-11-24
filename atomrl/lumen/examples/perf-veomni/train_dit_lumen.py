#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""VeOmni DiT training entry point with Lumen op replacement.

Mirrors tasks/train_dit.py, but wraps DiTTrainer._build_model so Lumen patches
the module tree after it is built and before FSDP2 wraps it. VeOmni source is
left untouched -- train.sh forwards its arguments to torchrun, so this file can
stand in for the task script.

Which ops get replaced is set by LUMEN_PATCH (comma-separated):

    linear   846 nn.Linear.forward -> lumen dispatch_gemm, with Lumen's own
             autograd.Function. BF16 in, BF16 out, so numerics should be
             unchanged.
    vae_conv 52 QwenImageCausalConv3d.forward in the frozen VAE -> Lumen's
             AITER-backed conv op. See lumen_vae_conv.py for why
             the T=1 causal rewrite is exact and which layers are
             deliberately left on torch.
    vae_conv_video
             vae_conv plus the shapes the T=1 rewrite cannot take. On a clip
             the T=1 identity holds only for the encoder's first chunk -- every
             later chunk carries a feature cache, so its leading frames are
             real data rather than zeros. This mode also rebinds the kernel
             underneath the module's own causal padding, which covers those,
             and takes the plain 2-D resamplers, which are faster through Lumen
             at video resolutions and slower at 256^2. Use it for Wan, LTX-2,
             MiniMax-H3; on Qwen-Image it adds nothing, because there every call
             is already reducible.
    conv_pad_in_kernel
             Modifier for vae_conv_video. The causal convolutions' stock forward
             F.pads a full copy of every input before a padding=0 convolution;
             this hands the (symmetric) spatial padding to the kernel instead,
             which applies it with range masks and no copy, and materialises
             only the temporal zeros the feature cache does not cover. See
             lumen_vae_conv._bind_causal_pad_in_kernel.
    vae_bf16 Cast the frozen VAE to BF16. NOT a Lumen patch and not free --
             it changes what the trainer computes. It is here because
             VeOmni loads this VAE in FP32
             (modeling_qwen_image_condition.py:83, hardcoded) while the AITER
             path is BF16/FP16-only, so without it `vae_conv` runs on
             torch. Combine the two to
             measure the kernel; use it alone to separate the cost of the dtype
             change from the benefit of the kernel.
    norm     NOT SAFE on this model, and refused below. The DiT's 241
             LayerNorms are elementwise_affine=False (AdaLayerNorm supplies
             scale/shift from a separate Linear). Lumen's _patch_norms swaps in
             LumenLayerNorm, which creates affine weights by default, adding 482
             trainable tensors the optimizer and FSDP2 never accounted for.

The rest are veomni_patches/ entries, applied before the trainer builds
anything (see that package for what each one does and how it was measured):

    sdpa_efficient    SDPA flash backend off, so the attention backward is
                      aiter's fused fmha_bwd. Makes the run nondeterministic.
    attn_triton_fwd   DiT attention forward on aiter's Triton kernel, backward
                      on the efficient one. Requires sdpa_efficient.
    rmsnorm_fuse      diffusers RMSNorm forward fused by torch.compile. Only
                      Qwen-Image's DiT uses diffusers' RMSNorm; Wan's is
                      torch.nn.RMSNorm, already one fused kernel.
    local_adamw       Fused AdamW on FSDP2 local shards. Bit-identical.
    qwen_offline_fix  VeOmni 573848a defect: Qwen-Image offline_training
                      crashes on its first step without it.

vae_conv* route through Lumen's in-package Conv2d/Conv3d implementation
(`atomrl.lumen.ops.conv`). Only `linear` and the convolution route import the
Lumen library.

The DiT is constructed under init_device=meta. The linear patch only rebinds
`forward` and never touches weights, so it is safe there. vae_conv reads
`weight[:, :, -1]` at patch time, which needs real values -- fine because the
condition model is loaded with weights and moved to the device for
online_training, never meta-initialised (dit_trainer.py:320).

    LUMEN_PATCH=vae_conv bash train.sh $EXAMPLE_DIR/train_dit_lumen.py <cfg> ...

LUMEN_TIME_VAE=1 additionally times vae.encode and reports its share of the
trainer's wall clock. It synchronises on every call, so it is for a separate
diagnostic run, not for a run whose step time will be quoted.

LUMEN_BATCH_HASH=1 prints an MD5 of every input the condition model's
get_condition / process_condition receive, per rank and call, so two
data-loading configurations can be shown to feed identical bytes in identical
order (compare_batch_hash.py) -- the only way to show it on Wan, whose loss does
not reproduce run to run. It copies every sample to the host; never time it.

Use run_10steps.sh or run_wan_10steps.sh rather than invoking this directly;
they set every variable the comparison needs to hold fixed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from veomni_patches import INSTALLERS as _VEOMNI_PATCHES

_PATCH = [p.strip() for p in os.environ.get("LUMEN_PATCH", "").split(",") if p.strip()]
_LUMEN_FEATURES = {
    "linear",
    "sdpa_efficient",
    "attn_triton_fwd",
    "rmsnorm_fuse",
    "local_adamw",
}


def _rank0(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[lumen] {msg}", flush=True)


def _enable_lumen(model):
    from atomrl.lumen.config import LumenConfig
    from torch import nn

    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    config = LumenConfig(
        scaling="none",
        lumen_linear="linear" in _PATCH,
        dit_sdpa_efficient="sdpa_efficient" in _PATCH,
        dit_sdpa_triton_forward="attn_triton_fwd" in _PATCH,
        diffusers_rmsnorm_fuse="rmsnorm_fuse" in _PATCH,
        local_shard_adamw="local_adamw" in _PATCH,
    )
    config.enable(model)

    if config.lumen_linear:
        patched = sum(
            1
            for module in model.modules()
            if isinstance(module, nn.Linear) and "forward" in module.__dict__
        )
        _rank0(f"lumen linear patch applied to {patched}/{n_linear} nn.Linear")
        if patched != n_linear:
            _rank0(f"WARNING: {n_linear - patched} nn.Linear were not patched")


def _get_vae(trainer, who):
    vae = getattr(trainer.condition_model, "vae", None)
    if vae is None:
        raise SystemExit(f"[lumen] LUMEN_PATCH={who} but condition_model has no .vae")
    return vae


def _cast_vae_bf16(trainer):
    """Run the frozen VAE's forward in BF16 but hand back FP32 latents.

    Casting the module alone breaks training a few lines downstream. The
    condition model derives the diffusion timestep from the latents' dtype
    (``timesteps[ids].to(dtype=sample_latents.dtype)``,
    modeling_qwen_image_condition.py:335, and the same line in
    modeling_wan_condition.py), and ``scale_noise`` then looks that timestep up
    in the FP32 schedule by exact equality. A BF16 timestep matches nothing, so
    the lookup returns an empty index tensor:

        IndexError: index 0 is out of bounds for dimension 0 with size 0

    Restoring FP32 on the way out keeps every downstream dtype identical to the
    baseline, so the only thing that changes is the precision the VAE convolves
    in -- which is the variable being measured.
    """
    import torch

    vae = _get_vae(trainer, "vae_bf16")
    before = vae.dtype
    vae.to(torch.bfloat16)

    original_encode = vae.encode

    def encode_fp32_out(*args, **kwargs):
        out = original_encode(*args, **kwargs)
        dist = out.latent_dist
        out.latent_dist = type(dist)(dist.parameters.float())
        return out

    vae.encode = encode_fp32_out
    _rank0(
        f"vae_bf16: condition_model.vae cast {before} -> {vae.dtype}, encode() upcasts latents back to float32"
    )


def _patch_vae_conv(trainer, video):
    import torch
    from lumen_vae_conv import patch_vae_convs

    who = "vae_conv_video" if video else "vae_conv"
    vae = _get_vae(trainer, who)
    if vae.dtype is not torch.bfloat16:
        _rank0(
            f"{who} WARNING: vae dtype is {vae.dtype}, but the AITER path is BF16/FP16-only. "
            "Lumen will demote every call to torch, so this measures the conv3d->conv2d "
            "rewrite alone. Add vae_bf16 to LUMEN_PATCH to exercise the kernel."
        )

    stats = patch_vae_convs(
        vae,
        video=video,
        verbose=_rank0,
        causal_pad_in_kernel="conv_pad_in_kernel" in _PATCH,
    )
    if not stats["patched"]:
        raise SystemExit(
            f"[lumen] {who} patched nothing -- refusing to run a comparison that changes nothing"
        )
    _rank0(f"{who} skipped (left on torch): {[n for n, _ in stats['skipped']]}")

    _rank0(f"{who} Lumen convolution: AITER")


_VAE_TIME = {"per_call": [], "shapes": []}


def _time_vae_encode(trainer):
    """Time every ``vae.encode``, to size it against the step.

    Set LUMEN_TIME_VAE=1. Off by default and deliberately not part of an A/B
    run: it synchronises on every call, which perturbs the step it is measuring.
    Run it once per mode on its own and read the share, not the step time.

    Without this the VAE's share of a step is an inference from a standalone
    encode benchmark, which ignores whatever else the condition model and the
    dataloader are doing in the same step, and assumes a clip shape the
    preprocessing may not actually produce. On Qwen-Image that inference put the
    convolutions at 0.07% of a step; on a video clip it is a different order of
    magnitude, and worth measuring rather than assuming.

    Calls are kept individually rather than summed. The first one carries
    AITER's JIT and torch's autotune, which on a video VAE's ~66 distinct
    shapes is large enough to swamp nine steady calls and invert the mean -- a
    total alone reported this patch as slower than the run it speeds up.
    """
    import torch

    vae = _get_vae(trainer, "LUMEN_TIME_VAE")
    original_encode = vae.encode

    def timed_encode(x, *args, **kwargs):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        out = original_encode(x, *args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        _VAE_TIME["per_call"].append(start.elapsed_time(end))
        shape = tuple(x.shape) if hasattr(x, "shape") else None
        if shape not in _VAE_TIME["shapes"]:
            _VAE_TIME["shapes"].append(shape)
        return out

    vae.encode = timed_encode
    _rank0(
        f"timing vae.encode (dtype {vae.dtype}); this synchronises, so do not read step time from this run"
    )


def _report_vae_time(wall_seconds):
    calls = _VAE_TIME["per_call"]
    if not calls:
        return
    _rank0(f"vae.encode input shapes seen: {_VAE_TIME['shapes']}")
    total = sum(calls)
    rest = calls[1:] or calls
    steady = sum(rest) / len(rest)
    _rank0(f"vae.encode: {len(calls)} calls, {total / 1000:.1f} s total")
    _rank0(
        f"vae.encode: first {calls[0]:.0f} ms (JIT + autotune), steady {steady:.0f} ms over {len(rest)}"
    )
    if wall_seconds:
        _rank0(
            f"vae.encode is {total / 1000 / wall_seconds * 100:.1f}% of {wall_seconds:.0f} s of trainer wall clock"
        )
        _rank0(
            f"  at the steady rate that would be {steady * len(calls) / 1000 / wall_seconds * 100:.1f}%"
        )


def _digest(obj, h):
    import torch

    if isinstance(obj, torch.Tensor):
        t = obj.detach().contiguous().cpu()
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _digest(x, h)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            h.update(str(k).encode())
            _digest(obj[k], h)
    else:
        h.update(repr(obj).encode())


def _install_batch_hash():
    """Hash every condition-model input; see LUMEN_BATCH_HASH in the module docstring."""
    import hashlib
    import importlib
    import inspect

    rank = int(os.environ.get("RANK", "0"))
    counts = {}
    patched = []
    for modname in (
        "veomni.models.diffusers.wan_t2v.wan_condition.modeling_wan_condition",
        "veomni.models.diffusers.qwen_image.qwen_image_condition.modeling_qwen_image_condition",
    ):
        try:
            mod = importlib.import_module(modname)
        # The unused model family's optional module may fail during import.
        except Exception:  # noqa: BLE001,S112 -- optional model boundary
            continue
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != modname:
                continue
            # get_condition sees raw samples (online only); process_condition sees
            # latents and text embeddings every step, online or offline.
            for method in ("get_condition", "process_condition"):
                if method not in cls.__dict__:
                    continue
                orig = getattr(cls, method)
                tag = f"{name}.{method}"

                def wrapped(self, *args, __orig=orig, __tag=tag, **kwargs):
                    h = hashlib.md5()
                    _digest((args, sorted(kwargs.items())), h)
                    n = counts[__tag] = counts.get(__tag, 0) + 1
                    print(
                        f"[lumen] batch_hash rank={rank} call={n} {__tag} md5={h.hexdigest()}",
                        flush=True,
                    )
                    return __orig(self, *args, **kwargs)

                setattr(cls, method, wrapped)
                patched.append(tag)
    _rank0(
        f"batch hash on {', '.join(patched) or '<nothing>'} -- verification only, do not time this run"
    )


def _report_backends():
    """Say what the patches actually ran, so a silent torch fallback is visible.

    Without this a run can look patched and still have executed nothing but
    torch: the convolution falls back on its own for inputs the kernel does not
    take, and the loss and step time would look identical for the wrong reason.
    """
    if not _PATCH:
        return
    if "vae_conv" in _PATCH or "vae_conv_video" in _PATCH:
        from atomrl.lumen.ops.conv import conv_calls

        _rank0(f"aiter conv calls: {conv_calls()}")
    if "attn_triton_fwd" in _PATCH:
        from atomrl.lumen.ops.attention import hf_patch as sdpa

        _rank0(f"sdpa routing: {sdpa.stats()}")
    if "rmsnorm_fuse" in _PATCH:
        from atomrl.lumen.ops.normalization import rmsnorm

        _rank0(f"rmsnorm_fuse calls: {rmsnorm.diffusers_rmsnorm_stats()}")


def _install_hook():
    time_vae = os.environ.get("LUMEN_TIME_VAE", "") not in ("", "0")
    if os.environ.get("LUMEN_BATCH_HASH", "") not in ("", "0"):
        _install_batch_hash()
    if not _PATCH and not time_vae:
        _rank0("LUMEN_PATCH empty -- running unmodified VeOmni (baseline)")
        return

    known = {
        "linear",
        "norm",
        "vae_conv",
        "vae_conv_video",
        "vae_bf16",
        "conv_pad_in_kernel",
        *_LUMEN_FEATURES,
        *_VEOMNI_PATCHES,
    }
    unknown = set(_PATCH) - known
    if unknown:
        raise SystemExit(f"[lumen] unknown LUMEN_PATCH entries: {sorted(unknown)}")
    if {"vae_conv", "vae_conv_video"} <= set(_PATCH):
        raise SystemExit("[lumen] vae_conv_video is a superset of vae_conv; pick one")
    if "conv_pad_in_kernel" in _PATCH and "vae_conv_video" not in _PATCH:
        raise SystemExit(
            "[lumen] conv_pad_in_kernel modifies vae_conv_video; add vae_conv_video"
        )
    if "attn_triton_fwd" in _PATCH and "sdpa_efficient" not in _PATCH:
        raise SystemExit(
            "[lumen] attn_triton_fwd keeps the efficient backward; add sdpa_efficient"
        )
    if "norm" in _PATCH:
        raise SystemExit(
            "[lumen] refusing LUMEN_PATCH=norm on this model: the DiT's 241 LayerNorms "
            "have elementwise_affine=False, and Lumen's replacement adds affine weights, "
            "which changes the parameter set under FSDP2. See this file's docstring."
        )

    # Before the trainer exists: these rebind classes and module attributes
    # (SDPA, RMSNorm.forward, AdamW.step, the condition model) that the model
    # and optimizer pick up when they are built.
    example_applied = [name for name in _VEOMNI_PATCHES if name in _PATCH]
    for name in example_applied:
        _VEOMNI_PATCHES[name]()
    if example_applied:
        _rank0(f"veomni_patches applied: {example_applied}")

    from veomni.trainer.dit_trainer import DiTTrainer

    original = DiTTrainer._build_model

    def _build_model_then_patch(self):
        original(self)

        # Order matters: vae_conv caches weight[:, :, -1] per layer, so the cast
        # has to happen first or the cached slices keep the old dtype.
        if "vae_bf16" in _PATCH:
            _cast_vae_bf16(self)
        if "vae_conv" in _PATCH or "vae_conv_video" in _PATCH:
            _patch_vae_conv(self, video="vae_conv_video" in _PATCH)

        if set(_PATCH) & _LUMEN_FEATURES:
            _enable_lumen(self.base.model)

        # Last, so the timer wraps the patched encode rather than being wrapped
        # by it -- otherwise vae_bf16's own encode wrapper hides the timing.
        if time_vae:
            _time_vae_encode(self)

    DiTTrainer._build_model = _build_model_then_patch
    _rank0(
        f"hook installed for LUMEN_PATCH={','.join(_PATCH) or '<none>'}, time_vae={time_vae}"
    )


if __name__ == "__main__":
    import time

    # First, so the veomni_patches rebindings exist before VeOmni and
    # diffusers import anything that could hold a reference to the originals.
    _install_hook()

    from veomni.arguments import parse_args
    from veomni.trainer.dit_trainer import DiTTrainer, VeOmniDiTArguments

    args = parse_args(VeOmniDiTArguments)
    trainer = DiTTrainer(args)
    _t0 = time.time()
    trainer.train()
    _report_vae_time(time.time() - _t0)
    _report_backends()
