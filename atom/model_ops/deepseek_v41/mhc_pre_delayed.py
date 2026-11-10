# SPDX-License-Identifier: MIT
"""The delayed mHC seam, including the sublayer's input RMSNorm.

V4's single `aiter.mhc_pre` cannot serve this model: the collapse applies the
pre-mix carried from the PREVIOUS sublayer while the one computed here is
carried to the next. HIP Mega-MHC and the Triton implementation cover that seam --
the owed post, the gate projection, Sinkhorn, the collapse with the carried
pre-mix and the sublayer's input RMSNorm. HIP uses one compute kernel on gfx950
and gfx1250; Triton remains available through ATOM_DSV41_MHC_BACKEND=triton.

The norm is the sublayer's own. Attention can request HIP's fused MXFP8
output alongside BF16 for its other readers; FFN keeps the BF16 interface.
Its consumers (wqkv_a's group32 GEMM, the bounded-replay row gather) read
row-major (T, H/32) E8M0 scales. gfx950 writes those; gfx1250 writes its
m32k4 GEMM A-scale layout unless asked for `scale_row_major`, so there the
fused output needs an AITER with that option, else the GEMM quantizes.
"""

import inspect
from functools import cache

import torch
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
    mhc_fused_post_pre_delayed_rmsnorm,
)

from atom.utils import envs

# Archs with HIP Mega-MHC kernels.
_HIP_ARCHS = ("gfx950", "gfx1250")


@cache
def _hip_ops():
    try:
        from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn
    except ImportError:
        return None
    return mega_mhc, mega_mhc_shuffle_fn


@cache
def _row_major_scale_options():
    """The mega_mhc options that give fused MXFP8 row-major (T, H/32) scales,
    or None when this arch/AITER cannot: gfx950 always writes them, gfx1250
    only through `scale_row_major` (m32k4 otherwise)."""
    arch, ops = get_gfx_runtime(), _hip_ops()
    if arch == "gfx950":
        return {}
    if (
        arch == "gfx1250"
        and ops is not None
        and "scale_row_major" in inspect.signature(ops[0]).parameters
    ):
        return {"scale_row_major": True}
    return None


def fuse_quant_supported() -> bool:
    """Whether HIP's fused MXFP8 output can have row-major (T, H/32) scales."""
    return _row_major_scale_options() is not None


def pack_mhc_fn(fn):
    """Select the backend and pack once after checkpoint loading, like V4."""
    backend = envs.ATOM_DSV41_MHC_BACKEND
    if backend not in ("auto", "hip", "triton"):
        raise ValueError("ATOM_DSV41_MHC_BACKEND must be auto, hip or triton")
    if backend == "triton":
        return None
    supported = (
        fn.is_cuda
        and fn.dtype == torch.float32
        and fn.ndim == 2
        and fn.shape[0] == 24
        and fn.shape[1] in (4 * 4096, 4 * 5120, 4 * 6144, 4 * 7168)
        and fn.is_contiguous()
        and get_gfx_runtime() in _HIP_ARCHS
    )
    if not supported:
        if backend == "hip":
            raise ValueError(
                "HIP Mega-MHC requires gfx950 or gfx1250, contiguous FP32 fn, "
                "hc_mult=4 and H in {4096, 5120, 6144, 7168}"
            )
        return None
    ops = _hip_ops()
    if ops is None:
        if backend == "hip":
            raise RuntimeError("HIP Mega-MHC requires AITER mega_mhc_shuffle_fn")
        return None
    return ops[1](fn)


def pre_delayed(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    post_mult,
    norm_eps,
    w_preshuffle_bf16=False,
    fuse_quant=False,
    sublayer_output=None,
    post_mix=None,
    combination=None,
):
    """One sublayer seam: optional post, the delayed pre, the input RMSNorm.

    Returns `(residual, normed_input, next_pre_mix, post_mix, combination)`.
    `residual` is the caller's own tensor when no post was requested.
    `hc_fn` is a HIP BF16 hi/lo pack when `w_preshuffle_bf16=True`,
    or the original FP32 weight for Triton otherwise.
    With `fuse_quant`, `normed_input` is `(bf16, fp8, e8m0_scale)` with
    row-major scales; it needs the HIP backend and `fuse_quant_supported()`.
    """
    outputs = v41_mhc_pre_delayed(
        residual,
        pre_mix,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        post_mult=post_mult,
        norm_eps=norm_eps,
        w_preshuffle_bf16=w_preshuffle_bf16,
        fuse_quant=fuse_quant,
        sublayer_output=sublayer_output,
        post_mix=post_mix,
        combination=combination,
    )
    # Keep the guarded op's outputs flat and preserve the no-post input alias
    # outside it. Only the attention-facing wrapper groups the norm outputs.
    if fuse_quant:
        outputs, quant, scale = outputs[:-2], outputs[-2], outputs[-1]
    outputs = (residual, *outputs) if sublayer_output is None else tuple(outputs)
    if fuse_quant:
        outputs = (outputs[0], (outputs[1], quant, scale), *outputs[2:])
    return outputs


def _pre_delayed_fake(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    sublayer_output=None,
    fuse_quant=False,
    **kwargs,
):
    *outer, hc, hidden = residual.shape
    outputs = [
        residual.new_empty((*outer, hidden)),
        residual.new_empty((*outer, hc), dtype=torch.float32),
        residual.new_empty((*outer, hc), dtype=torch.float32),
        residual.new_empty((*outer, hc, hc), dtype=torch.float32),
    ]
    if fuse_quant:
        outputs.extend(
            [
                residual.new_empty((*outer, hidden), dtype=torch.float8_e4m3fn),
                residual.new_empty((*outer, hidden // 32), dtype=torch.float8_e8m0fnu),
            ]
        )
    return (
        outputs if sublayer_output is None else [torch.empty_like(residual), *outputs]
    )


@torch_compile_guard(mutates_args=[], gen_fake=_pre_delayed_fake)
def v41_mhc_pre_delayed(
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
    norm_eps: float,
    w_preshuffle_bf16: bool = False,
    fuse_quant: bool = False,
    sublayer_output: torch.Tensor | None = None,
    post_mix: torch.Tensor | None = None,
    combination: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The seam as one guarded op, so Dynamo treats the launches as opaque."""
    *outer, hc_mult, hidden = residual.shape
    rows = residual.numel() // (hc_mult * hidden)
    with_post = sublayer_output is not None
    kernel = mhc_fused_post_pre_delayed_rmsnorm
    options = {}
    if fuse_quant and not w_preshuffle_bf16:
        raise ValueError("Fused mHC quantization requires the HIP backend")
    if fuse_quant and not fuse_quant_supported():
        raise ValueError(
            "Fused mHC quantization requires row-major scales: gfx950, or "
            "gfx1250 with AITER mega_mhc(scale_row_major=)"
        )
    if w_preshuffle_bf16:
        kernel = _hip_ops()[0]
        options["w_preshuffle_bf16"] = True
        options["out_dtype"] = "mxfp8" if fuse_quant else "bf16"
        if fuse_quant:
            options.update(_row_major_scale_options())
    new_residual, post, comb, normed, pre = kernel(
        residual.view(rows, hc_mult, hidden),
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        post_mult,
        sinkhorn_iters,
        pre_mix.reshape(rows, hc_mult),
        sublayer_out=sublayer_output.reshape(rows, hidden) if with_post else None,
        post_layer_mix=post_mix.reshape(rows, hc_mult) if with_post else None,
        comb_res_mix=(
            combination.reshape(rows, hc_mult, hc_mult) if with_post else None
        ),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        **options,
    )
    if fuse_quant:
        normed, quant, scale = normed
    outputs = [
        normed.view(*outer, hidden),
        pre.view(*outer, hc_mult),
        post.view(*outer, hc_mult),
        comb.view(*outer, hc_mult, hc_mult),
    ]
    if fuse_quant:
        outputs.extend(
            [
                quant.view(*outer, hidden),
                scale.view(torch.float8_e8m0fnu).view(*outer, hidden // 32),
            ]
        )
    return outputs if not with_post else [new_residual.view_as(residual), *outputs]
