# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""1x32 MXFP8 projections for the Qwen3.5 language-model layers (ATOM_MXFP8_1X32).

fp8 e4m3 values with one e8m0 scale per 32 K elements on BOTH operands, run by aiter's
gfx1250 Gluon GEMM ``gemm_afp8wfp8_preshuffle`` (scale_layout "n32k4").

Projections: per decoder layer (not MTP) ``self_attn.qkv_proj`` and ``self_attn.o_proj``, or
the fused GDN in_proj (prefix ``...linear_attn``, see qwen3_5.create_qkvzba_proj) and
``linear_attn.out_proj``. Their bf16 weights are quantized after loading (``convert_linear``):
``per_1x32_mx_quant_hip`` -> fp8 + e8m0, w preshuffled ``shuffle_weight(layout=(16, 16))``,
scales ``shuffle_mxfp8fp4_scale`` (n32k4).

Activation (x) scale layouts the GEMM takes, M = the GEMM's rows:
  * M >= 32: n32k4, rows padded to 32; byte of (row m, group g) at
        (m // 32) * K + (g // 4) * 128 + (m % 32) * 4 + g % 4          (K = columns)
    pad rows hold 0x7F (e8m0 1.0), as ``per_group_quant_hip(scale_layout_m32k4=True)`` writes;
  * M < 32: row-major (M, K // 32).
The standalone quant (``quant_standalone``) is per_group_quant_hip(..., scale_layout_m32k4=True)
for M >= 32 and per_1x32_mx_quant_hip(shuffle=False) for M < 32 (same fp8 codes).

A producer that quantizes its own output (ATOM_MXFP8_1X32_FUSED_ACT_QUANT) hands the Linear
``(x_fp8 [M, K], x_scale)`` with ``x_scale`` a flat uint8 buffer of ``scale_bytes(M, K)`` bytes
(``alloc``): the layout above (row-major in its first M * K / 32 bytes when M < 32). The flat
size has no M >= 32 branch, so fake (meta) functions need none either.

Bit-exactness with "producer bf16 output, then the standalone quant" (aiter
dynamic_per_group_quant_m32k4_kernel / the g32 path of dynamic_per_group_scaled_quant_kernel,
gfx1250, bf16 input):
  * amax over the bf16-rounded values of the group, from aiter's kAmaxFloor 1e-10, capped at
    448 * 2^119 (the hardware-convert path's cap);
  * e8m0 byte = RoundUp (fp_f32_to_e8m0_scale<RoundUp, FP8_E4M3>): the fp32 exponent field of
    amax * (1/448), plus one when its mantissa is non-zero (and the field is < 255);
  * value = RNE(x * 2^(127 - byte)) to e4m3 (v_cvt_scalef32_pk8_fp8_bf16 divides by the exact
    power of two), clamped to +-448 first (the quant clamps a degenerate group to +-448*scale;
    for a finite group |x| * 2^(127-byte) <= 448 * (1 + 2^-23) rounds to 448 either way).
  * an all-zero group gets byte 85 (2^-42) and zero codes.
"""

from __future__ import annotations

import logging
import re

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import Tensor

from atom.utils import envs

logger = logging.getLogger("atom")

GROUP = 32
E8M0_ONE = 0x7F  # pad rows of the n32k4 buffer
ZERO_GROUP_BYTE = 85  # e8m0 byte of an all-zero group (amax floor 1e-10 -> 2^-42)
FP8 = torch.float8_e4m3fn

# The in-scope projections: language-model decoder layers only (no MTP). The fused GDN in_proj's
# Linear prefix is the bare `...layers.N.linear_attn`.
_TARGET_RE = re.compile(
    r"^(?!.*mtp).*layers\.\d+\."
    r"(linear_attn|linear_attn\.out_proj|self_attn\.qkv_proj|self_attn\.o_proj)$"
)
_CONVERTED: list[str] = []
_NOTED: set = set()


def note(route: str) -> None:
    """Log the first use of each activation-quant route in this process (which producers fused)."""
    if route not in _NOTED:
        _NOTED.add(route)
        logger.info("MXFP8_1X32 x quant route: %s", route)


def enabled() -> bool:
    return bool(envs.ATOM_MXFP8_1X32)


def fused_act_quant() -> bool:
    """ATOM_MXFP8_1X32 with its producer fusion (ATOM_MXFP8_1X32_FUSED_ACT_QUANT, default 1)."""
    return enabled() and bool(envs.ATOM_MXFP8_1X32_FUSED_ACT_QUANT)


def is_target_prefix(prefix: str | None) -> bool:
    return bool(prefix) and _TARGET_RE.search(prefix) is not None


def applies(linear) -> bool:
    """Whether `linear` is (or, before loading, will be) a 1x32 MXFP8 projection: the flag is on,
    the prefix is in scope and the layer is unquantized (bf16 weights)."""
    if getattr(linear, "mx1x32", False):
        return True
    if not enabled() or not is_target_prefix(getattr(linear, "prefix", None)):
        return False
    from aiter import QuantType

    return linear.quant_type.value == QuantType.No.value


def takes_mx32(linear) -> bool:
    """A converted 1x32 projection (after load), or one that will be (at init)."""
    return linear is not None and applies(linear)


# --------------------------------------------------------------------------------------------
# Buffers / layouts
# --------------------------------------------------------------------------------------------
def scale_bytes(m: int, k: int) -> int:
    return (m + 31) // 32 * 32 * (k // GROUP)


def alloc(m: int, k: int, device) -> tuple[Tensor, Tensor]:
    """(x_fp8 [m, k], flat uint8 x_scale of scale_bytes(m, k)) for a producer to fill."""
    assert k % 128 == 0, k
    return (
        torch.empty((m, k), dtype=FP8, device=device),
        torch.empty(scale_bytes(m, k), dtype=torch.uint8, device=device),
    )


def gemm_scale_view(s_flat: Tensor, m: int, k: int) -> Tensor:
    """The flat x_scale as the GEMM reads it: (pad32(m) // 32, k) n32k4, or (m, k // 32)."""
    s = s_flat.view(torch.uint8)
    if m >= 32:
        return s.view(-1, k)
    return s[: m * (k // GROUP)].view(m, k // GROUP)


def quant_standalone(x2d: Tensor) -> tuple[Tensor, Tensor]:
    """bf16 [m, k] -> (fp8 [m, k], x_scale as the GEMM reads it)."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip, per_group_quant_hip

    m, k = x2d.shape
    x2d = x2d.contiguous()
    if m >= 32:
        q, s = per_group_quant_hip(
            x2d,
            quant_dtype=dtypes.fp8,
            group_size=GROUP,
            scale_type=dtypes.fp8_e8m0,
            scale_layout_m32k4=True,
        )
        return q, s.view(torch.uint8).view(-1, k)
    q, s = per_1x32_mx_quant_hip(
        x2d, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    s = s.view(torch.uint8)
    if s.shape != (m, k // GROUP) or not s.is_contiguous():
        s = s[:m, : k // GROUP].contiguous()
    return q.view(m, k), s


def quant_into(y2d: Tensor, q_out: Tensor, s_flat: Tensor) -> None:
    """The standalone quant of bf16 [m, k] written into a producer's buffers (`alloc`): the
    fallback of a producer route that has no fused epilogue."""
    from aiter import dtypes
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    m, k = y2d.shape
    if m == 0:
        return
    if m >= 32:
        dynamic_per_group_scaled_quant(
            q_out.view(m, k),
            y2d.contiguous(),
            s_flat.view(dtypes.fp8_e8m0).view(-1, k // GROUP),
            GROUP,
            scale_layout_m32k4=True,
        )
        return
    q, s = quant_standalone(y2d)
    q_out.view(m, k).copy_(q)
    s_flat.view(torch.uint8)[: s.numel()].copy_(s.reshape(-1))


def _quant_mx32_fake(y2d: Tensor) -> tuple[Tensor, Tensor]:
    return alloc(y2d.shape[0], y2d.shape[1], y2d.device)


@torch_compile_guard(gen_fake=_quant_mx32_fake, mutates_args=[])
def quant_mx32_op(y2d: Tensor) -> tuple[Tensor, Tensor]:
    """Functional standalone quant in the producer contract: (fp8 [m, k], flat uint8 x_scale)."""
    q, s = alloc(y2d.shape[0], y2d.shape[1], y2d.device)
    quant_into(y2d, q, s)
    return q, s


def reference(y_bf16: Tensor) -> tuple[Tensor, Tensor]:
    """For tests: the standalone quant of the bf16 rows as (fp8, flat uint8 scale) in the producer
    contract (`alloc`); for m < 32 only the first m * k / 32 bytes are defined."""
    m, k = y_bf16.shape
    q, s = quant_standalone(y_bf16)
    flat = torch.full(
        (scale_bytes(m, k),), E8M0_ONE, dtype=torch.uint8, device=y_bf16.device
    )
    flat[: s.numel()].copy_(s.reshape(-1))
    return q, flat


def valid_scale_bytes(m: int, k: int) -> int:
    """How many leading bytes of a flat x_scale the GEMM reads (all of it for m >= 32)."""
    return scale_bytes(m, k) if m >= 32 else m * (k // GROUP)


# --------------------------------------------------------------------------------------------
# Triton helpers shared by the producers
# --------------------------------------------------------------------------------------------
@triton.jit
def mx32_scale_offset(m, g, KG: tl.constexpr, N32K4: tl.constexpr):
    """Byte offset of the scale of (row m, 32-column group g); KG = K // 32 groups a row."""
    if N32K4:
        return (m // 32) * (KG * 32) + (g // 4) * 128 + (m % 32) * 4 + (g % 4)
    else:
        return m * KG + g


@triton.jit
def mx32_e8m0(amax):
    """Group amax (fp32, floored and capped) -> (e8m0 byte as int32, 2^(127 - byte) as fp32)."""
    u = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    e = ((u >> 23) & 0xFF).to(tl.int32)
    e = e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).to(tl.int32)
    inv = ((254 - tl.minimum(e, 253)).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    return e, inv


@triton.jit
def mx32_quant_groups(y):
    """y [G, 32] fp32 holding bf16-representable values -> (fp8 e4m3 [G, 32], e8m0 bytes [G])."""
    amax = tl.max(tl.abs(y), axis=1)
    amax = tl.minimum(tl.maximum(amax, 1e-10), 448.0 * 6.646139978924579e35)  # 2^119
    e, inv = mx32_e8m0(amax)
    v = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0)
    return v.to(tl.float8e4nv), e.to(tl.uint8)


@triton.jit
def _mx32_fill_kernel(q_ptr, s_ptr, start, M, K: tl.constexpr, N32K4: tl.constexpr):
    """Rows [start, M): what quantizing a zero row gives (fp8 0, byte 85); rows [M, pad32(M)) of an
    n32k4 buffer: 0x7F."""
    row = start + tl.program_id(0).to(tl.int64)
    KG: tl.constexpr = K // 32
    g = tl.arange(0, KG)
    if row < M:
        cols = tl.arange(0, K)
        tl.store(q_ptr + row * K + cols, tl.zeros([K], dtype=tl.uint8))
        tl.store(
            s_ptr + mx32_scale_offset(row, g, KG, N32K4),
            tl.full([KG], 85, dtype=tl.uint8),
        )
    else:
        tl.store(
            s_ptr + mx32_scale_offset(row, g, KG, N32K4),
            tl.full([KG], 0x7F, dtype=tl.uint8),
        )


def fill_rows(q: Tensor, s_flat: Tensor, start: int) -> None:
    """Rows from `start` of a producer's (q [M, K], flat scale): zero rows (fp8 0, byte 85) up to M,
    then the n32k4 pad rows (0x7F). One launch; nothing when there is nothing to fill.
    """
    m, k = q.shape
    end = (m + 31) // 32 * 32 if m >= 32 else m
    if start >= end:
        return
    _mx32_fill_kernel[(end - start,)](
        q.view(torch.uint8),
        s_flat.view(torch.uint8),
        start,
        m,
        K=k,
        N32K4=m >= 32,
        num_warps=4,
    )


# --------------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------------
def convert_linear(layer) -> None:
    """bf16 weight -> preshuffled fp8 + n32k4 e8m0 1x32 scales, in place (after loading)."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight

    assert not torch.cuda.is_current_stream_capturing(), layer.prefix
    w = layer.weight.data
    assert w.dim() == 2 and w.dtype == torch.bfloat16, (
        layer.prefix,
        w.dtype,
        tuple(w.shape),
    )
    n, k = w.shape
    assert n % 32 == 0 and k % 128 == 0, (layer.prefix, n, k)
    assert layer.bias is None, f"{layer.prefix}: 1x32 MXFP8 path has no bias"
    q, s = per_1x32_mx_quant_hip(
        w.contiguous(),
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    s = s.view(torch.uint8)[:n, : k // GROUP].contiguous()
    w_sh = shuffle_weight(q.view(torch.uint8).view(n, k), layout=(16, 16)).reshape(
        n // 16, k * 16
    )
    ws_sh = shuffle_mxfp8fp4_scale(s).view(torch.uint8).reshape(n // 32, k)
    layer.weight.data = w_sh.view(FP8).contiguous()
    layer.mx1x32_w_scale = ws_sh.contiguous()
    layer.mx1x32_nk = (n, k)
    layer.mx1x32 = True
    del q, s, w
    _CONVERTED.append(layer.prefix)
    logger.info(
        "MXFP8_1X32 weight %d: %s N=%d K=%d -> fp8 %s + e8m0 n32k4 %s",
        len(_CONVERTED),
        layer.prefix,
        n,
        k,
        tuple(layer.weight.shape),
        tuple(ws_sh.shape),
    )


def check_converted(model: torch.nn.Module) -> None:
    """After the post-load pass: in every language-model decoder layer (not the MTP drafter's) both
    projections -- qkv_proj + o_proj, or the GDN in_proj + out_proj -- are converted, read from the
    layer's own modules (so a projection whose prefix the pattern misses is caught), and no other
    Linear is: 80 a rank for Qwen3.5-35B-A3B. Called once per loaded model (the drafter's, whose
    projections are all MTP, checks nothing)."""
    if not enabled():
        return
    from atom.model_ops.linear import LinearBase

    expected, missed = [], []
    for name, mod in model.named_modules():
        kind = getattr(mod, "layer_type", None)
        if kind == "linear_attention":
            attn = getattr(mod, "linear_attn", None)
            names = ("in_proj_qkvzba", "out_proj")
        elif kind == "full_attention":
            attn = getattr(mod, "self_attn", None)
            names = ("qkv_proj", "o_proj")
        else:
            continue
        projs = [getattr(attn, n, None) for n in names]
        if any(p is None for p in projs):
            missed.append(f"{name}: no {names}")
            continue
        if any("mtp" in (p.prefix or "") for p in projs):
            continue
        for p in projs:
            expected.append(p.prefix)
            if not getattr(p, "mx1x32", False):
                missed.append(p.prefix)
    converted = [
        m.prefix
        for _, m in model.named_modules()
        if isinstance(m, LinearBase) and getattr(m, "mx1x32", False)
    ]
    if not expected and not converted:
        return  # e.g. the MTP drafter
    logger.info(
        "MXFP8_1X32: %d projections converted (%d expected, 2 per decoder layer)",
        len(converted),
        len(expected),
    )
    assert not missed, f"MXFP8_1X32: projections not converted: {missed}"
    assert sorted(converted) == sorted(expected), (
        f"MXFP8_1X32: converted {len(converted)} != expected {len(expected)}: "
        f"{sorted(set(converted) ^ set(expected))}"
    )


# --------------------------------------------------------------------------------------------
# GEMM (custom ops: opaque to Dynamo)
# --------------------------------------------------------------------------------------------
def _gemm(
    x: Tensor, x_scale: Tensor | None, w: Tensor, w_scale: Tensor, y: Tensor
) -> None:
    from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle

    m, k = x.shape
    if x_scale is None:
        note(f"standalone quant in the Linear (N={w.shape[0] * 16}, K={k})")
        xq, xs = quant_standalone(x)
    else:
        assert x.dtype == FP8 and x_scale.numel() == scale_bytes(m, k), (
            x.dtype,
            x_scale.shape,
            m,
            k,
        )
        xq, xs = x, gemm_scale_view(x_scale.reshape(-1), m, k)
    gemm_afp8wfp8_preshuffle(
        xq,
        w,
        xs,
        w_scale,
        dtype=y.dtype,
        y=y,
        scale_layout="n32k4",
        x_scale_group_size=GROUP,
        w_scale_block_size=(1, GROUP),
    )


def _mx1x32_gemm_fake(
    x: Tensor, x_scale: Tensor | None, w: Tensor, w_scale: Tensor, otype: torch.dtype
) -> Tensor:
    return torch.empty((x.shape[0], w.shape[0] * 16), dtype=otype, device=x.device)


@torch_compile_guard(gen_fake=_mx1x32_gemm_fake, mutates_args=[])
def mx1x32_gemm(
    x: Tensor, x_scale: Tensor | None, w: Tensor, w_scale: Tensor, otype: torch.dtype
) -> Tensor:
    """y = x @ w^T; x bf16 (quantized here) or fp8 with its flat x_scale (a fused producer)."""
    y = torch.empty((x.shape[0], w.shape[0] * 16), dtype=otype, device=x.device)
    if x.shape[0]:
        _gemm(x, x_scale, w, w_scale, y)
    return y


def _mx1x32_gemm_into_output_fake(
    x: Tensor, x_scale: Tensor | None, w: Tensor, w_scale: Tensor, out: Tensor
) -> None:
    return None


@torch_compile_guard(gen_fake=_mx1x32_gemm_into_output_fake, mutates_args=["out"])
def mx1x32_gemm_into_output(
    x: Tensor, x_scale: Tensor | None, w: Tensor, w_scale: Tensor, out: Tensor
) -> None:
    """mx1x32_gemm into the caller's (m, n) `out` (returns None: a mutating op)."""
    if x.shape[0]:
        _gemm(x, x_scale, w, w_scale, out)


def linear_forward(
    layer, x: Tensor, x_scale: Tensor | None, otype, out: Tensor | None
) -> Tensor:
    """LinearBase.forward of a converted projection (before its TP all-reduce)."""
    n, k = layer.mx1x32_nk
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if out is not None:
        y = out.view(m, n)
        mx1x32_gemm_into_output(x2, x_scale, layer.weight, layer.mx1x32_w_scale, y)
    else:
        y = mx1x32_gemm(x2, x_scale, layer.weight, layer.mx1x32_w_scale, otype)
    return y.view(*lead, n)
