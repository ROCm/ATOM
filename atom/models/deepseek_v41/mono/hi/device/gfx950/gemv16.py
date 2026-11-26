# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16-activation GEMV on gfx950: weights x bf16 activations, rows as the MFMA N, fp32 sums.

The engine of linears whose contract is a BF16 GEMM (bf16 x bf16 products, exact in fp32, accumulated in
fp32). One ``v_mfma_f32_16x16x32_bf16``
per 32 K; a **unit** is 16 weight rows x 128 K (four MFMAs), as in ``gemv``. Two weight formats:

- ``fp4``: weights whose values are MXFP4 values (a checkpoint that rounded a BF16 linear to
  MXFP4 and kept it BF16): stored losslessly as MXFP4 in ``gemv.pack_weight``'s unit layout
  (lane ``l``: row ``l % 16``, K ``32 j .. 32 j + 31`` of the unit, ``j = l // 16``, 16 B, and
  one E8M0 code), decoded to the exact bf16 values with ``v_cvt_scalef32_pk_bf16_fp4``;
- ``bf16``: plain bf16 weights, same lane mapping, 64 B per lane (``pack_bf16``).

**K order.** The MFMA reads lane ``l``'s eight A values and eight B values as K ``8 j ..
8 j + 7`` of its 32; any assignment of real K indices works if A and B agree. MFMA ``m``
(0..3) of a unit takes, in lane ``l``, real K ``32 j + 8 m .. 32 j + 8 m + 7`` for both: the
lane's weight dword ``m`` (fp4) and its activation bf16 at those K (``b_operands``). The sum
over K is the same set of exact products; only the fp32 order is the MFMA's.

Rows: lane ``l`` feeds activation row ``min(l % 16, rows - 1)``; lane ``l`` of the result
holds features ``4 (l // 16) + e`` of row ``l % 16`` (as ``gemv``).
"""

import flydsl.expr as fx
import torch
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .ops import CM_NT

OOB = (
    0x7FFFFFFF // 4
)  # a word offset no buffer reaches: the load returns 0 without traffic
UNIT_K = 128


# ------------------------------------------------------------------ packing (host)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """bf16 ``w`` [N, K] (N % 16 == 0, K % 128 == 0) -> flat bf16 in unit order: unit (rg, st)
    at element ``(rg * S + st) * 2048``, lane ``l``'s 32 values (row ``l % 16``, K ``32 (l //
    16) ..``) at ``32 l`` of it."""
    n, k = w.shape
    if n % 16 or k % UNIT_K:
        raise ValueError(
            f"pack_bf16 needs N % 16 == 0 and K % 128 == 0, got {tuple(w.shape)}"
        )
    s = k // UNIT_K
    # [rg, r, st, j, e] -> [rg, st, j, r, e]: lane l = 16 j + r
    return (
        w.reshape(n // 16, 16, s, 4, 32).permute(0, 2, 3, 1, 4).contiguous().reshape(-1)
    )


def pack_bf16_mmajor(w: torch.Tensor) -> torch.Tensor:
    """``pack_bf16``'s units with each unit's four MFMA slices contiguous: unit (rg, st) at element ``(rg * S + st) *
    2048``, then slice ``m`` (K ``32 j + 8 m ..`` of every lane) at ``512 m``, lane ``l``'s 8 values at ``8 l`` of it:
    one 16 B load instruction of a wave reads 1 KiB contiguous (``load_bf16_mmajor``), as MXFP8's halves do.
    """
    n, k = w.shape
    if n % 16 or k % UNIT_K:
        raise ValueError(
            f"pack_bf16_mmajor needs N % 16 == 0 and K % 128 == 0, got {tuple(w.shape)}"
        )
    s = k // UNIT_K
    # [rg, r, st, j, m, e] -> [rg, st, m, j, r, e]: lane l = 16 j + r
    return (
        w.reshape(n // 16, 16, s, 4, 4, 8)
        .permute(0, 2, 4, 3, 1, 5)
        .contiguous()
        .reshape(-1)
    )


# ------------------------------------------------------------------ device: weights


def load_fp4(r_w, unit, lane, valid=None):
    """Lane's 16 B (one Vector of 4 i32) of fp4 unit ``unit``, nontemporal; ``valid`` False
    reads zeros without memory traffic."""
    off = (unit * 64 + lane) * 4
    if valid is not None:
        off = valid.select(off, fx.Int32(OOB))
    return fx.Vector(
        bo.buffer_load(r_w, off, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )


def load_scale(r_s, unit, lane, valid=None):
    """Lane's E8M0 code (Int32) of fp4 unit ``unit`` (byte ``64 unit + lane`` of the scales)."""
    off = unit * 16 + lane // 4
    if valid is not None:
        off = valid.select(off, fx.Int32(OOB))
    v = fx.Int32(
        bo.buffer_load(r_s, off, vec_width=1, dtype=T.i32, cache_modifier=CM_NT)
    )
    return (v >> ((lane % 4) * 8)) & 255


def load_bf16(r_w, unit, lane, valid=None):
    """Lane's 64 B (four Vectors of 4 i32) of bf16 unit ``unit``, nontemporal."""
    out = []
    for m in range(4):
        off = (unit * 256 + lane * 4 + m) * 4
        if valid is not None:
            off = valid.select(off, fx.Int32(OOB))
        out.append(
            fx.Vector(
                bo.buffer_load(r_w, off, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
            )
        )
    return out


def load_bf16_mmajor(r_w, unit, lane, valid=None):
    """``load_bf16``'s 64 B of a ``pack_bf16_mmajor`` unit: slice ``m``'s 16 B at word ``unit * 1024 + 256 m + 4 lane``
    (each load instruction a contiguous 1 KiB), the same four Vectors (``bf16_operands``).
    """
    out = []
    for m in range(4):
        off = (unit * 256 + m * 64 + lane) * 4
        if valid is not None:
            off = valid.select(off, fx.Int32(OOB))
        out.append(
            fx.Vector(
                bo.buffer_load(r_w, off, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
            )
        )
    return out


def _bf16x8(words):
    return fx.Vector.from_elements(list(words), fx.Int32).bitcast(fx.BFloat16)


def decode_fp4(w, code):
    """An fp4 unit's lane dwords ``w`` (Vector of 4 i32) with E8M0 ``code`` -> the four MFMAs'
    A operands (8 bf16 each): dword ``m``'s 8 values (low nibble first) times 2^(code - 127),
    exact."""
    scale = (fx.Int32(code) << 23).bitcast(fx.Float32)
    out = []
    for m in range(4):
        words = []
        for b in range(4):
            pair = rocdl.cvt_scalef32_pk_bf16_fp4(
                T.vec(2, T.bf16), fx.Int32(w[m]), scale, b
            )
            words.append(fx.Vector(pair).bitcast(fx.Int32)[0])
        out.append(_bf16x8(words))
    return out


def bf16_operands(ws):
    """A bf16 unit's lane words (``load_bf16``) -> the four MFMAs' A operands."""
    return [_bf16x8([ws[m][e] for e in range(4)]) for m in range(4)]


def mfma(acc, a, b):
    """acc (vec4 f32) += 16x16x32 bf16."""
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, acc]))


def mfma_unit(acc, a_ops, b_ops):
    for m in range(4):
        acc = mfma(acc, a_ops[m], b_ops[m])
    return acc


# ------------------------------------------------------------------ device: activations in LDS
# One K slice of the activation sits in LDS as [rows][pitch] words of two bf16 (row-major K).


def b_operands(lds, row, j, unit, pitch):
    """Lane's four B operands of unit ``unit`` of the slice: bf16 at K ``128 unit + 32 j + 8 m
    ..`` of activation row ``row`` (``j`` = lane // 16)."""
    base = row * pitch + unit * 64 + j * 16
    out = []
    for m in range(4):
        v = fx.Vector(
            fx.ptr_load(
                lds + (base + 4 * m), result_type=fx.Vector.make_type(4, fx.Int32)
            )
        )
        out.append(_bf16x8([v[e] for e in range(4)]))
    return out
