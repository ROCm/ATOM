# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The W4A8 GEMV engine on gfx950: MXFP4 weights x MXFP8 activations, rows as the MFMA N.

One ``v_mfma_scale_f32_16x16x128_f8f6f4`` multiplies a **unit**: 16 weight rows
(output features) x 128 K of the FP4 weight (the A operand, E8M0 scale per
row and 32 K) by 128 K of up to 16 activation rows (the B operand, FP8 with an
E8M0 scale per row and 32 K). The step's rows are the MFMA N dimension: lane
``l`` feeds activation row ``min(l % 16, rows - 1)``, and lane ``l`` of the
result holds features ``4 (l // 16) + e`` (``e`` < 4) of row ``l % 16``.

Operand layouts (CDNA4 ISA §7.1, "F8F6F4"):

- A (FP4, 16 B per lane): lane ``l`` holds row ``l % 16``, K ``32 j .. 32 j + 31``
  (``j = l // 16``), two codes per byte, even element in the low nibble;
- B (FP8, 32 B per lane): lane ``l`` holds row ``l % 16``, K ``16 j .. 16 j + 15``
  then ``64 + 16 j .. 64 + 16 j + 15``;
- scales: lane ``l`` gives the code of (its row, K ``32 j .. 32 j + 31``), for A
  and B alike.

**Packed weight layout** (``pack_weight``; independent of the task shape and
the scenario): unit ``(rg, st)`` (row group ``rg`` of 16 rows, K step ``st``
of 128) is 1 KiB at byte ``(rg * S + st) * 1024`` (``S = K / 128``), lane
``l``'s 16 B at ``16 l``. That is aiter's ``shuffle_weight(layout=(16, 16))``.
Its scales are 64 B at ``(rg * S + st) * 64``, lane ``l``'s code at byte ``l``
(a task stages its scales in LDS: ``load_scales``).

**Weight formats** (``WeightFormat``, the A operand's format; the activation is always MXFP8):

- ``MXFP4``: as above, 16 B per lane, 1 KiB units (aiter's 16x16 shuffle);
- ``MXFP8``: E4M3 weights, E8M0 per row and 32 K (FP8 block-scaled checkpoints, e.g. 32 x 32 or
  128 x 128 blocks, convert exactly: ``weights.mx.fp8_block_to_mxfp8``). An FP8 A operand
  has the B operand's lane layout: lane ``l`` holds row ``l % 16``, K ``16 j .. 16 j + 15`` then
  ``64 + 16 j .. 64 + 16 j + 15``; 32 B per lane, 2 KiB units stored as two 1 KiB halves (half h: every
  lane's 16 B of K ``64 h ..``, lane order), the same scale layout as MXFP4.

Every helper is a free function of explicit values.
"""

from dataclasses import dataclass

import flydsl.expr as fx
import torch
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from atom.mono.device.mx import clamp_fp8, rcp_pow2

from . import buffer_ops as bo
from .ops import CM_NT, fp8_pack4, xred

FP8, FP4 = 0, 4  # the f8f6f4 MFMA's operand formats (cbsz / blgp)
FP8_MAX = 448.0
OOB = 0x7FFFFFFF  # an element offset no buffer reaches: the load returns 0


@dataclass(frozen=True)
class WeightFormat:
    """The A operand of the f8f6f4 MFMA: ``code`` its cbsz format, ``lane_words`` the
    VGPRs (32-bit words) one lane holds per 16 x 128 unit."""

    name: str
    code: int
    lane_words: int

    @property
    def unit_bytes(self) -> int:
        return 64 * 4 * self.lane_words

    def weight_bytes(self, n: int, k: int) -> int:
        return n * k * self.lane_words // 8

    def stored_k(self, k: int) -> int:
        """Bytes per weight row as stored before packing (FP4: two codes a byte)."""
        return k * self.lane_words // 8


MXFP4 = WeightFormat("mxfp4", FP4, 4)
MXFP8 = WeightFormat("mxfp8", FP8, 8)
FORMATS = {f.name: f for f in (MXFP4, MXFP8)}


# ------------------------------------------------------------------ packing (host)


def pack_weight(
    packed: torch.Tensor, codes: torch.Tensor, fmt: WeightFormat = MXFP4
) -> tuple[torch.Tensor, torch.Tensor]:
    """``packed``: MXFP4 [N, K/2] uint8 (low nibble = even element) or MXFP8 [N, K]
    (float8_e4m3fn or its uint8 bytes); ``codes`` [N, K/32] -> (weight bytes, scale bytes),
    flat, in unit order (module docstring). N must be a multiple of 16 and K of 128."""
    n, kb = packed.shape
    k = kb * 8 // fmt.lane_words
    if n % 16 or k % 128 or codes.shape != (n, k // 32):
        raise ValueError(
            f"pack_weight needs N % 16 == 0, K % 128 == 0, got {tuple(packed.shape)} {tuple(codes.shape)}"
        )
    s = k // 128
    if fmt is MXFP4:
        # weights: [rg, r, st, j, b] -> [rg, st, j, r, b]: lane l = 16 j + r, 16 B each
        w = (
            packed.reshape(n // 16, 16, s, 4, 16)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(-1)
        )
    elif fmt is MXFP8:
        # K within a step = 64 half + 16 j + b: [rg, r, st, half, j, b] -> [rg, st, half, j, r, b], lane l = 16 j + r:
        # 16 B a lane a half (K 16 j .. + 15, then 64 + 16 j .. + 15), each half's 64 lanes one contiguous 1 KiB
        # (one 16 B load instruction reads 1 KiB, not 16 B every 32 B)
        w = (
            packed.view(torch.uint8)
            .reshape(n // 16, 16, s, 2, 4, 16)
            .permute(0, 2, 3, 4, 1, 5)
        )
        w = w.contiguous().reshape(-1)
    else:
        raise ValueError(f"no packer for weight format {fmt.name}")
    # scales: [rg, r, st, j] -> [rg, st, j, r]: lane l = 16 j + r
    c = codes.reshape(n // 16, 16, s, 4).permute(0, 2, 3, 1).contiguous().reshape(-1)
    return w, c


# ------------------------------------------------------------------ device: weights


def load_weight(
    r_w, unit, lane, valid=None, fmt: WeightFormat = MXFP4, k_left=None, cm=CM_NT
):
    """Issue lane ``lane``'s weights of unit ``unit`` (= rg * S + st), nontemporal:
    ``fmt.lane_words`` words (MXFP4 16 B, MXFP8 32 B as two 16 B loads). ``valid`` False:
    reads 0 without touching memory (the buffer must carry its size). ``k_left``: the K
    elements of the unit's step that are live (from its start); a lane's chunk at or past it
    is zero padding, read as 0 the same way (MXFP4: lane l holds K 32 (l // 16) .. + 31;
    MXFP8: K 16 (l // 16) .. + 15, then + 64). ``cm``: the loads' cache-policy bits (default nontemporal).
    """
    halves = (
        fmt.lane_words // 4
    )  # 16 B loads a lane: half h of every lane is one contiguous 1 KiB
    offs = (
        [(unit * 64 + lane) * 4]
        if halves == 1
        else [((unit * halves + h) * 64 + lane) * 4 for h in range(halves)]
    )
    for h in range(halves):
        ok = valid
        if k_left is not None:
            live = fx.Int32((lane // 16) * (32 // halves) + 64 * h) < k_left
            ok = live if ok is None else ok & live
        if ok is not None:
            offs[h] = ok.select(offs[h], fx.Int32(OOB // 4 - (4 if halves > 1 else 0)))
    if halves == 1:
        return fx.Vector(
            bo.buffer_load(r_w, offs[0], vec_width=4, dtype=T.i32, cache_modifier=cm)
        )
    got = [
        fx.Vector(bo.buffer_load(r_w, o, vec_width=4, dtype=T.i32, cache_modifier=cm))
        for o in offs
    ]
    return fx.Vector.from_elements(
        [got[h][e] for h in range(2) for e in range(4)], fx.Int32
    )


def load_scales(r_s, chunk):
    """Issue 16 B chunk ``chunk`` of the scale codes (chunk = unit * 4 + lane // 16:
    the codes of lanes 16 (chunk % 4) .. + 15 of a unit), nontemporal."""
    return fx.Vector(
        bo.buffer_load(r_s, chunk * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )


def mfma_wxa8(acc, w, ws, x, xs, fmt: WeightFormat):
    """acc (vec4 f32) += (W unit in ``fmt``, scale ws) x (activation, FP8, scale xs)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [w, x, acc, fmt.code, FP8, 0, ws, 0, xs]
        )
    )


def widen_bf16(words, codes):
    """A lane's 32 FP8 values (8 words) times their E8M0 scales, as bf16: exact (an FP8 value times a power of two is
    a bf16 value). -> four 8-bf16 MFMA operands, word pairs in order (A and B widened alike keep every K position
    paired). ``codes`` = (code of words 0-3, code of words 4-7): in the 16x16x128 layout lane group j holds K
    [16j, 16j + 16) and [64 + 16j, 64 + 16j + 16), i.e. 32-blocks j // 2 and 2 + j // 2, not block j, whose code
    the MX MFMA's scale register carries (``b_codes_bf16``)."""
    scales = [(fx.Int32(c) << 23).bitcast(fx.Float32) for c in codes]
    vals = []
    for i, w in enumerate(words):
        scale = scales[i // 4]
        for hi in (False, True):
            v = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), fx.Int32(w).ir_value(), scale.ir_value(), hi
                )
            )
            vals += [v[0], v[1]]
    return [
        fx.Vector.from_elements(vals[8 * i : 8 * i + 8], fx.BFloat16) for i in range(4)
    ]


def mfma_bf16_wide(acc, wv, xv):
    """acc (vec4 f32) += the unit's 128 K as four bf16 MFMAs (``widen_bf16`` operands): exact products, fp32 sums."""
    from .ops import mfma_bf16

    for i in range(4):
        acc = mfma_bf16(wv[i], xv[i], acc)
    return acc


# ------------------------------------------------------------------ device: activation in LDS
# The activation of one K slice sits in LDS as [rows][pitch] words of four FP8
# (row-major K) and its codes as [rows][spitch] words (one code per word).


def _v4i():
    return fx.Vector.make_type(4, fx.Int32)


def b_operand(x_lds, xs_lds, row, j, st, pitch, spitch):
    """Lane's FP8 B operand (8 words) and its scale code for K step ``st`` of the slice;
    ``row`` = the lane's activation row, ``j`` = lane // 16."""
    off = row * pitch + st * 32 + j * 4
    a0 = fx.Vector(fx.ptr_load(x_lds + off, result_type=_v4i()))
    a1 = fx.Vector(fx.ptr_load(x_lds + (off + 16), result_type=_v4i()))
    x = fx.Vector.from_elements(
        [a0[e] for e in range(4)] + [a1[e] for e in range(4)], fx.Int32
    )
    xs = fx.ptr_load(xs_lds + (row * spitch + st * 4 + j))
    return x, xs


def b_codes_bf16(xs_lds, row, j, st, spitch):
    """The scale codes of the lane's two 16-value halves of ``b_operand``'s K step ``st`` (``widen_bf16``)."""
    lo = xs_lds + (row * spitch + st * 4 + (j >> 1))
    return fx.ptr_load(lo), fx.ptr_load(lo + 2)


# ------------------------------------------------------------------ device: MXFP8 quantization


def activation_code(amax):
    """E8M0 code of the smallest power of two >= amax / 448, clamped to [1, 254];
    0 for amax == 0 is replaced by 127 (``weights.mx.scale_codes``). Exact:
    448 = 1.75 * 2^8, so the ratio's ceiling is read off amax's exponent and
    mantissa (> 1.75 rounds up), no division."""
    bits = fx.Float32(amax).bitcast(fx.Int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    code = exponent - 8 + (mantissa > 0x600000).select(fx.Int32(1), fx.Int32(0))
    code = fx.max(fx.min(code, fx.Int32(254)), fx.Int32(1))
    return fx.Int32((fx.Float32(amax) == 0.0).select(fx.Int32(127), code))


def quant_word(vals, code):
    """Four f32 of one MX group with scale ``code`` -> a word of four E4M3 (RNE)."""
    inv = rcp_pow2(code)
    return fp8_pack4(*[clamp_fp8(v * inv) for v in vals])


def group_amax(vals, lanes):
    """max |v| over ``vals`` and over the ``lanes`` (2, 4 or 8) adjacent lanes that
    share an MX group (lane groups aligned to ``lanes``)."""
    m = fx.max(
        fx.max(abs_f(vals[0]), abs_f(vals[1])), fx.max(abs_f(vals[2]), abs_f(vals[3]))
    )
    off = 1
    while off < lanes:
        m = xred(m, off, fx.max)
        off *= 2
    return m


def abs_f(v):
    return (fx.Float32(v).bitcast(fx.Int32) & 0x7FFFFFFF).bitcast(fx.Float32)


def silu_f32(g):
    """silu(g) in fp32, no rounding (torch's fp32 ``F.silu``)."""
    from .ops import hw_exp2

    e = hw_exp2(g * fx.Float32(-1.4426950408889634))
    return g / (fx.Float32(1.0) + e)
