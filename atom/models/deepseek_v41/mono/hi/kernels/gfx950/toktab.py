# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::toktab::gather_kernel<8, 0>``.
"""The token-table gather of the first DSpark draft layer on gfx950 as a FlyDSL port of TileRT's
``toktab::gather_kernel<8, 0>``: kernarg ABI unchanged (288 B read by offset as 36 i64 words), one CTA of
256 threads a draft row (grid = rows).

That layer's q, kv and mHC mix depend on the token alone, so TileRT keeps them as per-token tables; row r
(position P = blk_pos[r] >= 0, token t = blk_tok[r]) copies
- q: table A[t] (8 heads x 512 bf16) -> Q[r], each head's dims 448..511 GPT-J RoPE'd at P (bf16 RNE);
- the 24 mHC mix coefficients: table C[t] -> hcA[r];
- kv: table B[t] (512 bf16), dims 448..511 RoPE'd at P, FP8 E4M3 with a power-of-two scale per 32
  (amax / 448 rounded up, at least 1e-4 / 448) -> bkv[r] / bks[r], or the window ring's slot P % 256 when
  bkv is null.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.hi.kernels.gfx950.common import (
    bf_rne_hi,
    bperm,
    f32c,
)
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.layer_tail import _ld
from atom.mono.device.ops import kernel_symbol, launcher, rsrc, traced

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 36
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
THREADS = 256
VERSION = "v1"


def _bf(lo_or_hi_word, hi):
    return (
        (lo_or_hi_word & Int32(-65536)).bitcast(F32)
        if hi
        else (lo_or_hi_word << 16).bitcast(F32)
    )


def _pack(even, odd):
    """Two f32 -> one bf16x2 word, RNE each (odd in the high half)."""
    return (bf_rne_hi(odd) & Int32(-65536)) | ((bf_rne_hi(even) >> 16) & 0xFFFF)


def _fp8x2(a, b, scale):
    o = fx.Vector.from_elements([Int32(0)], Int32).bitcast(fx.Int16)
    r = rocdl.cvt_scalef32_pk_fp8_f32(T.vec(2, T.i16), o.ir_value(), F32(a).ir_value(), F32(b).ir_value(),
                                      F32(scale).ir_value(), False)  # fmt: skip
    return fx.Vector(r).bitcast(Int32)[0] & 0xFFFF


@traced
def _q_rope(v, cs, sn):
    """GPT-J RoPE of four bf16 pairs: even' = cos x0 - (sin x1), odd' = cos x1 + (sin x0), bf16 RNE."""
    out = []
    for j in range_constexpr(4):
        x0, x1 = _bf(v[j], False), _bf(v[j], True)
        c, s = cs[j].bitcast(F32), sn[j].bitcast(F32)
        out.append(_pack(fmath.fma(c, x0, -(s * x1)), fmath.fma(c, x1, s * x0)))
    return out


@traced
def _entry(
    tid,
    r,
    qtab,
    kvtab,
    mixtab,
    blk_tok,
    blk_pos,
    rope,
    q_out,
    hca,
    win_kv,
    win_s,
    bkv,
    bks,
):
    pos = _ld(blk_pos, r)
    if pos >= 0:
        tok = _ld(blk_tok, r)
        a0 = _ld(qtab, tok * 2048 + tid * 4, 4)
        a1 = _ld(qtab, tok * 2048 + 1024 + tid * 4, 4)
        kvw = _ld(kvtab, tok * 256 + tid)
        mix = Int32(0)
        if tid < 24:
            mix = _ld(mixtab, tok * 24 + tid)
        c1, s1 = F32(1.0), F32(0.0)
        if tid > 223:
            c1 = _ld(rope, pos * 64 + (tid - 224)).bitcast(F32)
            s1 = _ld(rope, pos * 64 + 32 + (tid - 224)).bitcast(F32)
        d = (tid * 8) & 0x1F8
        qo = (r * 8192 + tid * 16) >> 2
        o0 = [a0[i] for i in range(4)]
        o1 = [a1[i] for i in range(4)]
        if d >= 448:
            cs = _ld(rope, pos * 64 + ((d - 448) >> 1), 4)
            sn = _ld(rope, pos * 64 + 32 + ((d - 448) >> 1), 4)
            o0 = _q_rope(a0, cs, sn)
            o1 = _q_rope(a1, cs, sn)
        bo.buffer_store(fx.Vector.from_elements(o0, Int32), rsrc(q_out), qo)
        bo.buffer_store(fx.Vector.from_elements(o1, Int32), rsrc(q_out), qo + 1024)
        if tid < 24:
            bo.buffer_store(mix, rsrc(hca), r * 24 + tid)
        y0, y1 = _bf(kvw, False), _bf(kvw, True)
        if tid > 223:
            p1 = s1 * y1
            p2 = c1 * y1
            y0 = (bf_rne_hi(fmath.fma(c1, y0, -p1)) & Int32(-65536)).bitcast(F32)
            y1 = (
                bf_rne_hi(fmath.fma(s1, _bf(kvw, False), p2)) & Int32(-65536)
            ).bitcast(F32)
        m = fx.max(abs(y0), abs(y1))
        lane = tid % 64
        for off in (8, 4, 2, 1):
            m = fx.max(m, bperm(m, lane ^ off))
        m = fx.max(m, f32c(0x38D1B717))
        sb = (m * f32c(0x3B124925)).bitcast(Int32) + 0x7FFFFF
        ex = (sb >> 23) & 0x1FF
        scale = (ex != 0).select(sb & Int32(-8388608), Int32(0x400000)).bitcast(F32)
        q8 = _fp8x2(y0, y1, scale)
        ring = bkv == 0
        idx = ring.select(pos & 0xFF, r)
        if ring:
            bo.buffer_store(
                q8.to(fx.Int16), rsrc(win_kv), idx * 512 + tid * 2, offset_is_bytes=True
            )
            if (tid & 15) == 0:
                bo.buffer_store(
                    ex.to(fx.Int8),
                    rsrc(win_s),
                    idx * 16 + (tid >> 4),
                    offset_is_bytes=True,
                )
        if ~ring:
            bo.buffer_store(
                q8.to(fx.Int16), rsrc(bkv), idx * 512 + tid * 2, offset_is_bytes=True
            )
            if (tid & 15) == 0:
                bo.buffer_store(
                    ex.to(fx.Int8),
                    rsrc(bks),
                    idx * 16 + (tid >> 4),
                    offset_is_bytes=True,
                )


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 36 kernarg words (grid = rows)."""
    name = kernel_symbol("tk_toktab_gather", heads=8, mode=0, v=VERSION)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def gather(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64,
    ):  # fmt: skip
        _entry(
            fx.thread_idx.x,
            fx.block_idx.x,
            a24,
            a25,
            a26,
            a27,
            a28,
            a29,
            a30,
            a31,
            a32,
            a33,
            a34,
            a35,
        )

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        gather(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35,
        ).launch(grid=(5,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
