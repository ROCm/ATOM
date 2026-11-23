# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::engram::engram_fused_kernel<6, false, 1>``.
"""Engram key / value projection and apply of the first engram layer on gfx950, a FlyDSL port of TileRT's
``engram_fused_kernel<6, false, 1>``: kernarg ABI unchanged (``GemvParams`` 0x00, ``ApplyParams`` 0x50,
the GEMV CTA count at 0x78; read as 16 i64 words), grid 224 x 512.

GEMV CTAs (s2 < n_gemv = 200): 16 of this rank's 3200 wkv rows (rows rank 3200 + 16 s2 ..) for the six
tokens, K = 6144 (wave w: K 768 w .. +767, 24 chunks of 32):
- A = the packed fp8 weights (nt, 16 B a lane at (s2 0x60 + 12 w) KiB + lane 16 + i KiB, i < 12; chunk
  j is half j % 2 of load j / 2), B = the token's gathered fp8 embedding (act row 2 tok + slot, 8 B a lane
  at w 768 + q 8 + 32 j), both to bf16 by v_cvt_scalef32_pk_bf16_fp8: the weights with scale 1, the
  embedding with 2^(ea + ew - 254) of the chunk's two e8m0 bytes (0 for lanes of tokens >= 6);
- 24 chained bf16 MFMAs 16x16x32 (K order), the 8 waves' accumulators summed by wave 0 in order;
- lane (tok, q) publishes features 4q..4q+3 as one 16 B LL entry {bf16x2, epoch} x 2 at entry
  (rank 3200 + 16 s2) / 4 + q + tok 0x1900 of every rank's area (tab[p] + 0x63000 / 0x12b000 by the
  epoch's parity; sc0 sc1).
Apply CTAs (s2 >= n_gemv, 24 = 6 rows x 4 mHC copies): engram_post's gated apply with x = X2[r 4 + c]
itself, written back in place (``engram_post._entry(direct=True)``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import bf_rne_hi
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.engram_post import LL_PAR
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.engram_post import _entry as _apply
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.layer_tail import (
    _ld,
    _lds_ld,
    _lds_st,
)
from atom.mono.device.ops import (
    CM_NT,
    CM_SYS,
    kernel_symbol,
    launcher,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 16
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 224, 512
VERSION = "v1"
LDS_WORDS = 2048  # 8 waves x 64 lanes x 4 f32 (the apply CTAs use the first 24 words)


def _fp8x8_bf16_scaled(w0, w1, scale):
    """Eight fp8 of two dwords -> bf16 MFMA operand, each value times ``scale`` (v_cvt_scalef32)."""
    out = []
    for w in (w0, w1):
        for hi in (False, True):
            p = rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2, as_ir_value(Int32(w)), as_ir_value(F32(scale)), hi
            )
            out.append(fx.Vector(p).bitcast(Int32)[0])
    return fx.Vector.from_elements(out, Int32).bitcast(fx.BFloat16)


def _byte(words, j):
    return (words[j // 4] >> (8 * (j % 4))) & 0xFF


@traced
def _gemv(tid, s2, lds, act, act_s, slot, wkv, wkv_s, nrows, tab, rank, npes, ep):
    lane, w = tid % 64, tid // 64
    h, q = lane % 16, (lane >> 4) & 3
    tok_ok = h < 6
    arow = slot + tok_ok.select(2 * h, Int32(0))
    # e8m0 bytes of the wave's 24 K chunks: weights (32-row blocks) and the token's embedding
    wsb = _ld(wkv_s, ((s2 >> 1) * 0xC0 + w * 24) >> 2, 4)
    wsb2 = _ld(wkv_s, ((s2 >> 1) * 0xC0 + w * 24 + 16) >> 2, 2)
    asb = _ld(act_s, (arow * 0xC0 + w * 24) >> 2, 4)
    asb2 = _ld(act_s, (arow * 0xC0 + w * 24 + 16) >> 2, 2)
    ab = arow * 0x1800 + w * 0x300 + q * 8
    av = [_ld(act, (ab + 32 * j) >> 2, 2) for j in range(24)]
    wb = ((s2 * 0x60 + w * 12) << 10) + lane * 16
    wv = [
        fx.Vector(
            bo.buffer_load(rsrc(wkv), (wb + i * 1024) >> 2, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
        )
        for i in range(12)
    ]  # fmt: skip
    pv = [_ld(tab, 2 * p, 2) for p in range(8)]
    ws_w = [wsb[0], wsb[1], wsb[2], wsb[3], wsb2[0], wsb2[1]]
    as_w = [asb[0], asb[1], asb[2], asb[3], asb2[0], asb2[1]]
    acc = fx.Vector.filled(4, 0.0, F32)
    for j in range_constexpr(24):
        sc = tok_ok.select(
            ((_byte(as_w, j) + _byte(ws_w, j)) << 23) + Int32(-1065353216), Int32(0)
        ).bitcast(
            F32
        )  # 2^(ea + ew - 254): + 0xC0800000 (-4.0) wraps the exponent sum
        a = _fp8x8_bf16_scaled(
            wv[j // 2][2 * (j % 2)], wv[j // 2][2 * (j % 2) + 1], F32(1.0)
        )
        b = _fp8x8_bf16_scaled(av[j][0], av[j][1], sc)
        acc = mfma_bf16(a, b, acc)
    # the 8 ranks' LL areas as buffer descriptors, built outside the divergent publish branch (inside
    # it LLVM wraps each descriptor in a waterfall loop)
    par = ((ep & 1) != 0).select(Int32(LL_PAR[1]), Int32(LL_PAR[0]))
    peers = [
        rsrc(((Int64(uniform(pv[p][1])) << 32) | Int64(fx.Uint32(uniform(pv[p][0])))) + par.to(Int64))
        for p in range(8)
    ]  # fmt: skip
    _lds_st(lds, w * 1024 + lane * 16, [acc[i].bitcast(Int32) for i in range(4)])
    gpu.barrier()
    if (tid < 64) & tok_ok & (npes > 0):
        f = [acc[i] for i in range(4)]
        for k in range_constexpr(1, 8):
            pk = _lds_ld(lds, k * 1024 + lane * 16, 4)
            f = [f[i] + pk[i].bitcast(F32) for i in range(4)]
        lo = (bf_rne_hi(f[1]) & Int32(-65536)) | ((bf_rne_hi(f[0]) >> 16) & 0xFFFF)
        hi = (bf_rne_hi(f[3]) & Int32(-65536)) | ((bf_rne_hi(f[2]) >> 16) & 0xFFFF)
        data = fx.Vector.from_elements([lo, ep, hi, ep], Int32)
        ent = ((rank * nrows + s2 * 16) >> 2) + q + h * 0x1900
        for p in range_constexpr(8):
            if p < npes:
                bo.buffer_store(data, peers[p], ent * 4, cache_modifier=CM_SYS)


@traced
def _entry(
    tid, s2, lds, a00, a01, a02, a03, a04, a05, a06, a07, a08, a10, a11, a12, a13, a15
):
    n_gemv = Int32(a15)
    if s2 < n_gemv:
        _gemv(tid, s2, lds, a00, a01, Int32(a02), a03, a04, Int32(a05), a06, Int32(a07),
              Int32(a07 >> 32), Int32(a08))  # fmt: skip
    else:
        _apply(tid, s2 - n_gemv, lds, a10, a10, None, None, a11, a12, Int32(a13),
               Int32(a13 >> 32), direct=True)  # fmt: skip


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 16 kernarg words."""
    name = kernel_symbol("tk_engram_fused", rows=6, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def engram_fused(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, lds, a00, a01, a02, a03, a04, a05, a06, a07, a08,
               a10, a11, a12, a13, a15)  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        engram_fused(
            a00,
            a01,
            a02,
            a03,
            a04,
            a05,
            a06,
            a07,
            a08,
            a09,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)

    return name, launch
