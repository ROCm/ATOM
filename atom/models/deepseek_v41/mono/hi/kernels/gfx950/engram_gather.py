# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::toktab::gather_kernel<8, 6>``.
"""The DSpark draft's token-table gather plus the engram n-gram lookup of the two engram layers on gfx950, a
FlyDSL port of TileRT's ``toktab::gather_kernel<8, 6>``: kernarg ABI unchanged (``engram::HashParams``
0x00..0xb7, rows at 0xb8, the token-table arguments 0xc0.., 288 B read as 36 i64 words), grid 6 + 12 x 256.

CTAs s2 < rows: ``toktab._entry`` (the draft row's q / kv / mHC mix from the token tables).
CTAs rows + 2 R + L (row R, engram layer L):
1. hash: cur = cmp(R), ctx_k (k = 1..3) = pad when pos < k, cmp(R - k) when vpos[R - k] == pos - k, else
   hist[(pos - k) & 15] (cmp(t) = token_map[vtok[t]] for 0 <= vtok < vocab, else pad); lane h < 24 (head:
   n-gram order 2 + h / 8) rolls cur m0 ^ ctx1 m1 (^ ctx2 m2 from order 3, ^ ctx3 m3 from order 4) over
   sign-extended i64 products, row = rolling % prime[L][h] + offset[L][h] -> LDS and hash[2R + L][h];
   layer 0's thread 0 then writes cur to hist[pos & 15];
2. this rank's 24 / npes heads (wave w: head rank 24 / npes + w): the head's table row (clamped to the
   rank's shard), each lane 4 fp8 x its e8m0 -> bf16 (RNE) -> re-quantized per 32 (8 lanes): scale =
   pow2ceil(max(amax, 1e-4) / 448), E4M3 of y / scale; published to every rank as 33 LL entries of 16 B
   (32 x {fp8 x4 of lanes 2i, 2i + 1, epoch}, then the 8 scale bytes) at entry ((2R + L) 24 + head) 33 + ..
   of tab[p] (+ 0x31800 for an odd epoch), sc0 sc1.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.deepseek_v41.mono.hi.kernels.gfx950.common import (
    bf_rne_hi,
    bperm,
    f32c,
)
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.layer_tail import (
    _ld,
    _lds_ld,
    _lds_st,
)
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.toktab import _entry as _toktab
from atom.mono.device.ops import CM_SYS, kernel_symbol, launcher, rsrc, traced, uniform

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 36
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
THREADS = 256
VERSION = "v1"
LL_PAR = 0x31800  # the LL area of an odd epoch
LDS_WORDS = 48  # the 24 heads' table rows (u64)


def _u64(v):
    """i64 from two dwords (lo, hi)."""
    return (Int64(v[1]) << 32) | Int64(fx.Uint32(v[0]))


def _bf_round(x):
    return (bf_rne_hi(x) & Int32(-65536)).bitcast(F32)


def _fp8x2(a, b):
    o = fx.Vector.from_elements([Int32(0)], Int32).bitcast(fx.Int16)
    r = rocdl.cvt_scalef32_pk_fp8_f32(T.vec(2, T.i16), o.ir_value(), F32(a).ir_value(), F32(b).ir_value(),
                                      F32(1.0).ir_value(), False)  # fmt: skip
    return fx.Vector(r).bitcast(Int32)[0] & 0xFFFF


def _bf16x2(w, hi):
    p = rocdl.cvt_scalef32_pk_bf16_fp8(
        T.bf16x2, as_ir_value(Int32(w)), as_ir_value(F32(1.0)), hi
    )
    return fx.Vector(p).bitcast(Int32)[0]


@traced
def _engram(tid, s2e, lds, vtok, vpos, hist, tmap, w04, emult, eprimes, eoffs, hashbuf, tbl, tbl_s,
            lo_row, hi_row, tab, w20, ep):  # fmt: skip
    lane, w = tid % 64, tid // 64
    R, L = s2e >> 1, s2e & 1
    vocab, pad = Int32(w04), Int32(w04 >> 32)
    rank, npes = Int32(w20), Int32(w20 >> 32)
    toks = [uniform(_ld(vtok, j)) for j in range(6)]
    poss = [uniform(_ld(vpos, j)) for j in range(6)]
    hs = [uniform(_ld(hist, i)) for i in range(16)]

    def pick(vals, i):
        out = vals[0]
        for j in range_constexpr(1, len(vals)):
            out = (i == j).select(vals[j], out)
        return out

    def cmp(j):
        t = pick(toks, j)
        ok = (t >= 0) & (t < vocab)
        return ok.select(uniform(_ld(tmap, ok.select(t, Int32(0)))), pad)

    pos = pick(poss, R)
    cur = cmp(R)
    ctx = []
    for k in range_constexpr(1, 4):
        p = pos - k
        rk = R - k
        match = (rk >= 0) & (pick(poss, fx.max(rk, Int32(0))) == p)
        c = match.select(cmp(fx.max(rk, Int32(0))), pick(hs, p & 15))
        ctx.append((pos < k).select(pad, c))
    if tid < 24:
        m = [_u64(_ld(emult, (L * 4 + i) * 2, 2)) for i in range(4)]
        prime = _u64(_ld(eprimes, (L * 24 + tid) * 2, 2))
        off = _u64(_ld(eoffs, (L * 24 + tid) * 2, 2))
        roll = (cur.to(Int64) * m[0]) ^ (ctx[0].to(Int64) * m[1])
        roll = (tid > 7).select(roll ^ (ctx[1].to(Int64) * m[2]), roll)
        roll = (tid > 15).select(roll ^ (ctx[2].to(Int64) * m[3]), roll)
        row = roll % prime + off
        lo, hi = Int32(row), Int32(row >> 32)
        _lds_st(lds, tid * 8, [lo, hi])
        bo.buffer_store(
            fx.Vector.from_elements([lo, hi], Int32),
            rsrc(hashbuf),
            (s2e * 24 + tid) * 2,
        )
    gpu.barrier()
    if (L == 0) & (tid == 0) & (pos > -1):
        bo.buffer_store(cur, rsrc(hist), pos & 15)
    hpr = Int32(24) // npes
    if tid < hpr * 64:
        head = rank * hpr + w
        rv = _lds_ld(lds, head * 8, 2)
        row = _u64(
            [uniform(rv[0]), uniform(rv[1])]
        )  # the wave's head: one row for all its lanes
        mine = (row >= lo_row) & (row < hi_row)
        local = mine.select(row, lo_row) - lo_row
        dw = _ld(tbl + local * 256, lane)
        sbw = _ld(tbl_s + local * 8, (lane >> 5) & 1)
        sb = (sbw >> (8 * ((lane >> 3) & 3))) & 0xFF
        sc = (sb << 23).bitcast(F32)
        p0, p1 = _bf16x2(dw, False), _bf16x2(dw, True)
        e = [(p0 << 16).bitcast(F32), (p0 & Int32(-65536)).bitcast(F32),
             (p1 << 16).bitcast(F32), (p1 & Int32(-65536)).bitcast(F32)]  # fmt: skip
        y = [_bf_round(sc * e[i]) for i in range(4)]
        a = fx.max(
            fx.max(fx.max(abs(y[0]), abs(y[1])), abs(y[2])),
            fx.max(abs(y[3]), abs(y[3])),
        )
        for x in (4, 2, 1):
            o = bperm(a, lane ^ x)
            a = fx.max(a, fx.max(o, o))
        a = fx.max(a, f32c(0x38D1B717))
        s = (a * f32c(0x3B124925)).bitcast(Int32)
        sbits = (s & 0x7F800000) + ((s & 0x7FFFFF) != 0).select(
            Int32(0x800000), Int32(0)
        )
        inv = (Int32(0x7F000000) - (sbits & 0x7F800000)).bitcast(F32)
        q8 = (_fp8x2(y[2] * inv, y[3] * inv) << 16) | _fp8x2(y[0] * inv, y[1] * inv)
        ebyte = (sbits >> 23) & 0xFF
        partner = Int32(
            fx.Int32(rocdl.ds_bpermute(T.i32, Int32(lane ^ 1) * 4, q8.ir_value()))
        )
        sbs = [
            Int32(rocdl.ds_bpermute(T.i32, Int32(8 * g) * 4, ebyte.ir_value()))
            for g in range(8)
        ]
        scw0 = sbs[0] | (sbs[1] << 8) | (sbs[2] << 16) | (sbs[3] << 24)
        scw1 = sbs[4] | (sbs[5] << 8) | (sbs[6] << 16) | (sbs[7] << 24)
        grp = (s2e * 24 + head) * 33
        par = ((ep & 1) != 0).select(Int32(LL_PAR), Int32(0))
        for pp in range_constexpr(8):
            if pp < npes:
                base = _u64(
                    [uniform(_ld(tab, 2 * pp)), uniform(_ld(tab, 2 * pp + 1))]
                ) + par.to(Int64)
                if (lane & 1) == 0:
                    bo.buffer_store(fx.Vector.from_elements([q8, ep, partner, ep], Int32), rsrc(base),
                                    (grp + (lane >> 1)) * 4, cache_modifier=CM_SYS)  # fmt: skip
                if lane == 0:
                    bo.buffer_store(fx.Vector.from_elements([scw0, ep, scw1, ep], Int32), rsrc(base),
                                    (grp + 32) * 4, cache_modifier=CM_SYS)  # fmt: skip


@traced
def _entry(tid, s2, lds, a):
    rows = Int32(a[23])
    if s2 < rows:
        _toktab(
            tid,
            s2,
            a[24],
            a[25],
            a[26],
            a[27],
            a[28],
            a[29],
            a[30],
            a[31],
            a[32],
            a[33],
            a[34],
            a[35],
        )
    else:
        s2e = s2 - rows
        L = s2e & 1
        tbl = (L == 0).select(a[9], a[10])
        tbl_s = (L == 0).select(a[11], a[12])
        lo_row = (L == 0).select(a[13], a[14])
        hi_row = (L == 0).select(a[15], a[16])
        _engram(tid, s2e, lds, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], tbl, tbl_s, lo_row,
                hi_row, a[19], a[20], Int32(a[21]))  # fmt: skip


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 36 kernarg words (grid = rows + 12)."""
    name = kernel_symbol("tk_toktab_gather", heads=8, mode=6, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def gather(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, lds,
               [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17,
                a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35])  # fmt: skip

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
        ).launch(grid=(18,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
