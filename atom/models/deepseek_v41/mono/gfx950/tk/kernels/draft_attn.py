# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::attn_fused::attn_fused_kernel<5, 2, true, 0, 3, 0, 8>``.
"""The DSpark draft layers' attention sublayer (5 rows: the token and 4 drafts) on gfx950 (TP8) as a
FlyDSL port of TileRT's ``attn_fused`` kernel: kernarg ABI unchanged (AttnArgs @0, ProjoArgs @112: 216 B
read by offset as 27 i64 words), grid 256 x 512.

- A1 (s2 < 40, row = s2 % 8 < 5, split = s2 / 8): splits 0..3 the window ring before the rows' smallest
  position (key = 32 split + kt + min pos - 128), split 4 the 5 draft tokens' fp8 KV (@0x30, scales
  @0x38; key kt < 5 valid when its row is). The attention body is the tail kernel's (``_a1_body``).
- A2 (s2 >= 192, row < 5, head = s2 / 8 - 24): the row's 5 partials merged, a dimension a thread.
- B1 / B2 / C1: the tail kernel's wo_a / cross-head fp8 / wo_b phases over 5 rows.
- C2: one-shot TP all-reduce (LL64 rows of a (CTA, row) 32-column slice: 28 + 4 columns, flag
  {epoch, epoch + @180}), the ranks summed in order 0..7, bf16 to @0xb8 [row][5120] (plain).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import (
    bf_rne_hi,
    bperm,
    exp_ocml,
    f32c,
    spin_pause,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.layer_tail import (
    AR_IN_LDS,
    AR_PAR,
    AR_SLOT,
    K_ROW,
    LDS_BYTES,
    MERGE_DEN,
    P_ROW,
    QX1,
    VALID_LDS,
    WOB_OUT_LDS,
    _a1_body,
    _any_valid,
    _cvt_bf16_fp8,
    _dppf,
    _ld,
    _ldf,
    _lds_ld,
    _lds_st,
    _lds_u16,
    _lds_vec,
    _max32,
    _sel_fetch,
    proj_b1,
    proj_b2,
    proj_c1,
    woa_fetch,
    wob_fetch,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_SYS,
    kernel_symbol,
    launcher,
    load_ptr64,
    memrealtime,
    readlane,
    rsrc,
    traced,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

ROWS, SPLITS = 5, 5
KARG_WORDS = 27  # AttnArgs (112 B) + ProjoArgs (104 B)
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 256, 512
CHUNKS = 20  # the (m, l) / O partial slots keep the tail kernel's 20 a row
VERSION = "v5"


@traced
def af_a1(tid, s2, q, win_kv, win_s, dkv, dkv_s, vpos, o_part, ml_part, epoch, lds, pl):
    """A (row, split) block: 32 keys a CTA staged as bf16 rows in LDS, then the tail's attention body."""
    row, ch = s2 % 8, s2 // 8
    lane, w = tid % 64, tid // 64
    lg, h = (tid >> 4) & 3, tid & 15
    kt = tid >> 4
    sl = w >> 1
    pos = _ld(vpos, row)
    vp = Int32(-1)
    if lane < ROWS:
        vp = _ld(vpos, lane)
    if pos >= 0:
        # the rows' smallest valid position (TileRT: min butterflies in 8-lane groups, lane 0's value)
        mv = (vp >= 0).select(vp, Int32(0x7FFFFFFF))
        for k in (1, 2, 4):
            mv = fx.min(mv, Int32(bperm(mv.bitcast(F32), lane ^ k).bitcast(Int32)))
        minpos = readlane(mv, 0)
        vk = Int32(
            bperm(vp.bitcast(F32), kt & 7).bitcast(Int32)
        )  # the position of draft row kt
        ok = Int32(0) != Int32(0)
        slot = Int32(0)
        if ch == SPLITS - 1:
            ok = (tid < 16 * ROWS) & (vk >= 0)
            slot = (tid < 16 * ROWS).select(kt, Int32(0))
        else:
            key = ch * 32 + kt + minpos - 128
            ok = key >= 0
            slot = key & 0xFF
        zero4 = fx.Vector.filled(4, 0, Int32)
        qv = [zero4, zero4, zero4, zero4]
        if h < 8:
            for m in range_constexpr(4):
                qv[m] = _ld(
                    q, (row * 8192 + h * 1024 + sl * 256 + lg * 16 + m * 64) // 4, 4
                )
        kw = [Int32(0)] * 16
        if ok:
            a = fx.Vector.filled(4, 0, Int32)
            b = fx.Vector.filled(4, 0, Int32)
            sb = Int32(0)
            if ch == SPLITS - 1:
                a = _ld(dkv, (slot * 512 + h * 32) // 4, 4)
                b = _ld(dkv, (slot * 512 + h * 32 + 16) // 4, 4)
                sb = Int32(
                    bo.buffer_load(rsrc(dkv_s), slot * 16 + h, vec_width=1, dtype=T.i8)
                ).to(Int32)
            else:
                a = _ld(win_kv, (slot * 512 + h * 32) // 4, 4)
                b = _ld(win_kv, (slot * 512 + h * 32 + 16) // 4, 4)
                sb = Int32(
                    bo.buffer_load(rsrc(win_s), slot * 16 + h, vec_width=1, dtype=T.i8)
                ).to(Int32)
            sc = ((sb & 0xFF) << 23 & 0x7F800000).bitcast(F32)
            words = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
            for i in range_constexpr(8):
                kw[2 * i] = _cvt_bf16_fp8(words[i], sc, False)
                kw[2 * i + 1] = _cvt_bf16_fp8(words[i], sc, True)
        for c in range_constexpr(4):
            _lds_st(lds, kt * K_ROW + h * 64 + c * 16, kw[4 * c : 4 * c + 4])
        if h == 0:
            _lds_st(lds, VALID_LDS + kt * 4, [ok.select(Int32(1), Int32(0))])
        gpu.barrier()
        anyv = _any_valid(lds)
        if anyv == 0:
            if tid < 8:
                at = (((row * CHUNKS + ch) * 8) | tid) * 4
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [f32c(0xFF800000).bitcast(Int32), epoch], Int32
                    ),
                    rsrc(ml_part),
                    at,
                )
                bo.buffer_store(
                    fx.Vector.from_elements([Int32(0), epoch], Int32),
                    rsrc(ml_part),
                    at + 2,
                )
        else:
            _a1_body(tid, row, ch, lane, w, lg, h, qv, o_part, ml_part, epoch, lds, pl)


@traced
def af_a2(tid, s2, vpos, sink, ropec, rope_len, o_part, ml_part, xout, epoch, lds):
    """Merge CTA (row = s2 % 8, head = s2 / 8 - 24): weights exp(m_j - M) of the 5 partials, the
    denominator exp(sink - M) + sum l_j exp(m_j - M); thread t merges dimension t (sc1 loads),
    inverse RoPE on the last 64 dims, bf16 pairs as LL packs {bf16x2, epoch} (sc1) at ProjoArgs+0x28.
    """
    row, head = s2 % 8, s2 // 8 - 24
    lane = tid % 64
    even = (tid & 1) == 0
    pos = _ld(vpos, row)
    out_word = (row * 2048 + head * 256 + (tid >> 1)) * 2
    if pos >= 0:
        cs = F32(0.0)
        sn = F32(0.0)
        if tid >= 448:
            trow = fx.min(pos, rope_len - 1)
            pr = (tid - 448) >> 1
            cs = _ldf(ropec, trow * 64 + pr)
            sn = _ldf(ropec, trow * 64 + 32 + pr)
        if tid < 64:
            snk = _ldf(sink, head)
            mm = f32c(0xFF800000)
            ll = F32(0.0)
            if lane < SPLITS:
                word = (row * 160 + head + lane * 8) * 4
                pm = _ld(ml_part, word, 2, CM_DEV)
                pl2 = _ld(ml_part, word + 2, 2, CM_DEV)
                while (pm[1] != epoch) | (pl2[1] != epoch):
                    spin_pause()
                    pm = _ld(ml_part, word, 2, CM_DEV)
                    pl2 = _ld(ml_part, word + 2, 2, CM_DEV)
                mm = pm[0].bitcast(F32)
                ll = pl2[0].bitcast(F32)
            ok = ll > 0.0
            mv = ok.select(mm, f32c(0xFF800000))
            big = _max32(mv, lane)
            e = ok.select(exp_ocml(mv - big, False), F32(0.0))
            lw = ok.select(ll * e, F32(0.0))
            if lane < SPLITS:
                _lds_st(lds, lane * 4, [e.bitcast(Int32)])
            tot = F32(0.0) + readlane(lw, 0)
            for j in range_constexpr(1, SPLITS):
                tot = tot + readlane(lw, j)
            if lane == 0:
                _lds_st(
                    lds, MERGE_DEN, [(exp_ocml(snk - big, False) + tot).bitcast(Int32)]
                )
        gpu.barrier()
        ev = _lds_vec(lds, 0, 4) + [
            _lds_ld(lds, 16).bitcast(F32)
        ]  # TileRT: ds_read_b128 + one word
        ov = [
            _ld(o_part, (row * 160 + head + i * 8) * 512 + tid, 1, CM_DEV).bitcast(F32)
            for i in range(SPLITS)
        ]
        a = (ev[0] > 0.0).select(fmath.fma(ev[0], ov[0], F32(0.0)), F32(0.0))
        for i in range_constexpr(1, SPLITS):
            a = a + (ev[i] > 0.0).select(ev[i] * ov[i], F32(0.0))
        o = a / _lds_ld(lds, MERGE_DEN).bitcast(F32)
        ob = bf_rne_hi(o)
        if tid >= 448:
            xs = (ob & Int32(-65536)).bitcast(F32)
            nb = _dppf(xs, QX1)
            x0 = even.select(xs, nb)
            x1 = even.select(nb, xs)
            ob = bf_rne_hi(even.select(cs * x0 + sn * x1, cs * x1 - sn * x0))
        hi = (ob >> 16) & 0xFFFF
        nh = Int32(_dppf(hi.bitcast(F32), QX1).bitcast(Int32))
        if even:
            bo.buffer_store(
                fx.Vector.from_elements([hi | (nh << 16), epoch], Int32),
                rsrc(xout),
                out_word,
                cache_modifier=CM_DEV,
            )
    else:
        if even:
            bo.buffer_store(
                fx.Vector.from_elements([Int32(0), epoch], Int32),
                rsrc(xout),
                out_word,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()


@traced
def af_c2(tid, s2, tab, me, npes, ehi, epoch, out, lds):
    """CTAs s2 < 160: one-shot TP all-reduce of the 5 x 32 wo_b partials. A (row r, peer) item is 2 LL64
    rows of 7 x 8 B (4 bf16 each: columns 0..27 and 28..31) + an 8 B flag {epoch, epoch + ehi} (sc0 sc1)
    at the peer's sym:projo slot of this rank. Receive (TileRT's mapping: item tid / 16, and + 32 for
    groups 0..2; item = r * 7 + slot) into AR_IN_LDS + item * 64, then each element summed over the
    ranks 0..7 in fp32, bf16 RNE, to out[r][32 s2 + col] (plain)."""
    par = (epoch & 1) * AR_PAR
    flag_hi = epoch + ehi
    i0 = tid >> 4
    j = tid & 7
    rq = (tid >> 3) & 1
    e = rq * 28 + j * 4
    data = (j != 7) & (e < 29)
    if npes >= 2:
        # Push: wave p < 7 -> the p-th peer (a wave-uniform base: a buffer descriptor is scalar; TileRT's
        # item-per-16-lanes push uses per-lane 64-bit global stores), rows 4 pass + lane / 16 (< 5);
        # the same bytes at the same addresses as TileRT's.
        wv, lane = tid // 64, tid % 64
        if wv < 7:
            dest = wv + (wv >= me).select(1, 0)
            base = load_ptr64(tab, dest)
            slot = me - (me > dest).select(1, 0)
            for ps in range_constexpr(2):
                r = 4 * ps + (lane >> 4)
                if r < ROWS:
                    val = fx.Vector.from_elements([epoch, flag_hi], Int32)
                    if data:
                        val = _lds_ld(lds, WOB_OUT_LDS + r * 64 + e * 2, 2)
                    off = slot * AR_SLOT + par + (s2 * ROWS + r) * 128 + rq * 64 + j * 8
                    bo.buffer_store(val, rsrc(base), off >> 2, cache_modifier=CM_SYS)
        gpu.barrier()
        own = load_ptr64(tab, me)
        for it in range_constexpr(2):
            i = i0 + 32 * it
            if i < ROWS * 7:
                r = i // 7
                p = i - r * 7
                rb = p * AR_SLOT + par + (s2 * ROWS + r) * 128 + rq * 64
                fl = _ld(own, (rb + 56) >> 2, 2, CM_SYS)
                while ~((fl[0] == epoch) & (fl[1] == flag_hi)):
                    spin_pause()
                    fl = _ld(own, (rb + 56) >> 2, 2, CM_SYS)
                if data:
                    d = _ld(own, (rb + j * 8) >> 2, 2, CM_SYS)
                    _lds_st(lds, AR_IN_LDS + i * 64 + e * 2, [d[0], d[1]])
        gpu.barrier()
    if tid < ROWS * 32:
        r, col = tid >> 5, tid & 31
        loc = WOB_OUT_LDS + r * 64 + col * 2
        outv = _lds_u16(lds, loc)
        if npes >= 2:
            acc = F32(0.0)
            for k in range_constexpr(8):
                rem = AR_IN_LDS + (r * 7 + k - (me < k).select(1, 0)) * 64 + col * 2
                acc = acc + (_lds_u16(lds, (me == k).select(loc, rem)) << 16).bitcast(
                    F32
                )
            outv = (bf_rne_hi(acc) >> 16) & 0xFFFF
        bo.buffer_store(
            outv.to(fx.Int16), rsrc(out), r * 5120 + s2 * 32 + col
        )  # i16 elements


@traced
def _entry(tid, s2, lds, pl, a00, a01, a02, a06, a07, a08, a09, a10, a11, a12, a13, a15, a16, a17, a18,
           a19, a20, a21, a22, a23):  # fmt: skip
    """Kernarg words (8 B each): AttnArgs q @0, window KV / scales @8 / @16, draft KV / scales @48 / @56,
    vpos @64, attn_sink @72, rope table @80 and its length @88, O partials @96, (m, l) @104; ProjoArgs
    wo_a / scales @120 / @128, wo_b / scales @136 / @144, the LL packs @152 (+0x28), the sym:projo table
    @160, mype | npes << 32 @168, the epoch @176 (and the flag's second-word offset @180), the output @184.
    """
    row = s2 % 8
    a1c = (s2 < 8 * SPLITS) & (row < ROWS)
    att = a1c | ((s2 >= 192) & (row < ROWS))
    # TileRT: the other CTAs wait ~1 us so the attention reads go out first, then fetch wo_a / wo_b; the
    # attention CTAs branch past both (no masked loads waiting on the wo_a / wo_b kernarg words)
    z4 = fx.Vector.filled(4, 0, Int32)
    pa = [z4, z4, Int32(0)]
    pb = [z4, z4, z4, z4, Int32(0)]
    if ~att:
        t0 = memrealtime()
        while (memrealtime() - t0) < 100:
            spin_pause()
        pa = woa_fetch(tid, s2, a15, a16, s2 >= 0)
        pb = wob_fetch(tid, s2, a17, a18, s2 < 160)
    ep = Int32(a22)
    if a1c:
        af_a1(tid, s2, a00, a01, a02, a06, a07, a08, a12, a13, ep, lds, pl)
    if (s2 >= 192) & (row < ROWS):
        pa = woa_fetch(
            tid, s2, a15, a16, s2 >= 0
        )  # TileRT: the merge CTAs fetch wo_a before the poll
        af_a2(tid, s2, a08, a09, a10, Int32(a11), a12, a13, a19, ep, lds)
    pa = _sel_fetch(a1c, woa_fetch(tid, s2, a15, a16, a1c), pa)
    pb = _sel_fetch(a1c, wob_fetch(tid, s2, a17, a18, a1c & (s2 < 160)), pb)
    proj_b1(tid, s2, pa, a19, ep, lds, ROWS)
    if ((s2 // 8) % 8) == 0:
        proj_b2(tid, s2, a19, ep, ROWS)
    if s2 < 160:
        proj_c1(tid, s2, pb, a19, ep, lds, ROWS)
        af_c2(
            tid, s2, a20, Int32(a21), Int32(a21 >> 32), Int32(a22 >> 32), ep, a23, lds
        )


def build():
    """(symbol, launcher) of the FlyDSL attn_fused kernel."""
    name = kernel_symbol("tk_attn_fused", phases=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_BYTES // 4, 16]
        p: fx.Array[fx.Int16, 16 * P_ROW, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_fused(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64,
    ):  # fmt: skip
        sm = fx.SharedAllocator().allocate(Smem).peek()
        _entry(fx.thread_idx.x, fx.block_idx.x, sm.w.ptr, sm.p.ptr, a00, a01, a02, a06, a07, a08, a09, a10,
               a11, a12, a13, a15, a16, a17, a18, a19, a20, a21, a22, a23)  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        attn_fused(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
