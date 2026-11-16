# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::mhc::mhc_v3_kg_kernel<6, 0, false, 10, 32, 8, 8>``.
# ruff: noqa: SIM102  (traced code: nested ifs are nested scf.if, not one fused condition)
"""mHC FULL (POST of the last sublayer + PRE of the next + the next mix-GEMV partials) for six rows on
gfx950, a FlyDSL port of TileRT's ``mhc_v3_kg`` kernel: kernarg ABI unchanged (X @0, mix @8, the
8-rank LL receive buffer @16, gamma @24, fn @32, ws @40, epoch @48, gamma-is-bf16 @52, Params @56;
256 B read as 32 i64 words), grid 220 x 256.

POST (CTAs 0..59, wave 0, row = cta / 10, 512 columns, 8 a lane) and GEMV (CTAs 60..219, threads 0..23:
row = t / 4, 8 of the slice's 32 columns) both receive y: 8 ranks x 4 LL slots {bf16x2, epoch} a lane
(sc0 sc1; rows by epoch parity 0x5000 apart, sources 0x28000 apart), y = bf16(((s0 + 0) + s1) .. + s7),
then X_new_j = bf16((((c[8+j] x0 + c[12+j] x1) + c[16+j] x2) + c[20+j] x3) + c[4+j] y).
- POST: optional mean (Params+0x50), then the MODE-2 PRE of X_new (``mhc_pre``'s arithmetic).
- GEMV: X_new to Params+0x78 (else ws+0x20000) and LDS; fn (24 x 4 x 32 f32, staged in LDS) . X_new as
  four column-mod-4 fma chains, (a0 + a1) + (a2 + a3), quad DPP sums -> ws partials.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath

from atom.models.deepseek_v41.mono.hi.kernels.gfx950.mhc_pre import (
    HC_OFF,
    QP_1032,
    QP_2301,
    _bf_hi,
    _bf_lo,
    _bf_round,
    _c,
    _dpp,
    _fp8x2,
    _ld,
    _pack_bf,
    _st,
    _umax1,
    _wave_sum,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_SYS,
    hw_rsq,
    kernel_symbol,
    launcher,
    rsrc,
    spin_pause,
    traced,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 32  # 256 B
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
ROWS, SLICES, KSLICES, MIX, HC, DIM = 6, 10, 160, 24, 4, 5120
PRE_CTAS = ROWS * SLICES
GRID, THREADS = PRE_CTAS + KSLICES, 256
LDS_WORDS = 3840  # 15360 B: fn [q 8][slot 96][4] f32 at 0, X_new [row 6][hc 4][32] f32 at 0x3000
XN_LDS = 0x3000 // 4
SRC_STRIDE, PAR_ROWS, ROW_BYTES = 0x28000, 64, 0x5000
VERSION = "v3"
_DBG = (
    __import__("os").environ.get("TK_KG_DBG", "0") == "1"
)  # bounded polls + timeout markers in ws
DBG_WS = 0x1F000 // 4  # ws words: [cta] recv-y timeout, [256 + cta] PRE-poll timeout
POLL_LIMIT = 20000 if _DBG else 0x7FFFFFFF


@traced
def _recv_ll_y(ll, epoch, row, col, ws=None, cta=None):
    """y of columns col..col+7 from the 8 ranks' {bf16x2, epoch} LL slots (4 a source, sc0 sc1), polled
    until every tag is the epoch, summed ((s0 + 0) + s1) .. + s7, RNE to bf16."""
    base = ((row + PAR_ROWS * (epoch & 1)) * ROW_BYTES + col * 4) >> 2  # dwords
    words = [base + (k * SRC_STRIDE >> 2) + 2 * s for k in range(8) for s in range(4)]
    v = [_ld(ll, w, 2, CM_SYS) for w in words]
    ok = v[0][1] == epoch
    for i in range_constexpr(1, 32):
        ok = ok & (v[i][1] == epoch)
    it = Int32(0)
    while ~ok & (it < POLL_LIMIT):
        spin_pause()
        rocdl.s_sleep(1)
        it = it + 1
        v = [_ld(ll, w, 2, CM_SYS) for w in words]
        ok = v[0][1] == epoch
        for i in range_constexpr(1, 32):
            ok = ok & (v[i][1] == epoch)
    if const_expr(_DBG):
        if ~ok:
            _st(ws, DBG_WS + cta, [Int32(0xDEAD) + (v[0][1] & 0xFFFF)])
    ys = []
    for s in range_constexpr(4):
        lo = _bf_lo(v[s][0]) + F32(0.0)
        hi = _bf_hi(v[s][0]) + F32(0.0)
        for k in range_constexpr(1, 8):
            lo = lo + _bf_lo(v[k * 4 + s][0])
            hi = hi + _bf_hi(v[k * 4 + s][0])
        ys.append(_bf_round(lo))
        ys.append(_bf_round(hi))
    return ys


@traced
def _recv_y(ll, epoch, row, col, ws=None, cta=None):
    """y of columns col..col+7 (f32 values of bf16). epoch 0 (every launch of the decode graph): the
    upstream kernel already all-reduced y, read plain as bf16 [row][5120] at @16; otherwise the 8-rank LL
    receive (``_recv_ll_y``)."""
    ys = [F32(0.0) for _ in range(8)]
    if epoch == 0:
        y4 = _ld(ll, (row * 0x2800 + col * 2) >> 2, 4)
        for q in range_constexpr(4):
            ys[2 * q] = _bf_lo(y4[q])
            ys[2 * q + 1] = _bf_hi(y4[q])
    else:
        yl = _recv_ll_y(ll, epoch, row, col, ws, cta)
        for e in range_constexpr(8):
            ys[e] = yl[e]
    return ys


def _x_new(xs, c, ys):
    """xs[h][8] (bf16 values), c[24] the mix row, ys[8] -> X_new[h][8] (bf16 values, f32)."""
    out = []
    for j in range(4):
        col = []
        for e in range(8):
            a = c[8 + j] * xs[0][e] + c[12 + j] * xs[1][e]
            a = a + c[16 + j] * xs[2][e]
            a = a + c[20 + j] * xs[3][e]
            a = a + c[4 + j] * ys[e]
            col.append(_bf_round(a))
        out.append(col)
    return out


def _load_x(X, row, col):
    xw = (row * 0xA000 + col * 2) // 4
    x = [_ld(X, xw + off // 4, 4) for off in HC_OFF]
    xs = [[None] * 8 for _ in range(4)]
    for h in range(4):
        for q in range(4):
            xs[h][2 * q] = _bf_lo(x[h][q])
            xs[h][2 * q + 1] = _bf_hi(x[h][q])
    return xs


def _load_mix(mixp, row):
    c = []
    for q in range(6):
        cv = _ld(mixp, row * MIX + 4 * q, 4)
        c += [cv[i].bitcast(F32) for i in range(4)]
    return c


@traced
def _post_pre(
    lane, cta, X, mixp, ll, epoch, gamma, gflag, ws, y_out, q8_out, sc_out, mean_out
):
    row = cta // SLICES
    col = (cta - row * SLICES) * 512 + lane * 8
    xs0 = _load_x(X, row, col)
    c = _load_mix(mixp, row)
    ys = _recv_y(ll, epoch, row, col, ws, cta)
    xs = _x_new(xs0, c, ys)
    _post_pre_core(lane, cta, xs, c, gamma, gflag, ws, y_out, q8_out, sc_out, mean_out)


@traced
def _post_pre_core(
    lane, cta, xs, c, gamma, gflag, ws, y_out, q8_out, sc_out, mean_out, sumsq_all=True
):
    """Given X_new (xs[h][8]) and the mix row c: optional mean, then the MODE-2 PRE (mhc_pre's
    arithmetic), the epoch counter max(1, old + 1) written back last."""
    row = cta // SLICES
    sl = cta - row * SLICES
    col = sl * 512 + lane * 8
    ep_word = (0x1E400 // 4) + cta
    old = _ld(ws, ep_word, 1)
    fp32_gamma = gflag == 0
    g32a = _ld(gamma, col, 4)
    g32b = _ld(gamma, col + 4, 4)
    gbf = _ld(gamma, col // 2, 4)
    if mean_out != 0:
        m = [((xs[0][e] + xs[1][e]) + xs[2][e]) + xs[3][e] for e in range(8)]
        m = [v * 0.25 for v in m]
        _st(
            mean_out,
            (row * 0x2800 + col * 2) // 4,
            [_pack_bf(m[2 * q], m[2 * q + 1]) for q in range(4)],
        )
    g = []
    for e in range_constexpr(8):
        gv32 = (g32a[e] if e < 4 else g32b[e - 4]).bitcast(F32)
        gvb = _bf_lo(gbf[e // 2]) if e % 2 == 0 else _bf_hi(gbf[e // 2])
        g.append(fp32_gamma.select(gv32, gvb))
    xp = []
    for e in range_constexpr(8):
        s = c[0] * xs[0][e] + c[1] * xs[1][e]
        s = s + c[2] * xs[2][e]
        s = s + c[3] * xs[3][e]
        xp.append(_bf_round(s))
    epn = _umax1(Int32(old) + 1)  # v_max_u32(1, old + 1)
    ss = xp[0] * xp[0]
    for e in range_constexpr(1, 8):
        ss = ss + xp[e] * xp[e]
    ssum = _wave_sum(ss)
    ll_word = (0x1E800 // 4) + row * 64
    if lane == 0:
        _st(ws, ll_word + sl * 2, [ssum.bitcast(Int32), epn], cm=CM_DEV)
    if const_expr(
        sumsq_all
    ):  # MODE 0 / kg: every hc's sum of squares for the next mix GEMV (not MODE 3)
        sa = xs[0][0] * xs[0][0]
        for h in range_constexpr(4):
            for e in range_constexpr(8):
                if h or e:
                    sa = sa + xs[h][e] * xs[h][e]
        sall = _wave_sum(sa)
        if lane == 0:
            _st(ws, (0x1E000 // 4) + row * 32 + sl, [sall.bitcast(Int32)])
    act = lane < SLICES
    slot = act.select(lane, Int32(0))
    v = _ld(ws, ll_word + slot * 2, 2, cm=CM_DEV)
    lo, hi = v[0], v[1]
    it2 = Int32(0)
    while act & (hi != epn) & (it2 < POLL_LIMIT):
        spin_pause()
        rocdl.s_sleep(1)
        it2 = it2 + 1
        v = _ld(ws, ll_word + slot * 2, 2, cm=CM_DEV)
        lo, hi = v[0], v[1]
    if const_expr(_DBG):
        if act & (hi != epn):
            _st(ws, DBG_WS + 256 + cta, [Int32(0xBEEF)])
    tot = _wave_sum(act.select(Int32(lo).bitcast(F32), F32(0.0)))
    ms = fmath.fma(tot, _c(0x394CCCCD), _c(0x1E3CE508))
    tiny = ms < _c(0x00800000)
    r = hw_rsq(tiny.select(ms * _c(0x4B800000), ms))
    r = tiny.select(r * _c(0x45800000), r)
    yv = [g[e] * (r * xp[e]) for e in range(8)]
    _st(
        y_out,
        (row * 0x2800 + col * 2) // 4,
        [_pack_bf(yv[2 * q], yv[2 * q + 1]) for q in range(4)],
    )
    if q8_out != 0:
        yb = [_bf_round(v) for v in yv]
        amax = fx.max(fx.max(abs(yb[0]), F32(0.0)), abs(yb[1]))
        for e in range_constexpr(2, 8):
            amax = fx.max(amax, abs(yb[e]))
        n1 = _dpp(amax, QP_1032)
        amax = fx.max(amax, fx.max(n1, n1))
        n2 = _dpp(amax, QP_2301)
        amax = fx.max(fx.max(amax, n2), _c(0x38D1B717))
        sc = (amax * _c(0x3B124925)).bitcast(Int32)
        sbits = (sc & 0x7F800000) + ((sc & 0x7FFFFF) == 0).select(
            Int32(0), Int32(0x800000)
        )
        inv = (Int32(0x7F000000) - (sbits & 0x7F800000)).bitcast(F32)
        q = [v * inv for v in yb]
        w01 = _fp8x2(q[0], q[1], Int32(0), False)
        w23 = _fp8x2(q[2], q[3], Int32(0), False)
        w45 = _fp8x2(q[4], q[5], Int32(0), False)
        w67 = _fp8x2(q[6], q[7], Int32(0), False)
        _st(
            q8_out,
            (row * 0x1400 + col) // 4,
            [(w01 & 0xFFFF) | (w23 << 16), (w45 & 0xFFFF) | (w67 << 16)],
        )
        if (lane % 4) == 0:
            bo.buffer_store(
                (sbits >> 23).to(fx.Int8),
                rsrc(sc_out),
                row * 160 + (col >> 5),
                offset_is_bytes=True,
            )
    if lane == 0:
        _st(ws, ep_word, [epn])


@traced
def _gemv(tid, k, X, mixp, ll, epoch, fn, ws, xn_out, w):
    """Mix-GEMV K slice k (32 columns of each hc): threads 0..23 rebuild X_new of their 8 columns."""
    col0 = k * 32
    # fn: thread t loads (mix t/32 + 8 i, hc (t/8)%4, quad t%8), staged [q][mix*4 + hc][4] in LDS
    fv = []
    for i in range_constexpr(3):
        t = tid + 256 * i
        fv.append(
            _ld(fn, (t >> 5) * 0x5000 + ((t >> 3) & 3) * 0x1400 + col0 + (t & 7) * 4, 4)
        )
    if tid < 24:
        row, cseg = tid >> 2, (tid & 3) * 8
        col = col0 + cseg
        xs0 = _load_x(X, row, col)
        c = _load_mix(mixp, row)
        ys = _recv_y(ll, epoch, row, col, ws, k + PRE_CTAS)
        xs = _x_new(xs0, c, ys)
        dst_ws = xn_out == 0
        for h in range_constexpr(4):
            word = (row * 0xA000 + HC_OFF[h] + col * 2) // 4
            vals = [_pack_bf(xs[h][2 * q], xs[h][2 * q + 1]) for q in range(4)]
            if dst_ws:
                _st(ws, (0x20000 // 4) + word, vals)
            else:
                _st(xn_out, word, vals)
            fx.ptr_store(
                fx.Vector.from_elements(xs[h][0:4], F32),
                w + (XN_LDS + row * 128 + h * 32 + cseg),
            )
            fx.ptr_store(
                fx.Vector.from_elements(xs[h][4:8], F32),
                w + (XN_LDS + row * 128 + h * 32 + cseg + 4),
            )
    for i in range_constexpr(3):
        t = tid + 256 * i
        fx.ptr_store(fv[i].bitcast(F32), w + ((t & 7) * 384 + (t >> 3) * 4))
    gpu.barrier()
    if tid < 192:
        t96 = tid % 96
        mix, hc = t96 >> 2, t96 & 3
        f = []
        for q in range_constexpr(8):
            fl = fx.Vector(
                fx.ptr_load(
                    w + (q * 384 + t96 * 4), result_type=fx.Vector.make_type(4, F32)
                )
            )
            f += [fl[j] for j in range(4)]
        r0 = (tid >= 96).select(Int32(1), Int32(0))
        for rr in range_constexpr(3):
            row = r0 + 2 * rr
            xv = []
            for q in range_constexpr(8):
                x4 = fx.Vector(fx.ptr_load(w + (XN_LDS + row * 128 + hc * 32 + 4 * q),
                                           result_type=fx.Vector.make_type(4, F32)))  # fmt: skip
                xv += [x4[j] for j in range(4)]
            a = [fmath.fma(f[j], xv[j], F32(0.0)) for j in range(4)]
            for q in range_constexpr(1, 8):
                for j in range_constexpr(4):
                    a[j] = fmath.fma(f[4 * q + j], xv[4 * q + j], a[j])
            s = (a[0] + a[1]) + (a[2] + a[3])
            s = s + _dpp(s, QP_1032, bound_ctrl=True)
            other = _dpp(s, QP_2301)
            if hc == 0:
                _st(ws, (row * KSLICES + k) * MIX + mix, [(s + other).bitcast(Int32)])


@traced
def _dispatch(
    tid,
    cta,
    w,
    X,
    mixp,
    ll,
    gamma,
    fn,
    ws,
    epoch,
    gflag,
    y_out,
    q8_out,
    sc_out,
    mean_out,
    xn_out,
):
    if cta >= PRE_CTAS:
        _gemv(tid, cta - PRE_CTAS, X, mixp, ll, epoch, fn, ws, xn_out, w)
    if (cta < PRE_CTAS) & (tid < 64):
        _post_pre(
            tid,
            cta,
            X,
            mixp,
            ll,
            epoch,
            gamma,
            gflag,
            ws,
            y_out,
            q8_out,
            sc_out,
            mean_out,
        )


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 32 kernarg words."""
    name = kernel_symbol("tk_mhc_v3_kg", rows=ROWS, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Float32, LDS_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mhc_kg(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
    ):  # fmt: skip
        w = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _dispatch(fx.thread_idx.x, fx.block_idx.x, w, a00, a01, a02, a03, a04, a05, Int32(a06), Int32(a06 >> 32),
                  a18, a19, a20, a21, a22)  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        mhc_kg(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
