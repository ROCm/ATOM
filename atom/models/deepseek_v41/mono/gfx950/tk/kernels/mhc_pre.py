# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::mhc::mhc_v3_kernel<6, 2, false, 10, 32, 0, 8, 0>``.
"""mHC PRE (MODE 2) for six rows on gfx950, a FlyDSL port of TileRT's kernel that keeps
its kernarg ABI (``Params`` 200 B + ``engram::HashParams`` 184 B, read by offset), grid
(220 x 256: CTAs 0..59 PRE, 60..219 mix-GEMV K slices), LDS and every rounding step, so a
graph launch of either is interchangeable and bit-equal.

PRE (CTA = row * 10 + slice, wave 0, lane = 8 columns):
  x_pre = ((p0 x0 + p1 x1) + p2 x2) + p3 x3   (bf16 X, f32 products), RNE -> bf16
  sumsq(x_pre) -> LL {f32, epoch} at ws+0x1e800 (sc1); sumsq(X, all hc) -> ws+0x1e000
  the row's 10 LL slots polled (s_sleep 1) -> rsq(sum / 5120 + 1e-20)
  y = gamma * (rsq * x_pre) -> bf16; optional mean (0.25 sum_h X) and 1x32 E4M3 + ue8m0
  epoch = max(1, ws[0x1e400 + 4 cta] + 1), written back last
GEMV (CTA 60 + k, 32 columns of each hc): ws[((row 160 + k) 24 + mix)] = sum fn * X
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import spin_pause
from atom.mono.device.ops import (
    CM_DEV,
    hw_rsq,
    kernel_symbol,
    launcher,
    readlane,
    rsrc,
    traced,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 48  # 384 B of kernarg as 48 i64: every field at its TileRT offset
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
ROWS, SLICES, KSLICES, MIX, HC, DIM = 6, 10, 160, 24, 4, 5120
PRE_CTAS = ROWS * SLICES
GRID = PRE_CTAS + KSLICES
THREADS = 256
LDS_WORDS = 3840  # 15360 B: fn staging 0..12287, X slice 12288..15359
HC_OFF = (0, 0x2800, 0x5000, 0x7800)  # bytes: X[row][hc] in a 0xa000-byte row
ISSUE = (
    3,
    2,
    1,
    0,
    6,
    7,
    5,
    4,
)  # TileRT's fn load order (byte offsets 48, 32, 16, 0, 96, 112, 80, 64)

# DPP controls
QP_1032, QP_2301, HALF_MIRROR, MIRROR, BCAST15, BCAST31 = (
    0xB1,
    0x4E,
    0x141,
    0x140,
    0x142,
    0x143,
)


def _dpp(v, ctrl, row_mask=0xF, bank_mask=0xF, bound_ctrl=False):
    w = update_dpp_i32(
        Int32(0), F32(v).bitcast(Int32), ctrl, row_mask, bank_mask, bound_ctrl
    )
    return Int32(w).bitcast(F32)


def _wave_sum(v):
    """TileRT's 64-lane f32 sum (its exact DPP tree), the total in every lane's readlane 63."""
    v = v + _dpp(v, QP_1032, bound_ctrl=True)
    v = v + _dpp(v, QP_2301, bound_ctrl=True)
    v = v + _dpp(v, HALF_MIRROR, bound_ctrl=True)
    v = v + _dpp(v, MIRROR, bound_ctrl=True)
    v = v + _dpp(v, BCAST15, row_mask=0xA)
    v = v + _dpp(v, BCAST31, row_mask=0xC)
    return readlane(v, 63)


def _bf_rne_bits(x):
    """f32 -> its bf16 RNE as the high half of an i32 (low half garbage), no NaN fixup."""
    b = F32(x).bitcast(Int32)
    return b + ((b >> 16) & 1) + 0x7FFF


def _bf_round(x):
    return (_bf_rne_bits(x) & Int32(-65536)).bitcast(F32)


def _pack_bf(lo, hi):
    return (_bf_rne_bits(hi) & Int32(-65536)) | ((_bf_rne_bits(lo) >> 16) & 0xFFFF)


def _bf_lo(w):
    return (Int32(w) << 16).bitcast(F32)


def _bf_hi(w):
    return (Int32(w) & Int32(-65536)).bitcast(F32)


def _fp8x2(a, b, old, hi):
    o = as_ir_value(fx.Vector.from_elements([Int32(old)], Int32).bitcast(fx.Int16))
    r = rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        o,
        as_ir_value(F32(a)),
        as_ir_value(F32(b)),
        as_ir_value(F32(1.0)),
        hi,
    )
    return fx.Vector(r).bitcast(Int32)[0]


def _ld(addr, word, n, cm=0):
    v = bo.buffer_load(rsrc(addr), word, vec_width=n, dtype=T.i32, cache_modifier=cm)
    return fx.Vector(v) if n > 1 else Int32(v)


def _st(addr, word, vals, cm=0):
    data = (
        fx.Vector.from_elements([Int32(x) for x in vals], Int32)
        if len(vals) > 1
        else Int32(vals[0])
    )
    bo.buffer_store(data, rsrc(addr), word, cache_modifier=cm)


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 48 kernarg words."""
    name = kernel_symbol("tk_mhc_v3_pre", rows=ROWS)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Float32, LDS_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mhc_pre(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64,
        a06: Int64, a07: Int64, a08: Int64, a09: Int64, a10: Int64, a11: Int64,
        a12: Int64, a13: Int64, a14: Int64, a15: Int64, a16: Int64, a17: Int64,
        a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64,
        a30: Int64, a31: Int64, a32: Int64, a33: Int64, a34: Int64, a35: Int64,
        a36: Int64, a37: Int64, a38: Int64, a39: Int64, a40: Int64, a41: Int64,
        a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        cta = fx.block_idx.x
        w = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _dispatch(
            tid,
            cta,
            w,
            a01,
            a03,
            Int32(a04),
            a05,
            a08,
            Int32(a09),
            a11,
            a12,
            a13,
            a14,
            a16,
        )

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64,
        a06: Int64, a07: Int64, a08: Int64, a09: Int64, a10: Int64, a11: Int64,
        a12: Int64, a13: Int64, a14: Int64, a15: Int64, a16: Int64, a17: Int64,
        a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64,
        a30: Int64, a31: Int64, a32: Int64, a33: Int64, a34: Int64, a35: Int64,
        a36: Int64, a37: Int64, a38: Int64, a39: Int64, a40: Int64, a41: Int64,
        a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        mhc_pre(
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
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
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
            a36,
            a37,
            a38,
            a39,
            a40,
            a41,
            a42,
            a43,
            a44,
            a45,
            a46,
            a47,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)

    return name, launch


@traced
def _dispatch(
    tid, cta, w, X, mixp, stride, fn, gamma, gflag, y_out, q8_out, sc_out, mean_out, ws
):
    """Kernarg fields by offset: X 0x8, mix 0x18, mix row stride 0x20 (i32), fn 0x28, gamma 0x40,
    gamma-is-bf16 0x48 (i32), y 0x58, fp8 0x60, ue8m0 0x68, mean 0x70, ws 0x80."""
    if cta >= PRE_CTAS:
        _gemv(tid, cta - PRE_CTAS, X, fn, ws, w)
    if (cta < PRE_CTAS) & (tid < 64):
        _pre(
            tid, cta, X, mixp, stride, gamma, gflag, y_out, q8_out, sc_out, mean_out, ws
        )


@traced
def _gemv(tid, k, X, fn, ws, w):
    """Mix-GEMV K slice k: fn (24 x 4 x 32 f32) . X (6 rows x 4 hc x 32) -> 144 partials."""
    col0 = k * 32
    t96 = tid % 96
    rb = (tid // 96) * 3
    mix, hc = t96 // 4, t96 % 4
    if tid < 96:
        fnv = [None] * 8
        base = mix * (HC * DIM) + hc * DIM + col0  # f32 words
        # TileRT's issue order: 48, 32, 16, 0, 96, 112, 80, 64 B
        for i in range_constexpr(8):
            c = ISSUE[i]
            fnv[c] = _ld(fn, base + 4 * c, 4)
        for c in range_constexpr(8):
            fx.ptr_store(fnv[c].bitcast(F32), w + (c * 384 + t96 * 4))
    xr = tid - 192
    if (tid >= 192) & (xr < 24):
        row, cg = xr // 4, (tid % 4) * 8
        xw = (row * 0xA000 + (col0 + cg) * 2) // 4
        for h in range_constexpr(4):
            v = _ld(X, xw + HC_OFF[h] // 4, 4)
            vals = []
            for q in range_constexpr(4):
                vals += [_bf_lo(v[q]), _bf_hi(v[q])]
            fx.ptr_store(
                fx.Vector.from_elements(vals[:4], F32),
                w + (3072 + row * 128 + h * 32 + cg),
            )
            fx.ptr_store(
                fx.Vector.from_elements(vals[4:], F32),
                w + (3072 + row * 128 + h * 32 + cg + 4),
            )
    gpu.barrier()
    if tid < 192:
        f = []
        for c in range_constexpr(8):
            fl = fx.Vector(
                fx.ptr_load(
                    w + (c * 384 + t96 * 4), result_type=fx.Vector.make_type(4, F32)
                )
            )
            for j in range_constexpr(4):
                f.append(fl[j])
        for r in range_constexpr(3):
            row = rb + r
            xs = []
            for c in range_constexpr(8):
                xv = fx.Vector(
                    fx.ptr_load(
                        w + (3072 + row * 128 + hc * 32 + 4 * c),
                        result_type=fx.Vector.make_type(4, F32),
                    )
                )
                for j in range_constexpr(4):
                    xs.append(xv[j])
            acc = fmath.fma(f[0], xs[0], F32(0.0))
            for j in range_constexpr(1, 20):
                acc = fmath.fma(f[j], xs[j], acc)
            for j in range_constexpr(20, 32):
                acc = acc + f[j] * xs[j]
            acc = acc + _dpp(acc, QP_1032, bound_ctrl=True)
            other = _dpp(acc, QP_2301)
            if hc == 0:
                _st(ws, (row * KSLICES + k) * MIX + mix, [(acc + other).bitcast(Int32)])


@traced
def _pre(lane, cta, X, mixp, stride, gamma, gflag, y_out, q8_out, sc_out, mean_out, ws):
    row = cta // SLICES
    sl = cta - row * SLICES
    col = sl * 512 + lane * 8
    ep_word = (0x1E400 // 4) + cta
    old = _ld(ws, ep_word, 1)
    xw = (row * 0xA000 + col * 2) // 4
    x = [_ld(X, xw + off // 4, 4) for off in HC_OFF]
    fp32_gamma = gflag == 0
    # a bf16 gamma (10240 B) read as f32 at col would run up to 10 KB past its end: that faults when the
    # tensor ends a pool page (the draft layers' mtp.*.mhc.attn_gamma in the graph), so the unused f32 read
    # takes elements 0..7
    c32 = fp32_gamma.select(col, Int32(0))
    g32a = _ld(gamma, c32, 4)
    g32b = _ld(gamma, c32 + 4, 4)
    gbf = _ld(gamma, col // 2, 4)
    pre = _ld(mixp, row * stride, 4).bitcast(F32)
    xs = [[None] * 8 for _ in range(4)]
    for h in range_constexpr(4):
        for q in range_constexpr(4):
            xs[h][2 * q] = _bf_lo(x[h][q])
            xs[h][2 * q + 1] = _bf_hi(x[h][q])
    if mean_out != 0:
        m = [((xs[0][c] + xs[1][c]) + xs[2][c]) + xs[3][c] for c in range(8)]
        m = [v * 0.25 for v in m]
        _st(
            mean_out,
            (row * 0x2800 + col * 2) // 4,
            [_pack_bf(m[2 * q], m[2 * q + 1]) for q in range(4)],
        )
    g = []
    for c in range_constexpr(8):
        gv32 = (g32a[c] if c < 4 else g32b[c - 4]).bitcast(F32)
        gvb = _bf_lo(gbf[c // 2]) if c % 2 == 0 else _bf_hi(gbf[c // 2])
        g.append(fp32_gamma.select(gv32, gvb))
    xp = []
    for c in range_constexpr(8):
        s = pre[0] * xs[0][c] + pre[1] * xs[1][c]
        s = s + pre[2] * xs[2][c]
        s = s + pre[3] * xs[3][c]
        xp.append(_bf_round(s))
    epoch = _umax1(Int32(old) + 1)  # v_max_u32(1, old + 1)
    ss = xp[0] * xp[0]
    for c in range_constexpr(1, 8):
        ss = ss + xp[c] * xp[c]
    ssum = _wave_sum(ss)
    ll_word = (0x1E800 // 4) + row * 64
    if lane == 0:
        _st(ws, ll_word + sl * 2, [ssum.bitcast(Int32), epoch], cm=CM_DEV)
    sa = xs[0][0] * xs[0][0]
    for h in range_constexpr(4):
        for c in range_constexpr(8):
            if h or c:
                sa = sa + xs[h][c] * xs[h][c]
    sall = _wave_sum(sa)
    if lane == 0:
        _st(ws, (0x1E000 // 4) + row * 32 + sl, [sall.bitcast(Int32)])
    # the row's 10 LL slots, lane l < 10 polls slot l
    act = lane < SLICES
    slot = act.select(lane, Int32(0))
    v = _ld(ws, ll_word + slot * 2, 2, cm=CM_DEV)
    lo, hi = v[0], v[1]
    while act & (hi != epoch):
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
        rocdl.s_sleep(1)
        v = _ld(ws, ll_word + slot * 2, 2, cm=CM_DEV)
        lo, hi = v[0], v[1]
    tot = _wave_sum(act.select(Int32(lo).bitcast(F32), F32(0.0)))
    ms = fmath.fma(tot, _c(0x394CCCCD), _c(0x1E3CE508))  # sum * (1/5120) + 1e-20, fused
    tiny = ms < _c(0x00800000)
    r = hw_rsq(tiny.select(ms * _c(0x4B800000), ms))
    r = tiny.select(r * _c(0x45800000), r)
    yv = [g[c] * (r * xp[c]) for c in range(8)]
    _st(
        y_out,
        (row * 0x2800 + col * 2) // 4,
        [_pack_bf(yv[2 * q], yv[2 * q + 1]) for q in range(4)],
    )
    if q8_out != 0:
        yb = [_bf_round(v) for v in yv]
        amax = fx.max(fx.max(abs(yb[0]), F32(0.0)), abs(yb[1]))
        for c in range_constexpr(2, 8):
            amax = fx.max(amax, abs(yb[c]))
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
        d0 = (w01 & 0xFFFF) | (w23 << 16)
        d1 = (w45 & 0xFFFF) | (w67 << 16)
        _st(q8_out, (row * 0x1400 + col) // 4, [d0, d1])
        if (lane % 4) == 0:
            bo.buffer_store(
                (sbits >> 23).to(fx.Int8),
                rsrc(sc_out),
                row * 160 + (col >> 5),
                offset_is_bytes=True,
            )
    if lane == 0:
        _st(ws, ep_word, [epoch])


def _c(bits):
    return Int32(bits).bitcast(F32)


def _umax1(v):
    return (Int32(v) == 0).select(Int32(1), Int32(v))
