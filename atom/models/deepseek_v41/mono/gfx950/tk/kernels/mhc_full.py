# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::mhc::mhc_v3_kernel<5, 0, false, 10, 32, 0, 8, 0>``.
"""mHC FULL (MODE 0: POST + PRE + the next mix-GEMV partials) for the DSpark draft layers' five rows on
gfx950, a FlyDSL port of TileRT's ``mhc_v3_kernel<5, 0, ..>``: kernarg ABI unchanged (``Params`` 200 B +
``engram::HashParams`` 184 B read as 48 i64 words: y @0x0, X @0x8, mix @0x18 + row stride @0x20, fn @0x28,
gamma @0x40 (bf16 when @0x48), y_out / fp8 / ue8m0 / mean @0x58..0x70, X_new copy @0x78, ws @0x80, epoch
@0xb8, 8-rank LL @0xc0), grid 210 x 256 (CTAs 0..49 POST + PRE, 50..209 GEMV K slices).

The structure is ``mhc_kg``'s (y, X_new, PRE identical) with these differences: y is the plain all-reduced
bf16 [row][5120] at @0x0 when @0xc0 is null (every decode-graph launch); the GEMV threads 0..95 hold
their fn slice in VGPRs and run all 5 rows, each as MODE 2's chain (20 fmas from 0, then 12 mul + add).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.mhc_post_pre import (
    _load_x,
    _post_pre_core,
    _recv_ll_y,
    _x_new,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.mhc_pre import (
    HC_OFF,
    ISSUE,
    QP_1032,
    QP_2301,
    _bf_hi,
    _bf_lo,
    _dpp,
    _ld,
    _pack_bf,
    _st,
)
from atom.mono.device.ops import kernel_symbol, launcher, traced

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 48  # 384 B
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
ROWS, SLICES, KSLICES, MIX, HC, DIM = 5, 10, 160, 24, 4, 5120
PRE_CTAS = ROWS * SLICES
GRID, THREADS = PRE_CTAS + KSLICES, 256
LDS_WORDS = 640  # X_new [row 5][hc 4][32] f32
VERSION = "v1"


@traced
def _get_y(y_in, ll, epoch, row, col):
    """y of columns col..col+7: the plain bf16 [row][5120] at @0x0 when the LL buffer is null, else the
    8-rank LL receive."""
    ys = [F32(0.0) for _ in range(8)]
    if ll == 0:
        y4 = _ld(y_in, (row * 0x2800 + col * 2) >> 2, 4)
        for q in range_constexpr(4):
            ys[2 * q] = _bf_lo(y4[q])
            ys[2 * q + 1] = _bf_hi(y4[q])
    else:
        yl = _recv_ll_y(ll, epoch, row, col)
        for e in range_constexpr(8):
            ys[e] = yl[e]
    return ys


def _load_mix(mixp, row, stride):
    c = []
    for q in range(6):
        cv = _ld(mixp, row * stride + 4 * q, 4)
        c += [cv[i].bitcast(F32) for i in range(4)]
    return c


@traced
def _gemv(tid, k, y_in, X, mixp, stride, ll, epoch, fn, ws, xn_out, w):
    col0 = k * 32
    mix, hc = tid >> 2, tid & 3
    fnv = [fx.Vector.filled(4, 0, Int32) for _ in range(8)]
    if tid < 96:
        base = mix * (HC * DIM) + hc * DIM + col0
        for i in range_constexpr(
            8
        ):  # TileRT's issue order: 48, 32, 16, 0, 96, 112, 80, 64 B
            c = ISSUE[i]
            fnv[c] = _ld(fn, base + 4 * c, 4)
    if tid < 4 * ROWS:
        row, cseg = tid >> 2, (tid & 3) * 8
        col = col0 + cseg
        xs0 = _load_x(X, row, col)
        c = _load_mix(mixp, row, stride)
        ys = _get_y(y_in, ll, epoch, row, col)
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
                w + (row * 128 + h * 32 + cseg),
            )
            fx.ptr_store(
                fx.Vector.from_elements(xs[h][4:8], F32),
                w + (row * 128 + h * 32 + cseg + 4),
            )
    gpu.barrier()
    if tid < 96:
        f = []
        for c in range_constexpr(8):
            f += [fnv[c][j].bitcast(F32) for j in range(4)]
        for row in range_constexpr(ROWS):
            xv = []
            for c in range_constexpr(8):
                x4 = fx.Vector(
                    fx.ptr_load(
                        w + (row * 128 + hc * 32 + 4 * c),
                        result_type=fx.Vector.make_type(4, F32),
                    )
                )
                xv += [x4[j] for j in range(4)]
            acc = fmath.fma(f[0], xv[0], F32(0.0))
            for j in range_constexpr(1, 20):
                acc = fmath.fma(f[j], xv[j], acc)
            for j in range_constexpr(20, 32):
                acc = acc + f[j] * xv[j]
            acc = acc + _dpp(acc, QP_1032, bound_ctrl=True)
            other = _dpp(acc, QP_2301)
            if hc == 0:
                _st(ws, (row * KSLICES + k) * MIX + mix, [(acc + other).bitcast(Int32)])


@traced
def _post(
    lane,
    cta,
    y_in,
    X,
    mixp,
    stride,
    ll,
    epoch,
    gamma,
    gflag,
    ws,
    y_out,
    q8_out,
    sc_out,
    mean_out,
):
    row = cta // SLICES
    col = (cta - row * SLICES) * 512 + lane * 8
    xs0 = _load_x(X, row, col)
    c = _load_mix(mixp, row, stride)
    ys = _get_y(y_in, ll, epoch, row, col)
    xs = _x_new(xs0, c, ys)
    _post_pre_core(lane, cta, xs, c, gamma, gflag, ws, y_out, q8_out, sc_out, mean_out)


@traced
def _dispatch(tid, cta, w, y_in, X, mixp, stride, fn, gamma, gflag, y_out, q8_out, sc_out, mean_out, xn_out, ws,
              epoch, ll):  # fmt: skip
    if cta >= PRE_CTAS:
        _gemv(tid, cta - PRE_CTAS, y_in, X, mixp, stride, ll, epoch, fn, ws, xn_out, w)
    if (cta < PRE_CTAS) & (tid < 64):
        _post(
            tid,
            cta,
            y_in,
            X,
            mixp,
            stride,
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
    """(kernel symbol, launcher) of the port; the launcher takes the 48 kernarg words."""
    name = kernel_symbol("tk_mhc_v3_full", rows=ROWS, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Float32, LDS_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mhc_full(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
    ):  # fmt: skip
        w = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _dispatch(fx.thread_idx.x, fx.block_idx.x, w, a00, a01, a03, Int32(a04), a05, a08, Int32(a09), a11, a12, a13,
                  a14, a15, a16, Int32(a23), a24)  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        mhc_full(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
