# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::mhc::mhc_v3_kernel<5|6, 3, false, 10, 32, 0, 8, 0>``.
"""mHC MODE 3 (POST + PRE, no mix GEMV) for 5 or 6 rows (TK_MHC3_ROWS) on gfx950, a FlyDSL port of
TileRT's ``mhc_v3_kernel<R, 3, ..>``: kernarg ABI of the other mhc_v3 modes (48 i64 words), grid R * 10 x
256, wave 0 of each (row, 512-column slice) CTA. y (plain @0x0 when the LL @0xc0 is null), X_new with the
mhc_cell order written back in place to X (@0x8), optional mean, then the MODE-2 PRE without the all-hc
sum of squares (``mhc_kg._post_pre_core``)."""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.mhc_full import _get_y, _load_mix
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.mhc_post_pre import (
    _load_x,
    _post_pre_core,
    _x_new,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.mhc_pre import (
    HC_OFF,
    _pack_bf,
    _st,
)
from atom.mono.device.ops import kernel_symbol, launcher, traced

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 48
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
ROWS = int(os.environ.get("TK_MHC3_ROWS", "6"))
SLICES = 10
GRID, THREADS = ROWS * SLICES, 256
VERSION = "v1"


@traced
def _post3(
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
    for h in range_constexpr(4):  # X_new back to X in place
        _st(
            X,
            (row * 0xA000 + HC_OFF[h] + col * 2) // 4,
            [_pack_bf(xs[h][2 * q], xs[h][2 * q + 1]) for q in range(4)],
        )
    _post_pre_core(
        lane, cta, xs, c, gamma, gflag, ws, y_out, q8_out, sc_out, mean_out, False
    )


@traced
def _dispatch(
    tid,
    cta,
    y_in,
    X,
    mixp,
    stride,
    gamma,
    gflag,
    y_out,
    q8_out,
    sc_out,
    mean_out,
    ws,
    epoch,
    ll,
):
    if tid < 64:
        _post3(
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
    name = kernel_symbol("tk_mhc_v3_mode3", rows=ROWS, v=VERSION)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mhc_mode3(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
    ):  # fmt: skip
        _dispatch(fx.thread_idx.x, fx.block_idx.x, a00, a01, a03, Int32(a04), a08, Int32(a09), a11, a12, a13, a14,
                  a16, Int32(a23), a24)  # fmt: skip

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
        mhc_mode3(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
