# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::mhc::mhc_v3_kernel<6, 1, false, 10, 32, 2, 8, 0>``.
"""mHC POST (MODE 1) of the six target rows plus the engram receive on gfx950, a FlyDSL port of TileRT's
``mhc_v3_kernel<6, 1, .., 2, ..>``: kernarg ABI unchanged (``Params`` 200 B + ``engram::HashParams`` 184 B
read as 48 i64 words: y @0x0, X @0x8, mix @0x10 (24 f32 a row), mean @0x70, 8-rank LL @0xc0; act
@0x150, act_s @0x158, tab_engram @0x160, rank | npes << 32 @0x168, epoch @0x170), grid 60 + 12 x 256.

POST (CTAs 0..59, wave 0, row = cta / 10, 512 columns, 8 a lane): y (plain bf16 [row][5120] when the LL
pointer is null, else the 8-rank LL receive), X_new_j = bf16(((((c[8+j] x0 + c[12+j] x1) + c[16+j] x2)
+ c[20+j] x3) + c[4+j] y)) written back over X's four copies; the optional mean 0.25 (((x0 + x1) + x2)
+ x3).
Engram receive (CTA 60 + 2R + L): the 24 x 33 LL entries ``engram_gather`` published for (row R, layer
L) in this rank's area (tab[rank] + 0x31800 for an odd epoch), polled (sc0 sc1, s_sleep 1), entry
(head, e) -> act[2R + L][head 256 + 8 e] (e < 32) or act_s[2R + L][head 8] (e = 32).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr, rocdl

from atom.models.deepseek_v41.mono.hi.kernels.gfx950.layer_tail import OOB, _ld, _ldm
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.mhc_full import _get_y
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.mhc_post_pre import _load_x, _x_new
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.mhc_pre import (
    HC_OFF,
    _pack_bf,
    _st,
)
from atom.mono.device.ops import (
    CM_SYS,
    kernel_symbol,
    launcher,
    rsrc,
    spin_pause,
    traced,
    uniform,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 48
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
ROWS, SLICES = 6, 10
POST_CTAS = ROWS * SLICES
GRID, THREADS = POST_CTAS + 2 * ROWS, 256
VERSION = "v1"
LL_PAR = 0x31800
ITEMS = 24 * 33
NB = 1 << 30
assert OOB * 4 > NB


@traced
def _post(lane, cta, y_in, X, mixp, mean_out, epoch, ll):
    row = cta // SLICES
    col = (cta - row * SLICES) * 512 + lane * 8
    xs0 = _load_x(X, row, col)
    c = []
    for q in range_constexpr(6):
        cv = _ld(mixp, row * 24 + 4 * q, 4)
        c += [cv[i].bitcast(F32) for i in range(4)]
    ys = _get_y(y_in, ll, epoch, row, col)
    xs = _x_new(xs0, c, ys)
    for h in range_constexpr(4):
        _st(
            X,
            (row * 0xA000 + HC_OFF[h] + col * 2) // 4,
            [_pack_bf(xs[h][2 * q], xs[h][2 * q + 1]) for q in range(4)],
        )
    if mean_out != 0:
        m = [(((xs[0][e] + xs[1][e]) + xs[2][e]) + xs[3][e]) * 0.25 for e in range(8)]
        _st(
            mean_out,
            (row * 0x2800 + col * 2) // 4,
            [_pack_bf(m[2 * q], m[2 * q + 1]) for q in range(4)],
        )


@traced
def _receive(tid, s18, act, act_s, tab, rank, ep):
    bv = _ld(tab, rank * 2, 2)
    base = (Int64(uniform(bv[1])) << 32) | Int64(fx.Uint32(uniform(bv[0])))
    base = base + ((ep & 1) != 0).select(Int32(LL_PAR), Int32(0)).to(Int64)
    items = [tid + 256 * k for k in range(4)]
    valid = [Int32(1) == 1, Int32(1) == 1, Int32(1) == 1, items[3] < ITEMS]
    heads = [valid[k].select(items[k] // 33, Int32(0)) for k in range(4)]
    es = [valid[k].select(items[k] - heads[k] * 33, Int32(0)) for k in range(4)]
    words = [((s18 * 24 + heads[k]) * 33 + es[k]) * 4 for k in range(4)]
    d, t = [], []
    for k in range_constexpr(4):
        for half in range_constexpr(2):
            v = _ldm(base, NB, words[k] + 2 * half, valid[k], 2, CM_SYS)
            d.append(Int32(v[0]))
            t.append(Int32(v[1]))
    ok = (~valid[0]) | (t[0] == ep)
    for i in range_constexpr(1, 8):
        ok = ok & ((~valid[i // 2]) | (t[i] == ep))
    while ~ok:
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
        rocdl.s_sleep(1)
        nd, nt = [], []
        for i in range_constexpr(8):
            need = valid[i // 2] & (t[i] != ep)
            v = _ldm(base, NB, words[i // 2] + 2 * (i % 2), need, 2, CM_SYS)
            nd.append(need.select(Int32(v[0]), d[i]))
            nt.append(need.select(Int32(v[1]), t[i]))
        d, t = nd, nt
        ok = (~valid[0]) | (t[0] == ep)
        for i in range_constexpr(1, 8):
            ok = ok & ((~valid[i // 2]) | (t[i] == ep))
    for k in range_constexpr(4):
        data = fx.Vector.from_elements([d[2 * k], d[2 * k + 1]], Int32)
        if valid[k]:
            if es[k] < 32:
                bo.buffer_store(
                    data, rsrc(act), (s18 * 6144 + heads[k] * 256 + es[k] * 8) >> 2
                )
            else:
                bo.buffer_store(data, rsrc(act_s), (s18 * 192 + heads[k] * 8) >> 2)


@traced
def _entry(tid, cta, a):
    if cta < POST_CTAS:
        if tid < 64:
            _post(tid, cta, a[0], a[1], a[2], a[14], Int32(a[23]), a[24])
    else:
        _receive(tid, cta - POST_CTAS, a[42], a[43], a[44], Int32(a[45]), Int32(a[46]))


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 48 kernarg words."""
    name = kernel_symbol("tk_mhc_v3_post_engram", rows=ROWS, v=VERSION)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mhc_post_engram(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
    ):  # fmt: skip
        _entry(fx.thread_idx.x, fx.block_idx.x,
               [a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18,
                a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37,
                a38, a39, a40, a41, a42, a43, a44, a45, a46, a47])  # fmt: skip

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
        mhc_post_engram(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
