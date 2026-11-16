# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::engram_post::engram_post_apply_kernel<6>``.
"""Engram's gated apply on gfx950 as a FlyDSL port of TileRT's ``engram_post_apply_kernel<6>``: kernarg
ABI unchanged (``PostApplyParams``, 56 B read as 7 i64 words), grid 24 = 6 rows x 4 mHC copies, 512
threads, every rounding step TileRT's.

CTA (row r = s2 / 4, copy c = s2 % 4); a lane owns columns 4 (t + 512 j), j < 3 (j = 2 only for t < 256):
- x = hcF[r][4 + c] Y + (hcF[r][20 + c] X3 + (hcF[r][16 + c] X2 + (hcF[r][8 + c] X0 + hcF[r][12 + c] X1)))
  (bf16 X / Y, f32 products and sums, no fma), rounded to bf16 (RNE);
- the engram key k (slot c) and value v (slot 4) of the row: two 16 B LL entries a 4-column group,
  {bf16x2, epoch} x 2, polled (sc0 sc1, s_sleep 1) in the rank's own buffer (``tab[rank]``, parity
  0x12b000 / 0x63000 by the epoch's low bit);
- per lane in column order: dot += (qk[c] x) k, sx += x^2, sk += k^2; wave sums (TileRT's DPP tree), the 8
  waves added in order;
- g = (dot (rsq(sk / 5120 + 1e-20) rsq(sx / 5120 + 1e-20))) / sqrt(5120); gate = sigmoid(sign(g)
  sqrt(max(|g|, 1e-6))) (OCML expf, IEEE sqrt and divide);
- X2[r 4 + c] = bf16(gate v + x).
TileRT has thread 0 finish the gate and hand it over LDS behind a second barrier; here every lane
computes it from the 8 waves' partials (same arithmetic, one barrier fewer).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath

from atom.models.deepseek_v41.mono.hi.kernels.gfx950.common import (
    bf_hi,
    bf_lo,
    bf_rne_hi,
    exp_ocml,
    f32c,
)
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.layer_tail import (
    _ld,
    _ldm,
    _lds_ld,
    _lds_st,
    _rsq_scaled,
)
from atom.models.deepseek_v41.mono.hi.kernels.gfx950.mhc_pre import _wave_sum
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

KARG_WORDS = 7
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
ROWS, HC, DIM = 6, 4, 5120
GRID, THREADS = ROWS * HC, 512
VERSION = "v1"
NB = 1 << 30  # buffer bound of the masked loads (their OOB index reads 0)
LL_PAR = (0x63000, 0x12B000)  # the LL area by the epoch's parity
ROW_ENTRIES, COPY_ENTRIES, VALUE_SLOT = 0x1900, 0x500, 0x1400
LDS_WORDS = 24  # [wave][dot, sum k^2, sum x^2]


def _bf_round(x):
    return (bf_rne_hi(x) & Int32(-65536)).bitcast(F32)


def _pack(lo, hi):
    return (bf_rne_hi(hi) & Int32(-65536)) | ((bf_rne_hi(lo) >> 16) & 0xFFFF)


def _halves(w):
    return [bf_lo(w), bf_hi(w)]


@traced
def _entry(tid, s2, lds, X, X2, Y, hcf, qk, tab, word6):
    lane, wv = tid % 64, tid // 64
    r, c = s2 >> 2, s2 & 3
    rank = Int32(word6)
    ep = Int32(word6 >> 32)
    # hcF[r]: [4 + c] scales Y, [8 + 4 h + c] scales X_h
    cy = uniform(_ld(hcf, r * 24 + 4 + c)).bitcast(F32)
    cx = [uniform(_ld(hcf, r * 24 + 8 + 4 * h + c)).bitcast(F32) for h in range(HC)]
    valid = [tid < 768, tid < 768, tid < 256]  # (t + 512 j) * 4 < 5120
    cols = [tid + 512 * j for j in range(3)]
    xw = [
        [_ldm(X, NB, (r * 0xA000 + h * 0x2800 + cols[j] * 8) >> 2, valid[j], 2) for h in range(HC)]
        for j in range(3)
    ]  # fmt: skip
    yw = [_ldm(Y, NB, (r * 0x2800 + cols[j] * 8) >> 2, valid[j], 2) for j in range(3)]
    qw = [_ldm(qk, NB, (c * 0x5000 + cols[j] * 16) >> 2, valid[j], 4) for j in range(3)]
    # the rank's own LL area
    bv = _ld(tab, rank * 2, 2)
    base = (Int64(uniform(bv[1])) << 32) | Int64(fx.Uint32(uniform(bv[0])))
    base = base + ((ep & 1) != 0).select(Int32(LL_PAR[1]), Int32(LL_PAR[0])).to(Int64)
    # dword index of each 8 B {bf16x2, epoch} half: j, (key lo, key hi, value lo, value hi)
    ent = []
    for j in range_constexpr(3):
        ki = r * ROW_ENTRIES + c * COPY_ENTRIES + cols[j]
        vi = r * ROW_ENTRIES + VALUE_SLOT + cols[j]
        ent.append([ki * 4, ki * 4 + 2, vi * 4, vi * 4 + 2])
    d, t = [], []
    for j in range_constexpr(3):
        for k in range_constexpr(4):
            v = _ldm(base, NB, ent[j][k], valid[j], 2, CM_SYS)
            d.append(Int32(v[0]))
            t.append(Int32(v[1]))
    ok = (~valid[0]) | (t[0] == ep)
    for i in range_constexpr(1, 12):
        ok = ok & ((~valid[i // 4]) | (t[i] == ep))
    while ~ok:
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
        rocdl.s_sleep(1)
        nd, nt = [], []
        for i in range_constexpr(12):
            need = valid[i // 4] & (t[i] != ep)
            v = _ldm(base, NB, ent[i // 4][i % 4], need, 2, CM_SYS)
            nd.append(need.select(Int32(v[0]), d[i]))
            nt.append(need.select(Int32(v[1]), t[i]))
        d, t = nd, nt
        ok = (~valid[0]) | (t[0] == ep)
        for i in range_constexpr(1, 12):
            ok = ok & ((~valid[i // 4]) | (t[i] == ep))
    dot, sx, sk = F32(0.0), F32(0.0), F32(0.0)
    xs = []
    for j in range_constexpr(3):
        xh = [_halves(xw[j][h][0]) + _halves(xw[j][h][1]) for h in range(HC)]
        yh = _halves(yw[j][0]) + _halves(yw[j][1])
        kh = _halves(d[4 * j]) + _halves(d[4 * j + 1])
        xj = []
        ndot, nsx, nsk = dot, sx, sk
        for e in range_constexpr(4):
            a = cx[0] * xh[0][e] + cx[1] * xh[1][e]
            a = cx[2] * xh[2][e] + a
            a = cx[3] * xh[3][e] + a
            x = _bf_round(cy * yh[e] + a)
            xj.append(x)
            ndot = ndot + (qw[j][e].bitcast(F32) * x) * kh[e]
            nsx = nsx + x * x
            nsk = nsk + kh[e] * kh[e]
        xs.append(xj)
        dot = valid[j].select(ndot, dot)
        sx = valid[j].select(nsx, sx)
        sk = valid[j].select(nsk, sk)
    wd, wk, wx = _wave_sum(dot), _wave_sum(sk), _wave_sum(sx)
    if lane == 0:
        _lds_st(lds, wv * 12, [wd.bitcast(Int32), wk.bitcast(Int32), wx.bitcast(Int32)])
    gpu.barrier()
    p = [_lds_ld(lds, 16 * i, 4) for i in range(6)]
    pw = [p[i // 4][i % 4].bitcast(F32) for i in range(24)]
    td, tk, tx = pw[0], pw[1], pw[2]
    for w in range_constexpr(1, 8):
        td = td + pw[3 * w]
        tk = tk + pw[3 * w + 1]
        tx = tx + pw[3 * w + 2]
    inv_d, eps = f32c(0x394CCCCD), f32c(0x1E3CE508)  # 1 / 5120, 1e-20
    rk = _rsq_scaled(tk * inv_d + eps)
    rx = _rsq_scaled(tx * inv_d + eps)
    g = (td * (rk * rx)) * f32c(0x3C64F92E)  # / sqrt(5120)
    sq = fmath.sqrt(fx.max(abs(g), f32c(0x358637BD)))
    s = ((sq.bitcast(Int32) & 0x7FFFFFFF) | ((g.bitcast(Int32) ^ Int32(-2147483648)) & Int32(-2147483648))).bitcast(F32)  # fmt: skip
    gate = F32(1.0) / (F32(1.0) + exp_ocml(s, False))
    ob = (r * 4 + c) * 0x2800
    for j in range_constexpr(3):
        vh = _halves(d[4 * j + 2]) + _halves(d[4 * j + 3])
        o = [gate * vh[e] + xs[j][e] for e in range(4)]
        if valid[j]:
            bo.buffer_store(
                fx.Vector.from_elements([_pack(o[0], o[1]), _pack(o[2], o[3])], Int32),
                rsrc(X2),
                (ob + cols[j] * 8) >> 2,
            )


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 7 kernarg words."""
    name = kernel_symbol("tk_engram_post_apply", rows=ROWS, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_WORDS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def engram_post_apply(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, lds, a00, a01, a02, a03, a04, a05, a06)

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        engram_post_apply(a00, a01, a02, a03, a04, a05, a06).launch(
            grid=(GRID,), block=(THREADS,), stream=stream
        )

    return name, launch
