# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::qkv::q_kv_xf_kg_kernel<6, true, 8, 16>``.
"""The ordinary layer's QKV kernel on gfx950 as a FlyDSL port of TileRT's, kernarg ABI unchanged
(25 explicit arguments, 376 B, read by offset as 47 i64 words), grid 487 x 512.

Roles (``runs/20261115-flydsl-tilert/m3/qkv_kg_notes.md``): CTA 0 the mHC finalize when the
``ws | 1`` flag at 0x30 is set (the rest then work as logical CTA wgid - 1); q_a / kv GEMV,
q-norm with XCC-local replicas, q_b + RoPE, kv-norm + window ring. Ported one role at a time,
each diffed bit-exact against the original by ``m2/x_replay.py``.
"""

# ruff: noqa: SIM102  (nested traced / const_expr ifs are separate scf.if regions)
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import (
    EPS6,
    bf_hi,
    bf_lo,
    bf_rne_hi,
    bf_round,
    pack_bf2,
    spin_pause,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import bperm as _bperm
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import exp_ocml as _exp_ocml
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import f32c as _c
from atom.mono.device.ops import memrealtime as _mrt104
from atom.mono.device.ops import spin_pause as _spin104
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    fp8x8_bf16_pk,
    hw_rcp,
    hw_rsq,
    kernel_symbol,
    launcher,
    mfma_bf16,
    permlane_swap,
    readlane,
    rsrc,
    traced,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 47  # 376 B of explicit kernarg as 47 i64
# TK_QKV_XF=1: the q_kv_xf_full<6,1,8,0,3,0> argument layout (58 words: + u8*, u64* at 0x30 / 0x38, the
# MhcFinalizeArgs at 0x40, + int and CompArgs after the PrepArgs: the tail arguments 11 words later), the
# kg roles mapped onto it (a probe: which outputs the compressor adds)
XF = os.environ.get("TK_QKV_XF", "0") == "1"
if XF:
    KARG_WORDS = 58
# TK_QKV_MERGED=1 (with the xf_full argument layout): q_kv_merged<6, true, 0, 2, true, 8, true, 64, false, 3>,
# the ratio-2 KV-source layers' QKV kernel with the compressor (grid 755)
MERGED = XF and os.environ.get("TK_QKV_MERGED", "0") == "1"
# the q latent for q_b / iq: per-XCD replicas at ws + XCC * 0x4000 (xf_full: + 0xc0000); merged: one copy at 0x38000
RBASE = 0x38000 if MERGED else (0xC0000 if XF else 0)
# the compressor: 2 (merged), or TK_QKV_COMP=1 with xf_full: q_kv_xf_full<6,1,8,1,3,64>, the ratio-1 KV-source
# layer's QKV kernel (layer 20: wkv only, every position its own entry; grid 621)
COMP = 2 if MERGED else (int(os.environ.get("TK_QKV_COMP", "0")) if XF else 0)
# TK_QKV_ROWS (the kg kernels): the draft-row count, 6 (q_kv_xf_kg<6, ..>) or 5 (<5, true, 8, 0>, the MTP
# layers': grid 486, the kv-norm also writing the block kv / scales)
QROWS = int(os.environ.get("TK_QKV_ROWS", "6"))
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 481 + QROWS, 512
if MERGED:
    GRID = 755
elif COMP == 1:
    GRID = 621
MIX, KSLICES = 24, 160
# TileRT template idx: 16 (layers fed by mhc_kg: 64 sums of squares a K slice) or 0 (10 already reduced
# sums a row at 0x1e000 + row * 128 B)
IDX = int(os.environ.get("TK_QKV_IDX", "16"))

QP_1032, QP_2301, HALF_MIRROR, MIRROR, BCAST15, BCAST31 = (
    0xB1,
    0x4E,
    0x141,
    0x140,
    0x142,
    0x143,
)
ROR4, ROR8, ROR12 = 0x124, 0x128, 0x12C


def _dpp(v, ctrl, row_mask=0xF, bank_mask=0xF, bound_ctrl=False):
    w = update_dpp_i32(
        Int32(0), F32(v).bitcast(Int32), ctrl, row_mask, bank_mask, bound_ctrl
    )
    return Int32(w).bitcast(F32)


def _wave_sum(v):
    v = v + _dpp(v, QP_1032)
    v = v + _dpp(v, QP_2301)
    v = v + _dpp(v, HALF_MIRROR)
    v = v + _dpp(v, MIRROR)
    v = v + _dpp(v, BCAST15, row_mask=0xA)
    v = v + _dpp(v, BCAST31, row_mask=0xC)
    return readlane(v, 63)


def _ld(addr, word, n=1):
    v = bo.buffer_load(rsrc(addr), word, vec_width=n, dtype=T.i32)
    return fx.Vector(v) if n > 1 else Int32(v)


def _ldf(addr, word):
    return _ld(addr, word).bitcast(F32)


def _stf(addr, word, v):
    bo.buffer_store(F32(v).bitcast(Int32), rsrc(addr), word)


def _div_nr(x, s):
    """x / s by v_rcp and two Newton-Raphson refinements, TileRT's sinkhorn division."""
    r = hw_rcp(s)
    e = fmath.fma(-s, r, F32(1.0))
    r = fmath.fma(e, r, r)
    q = x * r
    d = fmath.fma(-s, q, x)
    q = fmath.fma(d, r, q)
    d = fmath.fma(-s, q, x)
    return fmath.fma(d, r, q)


def _col_sum(v, bit2_clear):
    s = v + _dpp(v, ROR8, bound_ctrl=True)
    return s + bit2_clear.select(_dpp(s, ROR12), _dpp(s, ROR4))


def _quad_sum(v):
    s = v + _dpp(v, QP_1032, bound_ctrl=True)
    return s + _dpp(s, QP_2301, bound_ctrl=True)


@traced
def mhc_finalize(tid, ws, scale, base, out, nent):
    """CTA 0, wave w finalizes row e = w (nent <= 8 rows)."""
    lane, e = tid % 64, tid // 64
    if e < nent:
        mhc_finalize_row(lane, e, ws, scale, base, out, IDX == 0)


@traced
def mhc_finalize_row(lane, e, ws, scale, base, out, idx0):
    """One wave finalizes mHC row e: hcA[e][0..3] = sigmoid + 1e-6 (pre), [4..7] = 2 sigmoid (post),
    [8..23] = 20-round sinkhorn of the 4 x 4 comb logits. ws: mHC's workspace (GEMV partials at 0,
    per-row sums of squares at 0x1e000: 10 already reduced a row when idx0, else 640).
    """
    m = (lane >= 24).select(lane - 24, lane)
    pbase = (lane >= 24).select(Int32(0x780), Int32(0)) + m
    part = F32(0.0)
    if lane < 48:
        acc = _ldf(ws, e * (KSLICES * MIX) + pbase)
        for k in range_constexpr(1, 80):
            acc = acc + _ldf(ws, e * (KSLICES * MIX) + pbase + k * MIX)
        part = acc
    if const_expr(idx0):
        sq = [_ldf(ws, (0x1E000 // 4) + e * 32 + k) for k in range(10)]
        tot = sq[0] + sq[1]
        for k in range_constexpr(2, 10):
            tot = tot + sq[k]
    else:
        sq = [_ldf(ws, (0x1E000 // 4) + e * 640 + k * 64 + lane) for k in range(10)]
        tot = _wave_sum(sq[0]) + _wave_sum(sq[1])
        for k in range_constexpr(2, 10):
            tot = tot + _wave_sum(sq[k])
    ms = fmath.fma(tot, _c(0x384CCCCD), _c(0x1E3CE508))
    tiny = ms < _c(0x00800000)
    r = hw_rsq(tiny.select(ms * _c(0x4B800000), ms))
    r = tiny.select(r * _c(0x45800000), r)
    mixv = (part + _bperm(part, (lane + 24) % 64)) * r
    comb_in = _bperm(mixv, (lane + 8) % 64)
    if lane < 8:
        post = lane >= 4
        y = mixv * _ldf(scale, post.select(Int32(1), Int32(0))) + _ldf(base, lane)
        sig = F32(1.0) / (F32(1.0) + _exp_ocml(y, True))
        _stf(out, e * MIX + lane, post.select(sig + sig, sig + _c(EPS6)))
    if lane < 16:
        y = _ldf(scale, 2) * comb_in + _ldf(base, lane + 8)
        mx = fx.max(y, _dpp(y, QP_1032))
        mx = fx.max(mx, _dpp(mx, QP_2301))
        ex = _exp_ocml(y - mx, False)
        p = _div_nr(ex, _quad_sum(ex)) + _c(EPS6)
        bit2_clear = (lane & 4) == 0  # tid & 4
        v = _div_nr(p, _col_sum(p, bit2_clear) + _c(EPS6))
        for _ in range_constexpr(19):
            q = _div_nr(v, _quad_sum(v) + _c(EPS6))
            v = _div_nr(q, _col_sum(q, bit2_clear) + _c(EPS6))
        _stf(out, e * MIX + 8 + lane, v)


LDS_BYTES = 72560
X_ROW = 0x2800  # bf16 X row in LDS (5120 x 2 B)
XS_LDS = 61440  # activation scales (f32), 0x280 B a row
RED_LDS = 64256  # wave w's partials at w * 1024 + lane * 4 + {0, 256, 512, 768}
LL_QA = 0x29000  # {bf16x2, epoch} packs, 0x380 packs a token row (1280 q_a + 512 kv columns)
EPOCHS = 0x28000


def _e8m0_f32(byte):
    """ue8m0 byte -> f32 2^(e-127); e == 0 -> 2^-127 (0x400000), TileRT's decode."""
    b = Int32(byte) & 0xFF
    return (b == 0).select(Int32(0x400000), b << 23).bitcast(F32)


def _lds_ld(lds, byte, n):
    r = (
        fx.ptr_load(lds + (byte // 4), result_type=fx.Vector.make_type(n, Int32))
        if n > 1
        else None
    )
    return (
        fx.Vector(r)
        if n > 1
        else Int32(fx.ptr_load(lds + (byte // 4), result_type=Int32.ir_type))
    )


def _lds_st(lds, byte, vals):
    v = (
        fx.Vector.from_elements([Int32(x) for x in vals], Int32)
        if len(vals) > 1
        else Int32(vals[0])
    )
    fx.ptr_store(v, lds + (byte // 4))


def _fp8x8_bf16_ps(words, scale):
    """PS: eight fp8 of two dwords -> bf16 times ``scale`` (a power of two: exact) by
    v_cvt_scalef32_pk_bf16_fp8 -- fp8x8_bf16_pk with the block scale applied in the conversion."""
    out = []
    for w in words:
        for hi in (False, True):
            p = rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2, Int32(w).ir_value(), F32(scale).ir_value(), hi
            )
            out.append(fx.Vector(p).bitcast(Int32)[0])
    return fx.Vector.from_elements(out, Int32).bitcast(fx.BFloat16)


@traced
def qkv_gemv(tid, s16, xq, xs, qa_w, qa_s, kv_w, kv_s, ws, lds, ps=1):
    """Logical CTA s16 < 224: 8 rows of the 16-row tile s16 / 2 (q_a tiles 0..79, kv 80..111) for the
    6 tokens over K = 5120 (wave w: K tiles w*10 .. +9 of 64), LDS-reduced, published as LL packs.
    """
    lane, w = tid % 64, tid // 64
    pair, par = s16 // 2, s16 % 2
    is_kv = s16 > 159
    tile = is_kv.select(pair - 80, pair)
    wptr = is_kv.select(kv_w, qa_w)
    sptr = is_kv.select(kv_s, qa_s)
    old = _ld(ws, EPOCHS // 4 + s16)
    # TileRT's issue order: X (4 rounds of 16-B chunks), its scales, then this CTA's half of the
    # weight tile (nt, only the lanes of its 8 rows: the pair's other CTA loads the rest), so the
    # X -> LDS conversion waits on vmcnt(10) while the weights are still in flight
    xc, xok, xv = [], [], []
    for rnd in range_constexpr(4):
        c = tid | (rnd * 0x200)
        if rnd == 3:
            c = fx.min(c, Int32(QROWS * 320 - 1))
        ok = Int32(1) == 1
        if rnd == 2:
            ok = c <= QROWS * 320 - 1
        if rnd == 3:
            ok = tid < QROWS * 320 - 1536
        row, col = c // 320, c % 320
        xc.append((row, col))
        xok.append(ok)
        xv.append(_ld(xq, (row * 0x1400 + col * 16) // 4, 4))
    d = _ld(xs, fx.min(tid, Int32(QROWS * 40 - 1))) if const_expr(ps == 0) else Int32(0)
    xsw = [_ld(xs, (xc[q][0] * 160 + xc[q][1] // 2) // 4) for q in range(4)]  # PS: the chunks' scale dwords
    mine = ((lane >> 3) & 1) == par
    # this CTA's half of the tile only, exec-masked as TileRT's (zeros on the other lanes). The join
    # costs a conservative vmcnt(4) before the X conversion (TileRT's hipcc emits vmcnt(10)), but every
    # variant that issues loads on the other lanes too was slower (OOB mask, partner address,
    # num_records = 0 dummies: runs/20261115-flydsl-tilert/m3/qkv_kg_notes.md)
    wbase = ((tile * 0x50 + w * 10) << 10) + lane * 16
    zero4 = fx.Vector.filled(4, 0, Int32)
    wv = [zero4 for _ in range(10)]
    if mine:
        for i in range_constexpr(10):
            wv[i] = fx.Vector(
                bo.buffer_load(
                    rsrc(wptr),
                    (wbase + i * 0x400) // 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
    for rnd in range_constexpr(4):
        if xok[rnd]:
            row, col = xc[rnd]
            v = xv[rnd]
            # PS: X pre-scaled by its 32-block ue8m0 scale (row * 160 + col / 2) in the conversion
            sx = _e8m0_f32(xsw[rnd] >> (8 * ((col // 2) % 4)))
            lo = _fp8x8_bf16_ps([v[0], v[1]], sx).bitcast(Int32)
            hi = _fp8x8_bf16_ps([v[2], v[3]], sx).bitcast(Int32)
            _lds_st(lds, row * X_ROW + col * 32, [lo[0], lo[1], lo[2], lo[3]])
            _lds_st(lds, row * X_ROW + col * 32 + 16, [hi[0], hi[1], hi[2], hi[3]])
    if const_expr(ps == 0):  # the per-block X scales in LDS: only the unscaled path reads them
        if tid < QROWS * 40:
            f = [_e8m0_f32(d >> (8 * b)).bitcast(Int32) for b in range(4)]
            _lds_st(lds, XS_LDS + (tid // 40) * 0x280 + (tid % 40) * 16, f)
    sbytes = [_ld(sptr, ((tile // 2) * 0xA0 + w * 20) // 4 + q) for q in range(5)]
    tok_ok = (lane % 16) < QROWS
    tok = tok_ok.select(lane % 16, Int32(0))
    gpu.barrier()
    acc = [F32(0.0), F32(0.0), F32(0.0), F32(0.0)]
    if const_expr(ps != 0):
        # PS: both operands carry their power-of-two block scales -> one MFMA accumulator chain
        av = fx.Vector.filled(4, 0.0, F32)
        for i in range_constexpr(10):
            for j in range_constexpr(2):
                kb = 2 * i + j
                wsc = _e8m0_f32(sbytes[kb // 4] >> (8 * (kb % 4)))
                a = _fp8x8_bf16_ps([wv[i][2 * j], wv[i][2 * j + 1]], wsc)
                b = _lds_ld(
                    lds, tok * X_ROW + w * 1280 + kb * 64 + (lane // 16) * 16, 4
                ).bitcast(fx.BFloat16)
                av = mfma_bf16(a, b, av)
        for q in range_constexpr(4):
            acc[q] = av[q]
    else:
        for i in range_constexpr(10):
            for j in range_constexpr(2):
                kb = 2 * i + j
                a = fp8x8_bf16_pk([wv[i][2 * j], wv[i][2 * j + 1]])
                b = _lds_ld(
                    lds, tok * X_ROW + w * 1280 + kb * 64 + (lane // 16) * 16, 4
                ).bitcast(fx.BFloat16)
                m = mfma_bf16(a, b, fx.Vector.filled(4, 0.0, F32))
                wsc = _e8m0_f32(sbytes[kb // 4] >> (8 * (kb % 4)))
                xsc = _lds_ld(lds, XS_LDS + tok * 0x280 + (w * 20 + kb) * 4, 1).bitcast(F32)
                c = wsc * xsc
                for q in range_constexpr(4):
                    acc[q] = fmath.fma(m[q], c, F32(0.0) if kb == 0 else acc[q])
    acc01 = [acc[0], acc[1]]
    acc23 = [acc[2], acc[3]]
    if w != 0:
        base = w * 1024 + lane * 4 + RED_LDS
        _lds_st(lds, base, [acc01[0].bitcast(Int32)])
        _lds_st(lds, base + 256, [acc01[1].bitcast(Int32)])
        _lds_st(lds, base + 512, [acc23[0].bitcast(Int32)])
        _lds_st(lds, base + 768, [acc23[1].bitcast(Int32)])
    epoch = Int32(old) + 1
    gpu.barrier()
    if (w == 0) & ((lane >> 5) == par) & tok_ok:
        r = [acc01[0], acc01[1], acc23[0], acc23[1]]
        for src in range_constexpr(1, 8):
            base = src * 1024 + lane * 4 + RED_LDS
            r = [r[q] + _lds_ld(lds, base + 256 * q, 1).bitcast(F32) for q in range(4)]
        word = (LL_QA + 8 * (tok * 0x380 + pair * 8 + 2 * (lane >> 4))) // 4
        vals = [pack_bf2(r[0], r[1]), epoch, pack_bf2(r[2], r[3]), epoch]
        bo.buffer_store(
            fx.Vector.from_elements(vals, Int32), rsrc(ws), word, cache_modifier=CM_DEV
        )
    if tid == 0:
        _st_i(ws, EPOCHS // 4 + s16, epoch)
    return epoch


def _st_i(addr, word, v):
    bo.buffer_store(Int32(v), rsrc(addr), word)


QN_GAMMA_LDS = 15360  # q_norm gamma, 1280 f32
WAVE_SUMS_LDS = 0x11B60
ROW_SHR = 0x110


def _more(cond, n):
    """A poll loop's condition (``n``: the loop's reload count, unused)."""
    return cond


def _xcc_off():
    """This XCD's q-latent replica offset (none in the merged kernel: one copy)."""
    if const_expr(MERGED):
        return Int32(0)
    return (_xcc_id() << 14) & 0x1C000


def _xcc_id():
    return Int32(
        llvm.InlineAsmOp(
            T.i32,
            [],
            "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
            "=s",
            has_side_effects=True,
        ).result
    )


def _fp8x2_scaled(a, b, scale):
    o = fx.Vector.from_elements([Int32(0)], Int32).bitcast(fx.Int16)
    r = rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        o.ir_value(),
        F32(a).ir_value(),
        F32(b).ir_value(),
        F32(scale).ir_value(),
        False,
    )
    return fx.Vector(r).bitcast(Int32)[0]


@traced
def qkv_qnorm(tid, s16, epoch, ws, gamma, qr, lat, lat_s, lds):
    """Logical CTA s16 < 48 after its GEMV: row s16 / 8's q_a (1280) polled from the LL packs, RMS-normed,
    E4M3 + ue8m0 a 32-block, published as {fp8 x4, epoch | e << 24} packs into this XCD's replica
    (ws + XCC * 0x4000 + row * 0xa00); the row's leader (s16 % 8 == 0) also writes qr (bf16) and the
    optional fp8 latent / scales and its epoch slot."""
    gpu.barrier()
    if tid < 256:
        lane, wv = tid % 64, tid // 64
        g = _ld(gamma, tid * 4, 4)
        _lds_st(lds, QN_GAMMA_LDS + tid * 16, [g[0], g[1], g[2], g[3]])
        if tid < 64:
            g2 = _ld(gamma, (tid + 256) * 4, 4)
            _lds_st(lds, QN_GAMMA_LDS + (tid + 256) * 16, [g2[0], g2[1], g2[2], g2[3]])
        row = s16 // 8
        sq = F32(0.0)
        if tid < 160:
            word = (LL_QA + row * 0x1C00 + tid * 32) // 4
            v = _ll_packs4(ws, word)
            n_dbg = Int32(0)
            while _more(
                (v[1] != epoch) | (v[3] != epoch) | (v[5] != epoch) | (v[7] != epoch),
                n_dbg,
            ):
                n_dbg = n_dbg + 1
                spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
                rocdl.s_sleep(1)
                v = _ll_packs4(ws, word)
            _lds_st(lds, row * 0xA00 + tid * 16, [v[0], v[2], v[4], v[6]])
            a = [
                bf_lo(v[2 * k]) * bf_lo(v[2 * k]) + bf_hi(v[2 * k]) * bf_hi(v[2 * k])
                for k in range(4)
            ]
            sq = ((a[0] + a[1]) + a[2]) + a[3]
        t = sq
        for k in range_constexpr(4):
            t = t + _dpp(t, ROW_SHR + (1 << k), bound_ctrl=True)
        t = t + _dpp(t, BCAST15, bound_ctrl=True)
        top = t + _dpp(t, BCAST31, bound_ctrl=True)
        if lane == 63:
            _lds_st(lds, WAVE_SUMS_LDS + wv * 4, [top.bitcast(Int32)])
        gpu.barrier()
        leader = (s16 % 8) == 0
        if leader & (tid == 0):
            _st_i(ws, EPOCHS // 4 + 1016 + row, epoch)
        # the CTA of row 0 also bumps the slots of the unused rows up to 8 (1016 + QROWS .. 1023)
        if const_expr(QROWS == 6):
            if (s16 == 0) & ((tid & 0xFE) == 64):
                _st_i(ws, EPOCHS // 4 + 958 + tid, epoch)
        else:
            if (s16 == 0) & (tid >= 64) & (tid < 72 - QROWS):
                _st_i(ws, EPOCHS // 4 + 952 + QROWS + tid, epoch)
        w4 = _lds_ld(lds, WAVE_SUMS_LDS, 4)
        tot = (
            ((F32(0.0) + w4[0].bitcast(F32)) + w4[1].bitcast(F32)) + w4[2].bitcast(F32)
        ) + w4[3].bitcast(F32)
        ms = tot / F32(1280.0) + _c(0x1E3CE508)
        rstd = F32(1.0) / fmath.sqrt(ms)
        idx = fx.min(tid, Int32(159))
        xd = _lds_ld(lds, row * 0xA00 + idx * 16, 4)
        ga = _lds_ld(lds, QN_GAMMA_LDS + idx * 32, 4)
        gb = _lds_ld(lds, QN_GAMMA_LDS + idx * 32 + 16, 4)
        gm = [ga[0], ga[1], ga[2], ga[3], gb[0], gb[1], gb[2], gb[3]]
        y = []
        for c in range_constexpr(8):
            xc = bf_lo(xd[c // 2]) if c % 2 == 0 else bf_hi(xd[c // 2])
            y.append(bf_round(gm[c].bitcast(F32) * (rstd * xc)))
        amax = fx.max(fx.max(abs(y[0]), abs(y[1])), F32(0.0))
        for c in range_constexpr(2, 8):
            amax = fx.max(amax, abs(y[c]))
        n1 = _dpp(amax, QP_1032)
        amax = fx.max(amax, fx.max(n1, n1))
        n2 = _dpp(amax, QP_2301)
        if tid < 160:
            amax = fx.max(fx.max(amax, n2), _c(0x38D1B717))
            sb = (amax * _c(0x3B124925)).bitcast(Int32) + 0x7FFFFF
            e = (sb >> 23) & 0xFF
            scale = (e == 0).select(Int32(0x400000), sb & Int32(-8388608)).bitcast(F32)
            q01 = _fp8x2_scaled(y[0], y[1], scale)
            q23 = _fp8x2_scaled(y[2], y[3], scale)
            q45 = _fp8x2_scaled(y[4], y[5], scale)
            q67 = _fp8x2_scaled(y[6], y[7], scale)
            d0 = (q23 << 16) | (q01 & 0xFFFF)
            d1 = (q67 << 16) | (q45 & 0xFFFF)
            tag = (epoch & 0xFFFFFF) | (e << 24)
            rep = (RBASE + _xcc_off() + row * 0xA00 + tid * 16) // 4
            # merged: one copy read by every XCD -> sc1 (an XCD's own replica stays in its L2: plain)
            bo.buffer_store(
                fx.Vector.from_elements([d0, tag, d1, tag], Int32), rsrc(ws), rep,
                cache_modifier=CM_DEV if const_expr(MERGED) else 0,
            )  # fmt: skip
            if leader:
                if lat != 0:
                    bo.buffer_store(
                        fx.Vector.from_elements([d0, d1], Int32),
                        rsrc(lat),
                        (row * 0x500 + idx * 8) // 4,
                    )
                    if (idx % 4) == 0:
                        bo.buffer_store(
                            e.to(fx.Int8),
                            rsrc(lat_s),
                            row * 40 + idx // 4,
                            offset_is_bytes=True,
                        )
                packed = [pack_bf2(y[2 * q], y[2 * q + 1]) for q in range(4)]
                bo.buffer_store(
                    fx.Vector.from_elements(packed, Int32),
                    rsrc(qr),
                    (row * 0xA00 + idx * 16) // 4,
                )


def _ll_packs4(addr, word):
    """Four consecutive 8-B LL packs, one dwordx2 sc1 load each (TileRT's form)."""
    vals = []
    for k in range_constexpr(4):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(addr),
                word + 2 * k,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        vals += [v[0], v[1]]
    return fx.Vector.from_elements(vals, Int32)


QB_X_LDS = (
    5120 + QROWS * 2560
)  # fp8 q -> bf16, 2560 B a row, swizzled for the MFMA B operand
QB_XS_LDS = QB_X_LDS + QROWS * 2560  # q's ue8m0 scales as f32, 160 B a row
QB_RED_LDS = 35776  # wave w (1..3) partials at w * 1024 + lane * 4 + {0, 256, 512, 768}
QB_EPOCH_SLOT, QB_PREP_SLOT = 224, 488


@traced
def qkv_qb(tid, s16, ws, qb_w, qb_s, ropec, vpos, q_out, prep_d8, lds, psq=1):
    """Logical CTA 224 + t (t < 256): 16 q columns (tile t, head t / 32) for the 6 tokens over K = 1280
    (wave w: K tiles w*5 .. +4 of 64), from this XCD's replica of the fp8 q latent; RoPE on a head's
    last 64 dims (tiles 28..31), bf16 out."""
    t = s16 - 224
    if tid < 256:
        lane, w = tid % 64, tid // 64
        epoch = _ld(ws, EPOCHS // 4 + QB_EPOCH_SLOT + t) + 1
        sw = [_ld(qb_s, ((t // 2) * 40 + ((w * 10) & ~3)) // 4 + q) for q in range(3)]
        tok_ok = (lane % 16) < QROWS
        tok = tok_ok.select(lane % 16, Int32(0))
        pos = fx.max(_ld(vpos, tok), Int32(0))
        th = t % 32
        feat = th * 16 + 4 * (lane // 16)
        # G104: the q_b weights wait 150 ticks (1.5 us) so the q_a / kv GEMV weights get the HBM first
        t104 = _mrt104()
        while (_mrt104() - t104) < 150:
            _spin104()
        wv = []
        for i in range_constexpr(5):
            v = bo.buffer_load(
                rsrc(qb_w), (((t * 20 + w * 5) << 10) + lane * 16 + i * 0x400) // 4,
                vec_width=4, dtype=T.i32, cache_modifier=CM_NT,
            )  # fmt: skip
            wv.append(fx.Vector(v))
        rope = th > 27
        pair = rope.select((feat - 448) >> 1, Int32(0))
        tw = (pos * 256 + pair * 4) // 4
        cs = _ld(ropec, tw, 2)
        sn = _ld(ropec, tw + 32, 2)
        rep = (RBASE + _xcc_off()) // 4
        idx = [tid, tid + 256, tid + 512, fx.min(tid | 0x300, Int32(QROWS * 160 - 1))]
        e24 = epoch & 0xFFFFFF
        ch = _q_chunks(ws, rep, idx)
        n_dbg = Int32(0)
        while _more(~_tags_ok(ch, e24), n_dbg):
            n_dbg = n_dbg + 1
            spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
            rocdl.s_sleep(2)
            ch = _q_chunks(ws, rep, idx)
        for k in range_constexpr(4):
            c = idx[k]
            live = (tid | 0x300) < QROWS * 160 if k == 3 else (Int32(1) == 1)
            if live:
                row, cc = c // 160, c % 160
                sq8 = _e8m0_f32(ch[k][1] >> 24)  # PS: the 32-block scale
                lo = _fp8x8_bf16_ps([ch[k][0], ch[k][2]], sq8).bitcast(Int32)
                at = (
                    QB_X_LDS
                    + row * 2560
                    + (cc >> 3) * 128
                    + (cc & 3) * 32
                    + ((cc >> 2) & 1) * 16
                )
                _lds_st(lds, at, [lo[0], lo[1], lo[2], lo[3]])
                if const_expr(psq == 0):
                    if (cc & 3) == 0:
                        _lds_st(
                            lds,
                            QB_XS_LDS + row * 160 + (cc >> 2) * 4,
                            [_e8m0_f32(ch[k][1] >> 24).bitcast(Int32)],
                        )
        gpu.barrier()
        if tid == 0:
            _st_i(ws, EPOCHS // 4 + QB_EPOCH_SLOT + t, epoch)
        if (tid == 1) & (prep_d8 == 0) & (t <= 127):
            _st_i(ws, EPOCHS // 4 + QB_PREP_SLOT + t, epoch)
        qacc = [F32(0.0), F32(0.0), F32(0.0), F32(0.0)]
        if const_expr(psq != 0):
            qv = fx.Vector.filled(4, 0.0, F32)
            for i in range_constexpr(5):
                for j in range_constexpr(2):
                    kb = 2 * i + j
                    bi = 2 * (w & 1) + kb
                    wsc = _e8m0_f32(_pick_byte(sw, bi))
                    a = _fp8x8_bf16_ps([wv[i][2 * j], wv[i][2 * j + 1]], wsc)
                    b = _lds_ld(
                        lds,
                        QB_X_LDS + tok * 2560 + (w * 5 + i) * 128 + (lane // 16) * 32 + j * 16,
                        4,
                    )
                    qv = mfma_bf16(a, b.bitcast(fx.BFloat16), qv)
            for q in range_constexpr(4):
                qacc[q] = qv[q]
        else:
            for i in range_constexpr(5):
                for j in range_constexpr(2):
                    kb = 2 * i + j
                    a = fp8x8_bf16_pk([wv[i][2 * j], wv[i][2 * j + 1]])
                    b = _lds_ld(
                        lds,
                        QB_X_LDS
                        + tok * 2560
                        + (w * 5 + i) * 128
                        + (lane // 16) * 32
                        + j * 16,
                        4,
                    )
                    m = mfma_bf16(a, b.bitcast(fx.BFloat16), fx.Vector.filled(4, 0.0, F32))
                    bi = 2 * (w & 1) + kb
                    wsc = _e8m0_f32(_pick_byte(sw, bi))
                    xsc = _lds_ld(
                        lds, QB_XS_LDS + tok * 160 + (w * 10 + kb) * 4, 1
                    ).bitcast(F32)
                    c = wsc * xsc
                    for q in range_constexpr(4):
                        qacc[q] = fmath.fma(m[q], c, F32(0.0) if kb == 0 else qacc[q])
        acc01 = [qacc[0], qacc[1]]
        acc23 = [qacc[2], qacc[3]]
        if w != 0:
            base = w * 1024 + lane * 4 + QB_RED_LDS
            r4 = [acc01[0], acc01[1], acc23[0], acc23[1]]
            for q in range_constexpr(4):
                _lds_st(lds, base + 256 * q, [r4[q].bitcast(Int32)])
        gpu.barrier()
        if (w == 0) & tok_ok:
            r = [acc01[0], acc01[1], acc23[0], acc23[1]]
            for src in range_constexpr(1, 4):
                base = src * 1024 + lane * 4 + QB_RED_LDS
                r = [
                    r[q] + _lds_ld(lds, base + 256 * q, 1).bitcast(F32)
                    for q in range(4)
                ]
            x = [bf_round(v) for v in r]
            c0, c1 = cs[0].bitcast(F32), cs[1].bitcast(F32)
            s0, s1 = sn[0].bitcast(F32), sn[1].bitcast(F32)
            y0 = rope.select(fmath.fma(c0, x[0], -(s0 * x[1])), x[0])
            y1 = rope.select(fmath.fma(c0, x[1], s0 * x[0]), x[1])
            y2 = rope.select(fmath.fma(c1, x[2], -(s1 * x[3])), x[2])
            y3 = rope.select(fmath.fma(c1, x[3], s1 * x[2]), x[3])
            word = (tok * 8192 + (t // 32) * 1024 + feat * 2) // 4
            vals = [pack_bf2(y0, y1), pack_bf2(y2, y3)]
            bo.buffer_store(fx.Vector.from_elements(vals, Int32), rsrc(q_out), word)


# ---- the indexer's query prep (q_kv_xf_full: the layers whose indexer runs next) -----------------------------
IX_SCALE_OFF, IX_WEIGHT_OFF = (
    0x4000,
    0x4400,
)  # ws_ix: [tok][128] e8m0 bytes (head*4 + group), [tok][32] f32 weights
IW_SUM_LDS = 20480  # wave w's six row sums at + w*24 + r*4
IQ_X_LDS = 20480  # bf16 q latent x scale products [tok][1280], 0xa00 B a row
IQ_RED_LDS = 35840  # K-half 1's MFMA results: row half rh at + rh*1024 + lane*16
IQ_AMAX_LDS = 37888  # [rh][tok] f32 amax at + rh*64 + tok*4


@traced
def qkv_iw(tid, s16, xn, wproj, ix, lds):
    """Logical CTA 128 + h (h < 32): head h's indexer weight for the 6 tokens: Xn[r] . wproj[h] (bf16, 5120),
    a lane's fma chain over elements 8t..8t+7, 2048+8t.., (t < 128) 4096+8t.. in order, the wave's sum by
    xor-32..1 butterflies, waves (0+1)+(2+3), bf16 RNE x 1/64 -> f32 at ws_ix + 0x4400 + (r*32 + h)*4.
    """
    h = s16 - 128
    if tid < 256:
        lane, w = tid % 64, tid // 64
        offs = [tid * 16, 0x1000 + tid * 16]
        wv = [_ld(wproj, (h * 0x2800 + o) // 4, 4) for o in offs]
        xv = [[_ld(xn, (r * 0x2800 + o) // 4, 4) for o in offs] for r in range(6)]
        third = tid < 128
        w3 = fx.Vector.filled(4, 0, Int32)
        x3 = [fx.Vector.filled(4, 0, Int32) for _ in range(6)]
        if third:
            w3 = _ld(wproj, (h * 0x2800 + 0x2000 + tid * 16) // 4, 4)
            for r in range_constexpr(6):
                x3[r] = _ld(xn, (r * 0x2800 + 0x2000 + tid * 16) // 4, 4)
        a3 = []
        for r in range_constexpr(6):  # a row's whole chain before the next: its x registers free early
            a = F32(0.0)
            first = True
            for c in range_constexpr(2):
                for d in range_constexpr(4):
                    for hi in range_constexpr(2):
                        xe = bf_hi(xv[r][c][d]) if hi else bf_lo(xv[r][c][d])
                        we = bf_hi(wv[c][d]) if hi else bf_lo(wv[c][d])
                        a = fmath.fma(xe, we, F32(0.0) if first else a)
                        first = False
            a2 = a
            for d in range_constexpr(4):
                for hi in range_constexpr(2):
                    xe = bf_hi(x3[r][d]) if hi else bf_lo(x3[r][d])
                    we = bf_hi(w3[d]) if hi else bf_lo(w3[d])
                    a = fmath.fma(xe, we, a)
            a3.append(third.select(a, a2))
        for r in range_constexpr(6):
            v = a3[r]
            for k in (32, 16, 8, 4, 2, 1):
                v = v + _bperm(v, lane ^ k)
            if lane == 0:
                _lds_st(lds, IW_SUM_LDS + w * 24 + r * 4, [v.bitcast(Int32)])
        gpu.barrier()
        if tid < 6:
            p = [
                _lds_ld(lds, IW_SUM_LDS + q * 24 + tid * 4, 1).bitcast(F32)
                for q in range(4)
            ]
            t = (p[0] + p[1]) + (p[2] + p[3])
            _stf(ix, IX_WEIGHT_OFF // 4 + tid * 32 + h, bf_round(t) * _c(0x3C800000))


def _e2m1(v):
    """A clamped value's E2M1 code (RNE at the midpoints, sign from its bits)."""
    a = abs(v)
    n = (a > _c(0x3E800000)).select(Int32(1), Int32(0))
    n = n + (a >= _c(0x3F400000)).select(Int32(1), Int32(0))
    n = n + (a > _c(0x3FA00000)).select(Int32(1), Int32(0))
    n = n + (a >= _c(0x3FE00000)).select(Int32(1), Int32(0))
    n = n + (a > _c(0x40200000)).select(Int32(1), Int32(0))
    n = n + (a >= _c(0x40600000)).select(Int32(1), Int32(0))
    n = n + (a > _c(0x40A00000)).select(Int32(1), Int32(0))
    return n | ((v.bitcast(Int32) >> 28) & 8)


@traced
def qkv_iq(tid, s16, ws, wq, wq_s, ropec, rope_n, vpos, ix, lds):
    """Logical CTA t < 128: the indexer query's rows 32t..32t+31 (head t/4, dims (t%4)*32..) for the 6 tokens:
    i_wq_b (fp8) . (this XCD's fp8 q latent x 2^(e_q + e_w - 254), bf16 RNE) in 20 chained bf16 MFMAs a wave
    (wave w: rows (w&1)*16.., K half w>>1), the K halves summed, bf16 RNE, GPT-J RoPE on dims 64..127 (bf16 RNE),
    FP4 E2M1 with the group's power-of-two scale (amax/6 rounded up): ws_ix[tok][head*64 + dim/2] and the scale
    byte at + 0x4000 + tok*128 + head*4 + t%4; epoch slot QB_PREP_SLOT + t."""
    t = s16
    if tid < 256:
        lane, w = tid % 64, tid // 64
        rh, kh, g, h16 = w & 1, w >> 1, lane >> 4, lane & 15
        epoch = _ld(ws, EPOCHS // 4 + QB_PREP_SLOT + t) + 1
        row = t * 32 + rh * 16 + h16
        wb = row * 1280 + kh * 640 + g * 16
        wv = [_ld(wq, (wb + 64 * i) // 4, 4) for i in range(10)]
        tok_ok = h16 < 6
        tok = tok_ok.select(h16, Int32(0))
        pos = _ld(vpos, tok)
        pos_c = (pos > -1).select(fx.min(pos, Int32(rope_n) - 1), Int32(0))
        rq = fx.max(t % 4 - 2, Int32(0))
        pair = rq * 16 + rh * 8 + g * 2
        cs = _ld(ropec, pos_c * 64 + pair, 2)
        sn = _ld(ropec, pos_c * 64 + 32 + pair, 2)
        idx = [tid, tid + 256, tid + 512, fx.min(tid | 0x300, Int32(0x3BF))]
        wsb = []
        for k in range_constexpr(4):
            cc = idx[k] % 160
            wsb.append(
                Int32(
                    bo.buffer_load(
                        rsrc(wq_s), t * 40 + (cc >> 2), vec_width=1, dtype=T.i8
                    )
                ).to(Int32)
                & 0xFF
            )
        rep = (RBASE + _xcc_off()) // 4
        e24 = epoch & 0xFFFFFF
        ch = _q_chunks(ws, rep, idx)
        n_dbg = Int32(0)
        while _more(~_tags_ok(ch, e24), n_dbg):
            n_dbg = n_dbg + 1
            spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
            rocdl.s_sleep(2)
            ch = _q_chunks(ws, rep, idx)
        for k in range_constexpr(4):
            c = idx[k]
            live = (tid | 0x300) < 0x3C0 if k == 3 else (Int32(1) == 1)
            if live:
                r_, cc = c // 160, c % 160
                ssum = ((ch[k][1] >> 24) & 0xFF) + wsb[k]
                sc = ((ssum << 23) + Int32(-1065353216)).bitcast(
                    F32
                )  # + 0xc0800000 (-4.0): 2^(sum-254)
                b = fp8x8_bf16_pk([ch[k][0], ch[k][2]]).bitcast(Int32)
                outw = []
                for q in range_constexpr(4):
                    lo = (b[q] << 16).bitcast(F32) * sc
                    hi = (b[q] & Int32(-65536)).bitcast(F32) * sc
                    outw.append(pack_bf2(lo, hi))
                _lds_st(lds, IQ_X_LDS + r_ * 0xA00 + cc * 16, outw)
        gpu.barrier()
        if tid == 0:
            _st_i(ws, EPOCHS // 4 + QB_PREP_SLOT + t, epoch)
        xb = IQ_X_LDS + fx.min(h16, Int32(5)) * 0xA00 + kh * 0x500 + g * 32
        acc = fx.Vector.filled(4, 0.0, F32)
        for i in range_constexpr(10):
            for j in range_constexpr(2):
                a = fp8x8_bf16_pk([wv[i][2 * j], wv[i][2 * j + 1]])
                bx = _lds_ld(lds, xb + 128 * i + 16 * j, 4)
                bx = fx.Vector.from_elements(
                    [tok_ok.select(bx[q], Int32(0)) for q in range(4)], Int32
                )
                acc = mfma_bf16(a, bx.bitcast(fx.BFloat16), acc)
        if kh == 1:
            _lds_st(
                lds,
                IQ_RED_LDS + rh * 1024 + lane * 16,
                [acc[q].bitcast(Int32) for q in range(4)],
            )
        gpu.barrier()
        y = [F32(0.0), F32(0.0), F32(0.0), F32(0.0)]
        if kh == 0:
            o = _lds_ld(lds, IQ_RED_LDS + rh * 1024 + lane * 16, 4)
            x = [bf_round(acc[q] + o[q].bitcast(F32)) for q in range(4)]
            c0, c1 = cs[0].bitcast(F32), cs[1].bitcast(F32)
            s0, s1 = sn[0].bitcast(F32), sn[1].bitcast(F32)
            rope = (t % 4) >= 2
            y0 = rope.select(bf_round(fmath.fma(c0, x[0], -(s0 * x[1]))), x[0])
            y1 = rope.select(bf_round(fmath.fma(c0, x[1], s0 * x[0])), x[1])
            y2 = rope.select(bf_round(fmath.fma(c1, x[2], -(s1 * x[3]))), x[2])
            y3 = rope.select(bf_round(fmath.fma(c1, x[3], s1 * x[2])), x[3])
            y = [y0, y1, y2, y3]
            m = fx.max(
                fx.max(fx.max(abs(y0), F32(0.0)), abs(y1)), fx.max(abs(y2), abs(y3))
            )
            m = fx.max(m, _bperm(m, lane ^ 16))
            m = fx.max(m, _bperm(m, lane ^ 32))
            if g == 0:
                _lds_st(lds, IQ_AMAX_LDS + rh * 64 + h16 * 4, [m.bitcast(Int32)])
        gpu.barrier()
        if (kh == 0) & tok_ok:
            a0 = _lds_ld(lds, IQ_AMAX_LDS + tok * 4, 1).bitcast(F32)
            a1 = _lds_ld(lds, IQ_AMAX_LDS + 64 + tok * 4, 1).bitcast(F32)
            amax = fx.max(fx.max(a0, a1), _c(0x01C00000))
            qb = (amax * _c(0x3E2AAAAB)).bitcast(Int32)
            v4 = qb + 0x7FFFFF
            inv = (
                (qb < 0x7E800001)
                .select(Int32(0x7F000000) - (v4 & Int32(-8388608)), Int32(0x400000))
                .bitcast(F32)
            )
            nib = [
                _e2m1(fx.min(fx.max(y[q] * inv, F32(-6.0)), F32(6.0))) for q in range(4)
            ]
            word = nib[0] | (nib[1] << 4) | (nib[2] << 8) | (nib[3] << 12)
            head = t >> 2
            byte = tok * 2048 + head * 64 + (t % 4) * 16 + rh * 8 + g * 2
            bo.buffer_store(word.to(fx.Int16), rsrc(ix), byte, offset_is_bytes=True)
            if (rh == 0) & (g == 0):
                bo.buffer_store(
                    ((v4 >> 23) & 0xFF).to(fx.Int8), rsrc(ix), IX_SCALE_OFF + tok * 128 + head * 4 + t % 4,
                    offset_is_bytes=True,
                )  # fmt: skip


# ---- the KV compressor (q_kv_merged: ratio 2; xf_full<..,1,3,64>: ratio 1) -----------------------------------
COMP_LL = 0x80000  # ws_qkv: GEMV partials {f32, epoch} [K quarter 4][token 6][rows], rows * 8 B a (q, token)
COMP_ROWS = 512 if COMP == 1 else 1024  # wkv (ratio 1) or [wkv; wgate]
COMP_PACK = COMP_ROWS * 8
# epoch slots: GEMV CTA t, finalize row r (and fin + 6.. by row 0)
COMP_GEMV_SLOT, COMP_FIN_SLOT = (880, 1008) if COMP == 1 else (616, 872)
COMP_RED_LDS = 0  # waves 1..7: [w-1][4][64] f32
COMP_SUM_LDS = WAVE_SUMS_LDS  # the finalize's four wave sums of squares


@traced
def qkv_comp_gemv(tid, t, ws, c_w, xn, lds):
    """Compressor CTA t (0..255; ratio 1: 0..127): rows 16(t>>2).. of [wkv; wgate] (1024 x 5120; ratio 1: wkv,
    512 rows; bf16, MFMA-tile packed) over K quarter t&3, the 6 tokens of Xn: wave w chains 5 bf16 MFMAs (K steps
    w*5..), waves summed 0 + 1 + .. + 7; LL {f32, epoch} to ws_qkv + 0x80000."""
    lane, w = tid % 64, tid // 64
    rb, kq = t >> 2, t & 3
    g, h16 = lane >> 4, lane & 15
    epoch = _ld(ws, EPOCHS // 4 + COMP_GEMV_SLOT + t) + 1
    tok_ok = h16 < 6
    tok = tok_ok.select(h16, Int32(0))
    z4 = fx.Vector.filled(4, 0, Int32)
    xb = [z4, z4, z4, z4, z4]
    if tok_ok:
        for i in range_constexpr(5):
            xb[i] = _ld(
                xn, (tok * 10240 + (kq * 40 + w * 5) * 64 + i * 64 + g * 16) // 4, 4
            )
    # the weights stream once a step: non-temporal, as TileRT's
    wt = [
        fx.Vector(bo.buffer_load(rsrc(c_w), (((rb * 160 + kq * 40 + w * 5 + i) << 10) + lane * 16) // 4, vec_width=4,
                                 dtype=T.i32, cache_modifier=CM_NT))
        for i in range(5)
    ]  # fmt: skip
    acc = fx.Vector.filled(4, 0.0, F32)
    for i in range_constexpr(5):
        acc = mfma_bf16(wt[i].bitcast(fx.BFloat16), xb[i].bitcast(fx.BFloat16), acc)
    if w != 0:
        for e in range_constexpr(4):
            _lds_st(
                lds,
                COMP_RED_LDS + (w - 1) * 1024 + e * 256 + lane * 4,
                [acc[e].bitcast(Int32)],
            )
    gpu.barrier()
    if w == 0:
        if lane == 0:
            _st_i(ws, EPOCHS // 4 + COMP_GEMV_SLOT + t, epoch)
        if tok_ok:
            r = [acc[e] for e in range(4)]
            for src in range_constexpr(1, 8):
                r = [
                    r[e]
                    + _lds_ld(
                        lds, COMP_RED_LDS + (src - 1) * 1024 + e * 256 + lane * 4, 1
                    ).bitcast(F32)
                    for e in range(4)
                ]
            at = (COMP_LL + (kq * 6 + tok) * COMP_PACK + rb * 128 + g * 32) // 4
            bo.buffer_store(fx.Vector.from_elements([r[0].bitcast(Int32), epoch, r[1].bitcast(Int32), epoch], Int32),
                            rsrc(ws), at, cache_modifier=CM_DEV)  # fmt: skip
            bo.buffer_store(fx.Vector.from_elements([r[2].bitcast(Int32), epoch, r[3].bitcast(Int32), epoch], Int32),
                            rsrc(ws), at + 4, cache_modifier=CM_DEV)  # fmt: skip


def _comp_parts(ws, row, tid):
    """Token row's 4 quarters x (kv c, kv c+256, score c, score c+256) {f32, tag} words (c = tid; ratio 1: kv
    only)."""
    out = []
    for q in range(4):
        b = (COMP_LL + (q * 6 + row) * COMP_PACK + tid * 8) // 4
        out += [_ld_sc1(ws, b + o, 2) for o in (0, 512, 1024, 1536)[: COMP_ROWS // 256]]
    return out


def _comp_sums(p):
    """(((0 + q0) + q1) + q2) + q3 of each value."""
    n = len(p) // 4
    return [(((F32(0.0) + p[k][0].bitcast(F32)) + p[n + k][0].bitcast(F32)) + p[2 * n + k][0].bitcast(F32))
            + p[3 * n + k][0].bitcast(F32) for k in range(n)]  # fmt: skip


def _comp_ok(p, epoch):
    ok = p[0][1] == epoch
    for k in range(1, len(p)):
        ok = ok & (p[k][1] == epoch)
    return ok


def _e4m3_val(s8):
    """The E4M3 byte's value (TileRT's decode: normal (e << 23 | m << 20) + 0x3c000000, subnormal m 2^-9)."""
    e = (s8 >> 3) & 15
    m = s8 & 7
    nrm = (((e << 23) | (m << 20)) + 0x3C000000).bitcast(F32)
    sub = Int32(m).to(F32) * _c(0x3B000000)
    f = (e != 0).select(nrm, sub)
    return ((s8 & 0x80) != 0).select(-f, f)


@traced
def _comp_fp4(yv, lane, tid, ch, entry, store, kv0, kv1):
    """FP4 QAT of channel ch's value (group 16: E4M3 scale e4m3(max(amax, 6/512) / 6)) -> main_kv.0 / .1."""
    a = abs(yv)
    m = fx.max(a, _bperm(a, lane ^ 8))
    m = fx.max(m, _bperm(m, lane ^ 4))
    m = fx.max(m, _bperm(m, lane ^ 2))
    m = fx.max(fx.max(m, _bperm(m, lane ^ 1)), _c(0x3C400000))
    s8 = Int32(_fp8x2_scaled(m / F32(6.0), F32(0.0), F32(1.0))) & 0xFF
    q = yv / _e4m3_val(s8)
    nib = _e2m1(q)
    other = Int32(_bperm(nib.bitcast(F32), lane ^ 1).bitcast(Int32))
    if store & ((tid & 1) == 0):
        bo.buffer_store(
            (nib | (other << 4)).to(fx.Int8),
            rsrc(kv0),
            entry * 256 + (ch >> 1),
            offset_is_bytes=True,
        )
    if store & ((tid & 15) == 0):
        bo.buffer_store(
            s8.to(fx.Int8), rsrc(kv1), entry * 32 + (ch >> 4), offset_is_bytes=True
        )


@traced
def qkv_comp_fin(
    tid, r, ws, cnorm, state, kv0, kv1, maxent, latent, lat_idx, ropec, vpos, lds
):
    """Compressor finalize, token row r (ratio 2): the quarters summed; the row's (kv, score) into its state
    slot P & 15; an odd P pools with position P - 1 (the previous row of the step when it is that position,
    polled again, else its state slot): per channel softmax of the two scores (OCML exp, IEEE divides),
    kv_prev w_prev + kv w (fma), bf16; RMSNorm (eps 1e-20) x c_norm -> the latent (bf16, pre-RoPE) and its
    entry P >> 1; GPT-J RoPE (mul, mul, sub / add) on channels 448..511 at P - 1; FP4 with E4M3 group-16
    scales -> main_kv.0[entry] (256 B) / main_kv.1[entry] (32 B)."""
    if tid < 256:
        lane, w = tid % 64, tid // 64
        P = _ld(vpos, r)
        pprev = (r > 0).select(_ld(vpos, fx.max(r - 1, Int32(0))), Int32(-2))
        epoch = _ld(ws, EPOCHS // 4 + COMP_FIN_SLOT + r) + 1
        cn0 = _ldf(cnorm, tid)
        cn1 = _ldf(cnorm, tid + 256)
        live = P >= 0
        z2 = fx.Vector.filled(2, 0, Int32)
        p = [z2 for _ in range(16)]
        if live:
            p = _comp_parts(ws, r, tid)
            n_dbg = Int32(0)
            while _more(~_comp_ok(p, epoch), n_dbg):
                n_dbg = n_dbg + 1
                spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
                rocdl.s_sleep(2)
                p = _comp_parts(ws, r, tid)
        cur = _comp_sums(p)
        gpu.barrier()
        if tid == 0:
            _st_i(ws, EPOCHS // 4 + COMP_FIN_SLOT + r, epoch)
        if (r == 0) & (tid < 2):
            _st_i(ws, EPOCHS // 4 + COMP_FIN_SLOT + 6 + tid, epoch)
        odd = live & ((P & 1) == 1)
        lat0 = F32(0.0)
        lat1 = F32(0.0)
        if live:
            sw = (P & 15) * 1024 + tid
            for k in range_constexpr(4):
                _stf(state, sw + k * 256, cur[k])
        if odd:
            inrow = pprev == P - 1
            prv = [F32(0.0), F32(0.0), F32(0.0), F32(0.0)]
            if inrow:
                pp = _comp_parts(ws, fx.max(r - 1, Int32(0)), tid)
                n_dbg = Int32(0)
                while _more(~_comp_ok(pp, epoch), n_dbg):
                    n_dbg = n_dbg + 1
                    spin_pause()
                    rocdl.s_sleep(2)
                    pp = _comp_parts(ws, fx.max(r - 1, Int32(0)), tid)
                prv = _comp_sums(pp)
            else:
                sp = ((P - 1) & 15) * 1024 + tid
                prv = [_ldf(state, sp + k * 256) for k in range(4)]
            lat = []
            for k in range_constexpr(2):
                sprev, scur = prv[2 + k], cur[2 + k]
                mx = fx.max(sprev, scur)
                ep = _exp_ocml(sprev - mx, False)
                ec = _exp_ocml(scur - mx, False)
                sm = ep + ec
                wp = ep / sm
                wc = ec / sm
                lat.append(bf_round(fmath.fma(prv[k], wp, cur[k] * wc)))
            lat0, lat1 = lat[0], lat[1]
        if live & ~odd:
            if tid == 0:
                _st_i(lat_idx, r, Int32(-1))
        if ~live:
            if tid == 0:
                _st_i(lat_idx, r, Int32(-1))
        if odd:
            v = lat0 * lat0 + lat1 * lat1
            # every lane takes all six steps: a bpermute under a lane-0 branch reads lanes whose earlier
            # steps the compiler then sinks into that branch (lane 0 saw a stale partner: ~3/4 of the sum)
            for k in (32, 16, 8, 4, 2, 1):
                v = v + _bperm(v, lane ^ k)
            if lane == 0:
                _lds_st(lds, COMP_SUM_LDS + w * 4, [v.bitcast(Int32)])
        gpu.barrier()
        if odd:
            entry = P >> 1
            if tid == 0:
                _st_i(lat_idx, r, entry)
            w4 = _lds_ld(lds, COMP_SUM_LDS, 4)
            tot = (
                ((F32(0.0) + w4[0].bitcast(F32)) + w4[1].bitcast(F32))
                + w4[2].bitcast(F32)
            ) + w4[3].bitcast(F32)
            ms = fmath.fma(tot, _c(0x3B000000), _c(0x1E3CE508))
            rstd = F32(1.0) / fmath.sqrt(ms)
            y0 = bf_round((lat0 * rstd) * cn0)
            y1 = bf_round((lat1 * rstd) * cn1)
            bo.buffer_store(
                (bf_rne_hi(y0) >> 16).to(fx.Int16),
                rsrc(latent),
                r * 1024 + tid * 2,
                offset_is_bytes=True,
            )
            bo.buffer_store((bf_rne_hi(y1) >> 16).to(fx.Int16), rsrc(latent), r * 1024 + 512 + tid * 2,
                            offset_is_bytes=True)  # fmt: skip
            prow = (entry > -1).select(P & -2, Int32(0))
            pair = (tid >= 192).select((tid - 192) >> 1, Int32(0))
            cs = _ldf(ropec, prow * 64 + pair)
            sn = _ldf(ropec, prow * 64 + 32 + pair)
            oth = _bperm(y1, lane ^ 1)
            even = (tid & 1) == 0
            ye = even.select(y1, oth)
            yo = even.select(oth, y1)
            rot = even.select(cs * ye - sn * yo, sn * ye + cs * yo)
            y1r = (tid >= 192).select(bf_round(rot), y1)
            store = entry < Int32(maxent)
            _comp_fp4(y0, lane, tid, tid, entry, store, kv0, kv1)
            _comp_fp4(y1r, lane, tid, tid + 256, entry, store, kv0, kv1)


@traced
def qkv_comp_fin1(
    tid, r, ws, cnorm, kv0, kv1, maxent, latent, lat_idx, ropec, vpos, lds
):
    """Compressor finalize, token row r (ratio 1: every position P its own entry P): the quarters summed, bf16;
    RMSNorm (eps 1e-20) x c_norm -> the latent (bf16, pre-RoPE), latent_idx = P; GPT-J RoPE on channels 448..511
    at P; FP4 with E4M3 group-16 scales -> main_kv.0[P] (256 B) / main_kv.1[P] (32 B). A dead row (P < 0):
    latent_idx -1."""
    if tid < 256:
        lane, w = tid % 64, tid // 64
        P = _ld(vpos, r)
        epoch = _ld(ws, EPOCHS // 4 + COMP_FIN_SLOT + r) + 1
        cn0 = _ldf(cnorm, tid)
        cn1 = _ldf(cnorm, tid + 256)
        live = P >= 0
        z2 = fx.Vector.filled(2, 0, Int32)
        p = [z2 for _ in range(8)]
        if live:
            p = _comp_parts(ws, r, tid)
            n_dbg = Int32(0)
            while _more(~_comp_ok(p, epoch), n_dbg):
                n_dbg = n_dbg + 1
                spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
                rocdl.s_sleep(2)
                p = _comp_parts(ws, r, tid)
        cur = _comp_sums(p)
        prow = live.select(P, Int32(0))
        pair = (tid >= 192).select((tid - 192) >> 1, Int32(0))
        cs = _ldf(ropec, prow * 64 + pair)
        sn = _ldf(ropec, prow * 64 + 32 + pair)
        gpu.barrier()
        if tid == 0:
            _st_i(ws, EPOCHS // 4 + COMP_FIN_SLOT + r, epoch)
        if (r == 0) & (tid < 2):
            _st_i(ws, EPOCHS // 4 + COMP_FIN_SLOT + 6 + tid, epoch)
        if ~live:
            if tid == 0:
                _st_i(lat_idx, r, Int32(-1))
        lat0 = bf_round(cur[0])
        lat1 = bf_round(cur[1])
        if live:
            v = fmath.fma(lat1, lat1, lat0 * lat0)
            for k in (
                32,
                16,
                8,
                4,
                2,
                1,
            ):  # every lane takes all six steps (see qkv_comp_fin)
                v = v + _bperm(v, lane ^ k)
            if lane == 0:
                _lds_st(lds, COMP_SUM_LDS + w * 4, [v.bitcast(Int32)])
        gpu.barrier()
        if live:
            if tid == 0:
                _st_i(lat_idx, r, P)
            w4 = _lds_ld(lds, COMP_SUM_LDS, 4)
            tot = (
                ((F32(0.0) + w4[0].bitcast(F32)) + w4[1].bitcast(F32))
                + w4[2].bitcast(F32)
            ) + w4[3].bitcast(F32)
            ms = fmath.fma(tot, _c(0x3B000000), _c(0x1E3CE508))
            rstd = F32(1.0) / fmath.sqrt(ms)
            y0 = bf_round((lat0 * rstd) * cn0)
            y1 = bf_round((lat1 * rstd) * cn1)
            bo.buffer_store(
                (bf_rne_hi(y0) >> 16).to(fx.Int16),
                rsrc(latent),
                r * 1024 + tid * 2,
                offset_is_bytes=True,
            )
            bo.buffer_store((bf_rne_hi(y1) >> 16).to(fx.Int16), rsrc(latent), r * 1024 + 512 + tid * 2,
                            offset_is_bytes=True)  # fmt: skip
            oth = _bperm(y1, lane ^ 1)
            even = (tid & 1) == 0
            ye = even.select(y1, oth)
            yo = even.select(oth, y1)
            rot = even.select(cs * ye - sn * yo, sn * ye + cs * yo)
            y1r = (tid >= 192).select(bf_round(rot), y1)
            store = P < Int32(maxent)
            _comp_fp4(y0, lane, tid, tid, P, store, kv0, kv1)
            _comp_fp4(y1r, lane, tid, tid + 256, P, store, kv0, kv1)


def _pick_byte(words, b):
    """Byte b (compile-time or traced) of three little-endian i32 words."""
    wsel = (b // 4 == 0).select(words[0], (b // 4 == 1).select(words[1], words[2]))
    return (wsel >> ((b % 4) * 8)) & 0xFF


def _q_chunks(ws, rep, idx):
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(ws), rep + c * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        for c in idx
    ]


def _tags_ok(ch, e24):
    ok = (ch[0][1] & 0xFFFFFF) == e24
    for k in range_constexpr(4):
        for d in (1, 3):
            if k or d == 3:
                ok = ok & ((ch[k][d] & 0xFFFFFF) == e24)
    return ok


KV_WAVE_SUMS_LDS = 0x11B00  # four wave sums, 24 B apart
KV_LL = 0x1400  # kv columns in a token row's LL packs (pack 640..)


def _shfl_xor(v, lane, k):
    """f32 v of lane ^ k by an exact register move where one exists (DPP quad_perm for 1 / 2, row_ror 8
    for 8, v_permlane16 / 32_swap for 16 / 32: no LDS round trip), else ds_bpermute (k = 4).
    """
    if k == 1:
        return _dpp(v, QP_1032)
    if k == 2:
        return _dpp(v, QP_2301)
    if k == 8:
        return _dpp(v, ROR8)
    if k in (16, 32):
        vi = F32(v).bitcast(Int32)
        a, b = permlane_swap(
            k, vi, vi
        )  # each lane gets {own, partner}: the partner is in the other half
        upper = ((lane >> 4) & 1) == 1 if k == 16 else lane >= 32
        return upper.select(a, b).bitcast(F32)
    return _bperm(v, lane ^ k)


def _amax32(y, lane):
    """|y| max over a 32-lane half wave, floored at 1e-4 (TileRT's xor-16..1 shuffle)."""
    m = abs(y)
    for k in (16, 8, 4, 2):
        m = fx.max(m, _shfl_xor(m, lane, k))
    return fx.max(fx.max(m, _shfl_xor(m, lane, 1)), _c(0x38D1B717))


def _e8_scale(amax):
    """(e, scale): ue8m0 code and its f32 power of two rounding amax / 448 up, 0 -> 2^-127."""
    sb = (amax * _c(0x3B124925)).bitcast(Int32) + 0x7FFFFF
    e = (sb >> 23) & 0xFF
    return e, (e == 0).select(Int32(0x400000), sb & Int32(-8388608)).bitcast(F32)


@traced
def qkv_kvnorm(tid, s16, ws, vpos, kv_gamma, ropec, win_kv, win_s, alt_kv, alt_s, lds):
    """Logical CTA 480 + row: the row's 512 kv features (from the LL packs) RMS-normed, RoPE on the
    last 64, written as E4M3 + ue8m0 to the window ring slot pos % 256 (or the alternate buffer).
    """
    row = s16 - 480
    if tid < 256:
        lane, w = tid % 64, tid // 64
        epoch = _ld(ws, EPOCHS // 4 + s16) + 1
        pos = _ld(vpos, row)
        a = F32(0.0)
        b = F32(0.0)
        if pos >= 0:
            pw = (LL_QA + row * 0x1C00 + KV_LL + (tid // 2) * 8) // 4
            pa = _ld_sc1(ws, pw, 2)
            pb = _ld_sc1(ws, pw + 256, 2)
            n_dbg = Int32(0)
            while _more((pa[1] != epoch) | (pb[1] != epoch), n_dbg):
                n_dbg = n_dbg + 1
                spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload
                rocdl.s_sleep(1)
                pa = _ld_sc1(ws, pw, 2)
                pb = _ld_sc1(ws, pw + 256, 2)
            sh = (tid & 1) * 16
            a = ((pa[0] >> sh) << 16).bitcast(F32)
            b = ((pb[0] >> sh) << 16).bitcast(F32)
        gpu.barrier()
        if tid == 0:
            _st_i(ws, EPOCHS // 4 + s16, epoch)
        if (row == 0) & (tid < 8 - QROWS):  # the unused rows' slots up to 8
            _st_i(ws, EPOCHS // 4 + 480 + QROWS + tid, epoch)
        if pos >= 0:
            g1 = _ldf(kv_gamma, tid)
            g2 = _ldf(kv_gamma, tid + 256)
            rope = tid >= 192
            pair = rope.select((tid - 192) >> 1, Int32(0))
            tb = fx.max(pos, Int32(0)) * 64 + pair
            cs = _ldf(ropec, tb)
            sn = _ldf(ropec, tb + 32)
            v = fmath.fma(a, a, b * b)
            for k in range_constexpr(5):
                v = v + _shfl_xor(v, lane, 32 >> k)
            last = _shfl_xor(v, lane, 1)
            if lane == 0:
                _lds_st(lds, KV_WAVE_SUMS_LDS + w * 24, [(v + last).bitcast(Int32)])
            gpu.barrier()
            tot = F32(0.0)
            for q in range_constexpr(4):
                tot = tot + _lds_ld(lds, KV_WAVE_SUMS_LDS + q * 24, 1).bitcast(F32)
            rstd = F32(1.0) / fmath.sqrt(fmath.fma(tot, _c(0x3B000000), _c(0x1E3CE508)))
            alt = alt_kv != 0
            slot = alt.select(row, pos & 0xFF)
            kv_dst = alt.select(alt_kv, win_kv)
            s_dst = alt.select(alt_s, win_s)
            even = (tid & 1) == 0
            ya = bf_round(g1 * (a * rstd))
            ea, sa = _e8_scale(_amax32(ya, lane))
            na = _shfl_xor(ya, lane, 1)
            if even:
                q8 = _fp8x2_scaled(ya, na, sa)
                bo.buffer_store(
                    q8.to(fx.Int16),
                    rsrc(kv_dst),
                    slot * 512 + tid,
                    offset_is_bytes=True,
                )
            if (tid % 32) == 0:
                bo.buffer_store(
                    ea.to(fx.Int8),
                    rsrc(s_dst),
                    slot * 16 + tid // 32,
                    offset_is_bytes=True,
                )
            yb = bf_round(g2 * (b * rstd))
            nb = _shfl_xor(yb, lane, 1)
            if rope:
                x0 = even.select(yb, nb)
                x1 = even.select(nb, yb)
                yb = bf_round(even.select(cs * x0 - sn * x1, cs * x1 + sn * x0))
            eb, sbv = _e8_scale(_amax32(yb, lane))
            nb2 = _shfl_xor(yb, lane, 1)
            if even:
                q8 = _fp8x2_scaled(yb, nb2, sbv)
                bo.buffer_store(
                    q8.to(fx.Int16),
                    rsrc(kv_dst),
                    slot * 512 + 256 + tid,
                    offset_is_bytes=True,
                )
            if (tid % 32) == 0:
                bo.buffer_store(
                    eb.to(fx.Int8),
                    rsrc(s_dst),
                    slot * 16 + 8 + tid // 32,
                    offset_is_bytes=True,
                )


def _ld_sc1(addr, word, n):
    return fx.Vector(
        bo.buffer_load(
            rsrc(addr), word, vec_width=n, dtype=T.i32, cache_modifier=CM_DEV
        )
    )


def build_finalize_test():
    """Only CTA 0's finalize (when flagged): diff hcA against the original."""
    name = kernel_symbol(
        "tk_q_kv_xf_full_probe" if XF else "tk_q_kv_xf_kg",
        rows=QROWS,
        full=1,
        kv=8,
        idx=IDX,
        v=3,
    )

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_BYTES // 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def fin(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _test_entry(
            fx.thread_idx.x,
            fx.block_idx.x,
            lds,
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
            a27,
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
        )

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        fin(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch


@traced
def _test_entry(
    tid, cta, lds, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11,
    a27, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45,
):  # fmt: skip
    """All five roles: finalize, q_a / kv GEMV, q-norm, q_b + RoPE, kv-norm + window ring."""
    flag = (a06 & 1) != 0
    if flag & (cta == 0):
        mhc_finalize(tid, a07, a08, a09, a10, Int32(a11))
    s16 = flag.select(cta - 1, cta)
    if (s16 >= 0) & (s16 < 224) & ((cta != 0) | (~flag)):
        epoch = qkv_gemv(tid, s16, a00, a01, a02, a03, a04, a05, a06 & ~1, lds)
        if s16 < QROWS * 8:
            qkv_qnorm(tid, s16, epoch, a31, a32, a38, a44, a45, lds)
    if (s16 >= 224) & (s16 < 480):
        qkv_qb(tid, s16, a31, a33, a34, a36, a37, a39, a27, lds)
    if (s16 >= 480) & (s16 < 480 + QROWS):
        qkv_kvnorm(tid, s16, a31, a37, a35, a36, a40, a41, a42, a43, lds)


@traced
def _fin_entry(tid, cta, a06, a07, a08, a09, a10, a11):
    """The kernarg's 0x30 word (a06) carries the finalize flag in bit 0; MhcFinalizeArgs at
    0x38: mHC ws, attn_scale, attn_base, hcA (pointers), the row count at 0x58 (a11)."""
    if ((a06 & 1) != 0) & (cta == 0):
        mhc_finalize(tid, a07, a08, a09, a10, Int32(a11))


def build_xf():
    """q_kv_xf_full<6, true, 8, 0, 3, 0> (TK_QKV_XF=1, TK_QKV_IDX=0): its 58 argument words."""
    assert XF
    extra = {"comp": COMP} if COMP == 1 else {}
    name = kernel_symbol(
        "tk_q_kv_merged" if MERGED else "tk_q_kv_xf_full",
        rows=6,
        full=1,
        kv=8,
        idx=IDX,
        **extra,
        v=1,
    )

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_BYTES // 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def xff(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
        a56: Int64, a57: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        if const_expr(COMP == 1):
            _xf1_entry(fx.thread_idx.x, fx.block_idx.x, lds, a00, a01, a02, a03, a04, a05, a06,
                       a08, a09, a10, a11, a12, a16, a20, a21, a22, a25, a28, a30, a32,
                       a33, a34, a35, a37, a38, a39, a40, a41,
                       a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56)  # fmt: skip
        elif const_expr(MERGED):
            _merged_entry(fx.thread_idx.x, fx.block_idx.x, lds, a00, a01, a02, a03, a04, a05, a06,
                          a08, a09, a10, a11, a12, a16, a20, a21, a22, a25, a28, a30,
                          a33, a34, a35, a36, a37, a38, a39, a40, a41,
                          a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56)  # fmt: skip
        else:
            _xf_entry(fx.thread_idx.x, fx.block_idx.x, lds, a00, a01, a02, a03, a04, a05, a06,
                      a08, a09, a10, a11, a12, a16, a20, a21, a22, a25, a28, a30, a32,
                      a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56)  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
        a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
        a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
        a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
        a56: Int64, a57: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        xff(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
            a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
            a48, a49, a50, a51, a52, a53, a54, a55, a56, a57,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    # grids 487 / 621 / 755 > 256 CUs: two 512-thread CTAs a CU (LDS 2 x 72560 B fits) only at <= 128 VGPRs, 4
    # waves a SIMD (TileRT's: 125 / 125 / 128); left free the compiler took 134, one CTA a CU, the rest a round later
    launch.compile_hints = {**launch.compile_hints, "waves_per_eu": 4}
    return name, launch


@traced
def _xf_entry(
    tid, cta, lds, xq, xs, qa_w, qa_s, kv_w, kv_s, ws0, mws, mscale, mbase, hca, nrows, xn, iwq, iwq_s, iwp,
    ropec_i, ix, rope_n, qn_base, ws, qn_g, qb_w, qb_s, kv_g, ropec, vpos, qr, q_out, win_kv, win_s, alt_kv,
    alt_s, lat, lat_s,
):  # fmt: skip
    """q_kv_xf_full's roles: CTA 0 the mHC finalize when the row count (0x60) is >= 1 (the rest then logical
    CTA wgid - 1); GEMV 0..223, then q-norm on [base, base + 48) (base = int@0x100 rounded up to 8) and the
    indexer prep on 0..159 (iq 0..127, iw 128..159); q_b + RoPE 224..479; kv-norm + window 480..485.
    """
    flag = Int32(nrows) >= 1
    if flag & (cta == 0):
        mhc_finalize(tid, mws, mscale, mbase, hca, Int32(nrows))
    s16 = flag.select(cta - 1, cta)
    qn0 = (Int32(qn_base) + 7) & -8
    if (s16 >= 0) & (s16 < 224) & ((cta != 0) | (~flag)):
        epoch = qkv_gemv(tid, s16, xq, xs, qa_w, qa_s, kv_w, kv_s, ws0, lds)
        if (s16 >= qn0) & (s16 < qn0 + 48):
            qkv_qnorm(tid, s16 - qn0, epoch, ws, qn_g, qr, lat, lat_s, lds)
        if s16 < 128:
            qkv_iq(tid, s16, ws, iwq, iwq_s, ropec_i, rope_n, vpos, ix, lds)
        if (s16 >= 128) & (s16 < 160):
            qkv_iw(tid, s16, xn, iwp, ix, lds)
    if (s16 >= 224) & (s16 < 480):
        qkv_qb(tid, s16, ws, qb_w, qb_s, ropec, vpos, q_out, ix, lds)
    if (s16 >= 480) & (s16 < 486):
        qkv_kvnorm(tid, s16, ws, vpos, kv_g, ropec, win_kv, win_s, alt_kv, alt_s, lds)


@traced
def _xf1_entry(
    tid, cta, lds, xq, xs, qa_w, qa_s, kv_w, kv_s, ws0, mws, mscale, mbase, hca, nrows, xn, iwq, iwq_s, iwp,
    ropec_i, ix, rope_n, qn_base, cx, c_w, c_norm, kv0, kv1, maxent, latent, lat_idx, ws, qn_g, qb_w, qb_s, kv_g,
    ropec, vpos, qr, q_out, win_kv, win_s, alt_kv, alt_s, lat, lat_s,
):  # fmt: skip
    """q_kv_xf_full<6,1,8,1,3,64>'s roles: xf_full's (GEMV 0..223 with q-norm and the indexer prep) plus the
    ratio-1 compressor GEMV 224..351; q_b + RoPE 352..607; kv-norm + window 608..613; compressor finalize
    614..619."""
    flag = Int32(nrows) >= 1
    if flag & (cta == 0):
        mhc_finalize(tid, mws, mscale, mbase, hca, Int32(nrows))
    s16 = flag.select(cta - 1, cta)
    qn0 = (Int32(qn_base) + 7) & -8
    if (s16 >= 0) & (s16 < 224) & ((cta != 0) | (~flag)):
        epoch = qkv_gemv(tid, s16, xq, xs, qa_w, qa_s, kv_w, kv_s, ws0, lds)
        if (s16 >= qn0) & (s16 < qn0 + 48):
            qkv_qnorm(tid, s16 - qn0, epoch, ws, qn_g, qr, lat, lat_s, lds)
        if s16 < 128:
            qkv_iq(tid, s16, ws, iwq, iwq_s, ropec_i, rope_n, vpos, ix, lds)
        if (s16 >= 128) & (s16 < 160):
            qkv_iw(tid, s16, xn, iwp, ix, lds)
    if (s16 >= 224) & (s16 < 352):
        qkv_comp_gemv(tid, s16 - 224, ws, c_w, cx, lds)
    if (s16 >= 352) & (s16 < 608):
        qkv_qb(tid, s16 - 128, ws, qb_w, qb_s, ropec, vpos, q_out, ix, lds)
    if (s16 >= 608) & (s16 < 614):
        qkv_kvnorm(
            tid, s16 - 128, ws, vpos, kv_g, ropec, win_kv, win_s, alt_kv, alt_s, lds
        )
    if (s16 >= 614) & (s16 < 620):
        qkv_comp_fin1(
            tid,
            s16 - 614,
            ws,
            c_norm,
            kv0,
            kv1,
            maxent,
            latent,
            lat_idx,
            ropec,
            vpos,
            lds,
        )


@traced
def _merged_entry(
    tid, cta, lds, xq, xs, qa_w, qa_s, kv_w, kv_s, ws0, mws, mscale, mbase, hca, nrows, xn, iwq, iwq_s, iwp,
    ropec_i, ix, rope_n, cx, c_w, c_norm, state, kv0, kv1, maxent, latent, lat_idx, ws, qn_g, qb_w, qb_s, kv_g,
    ropec, vpos, qr, q_out, win_kv, win_s, alt_kv, alt_s, lat, lat_s,
):  # fmt: skip
    """q_kv_merged's roles: CTA 0 the mHC finalize (row count >= 1, the rest logical wgid - 1); GEMV 0..223
    (then the indexer prep: iq 0..127, iw 128..159); compressor GEMV 224..479; q-norm 480..485 (a CTA a row,
    one latent copy at 0x38000); q_b + RoPE 486..741; kv-norm + window 742..747; compressor finalize 748..753.
    """
    flag = Int32(nrows) >= 1
    if flag & (cta == 0):
        mhc_finalize(tid, mws, mscale, mbase, hca, Int32(nrows))
    s16 = flag.select(cta - 1, cta)
    if (s16 >= 0) & (s16 < 224) & ((cta != 0) | (~flag)):
        qkv_gemv(tid, s16, xq, xs, qa_w, qa_s, kv_w, kv_s, ws0, lds)
        if s16 < 128:
            qkv_iq(tid, s16, ws, iwq, iwq_s, ropec_i, rope_n, vpos, ix, lds)
        if (s16 >= 128) & (s16 < 160):
            qkv_iw(tid, s16, xn, iwp, ix, lds)
    if (s16 >= 224) & (s16 < 480):
        qkv_comp_gemv(tid, s16 - 224, ws, c_w, cx, lds)
    if (s16 >= 480) & (s16 < 486):
        row = s16 - 480
        epoch = _ld(ws, EPOCHS // 4 + 1016 + row) + 1
        qkv_qnorm(tid, row * 8, epoch, ws, qn_g, qr, lat, lat_s, lds)
    if (s16 >= 486) & (s16 < 742):
        qkv_qb(tid, s16 - 262, ws, qb_w, qb_s, ropec, vpos, q_out, ix, lds)
    if (s16 >= 742) & (s16 < 748):
        qkv_kvnorm(
            tid, s16 - 262, ws, vpos, kv_g, ropec, win_kv, win_s, alt_kv, alt_s, lds
        )
    if (s16 >= 748) & (s16 < 754):
        qkv_comp_fin(
            tid,
            s16 - 748,
            ws,
            c_norm,
            state,
            kv0,
            kv1,
            maxent,
            latent,
            lat_idx,
            ropec,
            vpos,
            lds,
        )


build = build_finalize_test
