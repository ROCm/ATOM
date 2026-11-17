# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::mega::mega_afm_tail_kernel<1, 384, 573440, 8, 14>``.
# ruff: noqa: SIM102  (traced code: nested ifs are nested scf.if, not one fused condition)
"""A layer body of DeepSeek-V4.1 on gfx950 (TP8) as a FlyDSL port of TileRT's mega tail kernel: the
kernarg ABI unchanged (AttnArgs @0, ProjoArgs @112, MhcFArgs @216, MoeArgs @328, TailArgs @456; 584 B
of explicit arguments read by offset as 73 i64 words), grid 256 x 512, ported one phase at a time
(``runs/20261115-flydsl-tilert/m4/tail_notes.md``), each diffed bit-exact by ``m2/x_replay8.py``.

Phase A1 (CTAs s2 < 160 with row = s2 % 8 < 6; chunk = s2 / 8): one (row, 32-key chunk) of the
flash-decoding attention over 8 heads x 512 dims; chunks 0..3 the window ring (fp8, e8m0 scales),
4..19 the indexer-selected main KV (FP4 x two fp8 scales, truncated to bf16). Unnormalized O (fp32)
to Attn+0x60, then {M, epoch} / {L, epoch} to Attn+0x68.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import (
    bf_rne_hi,
    bperm,
    exp_ocml,
    f32c,
    log1p_ocml,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    HALF_MIRROR as QK_HALF_MIRROR,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    MIRROR as QK_MIRROR,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    QP_1032 as QK_QP_1032,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    QP_2301 as QK_QP_2301,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    _col_sum as qk_col_sum,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    _div_nr as qk_div_nr,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    _dpp as qk_dpp,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    _quad_sum as qk_quad_sum,
)
from atom.mono.device.mx import FP4, FP8, mfma_scaled
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    hw_rsq,
    kernel_symbol,
    launcher,
    load_ptr64,
    memrealtime,
    mfma_bf16,
    permlane_swap,
    readlane,
    rsrc,
    spin_pause,
    traced,
    uniform,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

_AFM6_ENV = __import__("os").environ.get("TK_VARIANT", "tail") == "afm6"
KARG_WORDS = (
    57 if _AFM6_ENV else 73
)  # mega_afm<6,0>: 456 B, the tail kernel: 584 B of explicit kernarg
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 256, 512
LDS_BYTES = 95616

K_ROW = 0x410  # a key's 512 bf16 + 16 B pad in LDS
QK_PART = 0x8400 + 768  # [slice 4][key 32][head 16] f32 partial scores, 2048 B a slice
P_ROW = 40  # P[head][key] bf16 in its own LDS array, 40 entries (80 B) a head, as TileRT's 0x50 stride
MAX_LDS, SUM_LDS, VALID_LDS = (
    42752,
    43264,
    43776,
)  # per-wave [w][head] max / sum, per-key valid
CHUNKS = 20  # a row's partials: 4 window + 16 main-KV chunks of 32 keys
INV_SQRT_D = 0x3D3504F3  # 1 / sqrt(512)


def _ld(addr, word, n=1, cm=0):
    v = bo.buffer_load(rsrc(addr), word, vec_width=n, dtype=T.i32, cache_modifier=cm)
    return fx.Vector(v) if n > 1 else Int32(v)


def _lds_ld(lds, byte, n=1):
    if n == 1:
        return Int32(fx.ptr_load(lds + (byte // 4), result_type=Int32.ir_type))
    return fx.Vector(
        fx.ptr_load(lds + (byte // 4), result_type=fx.Vector.make_type(n, Int32))
    )


def _lds_st(lds, byte, vals):
    v = (
        fx.Vector.from_elements([Int32(x) for x in vals], Int32)
        if len(vals) > 1
        else Int32(vals[0])
    )
    fx.ptr_store(v, lds + (byte // 4))


def _lds_u16(lds, byte):
    """The u16 at an even LDS byte offset, from its dword."""
    return (_lds_ld(lds, byte & ~3) >> ((byte & 2) * 8)) & 0xFFFF


def _cvt_bf16_fp8(word, scale, hi):
    r = rocdl.cvt_scalef32_pk_bf16_fp8(
        T.bf16x2, as_ir_value(Int32(word)), as_ir_value(F32(scale)), hi
    )
    return fx.Vector(r).bitcast(Int32)[0]


def _cvt_f32_fp4(word, byte):
    v2 = fx.Vector.make_type(2, F32)
    r = fx.Vector(
        rocdl.cvt_scalef32_pk_f32_fp4(
            res=v2,
            src=as_ir_value(Int32(word)),
            scale=as_ir_value(F32(1.0)),
            src_sel_index=byte,
        )
    )
    return r[0], r[1]


def _cvt_f32_fp8x2(half):
    v2 = fx.Vector.make_type(2, F32)
    r = fx.Vector(
        rocdl.cvt_scalef32_pk_f32_fp8(
            res=v2,
            src=as_ir_value(Int32(half)),
            scale=as_ir_value(F32(1.0)),
            src_lo_hi_sel=False,
        )
    )
    return r[0], r[1]


def _trunc_pair(a, b):
    """Two f32 -> bf16 pair by truncation (TileRT's v_perm 0x7060302)."""
    return ((F32(a).bitcast(Int32) >> 16) & 0xFFFF) | (
        F32(b).bitcast(Int32) & Int32(-65536)
    )


def _mfma16k16(a2, b2, c):
    """16x16x16 bf16 from two i32 words each (4 bf16 a lane)."""
    a = fx.Vector.from_elements([Int32(x) for x in a2], Int32).bitcast(fx.Int16)
    b = fx.Vector.from_elements([Int32(x) for x in b2], Int32).bitcast(fx.Int16)
    return fx.Vector(
        rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [a.ir_value(), b.ir_value(), c])
    )


def _x(v, lane, k):
    return bperm(v, lane ^ k)


QX1, QX2, ROW_HALF_MIRROR, ROW_MIRROR = (
    0xB1,
    0x4E,
    0x141,
    0x140,
)  # DPP: quad xor 1 / 2, row mirrors
QROT1, QROT3 = (
    0x39,
    0x93,
)  # DPP quad_perm [1,2,3,0] / [3,0,1,2]: lane + 1 / + 3 on lanes % 4 == 0
ROW_SHL4, ROW_SHL8, ROW_SHL12 = (
    0x104,
    0x108,
    0x10C,
)  # DPP row_shl: lane + n inside a 16-lane row


def _dppf(v, ctrl):
    """f32 v of another lane by a DPP move (quad xor 1 / 2, row mirrors): the same values as the
    ds_bpermute it replaces without the LDS round trip (inactive sources read the lane's own value).
    """
    vi = F32(v).bitcast(Int32)
    return Int32(update_dpp_i32(vi, vi, ctrl, 0xF, 0xF, False)).bitcast(F32)


def _xsym(v, off, op):
    """op(v, v of lane ^ off) for a commutative op (max, IEEE add), off 16 or 32, by v_permlane{16,32}_swap
    (each lane gets {own, partner} back in some order: the same values as the ds_bpermute butterfly).
    """
    vi = F32(v).bitcast(Int32)
    a, b = permlane_swap(off, vi, vi)
    return op(a.bitcast(F32), b.bitcast(F32))


def _up32(v):
    """v of lane + 32 on lanes 0..31 (v_permlane32_swap: the second result's lower half)."""
    vi = F32(v).bitcast(Int32)
    return permlane_swap(32, vi, vi)[1].bitcast(F32)


def _fadd(a, b):
    return a + b


def _gather_sc(sbf, lane):
    """The scale bytes of lanes + 4 / + 8 / + 12 by DPP row_shl (read on lanes % 16 == 0 only)."""
    return tuple(
        Int32(_dppf(sbf, c).bitcast(Int32)) for c in (ROW_SHL4, ROW_SHL8, ROW_SHL12)
    )


def _max32(m, lane):
    """max over each 32-lane half: TileRT's quad xor 1, xor 2, row half-mirror, row mirror, permlane16 swap."""
    for c in (QX1, QX2, ROW_HALF_MIRROR, ROW_MIRROR):
        m = fx.max(m, _dppf(m, c))
    return _xsym(m, 16, fx.max)


def _lds_vec(lds, byte, n):
    """n f32 words at a 16 B aligned LDS byte offset by n / 4 ds_read_b128 (consecutive scalar reads are
    not merged past ds_read2_b32 here)."""
    out = []
    for k in range(n // 4):
        v4 = _lds_ld(lds, byte + 16 * k, 4)
        out += [v4[i].bitcast(F32) for i in range(4)]
    return out


def _any_valid(lds):
    """OR of the 32 per-key valid words (128 B at VALID_LDS) by 8 ds_read_b128 (was 32 ds_read_b32)."""
    anyv = Int32(0)
    for k in range_constexpr(8):
        v4 = _lds_ld(lds, VALID_LDS + k * 16, 4)
        anyv = anyv | ((v4[0] | v4[1]) | (v4[2] | v4[3]))
    return anyv


@traced
def attn_a1(
    tid, s2, q, win_kv, win_s, mkv, mkv_s, ilist, vpos, o_part, ml_part, epoch, lds, pl
):
    row, ch = s2 % 8, s2 // 8
    lane, w = tid % 64, tid // 64
    lg, h = (tid >> 4) & 3, tid & 15
    kt = tid >> 4  # the key this thread stages (0..31)
    # TileRT issues vpos, the indexer-list entry and Q together and tests pos afterwards: no load waits
    # on another (the list entry and Q loads masked to where they are used)
    pos = _ld(vpos, row)
    window = s2 < 32
    li = (
        row * 512 + ch * 32 + kt - 128
    )  # >= 0 on the non-window CTAs (ch >= 4), the only users
    tagged = (ilist & 1) != 0
    base = ilist - 1
    sl = w >> 1
    # (v9 issued the list entry and Q before the pos test: measured slightly slower, removed in v11. Its
    # `kv` first assigned under `if const_expr(...)` before the poll loop was not carried by the traced
    # while: the reload was dead code and the loop compiled to a load-free spin -> the v10 in-graph hang.
    # Poll loops here assign their polled value unconditionally right before the loop.)
    zero4 = fx.Vector.filled(4, 0, Int32)
    qv = [zero4, zero4, zero4, zero4]
    want = Int32(0)
    kv = fx.Vector.filled(2, 0, Int32)
    # v36, TileRT's order: vpos, the list's epoch word (plain), its entry (sys) and Q go out together,
    # pos is tested after (v9 tried this when other stages dominated; the per-layer profile shows
    # the layers right after the indexer pay the serial round trips most)
    if (~window) & tagged:
        want = _ld(base, 0, 1)
        kv = _ld(base, 64 + li * 2, 2, CM_SYS)
    if h < 8:
        for m in range_constexpr(4):
            qv[m] = _ld(
                q, (row * 8192 + h * 1024 + sl * 256 + lg * 16 + m * 64) // 4, 4
            )
    if pos >= 0:
        key = Int32(-1)
        if window:
            key = ch * 32 + pos + kt - 127
        else:
            if tagged:
                while kv[1] != want:
                    spin_pause()  # s_sleep is IntrNoMem: keep an opaque op beside it, as in every poll here
                    rocdl.s_sleep(1)
                    kv = _ld(base, 64 + li * 2, 2, CM_SYS)
                key = kv[0]
            else:
                key = _ld(ilist, li)
        ok = key >= 0
        # this thread's key, dims h*32 .. +31 -> 16 words of bf16 pairs
        kw = [Int32(0)] * 16
        if ok:
            if window:
                slot = key & 0xFF
                a = _ld(win_kv, (slot * 512 + h * 32) // 4, 4)
                b = _ld(win_kv, (slot * 512 + h * 32 + 16) // 4, 4)
                sb = bo.buffer_load(rsrc(win_s), slot * 16 + h, vec_width=1, dtype=T.i8)
                sc = ((Int32(sb).to(Int32) & 0xFF) << 23 & 0x7F800000).bitcast(F32)
                words = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
                for i in range_constexpr(8):
                    kw[2 * i] = _cvt_bf16_fp8(words[i], sc, False)
                    kw[2 * i + 1] = _cvt_bf16_fp8(words[i], sc, True)
            else:
                v = _ld(mkv, (key * 256 + h * 16) // 4, 4)
                s16 = bo.buffer_load(
                    rsrc(mkv_s),
                    key * 16 + h,  # offset in i16 units (32 B a key)
                    vec_width=1,
                    dtype=T.i16,
                )
                s0, s1 = _cvt_f32_fp8x2(Int32(s16).to(Int32) & 0xFFFF)
                for wd in range_constexpr(4):
                    sc = s0 if wd < 2 else s1
                    for byte in range_constexpr(4):
                        d0, d1 = _cvt_f32_fp4(v[wd], byte)
                        kw[wd * 4 + byte] = _trunc_pair(sc * d0, sc * d1)
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
def _a1_body(tid, row, ch, lane, w, lg, h, qv, o_part, ml_part, epoch, lds, pl):
    kg, sl = w & 1, w >> 1
    acc = fx.Vector.filled(4, 0.0, F32)
    for m in range_constexpr(4):
        a = _lds_ld(lds, (kg * 16 + h) * K_ROW + sl * 256 + lg * 16 + m * 64, 4)
        acc = mfma_bf16(a.bitcast(fx.BFloat16), qv[m].bitcast(fx.BFloat16), acc)
    for i in range_constexpr(4):
        _lds_st(
            lds,
            QK_PART + sl * 2048 + kg * 1024 + lg * 256 + i * 64 + h * 4,
            [acc[i].bitcast(Int32)],
        )
    gpu.barrier()
    ks = (w & 1) * 16 + lg * 4 + (w >> 1)
    p = [
        _lds_ld(lds, QK_PART + s * 2048 + ks * 64 + h * 4).bitcast(F32)
        for s in range(4)
    ]
    score = (((p[0] + p[1]) + p[2]) + p[3]) * f32c(INV_SQRT_D)
    valid = _lds_ld(lds, VALID_LDS + ks * 4) != 0
    score = valid.select(score, f32c(0xFF800000))
    mx = _xsym(_xsym(score, 16, fx.max), 32, fx.max)
    if lg == 0:
        _lds_st(lds, MAX_LDS + w * 64 + h * 4, [mx.bitcast(Int32)])
    gpu.barrier()
    big = _max8(lds, h)
    live = score != f32c(0xFF800000)
    e = live.select(exp_ocml(score - big, False), F32(0.0))
    sm = _xsym(_xsym(e, 16, _fadd), 32, _fadd)
    if lg == 0:
        _lds_st(lds, SUM_LDS + w * 64 + h * 4, [sm.bitcast(Int32)])
    fx.ptr_store((bf_rne_hi(e) >> 16).to(fx.Int16), pl + (h * P_ROW + ks))
    gpu.barrier()
    b0 = fx.Vector(
        fx.ptr_load(
            pl + (h * P_ROW + lg * 4), result_type=fx.Vector.make_type(4, fx.Int16)
        )
    ).bitcast(Int32)
    b1 = fx.Vector(
        fx.ptr_load(
            pl + (h * P_ROW + 16 + lg * 4), result_type=fx.Vector.make_type(4, fx.Int16)
        )
    ).bitcast(Int32)
    for t in range_constexpr(4):
        base = (w * 64 + h) * 2 + lg * 4 * K_ROW + t * 32
        a0 = [_lds_u16(lds, base + j * K_ROW) for j in range(4)]
        a1 = [_lds_u16(lds, base + 16 * K_ROW + j * K_ROW) for j in range(4)]
        r = _mfma16k16(
            [a0[0] | (a0[1] << 16), a0[2] | (a0[3] << 16)],
            [b0[0], b0[1]],
            fx.Vector.filled(4, 0.0, F32),
        )
        r = _mfma16k16(
            [a1[0] | (a1[1] << 16), a1[2] | (a1[3] << 16)], [b1[0], b1[1]], r
        )
        if h < 8:
            at = (
                ((((row * CHUNKS + ch) * 8) | h) << 11) + (w * 64 + lg * 4) * 4 + t * 64
            )
            bo.buffer_store(
                fx.Vector.from_elements([r[i].bitcast(Int32) for i in range(4)], Int32),
                rsrc(o_part),
                at // 4,
            )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid < 8:
        big8 = _max8(lds, tid)
        tot = F32(0.0)
        for k in range_constexpr(8):
            tot = tot + _lds_ld(lds, SUM_LDS + k * 64 + tid * 4).bitcast(F32)
        at = (((row * CHUNKS + ch) * 8) | tid) * 4
        bo.buffer_store(
            fx.Vector.from_elements([big8.bitcast(Int32), epoch], Int32),
            rsrc(ml_part),
            at,
        )
        bo.buffer_store(
            fx.Vector.from_elements([tot.bitcast(Int32), epoch], Int32),
            rsrc(ml_part),
            at + 2,
        )


MERGE_DEN = 160  # LDS: the merge denominator (exp(sink - M) + sum l e)
MERGE_ACC = 168  # LDS: [group 4][dim 512] f32 at group * 4096 + dim * 4


def _ldf(addr, word):
    return _ld(addr, word).bitcast(F32)


@traced
def attn_a2(tid, s2, vpos, sink, ropec, rope_len, o_part, ml_part, xout, epoch, lds):
    """Merge CTA (row = s2 % 8, head = s2 / 8 - 24): the row's 20 partials of one head weighted by
    exp(m_j - M) over l = exp(sink - M) + sum l_j exp(m_j - M), inverse RoPE on the last 64 dims, bf16
    to the wo_a input as LL packs {bf16x2, epoch} (sc1) at ProjoArgs+0x28."""
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
            NP = 4 if const_expr(_AFM6) else 20  # <6,0>: the 4 window chunks only
            if lane < NP:
                word = (row * 160 + head + lane * 8) * 4
                pm = _ld(ml_part, word, 2, CM_DEV)
                pl2 = _ld(ml_part, word + 2, 2, CM_DEV)
                # TileRT's pure spin (no s_sleep). spin_pause (a side-effecting s_nop) keeps the reloads
                # in the loop: without it LLVM treats them as invariant and drops the loop (a hang)
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
            if lane < NP:
                _lds_st(lds, lane * 4, [e.bitcast(Int32)])
            tot = F32(0.0) + readlane(lw, 0)
            for j in range_constexpr(1, 4 if _AFM6 else 20):
                tot = tot + readlane(lw, j)
            if lane == 0:
                _lds_st(
                    lds, MERGE_DEN, [(exp_ocml(snk - big, False) + tot).bitcast(Int32)]
                )
        gpu.barrier()
        k, t = tid >> 7, tid & 127
        PB = (
            1 if const_expr(_AFM6) else 5
        )  # partials a 128-thread block: chunk k*PB + i
        ev = [_lds_ld(lds, (k * PB + i) * 4).bitcast(F32) for i in range(PB)]
        ov = [
            _ld(
                o_part,
                (((row * 160 + head + k * PB * 8) << 11) + t * 16 + i * 0x4000) // 4,
                4,
            )
            for i in range(PB)
        ]
        acc = []
        for d in range_constexpr(4):
            a = (ev[0] > 0.0).select(
                fmath.fma(ev[0], ov[0][d].bitcast(F32), F32(0.0)), F32(0.0)
            )
            for i in range_constexpr(1, 1 if _AFM6 else 5):
                a = a + (ev[i] > 0.0).select(ev[i] * ov[i][d].bitcast(F32), F32(0.0))
            acc.append(a.bitcast(Int32))
        _lds_st(lds, MERGE_ACC + k * 4096 + t * 16, acc)
        gpu.barrier()
        g = [
            _lds_ld(lds, MERGE_ACC + q * 4096 + tid * 4).bitcast(F32) for q in range(4)
        ]
        o = (((g[0] + g[1]) + g[2]) + g[3]) / _lds_ld(lds, MERGE_DEN).bitcast(F32)
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
    gpu.barrier()  # TileRT's barrier closing the merge (its LDS is reused by wo_a)


WOA_X_LDS = (
    0xB800  # per wave [token 6][32 entries] bf16 pairs, 0x90 B a token, 0x6c0 B a wave
)
WOA_RED_LDS = 0  # per wave [token][32 rows] f32 partials, 2048 B a wave
WOA_PART = (
    0x140000  # ProjoArgs+0x28 + this: [ob 32][token 8][head 8][32 rows] {f32, epoch}
)
WOB_IN = 0x100000  # ProjoArgs+0x28 + this: [token][512] {bf16x2, epoch}, the wo_b input


def _e8_f32(bits_shifted):
    return (Int32(bits_shifted) & 0x7F800000).bitcast(F32)


def _pin4(v):
    """An empty side-effecting asm on the 4 dwords of v: LLVM sinks pure conversions to their first use (past
    a poll loop); this keeps them computed where it stands."""
    r = llvm.InlineAsmOp(T.vec(4, T.i32), [as_ir_value(v.bitcast(Int32))], "; pin $0", "=v,0",
                         has_side_effects=True).result  # fmt: skip
    return fx.Vector(r).bitcast(fx.BFloat16)


def _cvt4(words, scale):
    """Two words of fp8 -> four words of bf16 pairs, x 2^scale (cvt_scalef32_pk_bf16_fp8)."""
    out = []
    for w in words:
        out += [_cvt_bf16_fp8(w, scale, False), _cvt_bf16_fp8(w, scale, True)]
    return fx.Vector.from_elements(out, Int32).bitcast(fx.BFloat16)


def _tag_ok(v, epoch):
    """A {payload, tag} pack's tag as the 64-bit {tag, 0} TileRT compares against {epoch, 0}."""
    return v[1] == epoch


# debug builds (TK_STAMP=1): per-stage s_memrealtime stamps (``_stamp``); production builds emit nothing
_STAMP = __import__("os").environ.get("TK_STAMP", "0") == "1"
_DLY = 100  # entry delay (memrealtime ticks) of the non-attention CTAs
_AFM6 = (
    __import__("os").environ.get("TK_VARIANT", "tail") == "afm6"
)  # mega_afm_kernel<6,0,..>
OOB = 0x1FFFFFF0  # an element index past any num_records: the load returns 0 with no memory traffic


def _ldm(addr, nbytes, word, ok, n=1, cm=0, dtype=None):
    """A buffer load of ``n`` elements at ``word`` when ``ok``, else 0 without touching memory."""
    v = bo.buffer_load(rsrc(addr, nbytes), ok.select(word, Int32(OOB)), vec_width=n,
                       dtype=T.i32 if dtype is None else dtype, cache_modifier=cm)  # fmt: skip
    return fx.Vector(v) if n > 1 else v


def woa_fetch(tid, s2, wo_a, wo_a_s, ok):
    """wo_a: this CTA's 2 x 16 B a lane (nt) and its 16-bit e8m0 pair, issued when ``ok``."""
    lane, wv = tid % 64, tid // 64
    h = (s2 // 8) % 8
    ob = (s2 // 64) * 8 + s2 % 8
    wbase = ((ob * 8 + h) << 14) + wv * 2048 + lane * 16
    wa = _ldm(wo_a, 1 << 22, wbase >> 2, ok, 4, CM_NT)
    wb = _ldm(wo_a, 1 << 22, (wbase + 1024) >> 2, ok, 4, CM_NT)
    sc = (
        Int32(_ldm(wo_a_s, 4096, ob * 64 + h * 8 + wv, ok, dtype=T.i16)).to(Int32)
        & 0xFFFF
    )
    return [wa, wb, sc]


def wob_fetch(tid, s2, wo_b, wo_b_s, ok):
    """wo_b: this CTA's 4 x 16 B a lane (nt) and the e8m0 dword, issued when ``ok``."""
    lane, wv = tid % 64, tid // 64
    wbase = (s2 << 15) + wv * 4096 + lane * 16
    ws = [
        _ldm(wo_b, 160 << 15, (wbase + 1024 * i) >> 2, ok, 4, CM_NT) for i in range(4)
    ]
    return ws + [Int32(_ldm(wo_b_s, 160 * 32, s2 * 8 + wv, ok))]


def _sel_fetch(c, a, b):
    """Per element c ? a : b of two fetch lists."""
    out = []
    for x, y in zip(a, b):
        if isinstance(x, fx.Vector):
            out.append(
                fx.Vector.from_elements([c.select(x[i], y[i]) for i in range(4)], Int32)
            )
        else:
            out.append(c.select(x, y))
    return out


@traced
def _stamp(sp, tid, s2, k):
    """Debug builds (TK_STAMP): CTA s2's s_memrealtime at stage k -> sp[s2*32 + k] (u64)."""
    if const_expr(_STAMP):
        if (tid == 0) & (sp != 0):
            t = memrealtime()
            bo.buffer_store(fx.Vector.from_elements([Int32(t), Int32(t >> 32)], Int32), rsrc(sp),
                            (s2 * 32 + k) * 2)  # fmt: skip


@traced
def proj_b1(tid, s2, pf, xll, epoch, lds, rows=6):
    """wo_a for output block ob = (s2/64)*8 + s2%8 (32 rows) over K head h = (s2/8)%8 (512 dims; wave w
    the 64 dims w*64..): fp8 weights dequantized at conversion (e8m0 per 32 K), MFMA against the merged
    attention output polled from the LL packs, 8 waves summed, {f32, epoch} partials to WOA_PART.
    """
    lane, wv = tid % 64, tid // 64
    h = (s2 // 8) % 8
    ob = (s2 // 64) * 8 + s2 % 8
    wa, wb, sc = pf[0], pf[1], pf[2]
    s0, s1 = _e8_f32(sc << 23), _e8_f32(sc << 15)
    a0, a1 = _cvt4([wa[0], wa[1]], s0), _cvt4([wb[0], wb[1]], s0)
    a2, a3 = _cvt4([wa[2], wa[3]], s1), _cvt4([wb[2], wb[3]], s1)
    eb = h * 256 + wv * 32
    sent = _ld(xll, ((rows - 1) * 2048 + eb + 31) * 2, 2, CM_DEV)
    while ~_tag_ok(sent, epoch):
        spin_pause()
        sent = _ld(xll, ((rows - 1) * 2048 + eb + 31) * 2, 2, CM_DEV)
    r0 = (tid >> 5) & 1
    col = tid & 31
    words = [(eb + (r0 + 2 * j) * 2048 + col) * 2 for j in range(3)]
    if const_expr(rows == 6):
        v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
        while ~(_tag_ok(v[0], epoch) & _tag_ok(v[1], epoch) & _tag_ok(v[2], epoch)):
            spin_pause()
            v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
    else:  # rows 5: the third row (r0 + 4) exists for r0 = 0 only (masked load, counted as ready)
        has3 = r0 == 0
        v = [_ld(xll, words[0], 2, CM_DEV), _ld(xll, words[1], 2, CM_DEV),
             _ldm(xll, 1 << 24, words[2], has3, 2, CM_DEV)]  # fmt: skip
        while ~(
            _tag_ok(v[0], epoch) & _tag_ok(v[1], epoch) & (_tag_ok(v[2], epoch) | ~has3)
        ):
            spin_pause()
            v = [_ld(xll, words[0], 2, CM_DEV), _ld(xll, words[1], 2, CM_DEV),
                 _ldm(xll, 1 << 24, words[2], has3, 2, CM_DEV)]  # fmt: skip
    for j in range_constexpr(3):
        _lds_st(lds, WOA_X_LDS + wv * 0x6C0 + (r0 + 2 * j) * 0x90 + col * 4, [v[j][0]])
    tok = lane % 16
    zero4 = fx.Vector.filled(4, 0, Int32)
    b0 = zero4
    b1 = zero4
    if tok < rows:
        b0 = _lds_ld(lds, WOA_X_LDS + wv * 0x6C0 + tok * 0x90 + (lane // 16) * 16, 4)
        b1 = _lds_ld(
            lds, WOA_X_LDS + wv * 0x6C0 + tok * 0x90 + (lane // 16) * 16 + 64, 4
        )
    z = fx.Vector.filled(4, 0.0, F32)
    m0 = mfma_bf16(a0, b0.bitcast(fx.BFloat16), z)
    m1 = mfma_bf16(a1, b0.bitcast(fx.BFloat16), z)
    r_lo = mfma_bf16(a2, b1.bitcast(fx.BFloat16), m0)
    r_hi = mfma_bf16(a3, b1.bitcast(fx.BFloat16), m1)
    if tok < rows:
        at = WOA_RED_LDS + wv * 2048 + tok * 64 + (lane // 16) * 16
        _lds_st(lds, at, [r_lo[i].bitcast(Int32) for i in range(4)])
        _lds_st(lds, at + 512, [r_hi[i].bitcast(Int32) for i in range(4)])
    gpu.barrier()
    if tid < rows * 32:
        tk, r = tid >> 5, tid & 31
        at = WOA_RED_LDS + (r >> 4) * 512 + tk * 64 + (r & 15) * 4
        acc = F32(0.0)
        for w in range_constexpr(8):
            acc = acc + _lds_ld(lds, at + w * 2048).bitcast(F32)
        word = (WOA_PART // 8 + ob * 2048 + tk * 256 + h * 32 + r) * 2
        bo.buffer_store(
            fx.Vector.from_elements([acc.bitcast(Int32), epoch], Int32), rsrc(xll), word
        )
    gpu.barrier()


def _fp8_byte(q):
    o = fx.Vector.from_elements([Int32(0)], Int32).bitcast(fx.Int16)
    r = rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        o.ir_value(),
        F32(q).ir_value(),
        F32(0.0).ir_value(),
        F32(1.0).ir_value(),
        False,
    )
    return fx.Vector(r).bitcast(Int32)[0]


@traced
def proj_b2(tid, s2, xll, epoch, rows=6):
    """CTAs with h = 0: the 8 heads' wo_a partials of block ob summed, bf16, E4M3 round trip with the
    32-row block's ue8m0 scale, bf16 (truncated) pairs as LL packs {bf16x2, epoch} (sc1) at WOB_IN.
    """
    if tid < rows * 32:
        lane = tid % 64
        ob = (s2 // 64) * 8 + s2 % 8
        tk, r = tid >> 5, tid & 31
        words = [
            (WOA_PART // 8 + ob * 2048 + tk * 256 + hh * 32 + r) * 2 for hh in range(8)
        ]
        v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
        ok = _tag_ok(v[0], epoch)
        for hh in range_constexpr(1, 8):
            ok = ok & _tag_ok(v[hh], epoch)
        while ~ok:
            spin_pause()
            v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
            ok = _tag_ok(v[0], epoch)
            for hh in range_constexpr(1, 8):
                ok = ok & _tag_ok(v[hh], epoch)
        acc = F32(0.0)
        for hh in range_constexpr(8):
            acc = acc + v[hh][0].bitcast(F32)
        y = (bf_rne_hi(acc) & Int32(-65536)).bitcast(F32)
        m = abs(y)
        m = _max32(m, lane)
        m = fx.max(m, f32c(0x38D1B717))
        sv = (m * f32c(0x3B124925)).bitcast(Int32)
        sbits = (sv & 0x7F800000) + ((sv & 0x7FFFFF) == 0).select(
            Int32(0), Int32(0x800000)
        )
        scale = sbits.bitcast(F32)
        q = (F32(1.0) / scale) * y
        byte = _fp8_byte(q)
        v2 = fx.Vector.make_type(2, F32)
        dq = fx.Vector(
            rocdl.cvt_scalef32_pk_f32_fp8(
                res=v2,
                src=as_ir_value(byte),
                scale=as_ir_value(scale),
                src_lo_hi_sel=False,
            )
        )[0]
        bf = (dq.bitcast(Int32) >> 16) & 0xFFFF
        nb = Int32(_dppf(bf.bitcast(F32), QX1).bitcast(Int32))
        if (tid & 1) == 0:
            word = (WOB_IN // 8 + tk * 512 + ((ob * 32 + r) >> 1)) * 2
            bo.buffer_store(
                fx.Vector.from_elements([bf | (nb << 16), epoch], Int32),
                rsrc(xll),
                word,
                cache_modifier=CM_DEV,
            )


WOB_X_LDS = 0x5200  # per wave [token 6][128 K] bf16, 0x110 B a token, 0xcc0 B a wave
WOB_OUT_LDS = 0x4000  # [token 6][32 cols] bf16, 64 B a token: this CTA's wo_b partial
AR_IN_LDS = 0x4400  # [half 2][slot 7][3 tokens][32 cols] bf16: the peers' partials
AR_OUT_LDS = (
    0x15480  # [token 6][32 cols] f32 (bf16 values): the all-reduced o_proj, D's input
)
AR_SLOT, AR_PAR = 0xA0000, 0x50000  # sym:projo: a sender's slot, its epoch-parity half


@traced
def proj_c1(tid, s2, pf, xll, epoch, lds, rows=6):
    """CTAs s2 < 160: wo_b for hidden columns 32*s2.. (wave w: K = w*128..): fp8 weights in TileRT's
    packed order (32 KB a CTA, nt) dequantized at conversion (e8m0 a 32-K step), MFMA against the
    o_lora LL packs polled from WOB_IN, 8 waves summed (0 + w0 + .. + w7), bf16 RNE to WOB_OUT_LDS.
    """
    lane, wv = tid % 64, tid // 64
    # v29: no barrier here (B1 ends in one, B2 has no LDS). LLVM sinks these conversions past the poll;
    # pinning them before it (_pin4, v30) was neutral here and cost 0.33% in B1 (v31)
    w0, w1, w2, w3, sc = pf[0], pf[1], pf[2], pf[3], pf[4]
    k0, k1 = _e8_f32(sc << 23), _e8_f32(sc << 15)
    k2, k3 = _e8_f32(sc << 7), _e8_f32(sc >> 1)
    q00, q10 = _cvt4([w0[0], w0[1]], k0), _cvt4([w2[0], w2[1]], k0)
    q01, q11 = _cvt4([w0[2], w0[3]], k1), _cvt4([w2[2], w2[3]], k1)
    q02, q12 = _cvt4([w1[0], w1[1]], k2), _cvt4([w3[0], w3[1]], k2)
    q03, q13 = _cvt4([w1[2], w1[3]], k3), _cvt4([w3[2], w3[3]], k3)
    eb = WOB_IN // 4 + wv * 128
    sw = eb + (rows - 1) * 1024 + 63 * 2
    sent = _ld(xll, sw, 2, CM_DEV)
    while ~_tag_ok(sent, epoch):
        spin_pause()
        sent = _ld(xll, sw, 2, CM_DEV)
    words = [eb + t * 1024 + lane * 2 for t in range(rows)]
    v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
    ok = _tag_ok(v[0], epoch)
    for t in range_constexpr(1, rows):
        ok = ok & _tag_ok(v[t], epoch)
    while ~ok:
        spin_pause()
        v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
        ok = _tag_ok(v[0], epoch)
        for t in range_constexpr(1, rows):
            ok = ok & _tag_ok(v[t], epoch)
    xw = WOB_X_LDS + wv * 0xCC0
    for t in range_constexpr(rows):
        _lds_st(lds, xw + t * 0x110 + lane * 4, [v[t][0]])
    tok = lane % 16
    zero4 = fx.Vector.filled(4, 0, Int32)
    b0 = zero4
    b1 = zero4
    b2 = zero4
    b3 = zero4
    if tok < rows:
        xb = xw + tok * 0x110 + (lane // 16) * 16
        b0 = _lds_ld(lds, xb, 4)
        b1 = _lds_ld(lds, xb + 64, 4)
        b2 = _lds_ld(lds, xb + 128, 4)
        b3 = _lds_ld(lds, xb + 192, 4)
    z = fx.Vector.filled(4, 0.0, F32)
    bf = fx.BFloat16
    c0 = mfma_bf16(q00, b0.bitcast(bf), z)
    c1 = mfma_bf16(q10, b0.bitcast(bf), z)
    c0 = mfma_bf16(q01, b1.bitcast(bf), c0)
    c1 = mfma_bf16(q11, b1.bitcast(bf), c1)
    c0 = mfma_bf16(q02, b2.bitcast(bf), c0)
    c1 = mfma_bf16(q12, b2.bitcast(bf), c1)
    c0 = mfma_bf16(q03, b3.bitcast(bf), c0)
    c1 = mfma_bf16(q13, b3.bitcast(bf), c1)
    if tok < rows:
        at = WOA_RED_LDS + wv * 2048 + tok * 64 + (lane // 16) * 16
        _lds_st(lds, at, [c0[i].bitcast(Int32) for i in range(4)])
        _lds_st(lds, at + 512, [c1[i].bitcast(Int32) for i in range(4)])
    gpu.barrier()
    if tid < rows * 32:
        tk, r = tid >> 5, tid & 31
        at = WOA_RED_LDS + (r >> 4) * 512 + tk * 64 + (r & 15) * 4
        acc = F32(0.0)
        for w in range_constexpr(8):
            acc = acc + _lds_ld(lds, at + w * 2048).bitcast(F32)
        hi = (bf_rne_hi(acc) >> 16) & 0xFFFF
        nb = Int32(_dppf(hi.bitcast(F32), QX1).bitcast(Int32))
        if (r & 1) == 0:
            _lds_st(lds, WOB_OUT_LDS + tk * 64 + r * 2, [hi | (nb << 16)])
    gpu.barrier()


@traced
def proj_c2(tid, s2, tab, me, npes, epoch, lds, ptrs=None):
    """CTAs s2 < 160: one-shot TP all-reduce of the 6 x 32 wo_b partials. Push (waves 0..6, wave p
    -> the p-th peer skipping self, lanes 0..31 tokens 0..2 / 32..63 tokens 3..5): LL64 rows of
    7 x 8 B data + an 8 B {epoch, epoch} flag (sc0 sc1) into the peer's sym:projo slot of this rank
    (epoch-parity halves). Receive (TileRT's half-wave hw -> (hw / 7, hw % 7)) into AR_IN_LDS, then
    every element summed in rank order 0..7 in fp32, bf16 RNE, to AR_OUT_LDS as f32.
    """
    lane, wv = tid % 64, tid // 64
    par = (epoch & 1) * AR_PAR
    rq, j = (lane & 31) >> 3, lane & 7
    e = rq * 28 + j * 4
    data = (j != 7) & (e < 93)
    if npes >= 2:
        if wv < 7:
            dest = wv + (wv >= me).select(1, 0)
            if const_expr(ptrs is None):
                base = load_ptr64(tab, dest)
            else:
                base = ptrs[0]
            slot = me - (me > dest).select(1, 0)
            h = lane >> 5
            val = fx.Vector.from_elements([epoch, epoch], Int32)
            if data:
                val = _lds_ld(lds, WOB_OUT_LDS + h * 192 + e * 2, 2)
            off = slot * AR_SLOT + par + (s2 * 6 + h * 3) * 128 + rq * 64 + j * 8
            bo.buffer_store(val, rsrc(base), off >> 2, cache_modifier=CM_SYS)
        gpu.barrier()
        if wv < 7:
            hw = wv * 2 + (lane >> 5)
            h = hw // 7
            p = hw - h * 7
            if const_expr(ptrs is None):
                own = load_ptr64(tab, me)
            else:
                own = ptrs[1]
            row = p * AR_SLOT + par + (s2 * 6 + h * 3) * 128 + rq * 64
            fl = _ld(own, (row + 56) >> 2, 2, CM_SYS)
            while ~((fl[0] == epoch) & (fl[1] == epoch)):
                spin_pause()
                fl = _ld(own, (row + 56) >> 2, 2, CM_SYS)
            if data:
                d = _ld(own, (row + j * 8) >> 2, 2, CM_SYS)
                _lds_st(lds, AR_IN_LDS + hw * 192 + e * 2, [d[0], d[1]])
        gpu.barrier()
    if tid < 192:
        tk, col = tid >> 5, tid & 31
        loc = WOB_OUT_LDS + tk * 64 + col * 2
        out = _lds_u16(lds, loc) << 16
        if npes >= 2:
            h = (tk >= 3).select(1, 0)
            tr = tk - h * 3
            acc = F32(0.0)
            for r in range_constexpr(8):
                slot = r - (me < r).select(1, 0)
                rem = AR_IN_LDS + ((h * 7 + slot) * 3 + tr) * 64 + col * 2
                at = (me == r).select(loc, rem)
                acc = acc + (_lds_u16(lds, at) << 16).bitcast(F32)
            out = bf_rne_hi(acc) & Int32(-65536)
        _lds_st(lds, AR_OUT_LDS + tk * 128 + col * 4, [out])


MHC_X_LDS = (
    0x15880  # [token 6][hc 4][32 cols] f32 (bf16 values): the new residual streams
)
MHC_FN_PART = (
    0x2A800  # MhcF+0x50 + this: [row][s2 160][mix 24] {f32, epoch}, fn . X_new partials
)
MHC_SQ_PART = (
    0x66800  # MhcF+0x50 + this: [row][s2 160] {f32, epoch}, sum X_new^2 partials
)


def _ld_u16_f32(ptr, i):
    """The bf16 at element ``i`` of a buffer, as f32."""
    v = bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i16)
    return ((Int32(v).to(Int32) & 0xFFFF) << 16).bitcast(F32)


def mhc_d_fetch(tid, s2, x2, hca, fn):
    """D's inputs (residual streams, hcA, fn slices), issued before C1's poll as TileRT does."""
    row, col = fx.min(tid >> 5, Int32(5)), tid & 31
    okr = tid < 192
    g = s2 * 32 + col
    xs = [((Int32(_ldm(x2, 6 * 4 * 5120 * 2, row * 0x5000 + j * 0x1400 + g, okr, dtype=T.i16)).to(Int32)
            & 0xFFFF) << 16).bitcast(F32) for j in range(4)]  # fmt: skip
    c = []
    for q in range(6):
        cv = _ldm(hca, 6 * 24 * 4, row * 24 + q * 4, okr, 4)
        c += [cv[i].bitcast(F32) for i in range(4)]
    okf = (tid >= 256) & (tid < 352)
    m, hc = fx.max(tid - 256, Int32(0)) >> 2, tid & 3
    fb0 = m * 0x5000 + hc * 0x1400 + s2 * 32
    fv = []
    for q in range(8):
        v = _ldm(fn, 24 * 4 * 5120 * 4, fb0 + q * 4, okf, 4)
        fv += [v[i].bitcast(F32) for i in range(4)]
    return [xs, c, fv]


@traced
def mhc_d(tid, s2, x2, xres, hca, fn, xll, mep, lds, dpf=None):
    """CTAs s2 < 160, hidden columns g = 32*s2 + col: the mHC post / comb update of the 4 residual
    streams with the all-reduced o_proj y (X_new_j = c[8+j] x0 + c[12+j] x1 + c[16+j] x2 + c[20+j] x3
    + c[4+j] y, bf16) to X; the FFN input pre . X_new (bf16) as LL packs {bf16x2, moe epoch} at
    MhcF+0x50; per row the sum of X_new^2 and the 24 mixes' fn . X_new partials (LL, for [G']).
    """
    gpu.barrier()
    if tid < 192:
        row, col = tid >> 5, tid & 31
        g = s2 * 32 + col
        if const_expr(dpf is not None):
            xs, c = dpf[0], dpf[1]
        else:
            xs = [_ld_u16_f32(x2, row * 0x5000 + j * 0x1400 + g) for j in range(4)]
            c = []
            for q in range_constexpr(6):
                cv = _ld(hca, row * 24 + q * 4, 4)
                c += [cv[i].bitcast(F32) for i in range(4)]
        y = _lds_ld(lds, AR_OUT_LDS + row * 128 + col * 4).bitcast(F32)
        nb, nf = [], []
        for j in range_constexpr(4):
            a = c[8 + j] * xs[0] + c[12 + j] * xs[1]
            a = a + c[16 + j] * xs[2]
            a = a + c[20 + j] * xs[3]
            a = a + c[4 + j] * y
            r = bf_rne_hi(a)
            nb.append((r >> 16) & 0xFFFF)
            nf.append((r & Int32(-65536)).bitcast(F32))
        f = c[0] * nf[0] + c[1] * nf[1]
        f = f + c[2] * nf[2]
        f = f + c[3] * nf[3]
        fb = (bf_rne_hi(f) >> 16) & 0xFFFF
        for j in range_constexpr(4):
            _lds_st(
                lds, MHC_X_LDS + row * 512 + j * 128 + col * 4, [nf[j].bitcast(Int32)]
            )
        pf = Int32(_dppf(fb.bitcast(F32), QX1).bitcast(Int32))
        pn = [Int32(_dppf(nb[j].bitcast(F32), QX1).bitcast(Int32)) for j in range(4)]
        if (col & 1) == 0:
            bo.buffer_store(
                fx.Vector.from_elements([fb | (pf << 16), mep], Int32),
                rsrc(xll),
                row * 0x1400 + (g >> 1) * 2,
                cache_modifier=CM_DEV,
            )
            for j in range_constexpr(4):
                bo.buffer_store(
                    nb[j] | (pn[j] << 16),
                    rsrc(xres),
                    (row * 0x5000 + j * 0x1400 + g) >> 1,
                    cache_modifier=CM_DEV,
                )
        if col < 4:
            sb = MHC_X_LDS + row * 512 + col * 32
            xq = []
            for j in range_constexpr(4):
                v0 = _lds_ld(lds, sb + j * 128, 4)
                v1 = _lds_ld(lds, sb + j * 128 + 16, 4)
                xq += [v0[i].bitcast(F32) for i in range(4)]
                xq += [v1[i].bitcast(F32) for i in range(4)]
            sq = xq[0] * xq[0]
            for e in range_constexpr(1, 32):
                sq = sq + xq[e] * xq[e]
            s01 = sq + _dppf(sq, QX1)
            s = s01 + _dppf(s01, QX2)
            if col == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([s.bitcast(Int32), mep], Int32),
                    rsrc(xll),
                    (MHC_SQ_PART + row * 1280 + s2 * 8) >> 2,
                    cache_modifier=CM_DEV,
                )
    gpu.barrier()
    if (tid >= 256) & (tid < 352):
        m, hc = (tid - 256) >> 2, tid & 3
        fb0 = m * 0x5000 + hc * 0x1400 + s2 * 32
        if const_expr(dpf is not None):
            fv = dpf[2]
        else:
            fv = []
            for q in range_constexpr(8):
                v = _ld(fn, fb0 + q * 4, 4)
                fv += [v[i].bitcast(F32) for i in range(4)]
        for r in range_constexpr(6):
            xb = MHC_X_LDS + r * 512 + hc * 128
            xv = []
            for q in range_constexpr(8):
                v = _lds_ld(lds, xb + q * 16, 4)
                xv += [v[i].bitcast(F32) for i in range(4)]
            acc = fmath.fma(fv[0], xv[0], F32(0.0))
            for k in range_constexpr(1, 20):
                acc = fmath.fma(fv[k], xv[k], acc)
            for k in range_constexpr(20, 32):
                acc = acc + fv[k] * xv[k]
            s01 = acc + _dppf(acc, QX1)
            s = s01 + _dppf(s01, QX2)
            if hc == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([s.bitcast(Int32), mep], Int32),
                    rsrc(xll),
                    (MHC_FN_PART + r * 0x7800 + s2 * 192 + m * 8) >> 2,
                    cache_modifier=CM_DEV,
                )


RT_PART_LDS = (
    0x14DC0  # per wave [24 experts] f32 + sum x^2 at +96, 216 B a wave: router partials
)
X_RAW_LDS = 0  # the row's x as 5120 bf16 (dword t: entry t of the LL row), kept for the fp8 quant
ROW_SHR = (0x111, 0x112, 0x114, 0x118)


def _dppz(v, ctrl):
    """f32 DPP move, invalid source lanes read 0 (bound_ctrl), all rows written."""
    w = update_dpp_i32(Int32(0), F32(v).bitcast(Int32), ctrl, 0xF, 0xF, True)
    return Int32(w).bitcast(F32)


def _scan_sum(v):
    """TileRT's wave sum: row_shr 1/2/4/8 inclusive scans, then row_bcast 15 and 31; lane 63 ends
    with (row3 + row2) + (row1 + row0)."""
    for c in ROW_SHR:
        v = v + _dppz(v, c)
    v = v + _dppz(v, 0x142)
    return v + _dppz(v, 0x143)


def _rsq_scaled(ms):
    tiny = ms < f32c(0x00800000)
    r = hw_rsq(tiny.select(ms * f32c(0x4B800000), ms))
    return tiny.select(r * f32c(0x45800000), r)


@traced
def moe_router(tid, s2, gate_w, xll, gamma, gflag, ws_moe, mep, lds, g16=False):
    """CTAs s2 >= 160: row (s2-160)/16, shard sh = (s2%8)*2 + bit 3 of s2 owning experts sh*24..+24.
    Thread t holds x / gamma / gate_w elements 2t, 2t+1 (+1024 j, j < 5); per expert an fma chain
    over gamma*x in that order; wave sums by row_shr scans, 8 waves in order through LDS; logit =
    dot * rsq(sum x^2 / 5120 + 1e-20); score = sqrt(softplus(logit)) as {f32, moe epoch} (sc1) at
    ws_moe[row][384]. The row's x stays in LDS (X_RAW_LDS) for the fp8 quantization."""
    lane, wv = tid % 64, tid // 64
    row = (s2 - 160) >> 4
    sh = (s2 % 8) * 2 + ((s2 >> 3) & 1)
    # v35: the quantization's gamma (moe_prep) issued here, ahead of the router's own loads: the router
    # CTAs hold their x in LDS, so their prep otherwise waits one memory latency for it
    gq = _gamma16(tid, gamma, gflag) if const_expr(g16) else None
    w = [
        [_ld(gate_w, (sh * 24 + e) * 2560 + tid + 512 * j) for j in range(5)]
        for e in range(24)
    ]
    gv = fx.Vector.filled(10, 0.0, F32)
    if gflag != 0:
        gw = [_ld(gamma, tid + 512 * j) for j in range(5)]
        gl = []
        for j in range_constexpr(5):
            gl += [(gw[j] << 16).bitcast(F32), (gw[j] & Int32(-65536)).bitcast(F32)]
        gv = fx.Vector.from_elements(gl, F32)
    else:
        gl = []
        for j in range_constexpr(5):
            gl += [_ld(gamma, (tid + 512 * j) * 2).bitcast(F32)]
            gl += [_ld(gamma, (tid + 512 * j) * 2 + 1).bitcast(F32)]
        gv = fx.Vector.from_elements(gl, F32)
    sw = row * 0x1400 + 2559 * 2
    sent = _ld(xll, sw, 2, CM_DEV)
    while ~_tag_ok(sent, mep):
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload (v10 hang)
        rocdl.s_sleep(2)
        sent = _ld(xll, sw, 2, CM_DEV)
    words = [row * 0x1400 + (tid + 512 * j) * 2 for j in range(5)]
    v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
    ok = _tag_ok(v[0], mep)
    for j in range_constexpr(1, 5):
        ok = ok & _tag_ok(v[j], mep)
    while ~ok:
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload (v10 hang)
        rocdl.s_sleep(1)
        v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
        ok = _tag_ok(v[0], mep)
        for j in range_constexpr(1, 5):
            ok = ok & _tag_ok(v[j], mep)
    for j in range_constexpr(5):
        _lds_st(lds, X_RAW_LDS + (tid + 512 * j) * 4, [v[j][0]])
    xs = []
    for j in range_constexpr(5):
        xs += [(v[j][0] << 16).bitcast(F32), (v[j][0] & Int32(-65536)).bitcast(F32)]
    sq = fmath.fma(xs[0], xs[0], F32(0.0))
    for k in range_constexpr(1, 10):
        sq = fmath.fma(xs[k], xs[k], sq)
    gx = [gv[k] * xs[k] for k in range(10)]
    accs = []
    for e in range_constexpr(24):
        a = F32(0.0)
        for j in range_constexpr(5):
            lo = (w[e][j] << 16).bitcast(F32)
            hi = (w[e][j] & Int32(-65536)).bitcast(F32)
            a = fmath.fma(lo, gx[2 * j], a)
            a = fmath.fma(hi, gx[2 * j + 1], a)
        accs.append(_scan_sum(a))
    sq = _scan_sum(sq)
    if lane == 63:
        base = RT_PART_LDS + wv * 216
        for p in range_constexpr(12):
            _lds_st(
                lds,
                base + p * 8,
                [accs[2 * p].bitcast(Int32), accs[2 * p + 1].bitcast(Int32)],
            )
        _lds_st(lds, base + 96, [sq.bitcast(Int32)])
    gpu.barrier()
    if tid < 24:
        dot = _lds_ld(lds, RT_PART_LDS + tid * 4).bitcast(F32)
        ss = _lds_ld(lds, RT_PART_LDS + 96).bitcast(F32)
        for k in range_constexpr(1, 8):
            dot = dot + _lds_ld(lds, RT_PART_LDS + k * 216 + tid * 4).bitcast(F32)
            ss = ss + _lds_ld(lds, RT_PART_LDS + k * 216 + 96).bitcast(F32)
        rs = _rsq_scaled(fmath.fma(ss, f32c(0x394CCCCD), f32c(0x1E3CE508)))
        logit = dot * rs
        sp = (f32c(0x41A00000) < logit).select(
            logit, log1p_ocml(exp_ocml(logit, False))
        )
        score = fmath.sqrt(sp)
        bo.buffer_store(
            fx.Vector.from_elements([score.bitcast(Int32), mep], Int32),
            rsrc(ws_moe),
            (row * 384 + sh * 24 + tid) * 2,
            cache_modifier=CM_DEV,
        )
    return gq


QX_PART_LDS = 0x14D00  # [row][wave] f32 sum x^2 partials, 32 B a row
QX_SCALE_LDS = (
    30720  # [row][160] ue8m0 scale bytes of the fp8 x (row r's fp8 at r*5120)
)
SG_RED_LDS = (
    31680  # waves 1..7: [w][c 4][lane 64] f32 partials of the shared-expert MFMA
)
ACT_OFF = 0x6000  # ws_moe + this: [row][2016] {bf16x2, epoch} act LL, 0x1f80 B a row (shared first)


@traced
def _gamma16(tid, gamma, gflag):
    """gamma (bf16 when the flag is set, else f32) of elements 8t..8t+7 and 4096+8t..+7."""
    # the second half (elements 4096 + 8t) exists for t < 128 only: masked, a load past the 5120
    # elements faults when the tensor ends a pool page (seen in the graph, not in the replays)
    gv = fx.Vector.filled(16, 0.0, F32)
    lo = tid < 128
    if gflag != 0:
        gl = []
        for half in range_constexpr(2):
            for i in range_constexpr(4):
                if const_expr(half == 0):
                    d = _ld(gamma, tid * 4 + i)
                else:
                    d = Int32(_ldm(gamma, 10240, (tid + 512) * 4 + i, lo))
                gl += [(d << 16).bitcast(F32), (d & Int32(-65536)).bitcast(F32)]
        gv = fx.Vector.from_elements(gl, F32)
    else:
        gl = []
        for half in range_constexpr(2):
            for i in range_constexpr(8):
                if const_expr(half == 0):
                    gl += [_ld(gamma, tid * 8 + i).bitcast(F32)]
                else:
                    gl += [
                        Int32(_ldm(gamma, 20480, (tid + 512) * 8 + i, lo)).bitcast(F32)
                    ]
        gv = fx.Vector.from_elements(gl, F32)
    return gv


def _fp8x2(a, b):
    o = fx.Vector.from_elements([Int32(0)], Int32).bitcast(fx.Int16)
    r = rocdl.cvt_scalef32_pk_fp8_f32(
        T.vec(2, T.i16),
        o.ir_value(),
        F32(a).ir_value(),
        F32(b).ir_value(),
        F32(1.0).ir_value(),
        False,
    )
    return fx.Vector(r).bitcast(Int32)[0] & 0xFFFF


def _bfr(x):
    return (bf_rne_hi(x) & Int32(-65536)).bitcast(F32)


@traced
def _quant8(lane, xs, gs, rs, lds, at, sidx, gather):
    """8 elements: y = bf16(g * (rs * x)); the 32-element block's amax (quad), ue8m0 scale =
    pow2ceil(max(amax, 1e-4) / 448); E4M3(y * 2^-e) to LDS ``at``; the scale byte at sidx (lanes
    with ``gather`` write the 16-lane group's 4 bytes as a dword)."""
    y = [_bfr(gs[k] * (rs * xs[k])) for k in range(8)]
    m = abs(y[0])
    for k in range_constexpr(1, 8):
        m = fx.max(m, abs(y[k]))
    m = fx.max(m, _dppf(m, QX1))
    m = fx.max(fx.max(m, _dppf(m, QX2)), f32c(0x38D1B717))
    sv = (m * f32c(0x3B124925)).bitcast(Int32)
    sbits = (sv & 0x7F800000) + ((sv & 0x7FFFFF) == 0).select(Int32(0), Int32(0x800000))
    inv = (Int32(0x7F000000) - (sbits & 0x7F800000)).bitcast(F32)
    q = [y[k] * inv for k in range(8)]
    d0 = _fp8x2(q[0], q[1]) | (_fp8x2(q[2], q[3]) << 16)
    d1 = _fp8x2(q[4], q[5]) | (_fp8x2(q[6], q[7]) << 16)
    _lds_st(lds, at, [d0, d1])
    sb = (sbits >> 23) & 0xFF
    sbf = sb.bitcast(F32)
    s1, s2_, s3 = _gather_sc(sbf, lane)
    if gather:
        _lds_st(lds, QX_SCALE_LDS + sidx, [sb | (s1 << 8) | (s2_ << 16) | (s3 << 24)])


@traced
def quant_rows(tid, rows, gv, lds):
    """The MoE input x of ``rows`` (each 8 dwords of bf16 pairs a thread: elements 8t..8t+7, then
    4096+8t..+7, zero for t >= 128): per row the RMSNorm sum over the CTA (fma chain, row_shr wave
    scans, waves in order), y = bf16(gamma * (rsq * x)), 1x32 E4M3 + ue8m0 to LDS (row r at
    r*5120, scales at QX_SCALE_LDS + r*160)."""
    lane, wv = tid % 64, tid // 64
    xss = []
    for r in range_constexpr(len(rows)):
        xs = []
        for i in range_constexpr(8):
            xs += [
                (rows[r][i] << 16).bitcast(F32),
                (rows[r][i] & Int32(-65536)).bitcast(F32),
            ]
        xss.append(xs)
        sq = fmath.fma(xs[0], xs[0], F32(0.0))
        for k in range_constexpr(1, 16):
            sq = fmath.fma(xs[k], xs[k], sq)
        sq = _scan_sum(sq)
        if lane == 63:
            _lds_st(lds, QX_PART_LDS + r * 32 + wv * 4, [sq.bitcast(Int32)])
    gpu.barrier()
    for r in range_constexpr(len(rows)):
        pv = _lds_vec(
            lds, QX_PART_LDS + r * 32, 8
        )  # 2 ds_read_b128 (the 8 waves' partials)
        ss = pv[0]
        for k in range_constexpr(1, 8):
            ss = ss + pv[k]
        rs = _rsq_scaled(fmath.fma(ss, f32c(0x394CCCCD), f32c(0x1E3CE508)))
        xs = xss[r]
        _quant8(lane, xs[0:8], [gv[k] for k in range(8)], rs, lds, r * 5120 + tid * 8,
                r * 160 + (tid >> 2), (tid & 15) == 0)  # fmt: skip
        if tid < 128:
            _quant8(lane, xs[8:16], [gv[8 + k] for k in range(8)], rs, lds,
                    r * 5120 + 4096 + tid * 8, r * 160 + 128 + (tid >> 2), (tid & 15) == 0)  # fmt: skip
    gpu.barrier()


def _rows_ok(v, mep, lo):
    """Every polled entry tagged; a second-half entry (n % 8 >= 4) counts only for t < 128."""
    ok = _tag_ok(v[0], mep)
    for n in range(1, len(v)):
        ok = ok & ((_tag_ok(v[n], mep) | lo) if n % 8 >= 4 else _tag_ok(v[n], mep))
    return ok


@traced
def moe_prep(tid, s2, xll, gamma, gflag, sug, sug_s, ws_moe, mep, lds, gpre=None):
    """Every CTA's MoE input: routers quantize the row they already hold in LDS, routed CTAs
    (s2 < 144, row s2/24) poll and quantize theirs, shared-expert CTAs (144..159) all 6 rows and
    then run their shared-expert tiles."""
    if tid == 0:
        _lds_st(lds, TOPK_FLAG_LDS, [Int32(0)])
    gv = gpre if const_expr(gpre is not None) else _gamma16(tid, gamma, gflag)
    lo = tid >= 128
    if s2 >= 160:
        rw = [_lds_ld(lds, X_RAW_LDS + tid * 16 + 4 * i) for i in range(4)]
        rw += [
            lo.select(
                Int32(0), _lds_ld(lds, X_RAW_LDS + 8192 + (tid & 127) * 16 + 4 * i)
            )
            for i in range(4)
        ]
        quant_rows(tid, [rw], gv, lds)
    if s2 < 144:
        gvr = _gamma16(tid, gamma, gflag) if const_expr(gpre is not None) else gv
        rows = poll_x_rows(tid, xll, s2 // 24, 1, mep)
        quant_rows(tid, rows, gvr, lds)
    if (s2 >= 144) & (s2 < 160):
        gvs = _gamma16(tid, gamma, gflag) if const_expr(gpre is not None) else gv
        g0 = g_fetch(tid, s2, sug, sug_s, 0)
        rows6 = poll_x_rows(tid, xll, 0, 6, mep)
        quant_rows(tid, rows6, gvs, lds)
        shared_g(tid, s2, sug, sug_s, ws_moe, mep, lds, g0)


@traced
def poll_x_rows(tid, xll, row0, nrows, mep):
    """The rows row0..row0+nrows-1 of the MoE input LL (MhcF+0x50): sentinel = the last row's entry
    2559 (s_sleep 8), then each lane's entries 4t..4t+3 and (t < 128) 2048+4t..+3 (s_sleep 1).
    """
    sw = (row0 + nrows - 1) * 0x1400 + 2559 * 2
    sent = _ld(xll, sw, 2, CM_DEV)
    while ~_tag_ok(sent, mep):
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload (v10 hang)
        rocdl.s_sleep(8)
        sent = _ld(xll, sw, 2, CM_DEV)
    lo = tid >= 128  # no second half
    ents = [tid * 4 + i for i in range(4)] + [
        2048 + (tid & 127) * 4 + i for i in range(4)
    ]
    words = [(row0 + r) * 0x1400 + e * 2 for r in range(nrows) for e in ents]
    v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
    ok = _rows_ok(v, mep, lo)
    while ~ok:
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload (v10 hang)
        rocdl.s_sleep(1)
        v = [_ld(xll, wd, 2, CM_DEV) for wd in words]
        ok = _rows_ok(v, mep, lo)
    rows = []
    for r in range_constexpr(nrows):
        firsts = [v[r * 8 + i][0] for i in range(4)]
        seconds = [lo.select(Int32(0), v[r * 8 + 4 + i][0]) for i in range(4)]
        rows.append(firsts + seconds)
    return rows


def _u8_lds(lds, byte):
    return (_lds_ld(lds, byte & ~3) >> ((byte & 3) * 8)) & 0xFF


def g_fetch(tid, s2, sug, sug_s, it):
    """Shared-expert tile t0 + 16 it's weights (5 x 2 x 16 B a lane, nt) and 5 scale bytes, masked off
    past tile 35: issued ahead of use (tile 0 before the x poll, tile i+1 during tile i).
    """
    lane, wv = tid % 64, tid // 64
    q = lane >> 4
    tile = s2 - 144 + 16 * it
    ok = tile < 36
    wb = fx.min(tile, Int32(35)) * 0x14000 + wv * 0x2800 + lane * 16
    sb0 = (
        ((lane & 8) != 0).select(Int32(1440), Int32(0))
        + (fx.min(tile, Int32(35)) >> 2) * 160
        + wv * 20
        + q
    )
    out = []
    for j in range(5):
        a0 = _ldm(sug, 36 * 0x14000, (wb + j * 2048) >> 2, ok, 4, CM_NT)
        a1 = _ldm(sug, 36 * 0x14000, (wb + j * 2048 + 1024) >> 2, ok, 4, CM_NT)
        sa = Int32(_ldm(sug_s, 2880, sb0 + 4 * j, ok, dtype=T.i8)).to(Int32) & 0xFF
        out.append(
            (
                fx.Vector.from_elements(
                    [a0[d] for d in range(4)] + [a1[d] for d in range(4)], Int32
                ),
                sa,
            )
        )
    return out


@traced
def shared_g(tid, s2, sug, sug_s, ws_moe, mep, lds, pre=None):
    """CTAs 144..159: shared-expert tiles t0 = s2-144, +16, +32 (< 36) of 8 gate + 8 up rows each:
    wave w the K range 640w..+640 in 5 scaled 16x16x128 fp8 MFMAs (weights nt, 32x32-block scales),
    the 8 waves summed through LDS in order; SwiGLU (gate bf16 min 10, up bf16 clamp +-10,
    g / (1 + exp(-g)) * u, bf16) as LL {bf16x2, epoch} pairs (sc1) into the act rows."""
    lane, wv = tid % 64, tid // 64
    h, q = lane % 16, lane >> 4
    xr = h % 6
    t0 = s2 - 144
    nxt = pre
    # v39: the x operand (B) and its scales do not depend on the tile: read from LDS once and kept in
    # registers for the three tiles, as TileRT does (it re-read them before every MFMA of every tile)
    bvs, sbs = [], []
    for j in range_constexpr(5):
        xo = xr * 5120 + (wv * 5 + j) * 128 + q * 16
        b0 = _lds_ld(lds, xo, 4)
        b1 = _lds_ld(lds, xo + 64, 4)
        sbs.append(_u8_lds(lds, QX_SCALE_LDS + xr * 160 + wv * 20 + 4 * j + q))
        bvs.append(
            fx.Vector.from_elements(
                [b0[d] for d in range(4)] + [b1[d] for d in range(4)], Int32
            )
        )
    for it in range_constexpr(3):
        tile = t0 + 16 * it
        cur = nxt if const_expr(nxt is not None) else g_fetch(tid, s2, sug, sug_s, it)
        if const_expr(it < 2):
            nxt = g_fetch(tid, s2, sug, sug_s, it + 1)
        if tile < 36:
            acc = fx.Vector.filled(4, 0.0, F32)
            for j in range_constexpr(5):
                av, sa = cur[j]
                acc = mfma_scaled(av, bvs[j], acc, sa, sbs[j])
            if wv != 0:
                for i in range_constexpr(4):
                    _lds_st(
                        lds,
                        SG_RED_LDS + wv * 1024 + i * 256 + lane * 4,
                        [acc[i].bitcast(Int32)],
                    )
            gpu.barrier()
            if wv == 0:
                c = []
                for i in range_constexpr(4):
                    s = acc[i]
                    for w in range_constexpr(1, 8):
                        s = s + _lds_ld(
                            lds, SG_RED_LDS + w * 1024 + i * 256 + lane * 4
                        ).bitcast(F32)
                    c.append(s)
                up = [_up32(c[i]) for i in range(4)]
                if (lane < 32) & (h < 6):
                    a = []
                    for i in range_constexpr(4):
                        g = fx.min(_bfr(c[i]), F32(10.0))
                        u = fx.max(fx.min(_bfr(up[i]), F32(10.0)), F32(-10.0))
                        e = exp_ocml(g, True)
                        a.append((bf_rne_hi((g / (e + F32(1.0))) * u) >> 16) & 0xFFFF)
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [a[0] | (a[1] << 16), mep, a[2] | (a[3] << 16), mep], Int32
                        ),
                        rsrc(ws_moe),
                        (ACT_OFF + h * 0x1F80 + tile * 32 + q * 16) >> 2,
                        cache_modifier=CM_DEV,
                    )
            gpu.barrier()


def _row16_sum(v):
    """Butterfly sum of each 16-lane row (quad, quad, half-mirror, mirror)."""
    for c in (QK_QP_1032, QK_QP_2301, QK_HALF_MIRROR, QK_MIRROR):
        v = v + qk_dpp(v, c)
    return v


def _all_tags(vs, mep, mask_off):
    ok = _tag_ok(vs[0], mep) | mask_off
    for v in vs[1:]:
        ok = ok & (_tag_ok(v, mep) | mask_off)
    return ok


@traced
def mhc_gp(tid, s2, xll, scale, base, hcf, ll_on, llp, mep):
    """Wave 7 of CTAs 148..153 (row = s2 - 148): the FFN mHC finalize from D's partials. Lanes < 48
    sum 80 CTAs' fn . X_new partials of mix lane % 24 in CTA order (two batches of 40 LL loads);
    sum of squares: 16-lane row butterflies of CTAs lane, lane + 64, lane + 128, the 10 row sums in
    order; rsq(ss / 20480 + 1e-20); pre = sigmoid + 1e-6, post = 2 sigmoid, comb = 20 sinkhorn
    rounds; plain stores to hcF[row][24] and, when @520 is set, LL {f32, epoch} at @568 + row*192.
    """
    lane = tid % 64
    row = s2 - 148
    grp = (lane >= 24).select(Int32(1), Int32(0))
    m = lane - grp * 24
    off = lane >= 48
    fb = (MHC_FN_PART + row * 0x7800 + m * 8) >> 2
    acc = F32(0.0)
    for b in range_constexpr(2):
        words = [fb + (grp * 80 + b * 40 + k) * 48 for k in range(40)]
        v = [_ld(xll, w, 2, CM_DEV) for w in words]
        ok = _all_tags(v, mep, off)
        while ~ok:
            spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload (v10 hang)
            rocdl.s_sleep(1)
            v = [_ld(xll, w, 2, CM_DEV) for w in words]
            ok = _all_tags(v, mep, off)
        for k in range_constexpr(40):
            if const_expr(b == 0 and k == 0):
                acc = v[0][0].bitcast(F32)
            else:
                acc = acc + v[k][0].bitcast(F32)
    part = off.select(F32(0.0), acc)
    sb = (MHC_SQ_PART + row * 1280) >> 2
    hi = lane >= 32
    sw = [sb + lane * 2, sb + (lane + 64) * 2, sb + (fx.min(lane, Int32(31)) + 128) * 2]
    sv = [_ld(xll, w, 2, CM_DEV) for w in sw]
    ok = _tag_ok(sv[0], mep) & _tag_ok(sv[1], mep) & (_tag_ok(sv[2], mep) | hi)
    while ~ok:
        spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload (v10 hang)
        rocdl.s_sleep(1)
        sv = [_ld(xll, w, 2, CM_DEV) for w in sw]
        ok = _tag_ok(sv[0], mep) & _tag_ok(sv[1], mep) & (_tag_ok(sv[2], mep) | hi)
    ra = _row16_sum(sv[0][0].bitcast(F32))
    rb = _row16_sum(sv[1][0].bitcast(F32))
    rc = _row16_sum(hi.select(F32(0.0), sv[2][0].bitcast(F32)))
    tot = readlane(ra, 0) + readlane(ra, 16)
    for x, l in (
        (ra, 32),
        (ra, 48),
        (rb, 0),
        (rb, 16),
        (rb, 32),
        (rb, 48),
        (rc, 0),
        (rc, 16),
    ):
        tot = tot + readlane(x, l)
    rs = _rsq_scaled(fmath.fma(tot, f32c(0x384CCCCD), f32c(0x1E3CE508)))
    mixv = (part + bperm(part, (lane + 24) & 63)) * rs
    comb_in = bperm(mixv, (lane + 8) & 63)
    if lane < 8:
        post = lane >= 4
        y = mixv * _ld(scale, post.select(Int32(1), Int32(0))).bitcast(F32) + _ld(
            base, lane
        ).bitcast(F32)
        sig = F32(1.0) / (F32(1.0) + exp_ocml(y, True))
        val = post.select(sig + sig, sig + f32c(0x358637BD))
        bo.buffer_store(val.bitcast(Int32), rsrc(hcf), row * 24 + lane)
        if ll_on != 0:
            bo.buffer_store(fx.Vector.from_elements([val.bitcast(Int32), mep], Int32), rsrc(llp),
                            row * 48 + lane * 2, cache_modifier=CM_DEV)  # fmt: skip
    if lane < 16:
        y = _ld(scale, 2).bitcast(F32) * comb_in + _ld(base, lane + 8).bitcast(F32)
        mx = fx.max(y, qk_dpp(y, QK_QP_1032))
        mx = fx.max(mx, qk_dpp(mx, QK_QP_2301))
        ex = exp_ocml(y - mx, False)
        p = qk_div_nr(ex, qk_quad_sum(ex)) + f32c(0x358637BD)
        bit2_clear = (tid & 4) == 0
        cv = qk_div_nr(p, qk_col_sum(p, bit2_clear) + f32c(0x358637BD))
        for _ in range_constexpr(19):
            qv = qk_div_nr(cv, qk_quad_sum(cv) + f32c(0x358637BD))
            cv = qk_div_nr(qv, qk_col_sum(qv, bit2_clear) + f32c(0x358637BD))
        bo.buffer_store(cv.bitcast(Int32), rsrc(hcf), row * 24 + 8 + lane)
        if ll_on != 0:
            bo.buffer_store(fx.Vector.from_elements([cv.bitcast(Int32), mep], Int32), rsrc(llp),
                            row * 48 + 16 + lane * 2, cache_modifier=CM_DEV)  # fmt: skip


TOPK_LDS = 0x12C14  # the row's 6 routed expert ids in top-k order
TOPK_FLAG_LDS = (
    39872  # wave 0 -> waves 1..7: the needed top-k slots are in LDS (= moe epoch)
)
RW_LDS = 0x12BF4  # their routing weights (f32)
SIGN = -2147483648


def _all_eq(vs, x):
    ok = vs[0] == x
    for v in vs[1:]:
        ok = ok & (v == x)
    return ok


def _imax(a, b):
    return (a > b).select(a, b)


def _imin(a, b):
    return (a < b).select(a, b)


def _dpp_old(v, ctrl, old):
    return Int32(update_dpp_i32(Int32(old), Int32(v), ctrl, 0xF, 0xF, False))


def _wave_scan_red(v, op, neutral):
    """TileRT's DPP wave reduction (row_shr 1/2/4/8, row_bcast 15/31; lanes without a source read
    ``neutral``), the result read from lane 63 (wave-uniform)."""
    for c in ROW_SHR + (0x142, 0x143):
        v = op(v, _dpp_old(v, c, neutral))
    return readlane(v, 63)


def _wave_red(v, op):
    for off in (32, 16, 8, 4, 2, 1):
        v = op(v, Int32(bperm(v.bitcast(F32), _lane() ^ off).bitcast(Int32)))
    return v


def _lane():
    return fx.thread_idx.x % 64


@traced
def moe_topk(tid, row, gate_b, ws_moe, mep, lds, kneed=None, sp=None, s2=None):
    """Wave 0: the row's 384 router scores polled (LL), keys = gate_b + score as sortable u32;
    6 rounds of wave max with the smallest expert id on ties; routing weights
    1.5 * s_k / (((((s0 + 0) + s1) + ..) + s5) + 1e-20) of the raw scores, all to LDS.
    """
    lane = tid % 64
    # TileRT: the gate biases go out before the poll, and the poll reads the whole {score, tag} pack, so
    # no load is left after the tags arrive (v11 polled the tags alone and then re-read the scores and
    # the biases: two more memory round trips on the router -> H path)
    gb = [_ld(gate_b, lane + 64 * j).bitcast(F32) for j in range(6)]
    words = [(row * 384 + lane + 64 * j) * 2 for j in range(6)]
    pv = [_ld(ws_moe, w, 2, CM_DEV) for w in words]
    ok = _all_eq([p[1] for p in pv], mep)
    while ~ok:
        spin_pause()
        pv = [_ld(ws_moe, w, 2, CM_DEV) for w in words]
        ok = _all_eq([p[1] for p in pv], mep)
    if const_expr(sp is not None):
        _stamp(sp, tid, s2, 22)
    sc = [pv[j][0].bitcast(F32) for j in range(6)]
    ub = []
    for j in range_constexpr(6):
        kb = (gb[j] + sc[j]).bitcast(Int32)
        # TileRT's sortable u32 (negative: ~bits, else bits ^ sign) xor the sign bit, as a signed key;
        # (select(neg, ~kb, kb ^ sign) ^ sign hits an AMDGPU ISel failure in this kernel)
        ub.append(kb ^ (kb < 0).select(Int32(0x7FFFFFFF), Int32(0)))
    ids = [lane + 64 * j for j in range(6)]
    sel = []
    for r in range_constexpr(6):
        m = ub[0]
        for j in range_constexpr(1, 6):
            m = _imax(m, ub[j])
        m = _wave_scan_red(m, _imax, SIGN)
        cand = Int32(0x7FFFFFFF)
        for j in range_constexpr(6):
            cand = (ub[j] == m).select(_imin(cand, ids[j]), cand)
        e = _wave_scan_red(cand, _imin, 0x7FFFFFFF)
        sel.append(e)
        for j in range_constexpr(6):
            ub[j] = (ids[j] == e).select(Int32(SIGN), ub[j])
        if const_expr(kneed is not None):
            # TileRT: the slots up to the CTA's last needed k go out first, the other waves start H
            if lane == 0:
                _lds_st(lds, TOPK_LDS + 4 * r, [e])
                if kneed == r + 1:
                    _lds_st(lds, TOPK_FLAG_LDS, [mep])
            if const_expr(sp is not None):
                if kneed == r + 1:
                    _stamp(sp, tid, s2, 23)
    if const_expr(sp is not None):
        _stamp(sp, tid, s2, 24)
    ss = []
    for r in range_constexpr(6):
        blk = sel[r] >> 6
        ln = sel[r] & 63
        # (TileRT's v_cndmask chain + one v_readlane measured slower here: v15m1 0.82239 vs 0.82082)
        val = bperm(sc[0], ln)
        for j in range_constexpr(1, 6):
            val = (blk == j).select(bperm(sc[j], ln), val)
        ss.append(val)
    tot = ss[0] + F32(0.0)
    for r in range_constexpr(1, 6):
        tot = tot + ss[r]
    tot = tot + f32c(0x1E3CE508)
    if lane == 0:
        for r in range_constexpr(6):
            _lds_st(lds, TOPK_LDS + 4 * r, [sel[r]])
            _lds_st(lds, RW_LDS + 4 * r, [((ss[r] / tot) * F32(1.5)).bitcast(Int32)])


@traced
def moe_h(tid, row, s33, ug, ug_s, ws_moe, mep, lds):
    """Routed up/gate: item i = start + wave of this CTA's [6ng/5, 6n(g+1)/5) (n = (43 - s33%8) >> 3,
    g = s33/8) -> top-k slot k = i / n, inter tile (i % n)*8 + s33%8; 40 scaled 16x16x128 MFMAs
    (MXFP4 weights, fp8 x of LDS row 0); SwiGLU x routing weight -> act slot k+1 (LL, sc1).
    """
    lane, wv = tid % 64, tid // 64
    h, q = lane % 16, lane >> 4
    s17 = s33 % 8
    n = (43 - s17) >> 3
    g = s33 // 8
    start = (6 * n * g) // 5
    end = (6 * n * (g + 1)) // 5
    # v27: at most 6 items a CTA (n <= 5), so waves 6 and 7 are idle in H; wave 6 takes item slot 0 so
    # that wave 0 (the top-k wave: all 6 rounds + the routing weights first) carries no item and the
    # CTA's first item starts at the early top-k flag (TileRT gives slot 0 to wave 0)
    slot = (wv == 6).select(Int32(0), wv)
    i = start + slot
    act = (wv != 0) & (wv != 7) & (i < end)
    ic = fx.min(i, end - 1)
    k = ic // n
    tile = (ic % n) * 8 + s17
    acc = fx.Vector.filled(4, 0.0, F32)
    if act:
        e = uniform(_lds_ld(lds, TOPK_LDS + 4 * k))
        wb = e * 0x168000 + tile * 0xA000 + lane * 16
        sb = e * 0x16800 + tile * 2560 + h * 4
        # TileRT's A scales: 10 dwords a lane (row h, step 4o + q), spread over the 4 lane groups by 40
        # ds_bpermutes issued before the MFMA chain; each group's 4 step bytes packed into one register,
        # the MFMA picks its step's byte by op_sel (sa_byte). (v6 bpermuted inside the chain: slower;
        # v4..v16 loaded a dword a step: 40 loads.)
        sd10 = [_ld(ug_s, (sb + q * 64 + o * 256) >> 2) for o in range(10)]
        pk = []
        for o in range_constexpr(10):
            pw = Int32(0)
            for qq in range_constexpr(4):
                xb = Int32(bperm(sd10[o].bitcast(F32), h + 16 * qq).bitcast(Int32))
                pw = pw | (((xb >> (q * 8)) & 0xFF) << (8 * qq))
            pk.append(pw)
        for j in range_constexpr(40):
            av = _ld(ug, (wb + j * 1024) >> 2, 4, CM_NT)
            b0 = _lds_ld(lds, j * 128 + q * 16, 4)
            b1 = _lds_ld(lds, j * 128 + 64 + q * 16, 4)
            sbx = _u8_lds(lds, QX_SCALE_LDS + 4 * j + q)
            bv = fx.Vector.from_elements(
                [b0[d] for d in range(4)] + [b1[d] for d in range(4)], Int32
            )
            acc = mfma_scaled(av, bv, acc, pk[j // 4], sbx, FP4, FP8, sa_byte=j % 4)
    rocdl.s_waitcnt(
        vmcnt=0
    )  # v34: the shared down slice's LDS DMA (waves 1..6) lands before wave 0 reads it
    gpu.barrier()
    if act:
        up = [_up32(acc[t]) for t in range(4)]
        if (lane < 32) & (h == 0):
            w = _lds_ld(lds, RW_LDS + 4 * k).bitcast(F32)
            acts = []
            for t in range_constexpr(4):
                gg = fx.min(_bfr(acc[t]), F32(10.0))
                u = fx.max(fx.min(_bfr(up[t]), F32(10.0)), F32(-10.0))
                ex = exp_ocml(gg, True)
                acts.append(
                    (bf_rne_hi(((gg / (ex + F32(1.0))) * u) * w) >> 16) & 0xFFFF
                )
            bo.buffer_store(
                fx.Vector.from_elements(
                    [acts[0] | (acts[1] << 16), mep, acts[2] | (acts[3] << 16), mep],
                    Int32,
                ),
                rsrc(ws_moe),
                (ACT_OFF + row * 0x1F80 + (k + 1) * 0x480 + tile * 32 + q * 16) >> 2,
                cache_modifier=CM_DEV,
            )


@traced
def moe_routed(tid, s2, gate_b, ug, ug_s, ws_moe, mep, lds, sp=None, sdn=None):
    """Routers (s2 >= 160) and routed CTAs (s2 < 144): wave 0's top-k, then up/gate."""
    rt = s2 >= 160
    row = rt.select((s2 - 160) >> 4, s2 // 24)
    s33 = rt.select(((s2 >> 3) & 1) * 8 + s2 % 8, s2 % 24 + 16)
    if const_expr(sdn is not None):
        if (tid >= 64) & (tid < 448):
            # wave w (1..6) copies 1 KB a step: LDS SDN_LDS + kb*16 (M0, wave-uniform) + lane*16
            lane_ = tid % 64
            for j in range_constexpr(6):
                kb = uniform((tid // 64 - 1) * 64 + 384 * j)
                dst = fx.to_llvm_ptr(lds + ((SDN_LDS + kb * 16) // 4))
                rocdl.buffer_load_to_lds(rsrc(sdn), dst.ir_value() if hasattr(dst, "ir_value") else dst,
                                         as_ir_value(Int32(s33 * 0x9000 + (kb + lane_) * 16)), size_bytes=16)  # fmt: skip
    s17 = s33 % 8
    nn = (43 - s17) >> 3
    kneed = ((6 * nn * (s33 // 8 + 1)) // 5 - 1) // nn + 1
    if tid < 64:
        moe_topk(
            tid,
            row,
            gate_b,
            ws_moe,
            mep,
            lds,
            kneed,
            sp if const_expr(_STAMP) else None,
            s2,
        )
    else:
        while _lds_ld(lds, TOPK_FLAG_LDS) != mep:
            spin_pause()
    if const_expr(sp is not None):
        _stamp(sp, tid, s2, 11)
    moe_h(tid, row, s33, ug, ug_s, ws_moe, mep, lds)


SDN_LDS = 0x9BD0  # the shared expert's down slice (36 KB, s33's 128 rows x 288 fp8), copied during H
DN_ACT_LDS = 15024  # wave w's act as fp8 at w*384 (K padded to 384 with zeros)
DN_SC_LDS = 17712  # wave w's 9 act scales at w*16 (bytes 9..15: 0x7f)
DN_PART_LDS = 17824  # wave w's 128 down partials (f32) at w*512
MOE_Y_LDS = 0x14500  # this CTA's 128 bf16 (expert sum, pre all-reduce)
MOE_RX_LDS = 0x14600  # the peers' slots, 256 B each
MOE_SLOT, MOE_PAR = 0x30000, 0x18000  # tab_moe: a sender's slot, its epoch-parity half


@traced
def moe_down(tid, s2, row, s33, dn, dn_s, sdn, sdn_s, ws_moe, valid, mep, lds):
    """Waves 0..6 (0: the shared expert, w: top-k slot w-1) of routed CTAs: act slot w (288 bf16 LL)
    -> fp8 with ue8m0 per 32 (clamp +-448) in LDS; hidden columns 128*s33.. as 8 tiles x 3 K-steps of
    scaled 16x16x128 MFMA (shared: fp8 sdn with 32x32-block scales, routed: MXFP4 dn); the 7 sums in
    order (0 + shared + r0 + .. + r5), x row valid, bf16 -> MOE_Y_LDS."""
    lane, wv = tid % 64, tid // 64
    h, q = lane % 16, lane >> 4
    # TileRT issues the routed experts' down weights (nt) and scales before polling the act
    rok = (tid < 448) & (wv >= 1)
    exr = uniform(_lds_ld(lds, TOPK_LDS + 4 * (fx.max(wv, Int32(1)) - 1)))
    wbr = exr * 0xB4000 + s33 * 0x4800
    sbr = exr * 0xB400 + s33 * 0x480
    rw, rs16, rs8 = [], [], []
    for t in range_constexpr(8):
        for ks in range_constexpr(2):
            rw.append(
                _ldm(
                    dn,
                    384 * 0xB4000,
                    (wbr + t * 2304 + ks * 1024 + lane * 16) >> 2,
                    rok,
                    4,
                    CM_NT,
                )
            )
        rw.append(
            _ldm(
                dn,
                384 * 0xB4000,
                (wbr + t * 2304 + 2048 + h * 16) >> 2,
                rok & (q == 0),
                4,
                CM_NT,
            )
        )
        rs16.append(
            Int32(
                _ldm(
                    dn_s,
                    384 * 0xB400,
                    ((sbr + t * 144) >> 1) + lane,
                    rok,
                    dtype=T.i16,
                )
            ).to(Int32)
            & 0xFFFF
        )
        rs8.append(
            Int32(
                _ldm(
                    dn_s,
                    384 * 0xB400,
                    sbr + t * 144 + 128 + h,
                    rok & (q == 0),
                    dtype=T.i8,
                )
            ).to(Int32)
            & 0xFF
        )
    if tid < 448:
        if lane < 24:
            _lds_st(lds, DN_ACT_LDS + wv * 384 + 288 + lane * 4, [Int32(0)])
        if (lane >= 9) & (lane < 16):
            _lds_st(lds, DN_SC_LDS + wv * 16 + (lane & ~3), [Int32(0x7F7F7F7F)])
    if tid < 448:
        on = lane < 36
        aw = (ACT_OFF + row * 0x1F80 + wv * 0x480 + fx.min(lane, Int32(35)) * 32) >> 2
        v0 = _ld(ws_moe, aw, 4, CM_DEV)
        v1 = _ld(ws_moe, aw + 4, 4, CM_DEV)
        ok = ((v0[1] == mep) & (v0[3] == mep) & (v1[1] == mep) & (v1[3] == mep)) | ~on
        while ~ok:
            spin_pause()
            v0 = _ld(ws_moe, aw, 4, CM_DEV)
            v1 = _ld(ws_moe, aw + 4, 4, CM_DEV)
            ok = (
                (v0[1] == mep) & (v0[3] == mep) & (v1[1] == mep) & (v1[3] == mep)
            ) | ~on
        ys = []
        for d in (v0[0], v0[2], v1[0], v1[2]):
            ys += [(d << 16).bitcast(F32), (d & Int32(-65536)).bitcast(F32)]
        m = abs(ys[0])
        for k in range_constexpr(1, 8):
            m = fx.max(m, abs(ys[k]))
        m = fx.max(m, _dppf(m, QX1))
        m = fx.max(fx.max(m, _dppf(m, QX2)), f32c(0x38D1B717))
        sv = (m * f32c(0x3B124925)).bitcast(Int32)
        e = (sv >> 23) - 127 + ((sv & 0x7FFFFF) != 0).select(Int32(1), Int32(0))
        inv = (Int32(0x3F800000) - (e << 23)).bitcast(F32)
        qs = [fx.max(fx.min(ys[k] * inv, F32(448.0)), F32(-448.0)) for k in range(8)]
        d0 = _fp8x2(qs[0], qs[1]) | (_fp8x2(qs[2], qs[3]) << 16)
        d1 = _fp8x2(qs[4], qs[5]) | (_fp8x2(qs[6], qs[7]) << 16)
        sb = (e + 127) & 0xFF
        sbf = sb.bitcast(F32)
        s1, s2_, s3 = _gather_sc(sbf, lane)
        if on:
            _lds_st(lds, DN_ACT_LDS + wv * 384 + lane * 8, [d0, d1])
        if (lane & 15) == 0:
            if lane < 32:
                _lds_st(
                    lds,
                    DN_SC_LDS + wv * 16 + (lane >> 2),
                    [sb | (s1 << 8) | (s2_ << 16) | (s3 << 24)],
                )
            if lane == 32:
                _lds_st(lds, DN_SC_LDS + wv * 16 + 8, [Int32(0x7F7F7F00) | sb])
    if tid < 448:
        bvs, sbs = [], []
        for ks in range_constexpr(3):
            b0 = _lds_ld(lds, DN_ACT_LDS + wv * 384 + ks * 128 + q * 16, 4)
            b1 = _lds_ld(lds, DN_ACT_LDS + wv * 384 + ks * 128 + 64 + q * 16, 4)
            bvs.append(
                fx.Vector.from_elements(
                    [b0[d] for d in range(4)] + [b1[d] for d in range(4)], Int32
                )
            )
            sbs.append(_u8_lds(lds, DN_SC_LDS + wv * 16 + 4 * ks + q))
        lo2 = q < 2
        if wv == 0:
            for t in range_constexpr(8):
                c = fx.Vector.filled(4, 0.0, F32)
                for ks in range_constexpr(3):
                    if const_expr(ks < 2):
                        o = (
                            (h >> 2) * 1152
                            + t * 4608
                            + ks * 512
                            + (h & 3) * 64
                            + q * 16
                        )
                        a0 = _lds_ld(lds, SDN_LDS + o, 4)
                        a1 = _lds_ld(lds, SDN_LDS + o + 256, 4)
                        av = fx.Vector.from_elements(
                            [a0[d] for d in range(4)] + [a1[d] for d in range(4)], Int32
                        )
                        si = (t // 2) * 9 + ks * 4 + q
                    else:
                        o = (
                            (h >> 2) * 1152
                            + t * 4608
                            + 1024
                            + (h & 3) * 32
                            + (q & 1) * 16
                        )
                        a0 = _lds_ld(lds, SDN_LDS + o, 4)
                        av = fx.Vector.from_elements(
                            [lo2.select(a0[d], Int32(0)) for d in range(4)]
                            + [Int32(0)] * 4,
                            Int32,
                        )
                        si = (t // 2) * 9 + 8 + q * 0
                    sa = (
                        Int32(
                            bo.buffer_load(
                                rsrc(sdn_s), s33 * 36 + si, vec_width=1, dtype=T.i8
                            )
                        ).to(Int32)
                        & 0xFF
                    )
                    c = mfma_scaled(av, bvs[ks], c, sa, sbs[ks])
                if h == 0:
                    _lds_st(
                        lds,
                        DN_PART_LDS + (t * 16 + q * 4) * 4,
                        [c[i].bitcast(Int32) for i in range(4)],
                    )
        else:
            for t in range_constexpr(8):
                c = fx.Vector.filled(4, 0.0, F32)
                for ks in range_constexpr(3):
                    av = rw[t * 3 + ks]
                    if const_expr(ks < 2):
                        sa = (rs16[t] >> (8 * ks)) & 0xFF
                    else:
                        sa = (q == 0).select(rs8[t], Int32(0x7F))
                    c = mfma_scaled(av, bvs[ks], c, sa, sbs[ks], FP4, FP8)
                if h == 0:
                    _lds_st(lds, DN_PART_LDS + wv * 512 + (t * 16 + q * 4) * 4,
                            [c[i].bitcast(Int32) for i in range(4)])  # fmt: skip
    gpu.barrier()
    if tid < 128:
        acc = F32(0.0)
        for w in range_constexpr(7):
            acc = acc + _lds_ld(lds, DN_PART_LDS + w * 512 + tid * 4).bitcast(F32)
        yb = valid.select((bf_rne_hi(acc) >> 16) & 0xFFFF, Int32(0))
        nb = Int32(_dppf(yb.bitcast(F32), QX1).bitcast(Int32))
        if (tid & 1) == 0:
            _lds_st(lds, MOE_Y_LDS + tid * 2, [yb | (nb << 16)])
    gpu.barrier()


@traced
def moe_tp(tid, s2, row, s33, tab, me, npes, y, mep, lds, sp=None, ptrs=None, tla=None):
    """One-shot TP all-reduce of the 128 MoE outputs (LL64 like C2: wave p -> the p-th peer, lanes
    < 40 = 5 rows of 64 B; tab_moe slots 0x30000, parity halves 0x18000, work id row*42 + s33), the
    ranks summed in order 0..7 in fp32, bf16 to y[row][128*s33 + col] (plain) and MOE_Y_LDS.
    """
    lane, wv = tid % 64, tid // 64
    par = (mep & 1) * MOE_PAR
    wid = (row * 42 + s33) * 0x180
    rq, j = fx.min(lane, Int32(39)) >> 3, lane & 7
    e = rq * 28 + j * 4
    data = (j != 7) & (e < 125)
    if npes >= 2:
        if (wv < 7) & (lane < 40):
            dest = wv + (wv >= me).select(1, 0)
            if const_expr(ptrs is None):
                base = load_ptr64(tab, dest)
            else:
                base = ptrs[0]
            slot = me - (me > dest).select(1, 0)
            val = fx.Vector.from_elements([mep, mep], Int32)
            if data:
                val = _lds_ld(lds, MOE_Y_LDS + e * 2, 2)
            off = slot * MOE_SLOT + par + wid + rq * 64 + j * 8
            bo.buffer_store(val, rsrc(base), off >> 2, cache_modifier=CM_SYS)
    pre = None
    if const_expr(tla is not None):
        pre = tla[0]
        if tla[1]:
            pre = tail_loads(tid, row, s33, tla[2], tla[3], tla[4], tla[5])
    if npes >= 2:
        if const_expr(sp is not None):
            _stamp(sp, tid, s2, 17)
        gpu.barrier()
        if const_expr(sp is not None):
            _stamp(sp, tid, s2, 18)
        if (wv < 7) & (lane < 40):
            if const_expr(ptrs is None):
                own = load_ptr64(tab, me)
            else:
                own = ptrs[1]
            rb = wv * MOE_SLOT + par + wid + rq * 64
            fl = _ld(own, (rb + 56) >> 2, 2, CM_SYS)
            while ~((fl[0] == mep) & (fl[1] == mep)):
                spin_pause()
                fl = _ld(own, (rb + 56) >> 2, 2, CM_SYS)
            if data:
                d = _ld(own, (rb + j * 8) >> 2, 2, CM_SYS)
                _lds_st(lds, MOE_RX_LDS + wv * 256 + e * 2, [d[0], d[1]])
        if const_expr(sp is not None):
            _stamp(sp, tid, s2, 19)
        gpu.barrier()
        if const_expr(sp is not None):
            _stamp(sp, tid, s2, 20)
    if tid < 128:
        loc = MOE_Y_LDS + tid * 2
        out = _lds_u16(lds, loc)
        if npes >= 2:
            acc = F32(0.0)
            for r in range_constexpr(8):
                rem = MOE_RX_LDS + (r - (me < r).select(1, 0)) * 256 + tid * 2
                acc = acc + (_lds_u16(lds, (me == r).select(loc, rem)) << 16).bitcast(
                    F32
                )
            out = (bf_rne_hi(acc) >> 16) & 0xFFFF
        nb = Int32(_dppf(out.bitcast(F32), QX1).bitcast(Int32))
        if (tid & 1) == 0:
            bo.buffer_store(
                out | (nb << 16), rsrc(y), (row * 5120 + s33 * 128 + tid) >> 1
            )
            _lds_st(lds, MOE_Y_LDS + tid * 2, [out | (nb << 16)])
    if const_expr(sp is not None):
        _stamp(sp, tid, s2, 21)
    gpu.barrier()
    return pre


@traced
def moe_dj(tid, s2, dn, dn_s, sdn, sdn_s, ws_moe, vflags, tab, me_np, y, mep, lds, tl):
    """Routed CTAs (not the shared-expert ones): I down, then J (MoE TP all-reduce, mode 1)."""
    rt = s2 >= 160
    row = rt.select((s2 - 160) >> 4, s2 // 24)
    s33 = rt.select(((s2 >> 3) & 1) * 8 + s2 % 8, s2 % 24 + 16)
    valid = _ld(vflags, row) >= 0
    # J's peer base (wave w -> its peer) and own base: table loads that do not wait on I, issued first
    me_, wv_ = Int32(me_np), tid // 64
    jptrs = [
        load_ptr64(tab, fx.min(wv_ + (wv_ >= me_).select(1, 0), Int32(7))),
        load_ptr64(tab, me_),
    ]
    moe_down(tid, s2, row, s33, dn, dn_s, sdn, sdn_s, ws_moe, valid, mep, lds)
    _stamp(tl[12], tid, s2, 14)
    tla = None
    z4 = fx.Vector.filled(4, 0, Int32)
    tla = [[Int32(0)] * 5 + [z4] * 8, tl[8] != 0, tl[0], tl[1], tl[2], Int32(tl[3])]
    pre = moe_tp(tid, s2, row, s33, tab, me_, Int32(me_np >> 32), y, mep, lds, tl[12] if const_expr(_STAMP) else None,
                 jptrs, tla)  # fmt: skip
    _stamp(tl[12], tid, s2, 15)
    if tl[8] != 0:
        tail_next(tid, row, s33, tl[0], tl[1], tl[2], Int32(tl[3]), tl[4], tl[11], Int32(tl[11] != 0),
                  tl[5], tl[6], tl[7], tl[8], tl[10], mep, lds, pre)  # fmt: skip
    _stamp(tl[12], tid, s2, 16)


TAIL_X2_LDS = (
    0x16880  # [hc 4][128 cols] f32 (bf16 values): the post-FFN residual streams
)
TAIL_HC_LDS = 0x17280  # hcF[24] polled from G''s LL copy


@traced
def _pair_store(ptr, half_index, v16, lane, even):
    """Store the u16 v16 of consecutive threads as dwords from the even ones (half_index even)."""
    nb = Int32(_dppf(v16.bitcast(F32), QX1).bitcast(Int32))
    if even:
        bo.buffer_store(v16 | (nb << 16), rsrc(ptr), half_index >> 1)


@traced
def tail_loads(tid, row, s33, xres, fn_n, gam_n, gflag_n):
    """tail_next's global loads (residual streams and gamma for threads < 128, the next layer's attn_fn
    slice for the others), issued (v37, as TileRT) right after J's push, before its receive poll.
    """
    z4 = fx.Vector.filled(4, 0, Int32)
    xd = [Int32(0), Int32(0), Int32(0), Int32(0)]
    gd = Int32(0)
    wv8 = [z4, z4, z4, z4, z4, z4, z4, z4]
    if tid < 128:
        g = s33 * 128 + tid
        for hc in range_constexpr(4):
            xd[hc] = _ld(xres, (row * 0x5000 + hc * 0x1400 + g) >> 1, 1, CM_DEV)
        gd = _ld(gam_n, (gflag_n != 0).select(g >> 1, g))
    if tid >= 128:
        t = tid - 128
        r96 = t % 96
        wb = (
            (r96 >> 2) * 0x5000 + (r96 & 3) * 0x1400 + s33 * 128 + (t // 96) * 32
        )  # f32 index
        for i in range_constexpr(8):
            wv8[i] = _ld(fn_n, wb + 4 * i, 4)
    return xd + [gd] + wv8


@traced
def tail_next(tid, row, s33, xres, fn_n, gam_n, gflag_n, x2_out, mean_out, has_mean, x_out, xq_out,
              xs_out, ws_gemv, ll, mep, lds, pre=None):  # fmt: skip
    """After J: X2_j = (((c[8+j] x0 + c[12+j] x1) + c[16+j] x2) + c[20+j] x3) + c[4+j] y_moe with G''s
    hcF (bf16 to X2), the next layer's x = bf16(bf16(sum c_j X2_j) * gamma) with its 1x32 E4M3 / ue8m0
    (x, xq, xs), per 8 columns the sum of X2^2 and per 32 columns the 24 attn_fn . X2 partials (plain,
    the next layer's mHC finalize workspace: GEMV at ws, squares at ws + 0x1e000)."""
    lane, wv = tid % 64, tid // 64
    if (wv == 7) & (lane < 24):
        w = row * 48 + lane * 2
        v = _ld(ll, w, 2, CM_DEV)
        while ~_tag_ok(v, mep):
            spin_pause()  # s_sleep is IntrNoMem: alone it lets LLVM hoist the reload (v10 hang)
            rocdl.s_sleep(1)
            v = _ld(ll, w, 2, CM_DEV)
        _lds_st(lds, TAIL_HC_LDS + lane * 4, [v[0]])
    gpu.barrier()
    even = (tid & 1) == 0
    gx = F32(0.0)
    if tid < 128:
        g = s33 * 128 + tid
        xs4 = []
        for hc in range_constexpr(4):
            d = (
                pre[hc]
                if const_expr(pre is not None)
                else _ld(xres, (row * 0x5000 + hc * 0x1400 + g) >> 1, 1, CM_DEV)
            )
            xs4.append(even.select(d << 16, d & Int32(-65536)).bitcast(F32))
        c = _lds_vec(lds, TAIL_HC_LDS, 24)  # 6 ds_read_b128
        y = (_lds_u16(lds, MOE_Y_LDS + tid * 2) << 16).bitcast(F32)
        xb, hs = [], []
        for j in range_constexpr(4):
            a = c[8 + j] * xs4[0] + c[12 + j] * xs4[1]
            a = a + c[16 + j] * xs4[2]
            a = a + c[20 + j] * xs4[3]
            a = a + c[4 + j] * y
            r = bf_rne_hi(a)
            hs.append((r >> 16) & 0xFFFF)
            xb.append((r & Int32(-65536)).bitcast(F32))
        for j in range_constexpr(4):
            _pair_store(x2_out, row * 0x5000 + j * 0x1400 + g, hs[j], lane, even)
            _lds_st(lds, TAIL_X2_LDS + j * 512 + tid * 4, [xb[j].bitcast(Int32)])
        if has_mean != 0:
            mn = ((xb[0] + xb[1]) + xb[2]) + xb[3]
            _pair_store(
                mean_out,
                row * 0x1400 + g,
                (bf_rne_hi(mn * F32(0.25)) >> 16) & 0xFFFF,
                lane,
                even,
            )
        sq = []
        for j in range_constexpr(4):
            for k in range_constexpr(8):
                vk = bperm(xb[j], (lane & ~7) + k)
                sq.append(vk * vk)
        sacc = sq[0] + sq[1]
        for k in range_constexpr(2, 32):
            sacc = sacc + sq[k]
        if (tid & 7) == 0:
            bo.buffer_store(
                sacc.bitcast(Int32), rsrc(ws_gemv), 0x1E000 // 4 + row * 640 + (g >> 3)
            )
        gd = (
            pre[4]
            if const_expr(pre is not None)
            else _ld(gam_n, (gflag_n != 0).select(g >> 1, g))
        )
        gm = (
            (gflag_n != 0)
            .select(even.select(gd << 16, gd & Int32(-65536)), gd)
            .bitcast(F32)
        )
        p = c[0] * xb[0] + c[1] * xb[1]
        p = c[2] * xb[2] + p
        p = c[3] * xb[3] + p
        gx = _bfr(p) * gm
    gpu.barrier()
    if tid < 128:
        g = s33 * 128 + tid
        rb = bf_rne_hi(gx)
        xv = (rb & Int32(-65536)).bitcast(F32)
        m = abs(xv)
        m = _max32(m, lane)
        m = fx.max(m, f32c(0x38D1B717))
        sv = (m * f32c(0x3B124925)).bitcast(Int32)
        sbits = (sv & 0x7F800000) + ((sv & 0x7FFFFF) == 0).select(
            Int32(0), Int32(0x800000)
        )
        inv = (Int32(0x7F000000) - (sbits & 0x7F800000)).bitcast(F32)
        qb = _fp8x2(xv * inv, F32(0.0)) & 0xFF
        _pair_store(x_out, row * 0x1400 + g, (rb >> 16) & 0xFFFF, lane, even)
        q1 = Int32(
            _dppf(qb.bitcast(F32), QROT1).bitcast(Int32)
        )  # read on tid % 4 == 0 only
        q2 = Int32(_dppf(qb.bitcast(F32), QX2).bitcast(Int32))
        q3 = Int32(_dppf(qb.bitcast(F32), QROT3).bitcast(Int32))
        if (tid & 3) == 0:
            bo.buffer_store(
                qb | (q1 << 8) | (q2 << 16) | (q3 << 24),
                rsrc(xq_out),
                (row * 0x1400 + g) >> 2,
            )
        sbb = (sbits >> 23) & 0xFF
        t1 = Int32(_up32(sbb.bitcast(F32)).bitcast(Int32))  # read on lane 0 only
        # a CTA's 4 scale bytes (128 columns) from lanes 0 / 32 of waves 0 and 1
        sw1 = sbb
        if wv == 0:
            if lane == 0:
                _lds_st(lds, TAIL_HC_LDS + 96, [sbb | (t1 << 8)])
        if wv == 1:
            if lane == 0:
                _lds_st(lds, TAIL_HC_LDS + 100, [sw1 | (t1 << 8)])
    if tid >= 128:
        t = tid - 128
        grp = t // 96
        r96 = t % 96
        mix = r96 >> 2
        hc = r96 & 3
        wb = mix * 0x5000 + hc * 0x1400 + s33 * 128 + grp * 32  # f32 index
        wv8 = (
            pre[5:13]
            if const_expr(pre is not None)
            else [_ld(fn_n, wb + 4 * i, 4) for i in range(8)]
        )
        wf = []
        for i in range_constexpr(8):
            wf += [wv8[i][k].bitcast(F32) for k in range(4)]
        xv8 = [
            _lds_ld(lds, TAIL_X2_LDS + hc * 512 + grp * 128 + 16 * i, 4)
            for i in range(8)
        ]
        xf = []
        for i in range_constexpr(8):
            xf += [xv8[i][k].bitcast(F32) for k in range(4)]
        acc = []
        for cc in range_constexpr(4):
            a = fmath.fma(wf[cc], xf[cc], F32(0.0))
            for i in range_constexpr(1, 8):
                a = fmath.fma(wf[cc + 4 * i], xf[cc + 4 * i], a)
            acc.append(a)
        pp = (acc[0] + acc[1]) + (acc[2] + acc[3])
        s01 = pp + _dppf(pp, QX1)
        s = s01 + _dppf(s01, QX2)
        if hc == 0:
            bo.buffer_store(
                s.bitcast(Int32), rsrc(ws_gemv), (row * 160 + s33 * 4 + grp) * 24 + mix
            )
    gpu.barrier()
    if tid == 0:
        lo = _lds_ld(lds, TAIL_HC_LDS + 96)
        hi = _lds_ld(lds, TAIL_HC_LDS + 100)
        bo.buffer_store(
            (lo & 0xFFFF) | ((hi & 0xFFFF) << 16),
            rsrc(xs_out),
            (row * 0xA0 + s33 * 4) >> 2,
        )


def _max8(lds, h):
    s = [_lds_ld(lds, MAX_LDS + k * 64 + h * 4).bitcast(F32) for k in range(8)]
    m = fx.max(s[0], s[1])
    for k in (2, 4, 6):
        m = fx.max(fx.max(m, s[k]), s[k + 1])
    return m


def build():
    """(symbol, launcher) of the test kernel with the ported phases (A1 so far)."""
    name = kernel_symbol(
        "tk_mega_afm6" if _AFM6 else "tk_mega_tail",
        phases="full_v39_pf1_pf2" + ("_st" if _STAMP else ""),
    )

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_BYTES // 4, 16]
        p: fx.Array[fx.Int16, 16 * P_ROW, 16]

    if _AFM6:

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def mega_afm6(
            a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
            a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
            a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
            a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
            a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
            a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
            a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
            a56: Int64,
        ):  # fmt: skip
            sm = fx.SharedAllocator().allocate(Smem).peek()
            Z = Int64(0)
            _entry(
                fx.thread_idx.x,
                fx.block_idx.x,
                sm.w.ptr,
                sm.p.ptr,
                a00,
                a01,
                a02,
                a03,
                a04,
                a05,
                a08,
                a09,
                a10,
                a11,
                a12,
                a13,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
                a27,
                a28,
                a29,
                a30,
                a33,
                a34,
                a36,
                a31,
                a32,
                a35,
                a37,
                a42,
                a43,
                a44,
                a45,
                a46,
                a47,
                a48,
                a49,
                a50,
                a51,
                a52,
                a53,
                a54,
                a55,
                a56,
                Z,
                Z,
                Z,
                Z,
                Z,
                Z,
                Z,
                Z,
                Z,
                Z,
                Z,
            )

        @launcher(name)
        def launch(
            a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
            a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
            a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
            a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
            a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
            a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
            a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
            a56: Int64,
            stream: fx.Stream = _CURRENT_STREAM,
        ):  # fmt: skip
            mega_afm6(
                a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
                a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
                a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
                a48, a49, a50, a51, a52, a53, a54, a55, a56,
            ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    else:

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def mega_tail(
            a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
            a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
            a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
            a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
            a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
            a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
            a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
            a56: Int64, a57: Int64, a58: Int64, a59: Int64, a60: Int64, a61: Int64, a62: Int64, a63: Int64,
            a64: Int64, a65: Int64, a66: Int64, a67: Int64, a68: Int64, a69: Int64, a70: Int64, a71: Int64,
            a72: Int64,
        ):  # fmt: skip
            sm = fx.SharedAllocator().allocate(Smem).peek()
            _entry(
                fx.thread_idx.x,
                fx.block_idx.x,
                sm.w.ptr,
                sm.p.ptr,
                a00,
                a01,
                a02,
                a03,
                a04,
                a05,
                a08,
                a09,
                a10,
                a11,
                a12,
                a13,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
                a27,
                a28,
                a29,
                a30,
                a33,
                a34,
                a36,
                a31,
                a32,
                a35,
                a37,
                a42,
                a43,
                a44,
                a45,
                a46,
                a47,
                a48,
                a49,
                a50,
                a51,
                a52,
                a53,
                a54,
                a55,
                a56,
                a58,
                a59,
                a60,
                a61,
                a62,
                a63,
                a64,
                a65,
                a66,
                a71,
                a72,
            )

        @launcher(name)
        def launch(
            a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
            a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
            a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
            a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64, a30: Int64, a31: Int64,
            a32: Int64, a33: Int64, a34: Int64, a35: Int64, a36: Int64, a37: Int64, a38: Int64, a39: Int64,
            a40: Int64, a41: Int64, a42: Int64, a43: Int64, a44: Int64, a45: Int64, a46: Int64, a47: Int64,
            a48: Int64, a49: Int64, a50: Int64, a51: Int64, a52: Int64, a53: Int64, a54: Int64, a55: Int64,
            a56: Int64, a57: Int64, a58: Int64, a59: Int64, a60: Int64, a61: Int64, a62: Int64, a63: Int64,
            a64: Int64, a65: Int64, a66: Int64, a67: Int64, a68: Int64, a69: Int64, a70: Int64, a71: Int64,
            a72: Int64,
            stream: fx.Stream = _CURRENT_STREAM,
        ):  # fmt: skip
            mega_tail(
                a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
                a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31,
                a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47,
                a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63,
                a64, a65, a66, a67, a68, a69, a70, a71, a72,
            ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch


@traced
def _entry(
    tid,
    s2,
    lds,
    pl,
    a00,
    a01,
    a02,
    a03,
    a04,
    a05,
    a08,
    a09,
    a10,
    a11,
    a12,
    a13,
    a15,
    a16,
    a17,
    a18,
    a19,
    a20,
    a21,
    a22,
    a23,
    a27,
    a28,
    a29,
    a30,
    a33,
    a34,
    a36,
    a31,
    a32,
    a35,
    a37,
    a42,
    a43,
    a44,
    a45,
    a46,
    a47,
    a48,
    a49,
    a50,
    a51,
    a52,
    a53,
    a54,
    a55,
    a56,
    a58,
    a59,
    a60,
    a61,
    a62,
    a63,
    a64,
    a65,
    a66,
    a71,
    a72,
):
    """Kernarg: Attn q @0, window kv / scales @8 / @16, main kv / scales @24 / @32, the indexer key list
    @40 (bit 0: tagged LL), vpos @64, attn_sink @72, rope table @80 and its length @88, O partials @96,
    (m, l) @104; ProjoArgs: wo_a / scales @120 / @128, wo_b / scales @136 / @144, the LL packs @152
    (+0x28), the sym:projo table @160, mype | npes << 32 @168, the epoch @176 (+0x40); MhcFArgs: the
    residual streams in @216 / out @224, hcA @232, ffn_fn @240, the FFN-input LL region @296; the MoE
    epoch @440.
    """
    _stamp(a23, tid, s2, 0)
    att = ((s2 < (32 if const_expr(_AFM6) else 160)) | (s2 >= 192)) & ((s2 % 8) < 6)
    # TileRT: the other CTAs wait ~1 us so the attention reads go out first, then fetch wo_a / wo_b;
    # the attention CTAs branch past both (v22: no masked prefetch, so A1 / A2 never wait on the
    # wo_a / wo_b kernarg loads)
    z4 = fx.Vector.filled(4, 0, Int32)
    pa = [z4, z4, Int32(0)]
    pb = [z4, z4, z4, z4, Int32(0)]
    if ~att:
        t0 = memrealtime()
        while (memrealtime() - t0) < _DLY:
            spin_pause()
        pa = woa_fetch(tid, s2, a15, a16, s2 >= 0)
        pb = wob_fetch(tid, s2, a17, a18, s2 < 160)
    if (s2 < (32 if const_expr(_AFM6) else 160)) & ((s2 % 8) < 6):
        attn_a1(
            tid, s2, a00, a01, a02, a03, a04, a05, a08, a12, a13, Int32(a22), lds, pl
        )
    if (s2 >= 192) & ((s2 % 8) < 6):
        # TileRT: the merge CTAs fetch wo_a before polling the partials
        pa = woa_fetch(tid, s2, a15, a16, s2 >= 0)
        attn_a2(tid, s2, a08, a09, a10, Int32(a11), a12, a13, a19, Int32(a22), lds)
    _stamp(a23, tid, s2, 1)
    # the attention-chunk CTAs fetch after their chunk
    a1c = (s2 < (32 if const_expr(_AFM6) else 160)) & ((s2 % 8) < 6)
    pa = _sel_fetch(a1c, woa_fetch(tid, s2, a15, a16, a1c), pa)
    pb = _sel_fetch(a1c, wob_fetch(tid, s2, a17, a18, a1c & (s2 < 160)), pb)
    proj_b1(tid, s2, pa, a19, Int32(a22), lds)
    _stamp(a23, tid, s2, 2)
    if ((s2 // 8) % 8) == 0:
        proj_b2(tid, s2, a19, Int32(a22))
    _stamp(a23, tid, s2, 3)
    if s2 < 160:
        dpf = mhc_d_fetch(tid, s2, a27, a29, a30)
        me_, wv_ = (
            Int32(a21),
            tid // 64,
        )  # C2's peer / own bases: table loads that do not wait on C1
        cptrs = [
            load_ptr64(a20, fx.min(wv_ + (wv_ >= me_).select(1, 0), Int32(7))),
            load_ptr64(a20, me_),
        ]
        proj_c1(tid, s2, pb, a19, Int32(a22), lds)
        _stamp(a23, tid, s2, 5)
        proj_c2(tid, s2, a20, me_, Int32(a21 >> 32), Int32(a22), lds, cptrs)
        _stamp(a23, tid, s2, 7)
        mhc_d(tid, s2, a27, a28, a29, a30, a37, Int32(a55), lds, dpf)
        _stamp(a23, tid, s2, 8)
    gpre = fx.Vector.filled(16, 0.0, F32)
    if s2 >= 160:
        gpre = moe_router(
            tid, s2, a36, a37, a33, Int32(a34), a52, Int32(a55), lds, True
        )
        _stamp(a23, tid, s2, 9)
    moe_prep(tid, s2, a37, a33, Int32(a34), a47, a48, a52, Int32(a55), lds, gpre)
    _stamp(a23, tid, s2, 10)
    if (s2 >= 148) & (s2 < 154) & (tid >= 448):
        mhc_gp(tid, s2, a37, a31, a32, a35, Int32(a65 != 0), a71, Int32(a55))
    if (s2 < 144) | (s2 >= 160):
        moe_routed(tid, s2, a42, a43, a44, a52, Int32(a55), lds, a23, a49)
        _stamp(a23, tid, s2, 12)
        moe_dj(tid, s2, a45, a46, a49, a50, a52, a51, a53, a54, a56, Int32(a55), lds,
               [a28, a58, a59, a60, a61, a62, a63, a64, a65, a66, a71, a72, a23])  # fmt: skip
