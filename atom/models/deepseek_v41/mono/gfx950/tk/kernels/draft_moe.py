# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Ports the device semantics of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team)
# ``tilert::ops::ds_v4_1_flash::moe_rx::dsv41_moe_rx_kernel<5, 128, 8, false, true>``.
"""The DSpark draft layers' MoE on gfx950 (TP8) as a FlyDSL port of TileRT's ``moe_rx`` kernel: kernarg
ABI unchanged (236 B read by offset as 30 i64 words), grid 200 x 512, five draft rows of 40 CTAs each
(``runs/20261115-flydsl-tilert/m3/moe_rx_notes.md``), diffed bit-exact by ``m2/x_replay8.py``.

Row r = cta / 40, s3 = cta % 40. A row is valid when blk_pos[r] >= 0. CTAs s3 >= 24 score the row's
router (8 experts each); the valid rows' CTAs s3 <= 26 run one top-3 expert slot each (k = s3 / 9,
four 16-row up/gate tiles); the rest ("free" CTAs: s3 >= 27 of a valid row, all 40 of an invalid one,
numbered across the rows) take the shared expert's up/gate tiles and the FFN mHC finalize rows.
"""

# ruff: noqa: SIM102  (traced code: nested ifs are nested scf.if, not one fused condition)
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx950.tk.kernels.common import (
    bf_rne_hi,
    bperm,
    exp_ocml,
    f32c,
    log1p_ocml,
    spin_pause,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.layer_tail import (
    QX1,
    QX2,
    _bfr,
    _dppf,
    _fp8x2,
    _gather_sc,
    _ld,
    _ldm,
    _lds_ld,
    _lds_st,
    _lds_u16,
    _u8_lds,
    _up32,
)
from atom.models.deepseek_v41.mono.gfx950.tk.kernels.qkv import (
    ROW_SHR,
    _dpp,
    mhc_finalize_row,
)
from atom.mono.device.mx import FP4, FP8, mfma_scaled
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    CM_SYS,
    kernel_symbol,
    launcher,
    load_ptr64,
    readlane,
    rsrc,
    traced,
    uniform,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

KARG_WORDS = 30
LAYOUT = ",".join(f"{8 * i}:8" for i in range(KARG_WORDS))
GRID, THREADS = 200, 512
ROWS, CTAS_ROW, EXPERTS, TOPK = 5, 40, 128, 3
VERSION = "v1"

# LDS (bytes)
X8_LDS = 0  # the row's fp8 x (5120 B), then its 160 ue8m0 scales at 5120
XS8_LDS = 5120
RX_LDS = 5280  # router: the row's bf16 x (10240 B), wave w's K segment at w * 1280
RACC_LDS = 15536  # router: the waves' 4x4x4 accumulators, 16 B a lane
SG_RED_LDS = 23808  # shared expert: waves 1..7's partials, 1 KB a wave
RT_RED_LDS = 30976  # routed: the K-half-1 waves' partials, 1 KB a tile
TOPK_FLAG = 38064
W_LDS = 38068  # the three routing weights (f32)
ID_LDS = 38100  # the three expert ids
ACT8_LDS = 38144  # down: the four act slots as fp8 (shared, routed 0..2), 288 B each
ASC_LDS = 39296  # their ue8m0 scales, 32 B a slot (9 used)
DPART_LDS = 39424  # wave w's 128 down partials (f32) at w * 512
MOE_Y_LDS = 43520  # this CTA's 128 bf16 (expert sum, pre all-reduce)
MOE_RX_LDS = 43776  # the 7 peers' 128 bf16, 256 B each
LDS_BYTES = 45568

ACT_OFF, ACT_ROW, ACT_SLOT = (
    0x6000,
    0x3F00,
    0x900,
)  # ws_moe: [row][slot][tile] {bf16x2, epoch} act LL
UG_EXPERT, UG_TILE, UGS_EXPERT = (
    0x168000,
    0xA000,
    0x16800,
)  # MXFP4 up/gate: an expert, a 16-row tile; scales
SUG_TILE = 0x14000  # the shared expert's fp8 up/gate, a 16-row tile
DN_EXPERT, DN_CTA, DN_TILE, DNS_EXPERT = (
    0xB4000,
    0x4800,
    2304,
    0xB400,
)  # MXFP4 down: expert, 128 cols, 16 cols
SDN_BYTES, SDNS_BYTES = (
    5120 * 288,
    1440,
)  # the shared expert's fp8 down and its 32x32-block scales
MOE_SLOT, MOE_PAR = 0x30000, 0x18000  # tab_moe: a sender's slot, its epoch-parity half
SIGN = -2147483648


def _cfence():
    """A compiler memory barrier (empty asm, memory clobber): LDS stores stay in program order."""
    llvm.InlineAsmOp(None, [], "", "~{memory}", has_side_effects=True)


def _st_i(addr, word, v):
    bo.buffer_store(Int32(v), rsrc(addr), word)


def _mfma4(a2, b2, c):
    """v_mfma_f32_4x4x4_16b_bf16 from two i32 words each (4 bf16 a lane)."""
    a = fx.Vector.from_elements([Int32(x) for x in a2], Int32).bitcast(fx.Int16)
    b = fx.Vector.from_elements([Int32(x) for x in b2], Int32).bitcast(fx.Int16)
    av, bv, cv, cbsz, abid, blgp = rocdl._split_mfma_operands(
        [a.ir_value(), b.ir_value(), c]
    )
    return fx.Vector(
        rocdl.mfma_f32_4x4x4bf16_1k(
            T.vec(4, T.f32), av, bv, cv, cbsz, abid, blgp
        ).result
    )


def _score(logit):
    """sqrt(softplus(logit)): logit itself above 20, else log1p(exp(logit)) (OCML), IEEE sqrt."""
    sp = (f32c(0x41A00000) < logit).select(logit, log1p_ocml(exp_ocml(logit, False)))
    return fmath.sqrt(sp)


@traced
def router(tid, r, j, xn, gate_w, ws_moe, mep, lds):
    """Router CTA j (0..15) of row r: experts 8j..8j+7. Wave w takes K segment 640w.. of all eight
    (TileRT's packed gate_w: 10 KB a wave, lane 4b+i holds expert i + 4(b>>3), K (b&7)*4 of each 32)
    in 20 chained 4x4x4 bf16 MFMAs; wave 0 sums block (b&7) partials in order, then the 8 waves by
    DPP row_shr 1, 2, 4; score = sqrt(softplus(logit)) as {f32, epoch} (sc1) at ws_moe + r*1024.
    """
    lane, w = tid % 64, tid // 64
    order = (0, 1, 2, 3, 5, 6, 4, 7, 8, 9)
    wt = {}
    for t in order:
        wt[t] = _ld(gate_w, (((j * 80 + w * 10 + t) << 10) + lane * 16) >> 2, 4, CM_NT)
    xb = (r * 10240 + w * 1280 + lane * 16) >> 2
    xa = _ld(xn, xb, 4)
    x2 = fx.Vector.filled(4, 0, Int32)
    if lane < 16:
        x2 = _ld(xn, xb + 256, 4)
    _lds_st(lds, RX_LDS + w * 1280 + lane * 16, [xa[d] for d in range(4)])
    if lane < 16:
        _lds_st(lds, RX_LDS + w * 1280 + 1024 + lane * 16, [x2[d] for d in range(4)])
    xo = RX_LDS + w * 1280 + ((lane >> 2) & 7) * 8
    acc = fx.Vector.filled(4, 0.0, F32)
    for t in range_constexpr(10):
        for hf in range_constexpr(2):
            b = _lds_ld(lds, xo + 128 * t + 64 * hf, 2)
            acc = _mfma4([wt[t][2 * hf], wt[t][2 * hf + 1]], [b[0], b[1]], acc)
    _lds_st(lds, RACC_LDS + tid * 16, [acc[d].bitcast(Int32) for d in range(4)])
    gpu.barrier()
    if w == 0:
        src = lane & 7
        e = (lane >> 3) & 3
        base = RACC_LDS + src * 1024 + ((lane >> 5) & 1) * 512 + e * 4
        v = F32(0.0)
        for k in range_constexpr(8):
            v = v + _lds_ld(lds, base + k * 64).bitcast(F32)
        v = v + _dpp(v, ROW_SHR + 1, bound_ctrl=True)
        v = v + _dpp(v, ROW_SHR + 2, bound_ctrl=True)
        v = v + _dpp(v, ROW_SHR + 4, bound_ctrl=True)
        if (lane & 7) == 7:
            bo.buffer_store(
                fx.Vector.from_elements([_score(v).bitcast(Int32), mep], Int32),
                rsrc(ws_moe),
                (r * 1024 + (j * 8 + (lane >> 3)) * 8) >> 2,
                cache_modifier=CM_DEV,
            )


@traced
def load_x(tid, r, xq, xs, lds):
    """The row's fp8 x (320 x 16 B) and its 160 scale bytes into LDS 0 / 5120; the top-k flag reset."""
    if tid == 0:
        _lds_st(lds, TOPK_FLAG, [Int32(0)])
    if tid < 320:
        v = _ld(xq, (r * 5120 + tid * 16) >> 2, 4)
        _lds_st(lds, X8_LDS + tid * 16, [v[d] for d in range(4)])
    if (tid >= 320) & (tid < 330):
        v = _ld(xs, (r * 160 + (tid - 320) * 16) >> 2, 4)
        _lds_st(lds, X8_LDS + tid * 16, [v[d] for d in range(4)])


def _swiglu(g, u, w=None):
    """TileRT's SwiGLU: gate bf16 min 10, up bf16 clamped to +-10, g / (1 + exp(-g)) * u (x w), bf16."""
    gg = fx.min(_bfr(g), F32(10.0))
    uu = fx.max(fx.min(_bfr(u), F32(10.0)), F32(-10.0))
    a = (gg / (exp_ocml(gg, True) + F32(1.0))) * uu
    if w is not None:
        a = a * w
    return (bf_rne_hi(a) >> 16) & 0xFFFF


@traced
def shared_up(tid, t, xq, xs, sug, sug_s, ws_moe, mep, lds):
    """Shared-expert tile t (16 rows: 8 gate + 8 up) for the five rows (column h: x row h % 5): wave w
    the K range 640w..+640 in 5 scaled 16x16x128 fp8 MFMAs (weights nt, x and its scales read from
    global), waves summed 0 + 1 + .. + 7 through LDS; SwiGLU -> act slot 0 of rows h < 5 (LL, sc1).
    """
    lane, w = tid % 64, tid // 64
    h, q = lane % 16, lane >> 4
    xr = h % 5
    wb = t * SUG_TILE + w * 0x2800 + lane * 16
    sb = ((lane & 8) != 0).select(Int32(1440), Int32(0)) + (t >> 2) * 160 + w * 20 + q
    acc = fx.Vector.filled(4, 0.0, F32)
    for jj in range_constexpr(5):
        a0 = _ld(sug, (wb + jj * 2048) >> 2, 4, CM_NT)
        a1 = _ld(sug, (wb + jj * 2048 + 1024) >> 2, 4, CM_NT)
        sa = (
            Int32(bo.buffer_load(rsrc(sug_s), sb + 4 * jj, vec_width=1, dtype=T.i8)).to(
                Int32
            )
            & 0xFF
        )
        xo = xr * 5120 + w * 640 + jj * 128 + q * 16
        b0 = _ld(xq, xo >> 2, 4)
        b1 = _ld(xq, (xo + 64) >> 2, 4)
        sx = Int32(
            bo.buffer_load(
                rsrc(xs), xr * 160 + w * 20 + 4 * jj + q, vec_width=1, dtype=T.i8
            )
        )
        sx = sx.to(Int32) & 0xFF
        av = fx.Vector.from_elements(
            [a0[d] for d in range(4)] + [a1[d] for d in range(4)], Int32
        )
        bv = fx.Vector.from_elements(
            [b0[d] for d in range(4)] + [b1[d] for d in range(4)], Int32
        )
        acc = mfma_scaled(av, bv, acc, sa, sx)
    if w != 0:
        for i in range_constexpr(4):
            _lds_st(
                lds,
                SG_RED_LDS + (w - 1) * 1024 + i * 256 + lane * 4,
                [acc[i].bitcast(Int32)],
            )
    gpu.barrier()
    if w == 0:
        c = []
        for i in range_constexpr(4):
            s = acc[i]
            for ww in range_constexpr(1, 8):
                s = s + _lds_ld(
                    lds, SG_RED_LDS + (ww - 1) * 1024 + i * 256 + lane * 4
                ).bitcast(F32)
            c.append(s)
        up = [_up32(c[i]) for i in range(4)]
        if (lane < 32) & (h < 5):
            a = [_swiglu(c[i], up[i]) for i in range(4)]
            bo.buffer_store(
                fx.Vector.from_elements(
                    [a[0] | (a[1] << 16), mep, a[2] | (a[3] << 16), mep], Int32
                ),
                rsrc(ws_moe),
                (ACT_OFF + h * ACT_ROW + t * 32 + q * 16) >> 2,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()


@traced
def topk(tid, r, gate_b, ws_moe, mep, lds, kneed):
    """Wave 0: the row's 128 {score, tag} polled (sc1), keys = gate_b + score as sortable signed keys;
    3 rounds of wave max with the smallest expert id on ties; routing weights
    1.5 * s_k / ((((0 + s0) + s1) + s2) + 1e-20) of the raw scores -> LDS (ids, weights). As TileRT's:
    each id goes to LDS as it is found and the flag (= epoch) after round ``kneed``, so that the
    other waves start their weight loads while this wave finishes the rounds and the weights.
    """
    lane = tid % 64
    gb = [_ld(gate_b, lane + 64 * hh).bitcast(F32) for hh in range(2)]
    words = [(r * 1024 + (lane + 64 * hh) * 8) >> 2 for hh in range(2)]
    pv = [_ld(ws_moe, wd, 2, CM_DEV) for wd in words]
    ok = (pv[0][1] == mep) & (pv[1][1] == mep)
    while ~ok:
        spin_pause()
        pv = [_ld(ws_moe, wd, 2, CM_DEV) for wd in words]
        ok = (pv[0][1] == mep) & (pv[1][1] == mep)
    sc = [pv[hh][0].bitcast(F32) for hh in range(2)]
    ub = []
    for hh in range_constexpr(2):
        kb = (gb[hh] + sc[hh]).bitcast(Int32)
        ub.append(kb ^ (kb < 0).select(Int32(0x7FFFFFFF), Int32(0)))
    ids = [lane, lane + 64]
    sel, ss = [], []
    for rnd in range_constexpr(TOPK):
        m = (ub[0] > ub[1]).select(ub[0], ub[1])
        for off in (32, 16, 8, 4, 2, 1):
            o = Int32(bperm(m.bitcast(F32), lane ^ off).bitcast(Int32))
            m = (m > o).select(m, o)
        cand = (ub[1] == m).select(ids[1], Int32(0x7FFFFFFF))
        cand = (ub[0] == m).select(ids[0], cand)
        for off in (32, 16, 8, 4, 2, 1):
            o = Int32(bperm(cand.bitcast(F32), lane ^ off).bitcast(Int32))
            cand = (cand < o).select(cand, o)
        e = uniform(cand)
        val = (e >= 64).select(readlane(sc[1], e & 63), readlane(sc[0], e & 63))
        sel.append(e)
        ss.append(val)
        if lane == 0:
            _lds_st(lds, ID_LDS + 4 * rnd, [e])
            # the id must be in LDS before the flag: without a fence LLVM may sink the id store past it
            # (v3 faulted: the round-3 id store was merged into a later one, the flag went out first)
            _cfence()
            if kneed == rnd + 1:
                _lds_st(lds, TOPK_FLAG, [mep])
        for hh in range_constexpr(2):
            ub[hh] = (ids[hh] == e).select(Int32(SIGN), ub[hh])
    tot = ss[0] + F32(0.0)
    for k in range_constexpr(1, TOPK):
        tot = tot + ss[k]
    tot = tot + f32c(0x1E3CE508)
    if lane == 0:
        for k in range_constexpr(TOPK):
            _lds_st(lds, W_LDS + 4 * k, [((ss[k] / tot) * F32(1.5)).bitcast(Int32)])


@traced
def routed_up(tid, r, k, p, ug, ug_s, ws_moe, mep, lds):
    """Top-k slot k's expert, tiles 4p .. 4p+3 (pair = tid >> 7) of its 576 up/gate rows: wave pair
    K halves kh = (tid >> 6) & 1 in 20 scaled 16x16x128 MFMAs each (MXFP4 weights nt, the per-lane
    scale bytes of TileRT's layout; fp8 x and its scales from LDS); half 1's partials added to half
    0's; SwiGLU x the routing weight -> act slot k+1 (LL, sc1)."""
    lane = tid % 64
    h, q = lane % 16, lane >> 4
    pair, kh = tid >> 7, (tid >> 6) & 1
    tile = 4 * p + pair
    e = uniform(_lds_ld(lds, ID_LDS + 4 * k))
    wb = e * UG_EXPERT + tile * UG_TILE + kh * 20 * 1024 + lane * 16
    sb = e * UGS_EXPERT + (tile * 64 + lane) * 40 + kh * 20
    s4 = _ld(ug_s, sb >> 2, 4)
    s1 = _ld(ug_s, (sb + 16) >> 2)
    sw = [s4[0], s4[1], s4[2], s4[3], s1]
    acc = fx.Vector.filled(4, 0.0, F32)
    for jj in range_constexpr(20):
        av = _ld(ug, (wb + jj * 1024) >> 2, 4, CM_NT)
        xo = X8_LDS + kh * 2560 + jj * 128 + q * 16
        b0 = _lds_ld(lds, xo, 4)
        b1 = _lds_ld(lds, xo + 64, 4)
        sbx = _u8_lds(lds, XS8_LDS + (kh * 20 + jj) * 4 + q)
        bv = fx.Vector.from_elements(
            [b0[d] for d in range(4)] + [b1[d] for d in range(4)], Int32
        )
        acc = mfma_scaled(av, bv, acc, sw[jj // 4], sbx, FP4, FP8, sa_byte=jj % 4)
    if kh == 1:
        for i in range_constexpr(4):
            _lds_st(
                lds,
                RT_RED_LDS + pair * 1024 + i * 256 + lane * 4,
                [acc[i].bitcast(Int32)],
            )
    gpu.barrier()
    if kh == 0:
        c = [
            acc[i]
            + _lds_ld(lds, RT_RED_LDS + pair * 1024 + i * 256 + lane * 4).bitcast(F32)
            for i in range(4)
        ]
        up = [_up32(c[i]) for i in range(4)]
        if (lane < 32) & (h == 0):
            wk = _lds_ld(lds, W_LDS + 4 * k).bitcast(F32)
            a = [_swiglu(c[i], up[i], wk) for i in range(4)]
            bo.buffer_store(
                fx.Vector.from_elements(
                    [a[0] | (a[1] << 16), mep, a[2] | (a[3] << 16), mep], Int32
                ),
                rsrc(ws_moe),
                (ACT_OFF + r * ACT_ROW + (k + 1) * ACT_SLOT + tile * 32 + q * 16) >> 2,
                cache_modifier=CM_DEV,
            )


def _unit(u):
    """A routed act K unit (32 elements) -> (slot 0..2, chunk 0..8): units 0..23 the slots' chunks 0..7,
    24..26 their tails (K 256..287)."""
    lo = u < 24
    return lo.select(u >> 3, u - 24), lo.select(u & 7, Int32(8))


@traced
def down_act(tid, r, ws_moe, mep, lds):
    """Waves 0..3 = act slots 0..3 (shared, routed 0..2): 288 bf16 a slot (LL, lanes < 36, sc1) -> fp8
    with ue8m0 per 32 (clamp +-448) at ACT8_LDS + slot * 288, scales at ASC_LDS + slot * 32.
    """
    lane, w = tid % 64, tid // 64
    if tid < 256:
        on = lane < 36
        aw = (ACT_OFF + r * ACT_ROW + w * ACT_SLOT + fx.min(lane, Int32(35)) * 32) >> 2
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
        for kk in range_constexpr(1, 8):
            m = fx.max(m, abs(ys[kk]))
        m = fx.max(m, _dppf(m, QX1))
        m = fx.max(fx.max(m, _dppf(m, QX2)), f32c(0x38D1B717))
        sv = (m * f32c(0x3B124925)).bitcast(Int32)
        ex = (sv >> 23) - 127 + ((sv & 0x7FFFFF) != 0).select(Int32(1), Int32(0))
        inv = (Int32(0x3F800000) - (ex << 23)).bitcast(F32)
        qs = [fx.max(fx.min(ys[kk] * inv, F32(448.0)), F32(-448.0)) for kk in range(8)]
        d0 = _fp8x2(qs[0], qs[1]) | (_fp8x2(qs[2], qs[3]) << 16)
        d1 = _fp8x2(qs[4], qs[5]) | (_fp8x2(qs[6], qs[7]) << 16)
        sb = (ex + 127) & 0xFF
        s1, s2_, s3_ = _gather_sc(sb.bitcast(F32), lane)
        if on:
            _lds_st(lds, ACT8_LDS + w * 288 + lane * 8, [d0, d1])
        if ((lane & 15) == 0) & (lane < 48):
            _lds_st(
                lds,
                ASC_LDS + w * 32 + (lane >> 2),
                [sb | (s1 << 8) | (s2_ << 16) | (s3_ << 24)],
            )


@traced
def down_fetch(tid, s3, dn, dn_s, sdn, sdn_s, lds):
    """The down weights and scales of ``down_mma``, issued before the act poll (as TileRT's): waves 0..6
    the routed MXFP4 tiles of their unit (nt), waves 7, 0, 1 the shared fp8 tiles of their K step; loads
    a wave does not need go out of bounds (no memory traffic)."""
    lane, w = tid % 64, tid // 64
    h, q = lane % 16, lane >> 4
    r1 = w < 7
    u = 4 * w + q
    uok = u < 27
    us, uc = _unit(uok.select(u, Int32(0)))
    e = _lds_ld(lds, ID_LDS + 4 * us)
    ks = (w + 1) & 7
    r2 = ks < 3
    k1 = ks * 128 + q * 16
    ok1 = r2 & (k1 < 288)
    ok2 = r2 & (k1 < 224)
    blk = ks * 4 + q
    bk = r2 & (blk < 9)
    o1 = (k1 >> 5) * 128 + (k1 & 16)
    o2 = ((k1 + 64) >> 5) * 128 + (k1 & 16)
    ra, rs, sa1, sa2, ss = [], [], [], [], []
    for t in range_constexpr(8):
        col = s3 * 128 + 16 * t + h
        ra.append(_ldm(dn, EXPERTS * DN_EXPERT,
                       (e * DN_EXPERT + s3 * DN_CTA + t * DN_TILE + (h >> 2) * 576 + uc * 64 + (h & 3) * 16) >> 2,
                       r1, 4, CM_NT))  # fmt: skip
        rs.append(Int32(_ldm(dn_s, EXPERTS * DNS_EXPERT, e * DNS_EXPERT + col * 9 + uc, r1, dtype=T.i8)).to(Int32)
                  & 0xFF)  # fmt: skip
        ab = (col >> 2) * 1152 + (col & 3) * 32
        sa1.append(_ldm(sdn, SDN_BYTES, (ab + o1) >> 2, ok1, 4))
        sa2.append(_ldm(sdn, SDN_BYTES, (ab + o2) >> 2, ok2, 4))
        ss.append(
            Int32(_ldm(sdn_s, SDNS_BYTES, (col >> 5) * 9 + blk, bk, dtype=T.i8)).to(
                Int32
            )
            & 0xFF
        )
    return ra, rs, sa1, sa2, ss


@traced
def down_mma(tid, pre, lds):
    """This CTA's 128 hidden columns (8 tiles of 16) from the four act slots. Waves 0..6: one 128-K step
    of the routed slots' concatenated act (units 4w..4w+3 of 32, unit 27 empty), MXFP4 dn (TileRT's
    draft layout: [col/4][chunk 9][col%4][16 B]) x fp8 act, C = 0; waves 7, 0, 1: the shared expert's
    K step (w + 1) & 7 (fp8 sdn [col/4][K/32][col%4][32], 32x32-block scales) onto that. Each wave's
    8 tiles -> DPART_LDS + w * 512. ``pre``: ``down_fetch``'s loads."""
    ra, rs, sa1, sa2, ss = pre
    lane, w = tid % 64, tid // 64
    h, q = lane % 16, lane >> 4
    qa, hb = lane >> 5, (lane >> 4) & 1
    z4 = fx.Vector.filled(4, 0.0, F32)
    acc = [z4 for _ in range(8)]
    if w < 7:
        u = 4 * w + q
        uok = u < 27
        us, uc = _unit(uok.select(u, Int32(0)))
        bl = []
        for du in range_constexpr(2):
            ux = 4 * w + 2 * du + qa
            bok = ux < 27
            bs, bc = _unit(bok.select(ux, Int32(0)))
            bv4 = _lds_ld(lds, ACT8_LDS + (bs + 1) * 288 + bc * 32 + hb * 16, 4)
            bl += [bok.select(bv4[d], Int32(0)) for d in range(4)]
        bv = fx.Vector.from_elements(bl, Int32)
        sb = uok.select(_u8_lds(lds, ASC_LDS + (us + 1) * 32 + uc), Int32(0x7F))
        for t in range_constexpr(8):
            sa = uok.select(rs[t], Int32(0x7F))
            acc[t] = mfma_scaled(ra[t], bv, z4, sa, sb, FP4, FP8)
    ks = (w + 1) & 7
    if ks < 3:
        k1 = ks * 128 + q * 16
        ok1 = k1 < 288
        ok2 = k1 < 224
        b1 = _lds_ld(lds, ACT8_LDS + ok1.select(k1, Int32(0)), 4)
        b2 = _lds_ld(lds, ACT8_LDS + ok2.select(k1 + 64, Int32(0)), 4)
        bv = fx.Vector.from_elements([ok1.select(b1[d], Int32(0)) for d in range(4)]
                                     + [ok2.select(b2[d], Int32(0)) for d in range(4)], Int32)  # fmt: skip
        blk = ks * 4 + q
        bk = blk < 9
        sb = bk.select(_u8_lds(lds, ASC_LDS + bk.select(blk, Int32(0))), Int32(0x7F))
        for t in range_constexpr(8):
            av = fx.Vector.from_elements(
                [sa1[t][d] for d in range(4)] + [sa2[t][d] for d in range(4)], Int32
            )
            sa = bk.select(ss[t], Int32(0x7F))
            acc[t] = mfma_scaled(av, bv, acc[t], sa, sb)
    if h == 0:
        for t in range_constexpr(8):
            _lds_st(
                lds,
                DPART_LDS + w * 512 + (t * 16 + q * 4) * 4,
                [acc[t][i].bitcast(Int32) for i in range(4)],
            )


@traced
def down_sum(tid, valid, lds):
    """tid < 128: column tid's 8 wave partials summed in order from 0, bf16 (0 for an invalid row)."""
    if tid < 128:
        acc = F32(0.0)
        for ww in range_constexpr(8):
            acc = acc + _lds_ld(lds, DPART_LDS + ww * 512 + tid * 4).bitcast(F32)
        yb = valid.select((bf_rne_hi(acc) >> 16) & 0xFFFF, Int32(0))
        nb = Int32(_dppf(yb.bitcast(F32), QX1).bitcast(Int32))
        if (tid & 1) == 0:
            _lds_st(lds, MOE_Y_LDS + tid * 2, [yb | (nb << 16)])


@traced
def moe_tp(tid, cta, r, s3, tab, me, npes, solo, y, mep, lds):
    """One-shot TP all-reduce of the CTA's 128 outputs (LL64: wave p -> the p-th peer, lanes < 40 = 5
    lines of 64 B, the last qword of each the {epoch, epoch} tag; tab_moe slots 0x30000, parity halves
    0x18000, work id cta * 0x180), the ranks summed in order 0..7 in fp32, bf16 -> y[r][128*s3 + col].
    Without peers (npes < 2 or the solo flag) y is the CTA's own sum."""
    lane, wv = tid % 64, tid // 64
    par = (mep & 1) * MOE_PAR
    wid = cta * 0x180
    rq, j = fx.min(lane, Int32(39)) >> 3, lane & 7
    e = rq * 28 + j * 4
    data = (j != 7) & (e < 125)
    tp = (npes >= 2) & (solo == 0)
    if tp:
        if (wv < 7) & (lane < 40):
            dest = wv + (wv >= me).select(1, 0)
            base = load_ptr64(tab, dest)
            slot = me - (me > dest).select(1, 0)
            val = fx.Vector.from_elements([mep, mep], Int32)
            if data:
                val = _lds_ld(lds, MOE_Y_LDS + e * 2, 2)
            off = slot * MOE_SLOT + par + wid + rq * 64 + j * 8
            bo.buffer_store(val, rsrc(base), off >> 2, cache_modifier=CM_SYS)
        gpu.barrier()
        if (wv < 7) & (lane < 40):
            own = load_ptr64(tab, me)
            rb = wv * MOE_SLOT + par + wid + rq * 64
            fl = _ld(own, (rb + 56) >> 2, 2, CM_SYS)
            while ~((fl[0] == mep) & (fl[1] == mep)):
                spin_pause()
                fl = _ld(own, (rb + 56) >> 2, 2, CM_SYS)
            if data:
                d = _ld(own, (rb + j * 8) >> 2, 2, CM_SYS)
                _lds_st(lds, MOE_RX_LDS + wv * 256 + e * 2, [d[0], d[1]])
        gpu.barrier()
    if tid < 128:
        loc = MOE_Y_LDS + tid * 2
        out = _lds_u16(lds, loc)
        if tp:
            acc = F32(0.0)
            for rr in range_constexpr(8):
                rem = MOE_RX_LDS + (rr - (me < rr).select(1, 0)) * 256 + tid * 2
                acc = acc + (_lds_u16(lds, (me == rr).select(loc, rem)) << 16).bitcast(
                    F32
                )
            out = (bf_rne_hi(acc) >> 16) & 0xFFFF
        bo.buffer_store(
            out.to(fx.Int16),
            rsrc(y),
            (r * 5120 + s3 * 128 + tid) * 2,
            offset_is_bytes=True,
        )


@traced
def _entry(tid, cta, lds, xn, xq, xs, gate_w, gate_b, ug, ug_s, sug, sug_s, blk_pos, ws_moe, epoch,
           ws_mhc, mscale, mbase, hcf, nfin, dn, dn_s, sdn, sdn_s, tab, me, npes, y, ids_out, w_out,
           solo):  # fmt: skip
    r = cta // CTAS_ROW
    s3 = cta - r * CTAS_ROW
    mep = Int32(epoch)
    lane, w = tid % 64, tid // 64
    if s3 >= 24:
        router(tid, r, s3 - 24, xn, gate_w, ws_moe, mep, lds)
    pos = [_ld(blk_pos, i) for i in range(ROWS)]
    valid = _ld(blk_pos, r) >= 0
    load_x(tid, r, xq, xs, lds)
    # the free CTAs: s3 >= 27 of a valid row, all 40 of an invalid one, numbered across the rows
    before = Int32(0)
    total = Int32(0)
    for i in range_constexpr(ROWS):
        ci = (pos[i] >= 0).select(Int32(13), Int32(40))
        before = before + (r > i).select(ci, Int32(0))
        total = total + ci
    s31 = before + valid.select(s3 - 27, s3)
    free = (s3 > 26) | ~valid
    gpu.barrier()
    if free:
        fr = total - s31 - 1
        if (w == 7) & (fr < Int32(nfin)):
            mhc_finalize_row(lane, fr, ws_mhc, mscale, mbase, hcf, True)
        t = s31
        while t <= 35:
            shared_up(tid, t, xq, xs, sug, sug_s, ws_moe, mep, lds)
            t = t + total
    if valid:
        # a routed CTA needs its slot's id only (k = s3 / 9): the other waves go on after round k + 1
        kneed = (s3 <= 26).select(s3 // 9 + 1, Int32(TOPK))
        if w == 0:
            topk(tid, r, gate_b, ws_moe, mep, lds, kneed)
        if w != 0:
            while _lds_ld(lds, TOPK_FLAG) != mep:
                spin_pause()
        if s3 <= 26:
            routed_up(tid, r, s3 // 9, s3 % 9, ug, ug_s, ws_moe, mep, lds)
        pre = down_fetch(tid, s3, dn, dn_s, sdn, sdn_s, lds)
        down_act(tid, r, ws_moe, mep, lds)
        gpu.barrier()
        if (s3 == 0) & (tid < 3):
            if ids_out != 0:
                _st_i(ids_out, r * 3 + tid, _lds_ld(lds, ID_LDS + 4 * tid))
            if w_out != 0:
                _st_i(w_out, r * 3 + tid, _lds_ld(lds, W_LDS + 4 * tid))
        down_mma(tid, pre, lds)
    if ~valid:
        if (s3 == 0) & (tid < 3):
            if ids_out != 0:
                _st_i(ids_out, r * 3 + tid, Int32(-1))
            if w_out != 0:
                _st_i(w_out, r * 3 + tid, Int32(0))
    gpu.barrier()
    down_sum(tid, valid, lds)
    gpu.barrier()
    moe_tp(tid, cta, r, s3, tab, me, npes, solo, y, mep, lds)


def build():
    """(kernel symbol, launcher) of the port; the launcher takes the 30 kernarg words."""
    name = kernel_symbol("tk_moe_rx", rows=ROWS, v=VERSION)

    @fx.struct
    class Smem:
        w: fx.Array[fx.Int32, LDS_BYTES // 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def moe_rx(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64,
    ):  # fmt: skip
        lds = fx.SharedAllocator().allocate(Smem).peek().w.ptr
        _entry(fx.thread_idx.x, fx.block_idx.x, lds, a00, a01, a02, a03, a04, a05, a06, a09, a10, a13, a14,
               Int32(a17), a22, a23, a24, a25, Int32(a26), a07, a08, a11, a12, a15, Int32(a16), Int32(a16 >> 32),
               a18, a19, a20, Int32(a29))  # fmt: skip

    @launcher(name)
    def launch(
        a00: Int64, a01: Int64, a02: Int64, a03: Int64, a04: Int64, a05: Int64, a06: Int64, a07: Int64,
        a08: Int64, a09: Int64, a10: Int64, a11: Int64, a12: Int64, a13: Int64, a14: Int64, a15: Int64,
        a16: Int64, a17: Int64, a18: Int64, a19: Int64, a20: Int64, a21: Int64, a22: Int64, a23: Int64,
        a24: Int64, a25: Int64, a26: Int64, a27: Int64, a28: Int64, a29: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        moe_rx(
            a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15,
            a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29,
        ).launch(grid=(GRID,), block=(THREADS,), stream=stream)  # fmt: skip

    return name, launch
