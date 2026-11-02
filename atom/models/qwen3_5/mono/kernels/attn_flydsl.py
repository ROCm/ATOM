# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  (traced stage closures read the loop's values on purpose: the tracer unrolls)
"""Full-attention stages 3-4 (split, merge) under ``ATOM_PA_FLYDSL=1``: aiter's FlyDSL paged decode
(work plan, tile, reduce; aiter 59fae2078 ``kernels/pa_decode*``), bit for bit. A split task is one plan
slot covering all of its request's rows.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from atom.models.qwen3_5.mono.kernels.common import sigmoid, sigmoid_precise
from atom.models.qwen3_5.mono.kernels.ops import i64_of, lds_i64, lds_ld, lds_st
from atom.models.qwen3_5.mono.layout import BS, FLYDSL_MAX_PARTS, flydsl_capacity
from atom.mono.device.mx import mfma_scaled
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    exp,
    fp8_pack4,
    global_load,
    hw_exp2,
    hw_rcp,
    mfma_fp8,
    read_lane_i32,
    traced,
    uniform,
    wave_max,
    xshfl,
)
from atom.mono.device.sync import Addr
from atom.mono.plan.execution import THREADS, WAVES
from atom.mono.plan.trace import Space

TILE = 256  # keys of a FlyDSL tile (its context_partition_size)
MAX_PARTS = FLYDSL_MAX_PARTS  # partitions of a request at most
NEG_INF = float("-inf")
INV_FP8 = 1.0 / 448.0  # f32 0x3B124925
NRM_EPS = 1.0e-8 / 448.0  # f32 0x2DC45764
LOG2E = 1.4426950408889634
LN2 = 1.0 / 1.4426950408889634  # f32 0x3F317218


def flydsl_lds_words(T_SEQ: int, HD: int) -> int:
    """LDS words the split and merge stages use."""
    MTR = 16 * (-(-(8 * T_SEQ) // 16))
    split = (
        MTR * (2 * (HD // 4 + 2) + 1)  # Q bytes, P' bytes, row scales
        + 2 * T_SEQ * (HD // 4 + 1)  # this launch's K / V bytes and scales
        + 2 * HD  # its V bytes by head dim (8 keys a dim)
        + 3
        * TILE
        // 16  # block ids of three tiles (a ring: this, the next, the one after)
        + 4  # the task's context length, last block, block-table row, end tile
        + 2 * TILE  # the tile's k / v scales
        + MTR * (WAVES + 4)  # per-wave row maxima, warp sums
        + 4 * MTR * 4  # the warp halves' exp-sum hand-off (64 lanes a 16-row tile)
    )
    return max(
        split, MAX_PARTS * HD // 2
    )  # merge: a row's partition outputs, bf16 pairs


def plan(context_lens, Q, CAP):
    """aiter's ``pa_decode_plan`` on the first ``Q`` context lengths: per request (lens, tiles, first slot,
    slots)."""
    lens = [uniform(g32(context_lens, i)) for i in range(Q)]
    tiles = [(fx.max(L_, 0) + TILE - 1) // TILE for L_ in lens]
    tot = tiles[0]
    live = (tiles[0] > 0).select(fx.Int32(1), fx.Int32(0))
    for t_ in tiles[1:]:
        tot = tot + t_
        live = live + (t_ > 0).select(fx.Int32(1), fx.Int32(0))
    tot = fx.max(tot, 1)
    rest = CAP - live
    starts, cnts, c_incl, s_ = [], [], fx.Int32(0), fx.Int32(0)
    for t_ in tiles:
        c_prev = c_incl
        c_incl = c_incl + t_
        ne = (t_ > 0).select(fx.Int32(1), fx.Int32(0))
        cnt = fx.min(
            fx.min(ne + (c_incl * rest) // tot - (c_prev * rest) // tot, t_), MAX_PARTS
        )
        starts.append(s_)
        cnts.append(cnt)
        s_ = s_ + cnt
    return lens, tiles, starts, cnts


def pick(i, vals):
    """vals[i] for a runtime (uniform) i."""
    out = vals[0]
    for j in range(1, len(vals)):
        out = (i == j).select(vals[j], out)
    return out


def g32(base, i):
    """i32 ``i`` of the global buffer at ``base``."""
    return fx.Int32(global_load(base + fx.Int64(i) * 4))


@traced
def flydsl_split_merge(ctx):
    """Stages ``split`` (a task a plan slot) and ``merge`` (a task a row, head)."""
    G = ctx["G"]
    H = ctx["H"]
    HD = ctx["HD"]
    OUT_K = ctx["OUT_K"]
    QG = ctx["QG"]
    QKF_O = ctx["QKF_O"]
    S = ctx["S"]
    T_SEQ = ctx["T_SEQ"]
    bt_stride = ctx["bt_stride"]
    k_cache = ctx["k_cache"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    poll = ctx["poll"]
    put2 = ctx["put2"]
    put_bf = ctx["put_bf"]
    start = ctx["start"]
    tid = ctx["tid"]
    v_cache = ctx["v_cache"]
    wave = ctx["wave"]
    xs = ctx["xs"]
    # by 64-bit address, not buffer descriptors (SGPR pressure)
    block_tables, context_lens = ctx["block_tables"], ctx["context_lens"]
    k_scale, v_scale = ctx["k_scale"], ctx["v_scale"]
    sig = sigmoid if const_expr(QKF_O) else sigmoid_precise

    Q = S // T_SEQ  # requests
    R = H * T_SEQ  # a request's query rows (token-major: row = qi * H + head)
    MT = -(-R // 16)  # 16-row MFMA tiles
    MTR = 16 * MT
    CAP = flydsl_capacity(Q)
    KVB = HD * BS
    assert H == 8 and HD == 256 and BS == 16 and WAVES == 8 and THREADS == 512
    # FlyDSL's d256 pipe: K128 scaled MFMA up to 32 rows, K32 past it (same operand bytes)
    K128 = R <= 32
    # LDS layout (words)
    SQ = HD // 4 + 2
    XQ8 = 0
    XP8 = XQ8 + MTR * SQ
    XQS = XP8 + MTR * SQ
    XKN = XQS + MTR
    XVN = XKN + T_SEQ * HD // 4
    XSN = XVN + T_SEQ * HD // 4
    XKS = XSN + 2 * T_SEQ
    XVS = XKS + TILE
    XMX = XVS + TILE
    XLX = XMX + WAVES * MTR
    XW = XLX + 4 * MT * 64
    XVT = XW + 4 * MTR
    XBT = XVT + 2 * HD  # block ids of three tiles, by tile % 3
    # the task's scalars, re-read in the tile loop (VGPRs, not SGPRs)
    XSC = XBT + 3 * TILE // BS
    # T_SEQ <= 7: the V splice's arithmetic shift needs byte 7 of vt64 zero
    assert XSC + 4 <= flydsl_lds_words(T_SEQ, HD) and T_SEQ <= 7
    LOG2E_SCALE = ctx["SCALE"] * LOG2E
    g = lane // 16
    n = lane % 16
    fw = wave // 2  # FlyDSL's warp
    half = wave % 2  # which half of the warp's chunks

    def mfma_chain(a_packs, b_packs, acc):
        """acc += A x B over 8 FP8 packs, as FlyDSL issues it."""
        if const_expr(K128):
            for inst in range_constexpr(2):
                a = fx.Vector.from_elements(
                    a_packs[4 * inst : 4 * inst + 4], dtype=fx.Int64
                ).bitcast(fx.Int32)
                b = fx.Vector.from_elements(
                    b_packs[4 * inst : 4 * inst + 4], dtype=fx.Int64
                ).bitcast(fx.Int32)
                one = fx.Int32(0x7F7F7F7F)  # E8M0 1.0, every byte
                acc = mfma_scaled(a, b, acc, one, one)
        else:
            for p in range_constexpr(8):
                acc = mfma_fp8(a_packs[p], b_packs[p], acc)
        return acc

    def lds_vec4(idx):
        return fx.Vector(
            fx.ptr_load(xs + idx, result_type=fx.Vector.make_type(4, fx.Int32))
        )

    def kv_vec4(pool, blk, byte):
        """Four words at ``byte`` of cache block ``blk`` of ``pool``."""
        return fx.Vector(global_load(pool + fx.Int64(blk) * KVB + fx.Int64(byte), 4))

    # ================================ 3. split: a task a live plan slot (each task replans: SGPRs)
    _, _, starts, cnts = plan(context_lens, Q, CAP)
    n_live = starts[Q - 1] + cnts[Q - 1]
    for tt in range(start("split"), n_live, G):
        slot = fx.Int32(tt)
        gpu.barrier()
        lens, tiles, starts, cnts = plan(context_lens, Q, CAP)
        req = fx.Int32(0)
        for i in range_constexpr(1, Q):
            req = (slot >= starts[i]).select(fx.Int32(i), req)
        L = pick(req, lens)
        k_ = slot - pick(req, starts)
        cnt = fx.max(pick(req, cnts), 1)
        nt = pick(req, tiles)
        tb = (k_ * nt) // cnt
        te = ((k_ + 1) * nt) // cnt
        s0 = req * T_SEQ  # the request's first row
        pos0 = L - T_SEQ  # this launch's keys: pos0 .. L - 1
        last_blk = fx.max(fx.min((L - 1) // BS, bt_stride - 1), 0)

        # Q -> FP8 with a per-row scale (padded rows zero); this launch's K / V codes -> bytes
        NKW = T_SEQ * 2 * HD // 4
        NKR = -(-NKW // THREADS)
        if tid == 0:
            for i_, v_ in enumerate((L, last_blk, s0 * bt_stride, te)):
                lds_st(xs, XSC + i_, fx.Int32(v_).bitcast(fx.Float32))
        rows = [rr * WAVES + wave for rr in range(MT * 2)]
        got = poll(
            [
                (
                    mb("q"),
                    ((s0 + fx.min(row // H, T_SEQ - 1)) * H + row % H) * HD
                    + 4 * lane
                    + 2 * e,
                    2,
                )
                for row in rows
                for e in range(2)
            ]
        )
        for rr in range_constexpr(MT * 2):
            row = rows[rr]
            inrow = row < R
            qv = [
                inrow.select(
                    got[2 * rr + e // 2][e % 2].bitcast(fx.Float32), fx.Float32(0.0)
                )
                for e in range(4)
            ]
            amax = wave_max(
                fx.max(
                    fx.max(fmath.absf(qv[0]), fmath.absf(qv[1])),
                    fx.max(fmath.absf(qv[2]), fmath.absf(qv[3])),
                )
            )
            qs = amax * INV_FP8
            inv = hw_rcp(fx.max(qs, fx.Float32(1.0e-20)))
            lds_st(
                xs,
                XQ8 + row * SQ + lane,
                fp8_pack4(*[x * inv for x in qv]).bitcast(fx.Float32),
            )
            if lane == 0:
                lds_st(xs, XQS + row, inrow.select(qs * LOG2E_SCALE, fx.Float32(0.0)))
        wks = [fx.min(tid + rnd * THREADS, NKW - 1) for rnd in range(NKR)]
        kvb = [
            (s0 + wk // (2 * HD // 4)) * (2 * HD + 2)
            + ((wk // (HD // 4)) % 2) * HD
            + (wk % (HD // 4)) * 4
            for wk in wks
        ]
        gk = poll([(mb("kvnew"), b_ + 2 * i, 2) for b_ in kvb for i in range(2)])
        for rnd in range_constexpr(NKR):
            wk = wks[rnd]
            w = fp8_pack4(
                *[
                    gk[2 * rnd + i][e].bitcast(fx.Float32)
                    for i in range(2)
                    for e in range(2)
                ]
            )
            if tid + rnd * THREADS < NKW:
                i_ = wk // (2 * HD // 4)
                kv_ = (wk // (HD // 4)) % 2
                lds_st(
                    xs,
                    XKN + kv_ * (T_SEQ * HD // 4) + i_ * (HD // 4) + wk % (HD // 4),
                    w.bitcast(fx.Float32),
                )
        if tid < 2 * T_SEQ:
            sc2 = poll(
                [(mb("kvnew"), (s0 + tid // 2) * (2 * HD + 2) + 2 * HD + tid % 2, 1)]
            )
            lds_st(xs, XSN + tid, sc2[0][0].bitcast(fx.Float32))
        gpu.barrier()
        # this launch's V bytes by head dim, for the PV operand splice
        if tid < HD:
            vt = [fx.Int32(0), fx.Int32(0)]
            for i_ in range_constexpr(T_SEQ):
                nw = lds_ld(xs, XVN + i_ * (HD // 4) + tid // 4).bitcast(fx.Int32)
                vt[i_ // 4] = vt[i_ // 4] | (
                    ((nw >> (tid % 4) * 8) & 0xFF) << (i_ % 4) * 8
                )
            lds_st(xs, XVT + 2 * tid, vt[0].bitcast(fx.Float32))
            lds_st(xs, XVT + 2 * tid + 1, vt[1].bitcast(fx.Float32))
        gpu.barrier()

        NB = TILE // BS  # cache blocks a tile
        NS = 2 * MT + 2 * MT * 4  # carried softmax state: m, l, o

        def sc_(i):
            """Task scalar ``i``: context length, last block, block-table row, end tile."""
            return lds_ld(xs, XSC + i).bitcast(fx.Int32)

        def ids_of(tl):
            """Tile ``tl``'s block ids, a thread a block."""
            return g32(
                block_tables,
                sc_(2) + fx.min(fx.max(tl * NB + tid % NB, 0), sc_(1)),
            )

        def bid(tl, i):
            """Block ``i`` of tile ``tl`` (3-slot ring)."""
            return lds_ld(xs, XBT + (tl % 3) * NB + i).bitcast(fx.Int32)

        def load_k(tl):
            """This wave's two 16-key K chunks."""
            kw = []
            for c in range_constexpr(2):
                blk = uniform(bid(tl, 2 * wave + c))
                for q in range_constexpr(4):
                    w4 = kv_vec4(k_cache, blk, ((4 * q + g) * BS + n) * 16)
                    kw += [w4[e] for e in range(4)]
            return kw

        def load_v(tl):
            """This wave's two 16-dim V groups."""
            vw = []
            for j in range_constexpr(2):
                for kq in range_constexpr(4):
                    w4 = kv_vec4(
                        v_cache, bid(tl, 4 * g + kq), ((2 * wave + j) * 16 + n) * 16
                    )
                    vw += [w4[e] for e in range(4)]
            return vw

        def load_s(tl):
            """k / v scale of key ``tid % TILE``."""
            blk = bid(tl, (tid % TILE) // BS)
            return [
                g32(k_scale, blk * BS + tid % BS).bitcast(fx.Float32),
                g32(v_scale, blk * BS + tid % BS).bitcast(fx.Float32),
            ]

        def tile_body(t, carry, patch):
            """One tile; carry = (m, l, o, this tile's prefetched K, V, scales). ``patch``: the tile may hold
            this launch's keys or keys past the context."""
            m_p = carry[:MT]
            l_p = carry[MT : 2 * MT]
            o_p = carry[2 * MT : NS]
            kw = carry[NS : NS + 32]
            vw = carry[NS + 32 : NS + 64]
            ks_, vs_ = carry[NS + 64], carry[NS + 65]
            kbase = t * TILE
            L = sc_(0)
            pos0 = L - T_SEQ
            tn = fx.min(t + 1, sc_(3) - 1)
            idn = ids_of(t + 2)  # into the ring at the tile's end
            if const_expr(patch):
                # K: this launch's keys from LDS
                for c in range_constexpr(2):
                    key = kbase + 32 * wave + 16 * c + n
                    new = (key >= pos0) & (key < L)
                    for q in range_constexpr(4):
                        kn = lds_vec4(
                            XKN
                            + fx.min(fx.max(key - pos0, 0), T_SEQ - 1) * (HD // 4)
                            + (4 * q + g) * 4
                        )
                        for e in range_constexpr(4):
                            kw[(c * 4 + q) * 4 + e] = new.select(
                                kn[e], fx.Int32(kw[(c * 4 + q) * 4 + e])
                            )
                # V: keys < pos0 cached, pos0 .. L - 1 from vt64, >= L zero
                for j in range_constexpr(2):
                    vt64 = lds_i64(xs, XVT + 2 * ((2 * wave + j) * 16 + n))
                    for kq in range_constexpr(4):
                        for e in range_constexpr(4):
                            key0 = kbase + 64 * g + 16 * kq + 4 * e
                            nc = fx.min(fx.max(pos0 - key0, 0), 4)
                            keep = (nc >= 4).select(
                                fx.Int32(-1), (fx.Int32(1) << nc * 8) - 1
                            )
                            rel = key0 - pos0
                            up = fx.Int32(fx.min(fx.max(rel, 0), 7) * 8)
                            dn = fx.Int32(fx.min(fx.max(-rel, 0), 3) * 8)
                            sh = (vt64 >> fx.Int64(up)) << fx.Int64(dn)
                            nw = (rel >= 8).select(
                                fx.Int32(0), fx.Int32(sh & 0xFFFFFFFF)
                            )
                            i_ = (j * 4 + kq) * 4 + e
                            vw[i_] = (fx.Int32(vw[i_]) & keep) | (nw & ~keep)
                key = kbase + tid % TILE
                i_ = fx.min(fx.max(key - pos0, 0), T_SEQ - 1)
                ks_ = (key >= pos0).select(lds_ld(xs, XSN + 2 * i_), ks_)
                vs_ = (key >= pos0).select(lds_ld(xs, XSN + 2 * i_ + 1), vs_)
                ks_ = (key < L).select(ks_, fx.Float32(0.0))
                vs_ = (key < L).select(vs_, fx.Float32(0.0))
            if tid < TILE:
                lds_st(xs, XKS + tid, ks_)
                lds_st(xs, XVS + tid, vs_)
            gpu.barrier()
            # --- QK, masked scores, row maxima
            vmax = wave_max(
                fx.max(
                    fx.max(lds_ld(xs, XVS + 4 * lane), lds_ld(xs, XVS + 4 * lane + 1)),
                    fx.max(
                        lds_ld(xs, XVS + 4 * lane + 2), lds_ld(xs, XVS + 4 * lane + 3)
                    ),
                )
            )
            sc = [[None] * MT for _ in range(2)]
            for m in range_constexpr(MT):
                row = 16 * m + n
                qb = [
                    lds_i64(
                        xs,
                        XQ8 + row * SQ + ((4 * (p // 2) + g) * 16 + 8 * (p % 2)) // 4,
                    )
                    for p in range(8)
                ]
                scale_r = lds_ld(xs, XQS + row)
                qi = row // H
                bound = L - (T_SEQ - 1 - qi)
                for c in range_constexpr(2):
                    kq_ = [
                        kw[(c * 4 + p // 2) * 4 + 2 * (p % 2) + e]
                        for p in range(8)
                        for e in range(2)
                    ]
                    ka = [i64_of(kq_[2 * p], kq_[2 * p + 1]) for p in range(8)]
                    acc = mfma_chain(ka, qb, fx.Vector.filled(4, 0.0, fx.Float32))
                    vals = []
                    for r in range_constexpr(4):
                        kk = 32 * wave + 16 * c + 4 * g + r
                        s_ = (acc[r] * lds_ld(xs, XKS + kk)) * scale_r
                        ok = (row < R) & (kbase + kk < bound)
                        vals.append(ok.select(s_, fx.Float32(NEG_INF)))
                    sc[c][m] = vals
                mx = fx.max(
                    fx.max(
                        fx.max(sc[0][m][0], sc[0][m][1]),
                        fx.max(sc[0][m][2], sc[0][m][3]),
                    ),
                    fx.max(
                        fx.max(sc[1][m][0], sc[1][m][1]),
                        fx.max(sc[1][m][2], sc[1][m][3]),
                    ),
                )
                mx = fx.max(mx, xshfl(mx, 16))
                mx = fx.max(mx, xshfl(mx, 32))
                if g == 0:
                    lds_st(xs, XMX + wave * MTR + row, mx)
            kw_n = load_k(tn)  # the next tile's K, in K's registers
            gpu.barrier()
            # --- tile max, P, P', exp sum (first half)
            nrm = hw_rcp(fmath.fma(vmax, fx.Float32(INV_FP8), fx.Float32(NRM_EPS)))
            vms = vmax * INV_FP8
            m_n, corr, rs = [], [], []
            for m in range_constexpr(MT):
                row = 16 * m + n
                tmax = lds_ld(xs, XMX + row)
                for w_ in range_constexpr(1, WAVES):
                    tmax = fx.max(tmax, lds_ld(xs, XMX + w_ * MTR + row))
                mn = fx.max(fx.Float32(m_p[m]), tmax)
                safe = (mn > NEG_INF).select(mn, fx.Float32(0.0))
                corr.append(hw_exp2(fx.Float32(m_p[m]) - safe))
                m_n.append(mn)
                rc = []
                for c in range_constexpr(2):
                    ps = [hw_exp2(sc[c][m][r] - safe) for r in range(4)]
                    kk0 = 32 * wave + 16 * c + 4 * g
                    pq = [(ps[r] * lds_ld(xs, XVS + kk0 + r)) * nrm for r in range(4)]
                    lds_st(
                        xs,
                        XP8 + row * SQ + kk0 // 4,
                        fp8_pack4(*pq).bitcast(fx.Float32),
                    )
                    rc.append(((ps[0] + ps[1]) + ps[2]) + ps[3])
                rs.append(rc)
                if half == 0:
                    lds_st(xs, XLX + (fw * MT + m) * 64 + lane, rc[0] + rc[1])
            gpu.barrier()
            # --- exp sum (second half), PV
            for m in range_constexpr(MT):
                if half == 1:
                    ls = (lds_ld(xs, XLX + (fw * MT + m) * 64 + lane) + rs[m][0]) + rs[
                        m
                    ][1]
                    ws_ = ls + xshfl(ls, 16)
                    ws_ = ws_ + xshfl(ws_, 32)
                    if g == 0:
                        lds_st(xs, XW + fw * MTR + 16 * m + n, ws_)
            o_n = []
            for j in range_constexpr(2):
                vq = [
                    vw[(j * 4 + i // 2) * 4 + 2 * (i % 2) + e]
                    for i in range(8)
                    for e in range(2)
                ]
                va = [i64_of(vq[2 * i], vq[2 * i + 1]) for i in range(8)]
                for m in range_constexpr(MT):
                    row = 16 * m + n
                    pb = [
                        lds_i64(xs, XP8 + row * SQ + 16 * g + 2 * i) for i in range(8)
                    ]
                    pv = mfma_chain(va, pb, fx.Vector.filled(4, 0.0, fx.Float32))
                    for r in range_constexpr(4):
                        o_n.append(
                            fmath.fma(
                                fx.Float32(o_p[(j * MT + m) * 4 + r]),
                                corr[m],
                                pv[r] * vms,
                            )
                        )
            vw_n = load_v(tn)  # the next tile's V and scales
            s_n = load_s(tn)
            if tid < NB:
                lds_st(xs, XBT + ((t + 2) % 3) * NB + tid, idn.bitcast(fx.Float32))
            gpu.barrier()
            l_n = []
            for m in range_constexpr(MT):
                row = 16 * m + n
                gs = fx.Float32(0.0)
                for w_ in range_constexpr(4):
                    gs = gs + lds_ld(xs, XW + w_ * MTR + row)
                l_n.append(fmath.fma(fx.Float32(l_p[m]), corr[m], gs))
            return m_n + l_n + o_n + kw_n + vw_n + s_n

        def wrap(carry):
            return (
                [fx.Float32(x) for x in carry[:NS]]
                + [fx.Int32(x) for x in carry[NS : NS + 64]]
                + [fx.Float32(x) for x in carry[NS + 64 :]]
            )

        if tid < 2 * NB:
            tl = tb + tid // NB
            lds_st(xs, XBT + (tl % 3) * NB + tid % NB, ids_of(tl).bitcast(fx.Float32))
        gpu.barrier()
        init = (
            [fx.Float32(NEG_INF)] * MT
            + [fx.Float32(0.0)] * MT
            + [fx.Float32(0.0)] * (2 * MT * 4)
        )
        init += load_k(tb) + load_v(tb) + load_s(tb)
        # tiles before this launch's keys, then the patched rest
        ta = fx.max(fx.min(pos0 // TILE, te), tb)
        for ti, carry in range(tb, ta, fx.Int32(1), init=init):
            res = yield tile_body(fx.Int32(ti), wrap(carry), False)
        mid = wrap(res)
        for ti, carry in range(ta, te, fx.Int32(1), init=mid):
            res2 = yield tile_body(fx.Int32(ti), wrap(carry), True)
        fin = [fx.Float32(x) for x in res2[:NS]]
        for m in range_constexpr(MT):
            row = 16 * m + n
            lm = fin[MT + m]
            inv_l = hw_rcp((lm > 0.0).select(lm, fx.Float32(1.0)))
            for j in range_constexpr(2):
                dim = (2 * wave + j) * 16 + 4 * g
                o4 = [fin[2 * MT + (j * MT + m) * 4 + r] * inv_l for r in range(4)]
                if row < R:
                    put_bf(mb("sp_out"), (slot * R + row) * HD + dim, o4)
            if (wave == 0) & (g == 0) & (row < R):
                put2(mb("sp_ml"), (slot * R + row) * 2, fin[m] * LN2, lm)

    # ================================ 4. merge (aiter's reduce), output gate -> gated
    for tt in range(start("merge"), S * H, G):
        tt = fx.Int32(tt)
        s = tt // H
        hh = tt % H
        gpu.barrier()
        req = s // T_SEQ
        row = (s % T_SEQ) * H + hh
        _, _, starts, cnts = plan(context_lens, Q, CAP)
        first = pick(req, starts)
        cnt = pick(req, cnts)
        # a request without partitions (padded) polls a pair every launch writes instead
        pin = lane < cnt
        has = cnt > 0
        src = Addr("sp_ml", Space.SCRATCH, has.select(mb("sp_ml").value, mb("q").value))
        at = has.select(
            ((first + fx.min(lane, fx.max(cnt - 1, 0))) * R + row) * 2, fx.Int32(0)
        )
        ml = poll([(src, at, 2)])[0]
        mj = pin.select(ml[0].bitcast(fx.Float32), fx.Float32(NEG_INF))
        lj = pin.select(ml[1].bitcast(fx.Float32), fx.Float32(0.0))
        gmax = wave_max(mj)
        safe = (gmax > NEG_INF).select(gmax, fx.Float32(0.0))
        x = lj * (mj > NEG_INF).select(exp(mj - safe), fx.Float32(0.0))
        tot = x
        for o_ in (32, 16, 8, 4, 2, 1):
            tot = tot + xshfl(tot, o_)
        tot0 = read_lane_i32(tot.bitcast(fx.Int32), 0).bitcast(fx.Float32)
        wj = x * hw_rcp((tot0 > 0.0).select(tot0, fx.Float32(1.0)))
        if wave == 0:
            lds_st(misc, lane, pin.select(wj, fx.Float32(0.0)))  # w = 0 past cnt
        gpu.barrier()
        # partition outputs -> LDS (one poll), then summed in order; past cnt w = 0, as aiter pads
        tq = tid % (HD // 2)
        pq4 = tid // (HD // 2)
        at = [
            has.select(
                ((first + fx.min(pq4 * 16 + i, fx.max(cnt - 1, 0))) * R + row) * HD // 2
                + tq,
                fx.Int32(0),
            )
            for i in range(MAX_PARTS // 4)
        ]
        src = Addr(
            "sp_out", Space.SCRATCH, has.select(mb("sp_out").value, mb("q").value)
        )
        parts = poll([(src, a_, 1) for a_ in at], batch=MAX_PARTS // 4)
        for i in range_constexpr(MAX_PARTS // 4):
            word = has.select(
                parts[i][0], fx.Int32(0)
            )  # padded: not q's words (inf / nan halves)
            lds_st(xs, (pq4 * 16 + i) * (HD // 2) + tq, word.bitcast(fx.Float32))
        gpu.barrier()
        o0 = fx.Float32(0.0)
        o1 = fx.Float32(0.0)
        for p in range_constexpr(MAX_PARTS):
            v0, v1 = bf2_f32(lds_ld(xs, p * (HD // 2) + tq).bitcast(fx.Int32))
            wp = lds_ld(misc, p)
            o0 = o0 + v0 * wp
            o1 = o1 + v1 * wp
        gates = poll([(mb("qg"), s * QG + hh * 2 * HD + HD + 2 * tq, 2)])[0]
        sg = [sig(gates[e].bitcast(fx.Float32)) for e in range(2)]
        if tid < HD // 2:
            if const_expr(QKF_O):
                put2(
                    mb("gated"),
                    s * OUT_K + hh * HD + 2 * tq,
                    sg[0] * bf16_round(o0),
                    sg[1] * bf16_round(o1),
                )
            else:
                put_bf(
                    mb("gated"),
                    s * OUT_K + hh * HD + 2 * tq,
                    [
                        bf16_round(bf16_round(sg[e]) * bf16_round(o))
                        for e, o in enumerate((o0, o1))
                    ],
                )
