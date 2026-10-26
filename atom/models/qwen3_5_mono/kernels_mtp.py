# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5-35B-A3B fused decoder layer for MTP verify steps: R requests of SPEC tokens
each (the anchor and its drafts, consecutive positions of one sequence), T = SPEC R
tokens, up to 16 G. One persistent gfx1250 FlyDSL kernel a layer, as kernels.py; the
tokens travel in groups of 16, the columns of one WMMA B tile.

What changes from one-token-a-request decode (kernels.py):

* GDN: a request's SPEC tokens update one state in order. The conv state is a window
  of 3 + SPEC - 1 rows read from row ``num_accepted - 1`` (ATOM's rollback); the
  recurrent state is read from slot ``spec_state_indices[r, num_accepted - 1]`` and
  the state after token j stored to ``spec_state_indices[r, j]``.
* Attention: CTA (request, chunk); the B columns are the request's SPEC tokens x 4 q
  heads (16: a full tile), so a pass over the cache serves every token. Token j sees
  positions <= L - SPEC + j; the SPEC new K / V are patched into their owners'
  fragments.
* Projections: weights stationary in registers, a group of 16 tokens at a time
  through LDS. N1 / N2 a wave a token.
* MoE: M3 is token-major (CTA b: token b % T, its share of the 128 (expert, tile)
  tasks), so a CTA quantizes one token's activations; M4 a wave a token (its 8 experts
  accumulated in registers), the shared expert with the tokens as columns.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, ptrtoint, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen3_5_mono.config import (
    ATT_D,
    ATT_QH,
    ATT_QKVG,
    ATT_SCALE,
    BLOCKS,
    EPS,
    GDN_D,
    GDN_KH,
    GDN_QKV,
    GDN_QKVZ,
    GDN_VH,
    GDN_Z,
    HIDDEN,
    INTER,
    N_EXPERTS,
    ROUTER_ROWS,
    SH_INTER,
    SPEC,
    THREADS,
    TOP_K,
    TP,
    WAVE,
    WAVES,
)
from atom.models.qwen3_5_mono.layout import (
    A1_TILES,
    G1_TILES,
    M1_RT_TILES,
    M1_TILES,
    OUT_TILES,
    STAMPS,
    ACT_SLOTS,
    ACTQ_PAIRS,
    LOGIT_STRIDE,
    MBOX,
    SCRATCH,
    arg_index,
    pair_index,
)
from atom.models.qwen3_5_mono.prims import (
    AGENT,
    CM_DEV,
    CM_SYS,
    poll1,
    poll2,
    poll_words_sys,
    put1,
    put2,
    bf16_at,
    bf16_pack,
    bf16_unpack,
    bf16r,
    counter_base,
    dot,
    lds_addr,
    lds_barrier,
    lds_ld,
    lds_ld4,
    lds_ld4w,
    lds_ld8w,
    lds_ldi,
    lds_st,
    lds_st4,
    lds_sti,
    ld_dw,
    ld_f32,
    prefetch_line,
    uniform,
    rsrc_at,
    sigmoid,
    silu,
    softplus,
    st_dw,
    st_f32,
    wave_max,
    wave_sum,
    wmma_bf16,
    wmma_fp4_fp8,
    mx8_quant,
    abs_f,
    lds_atomic_or,
    popc,
    col8_sum,
    lane_val,
)

NEG = -1.0e30
W13_EXPERT_BYTES = 512 * 1024  # [512 rows (gate / up interleaved, padded), 1024 B]
W13S_EXPERT_BYTES = 16 * 2048
W2_EXPERT_BYTES = 2048 * 128
W2S_EXPERT_BYTES = 64 * 256
CONV_ROWS = 3 + SPEC - 1  # the conv state window (ATOM: kernel - 1 + drafts)
TG = 16  # tokens a group (WMMA B columns)
RG = TG // SPEC  # requests a group
M3_CHUNK = 8  # M3 tasks a CTA in flight (registers)
# LDS row strides (dwords), padded off powers of two: a WMMA fragment read hits 16 rows
# (tokens / heads) at once, which at a 4 KB stride all land in the same banks
XS = HIDDEN // 2 + 4  # xb: a token's 2048 bf16
GS = 512 + 4  # gb: a token's 1024 bf16 (out_proj / o_proj input)
SHS = SH_INTER // 2 + 4  # shb: a token's shared activations
QS = ATT_D // 2 + 4  # qb: a column's q head
PS = 32 + 4  # pb: a column's 64 P values
WRS = 16 + 4  # wr: a (wave, column)'s 16 rows
SPS = 16 + 4  # sp: a (block, K step, column)'s 16 positions
_ABL = __import__("os").environ.get("MONO_ABL", "")  # experiments: stages switched off


@functools.lru_cache(maxsize=None)
def _lds_struct(groups: int):
    tmax = TG * groups

    @fx.struct
    class _Lds:
        red: fx.Array[fx.Float32, 256, 16]
        nw: fx.Array[fx.Float32, HIDDEN, 16]  # ln1_w | ln2_w, packed bf16 (N1 / N2 read them from here)
        xb: fx.Array[fx.Float32, TG * XS, 16]  # a group's x, packed bf16 (WMMA B)
        gb: fx.Array[fx.Float32, TG * GS, 16]  # a group's out_proj / o_proj input
        wr: fx.Array[fx.Float32, WAVES * TG * WRS, 16]  # a tile's per-wave partials
        y16: fx.Array[fx.Float32, 256, 16]
        part: fx.Array[fx.Float32, tmax * 16, 16]  # the CTA's 16 dims of every token
        rcv: fx.Array[fx.Float32, TP * tmax * 16, 16]
        sel_e: fx.Array[fx.Float32, tmax * TOP_K, 16]  # i32 bits
        sel_w: fx.Array[fx.Float32, tmax * TOP_K, 16]
        sg: fx.Array[fx.Float32, tmax, 16]
        slot: fx.Array[fx.Float32, tmax * TOP_K * 16, 16]  # M4: a slot's 16 rows (expert-major)
        cnt: fx.Array[fx.Float32, 2 * N_EXPERTS, 16]  # the sort (i32 bits): an expert's 64-token bitmap
        off: fx.Array[fx.Float32, N_EXPERTS, 16]  # its first in the list
        pos: fx.Array[fx.Float32, tmax * TOP_K, 16]  # a slot's rank among its expert's
        list_: fx.Array[fx.Float32, tmax * TOP_K, 16]  # the slots, expert-major
        grp: fx.Array[fx.Float32, tmax * TOP_K // 16 + N_EXPERTS, 16]  # groups: e | start << 8 | count << 20
        ngrp: fx.Array[fx.Float32, 16, 16]
        shb: fx.Array[fx.Float32, tmax * SHS, 16]  # shared activations, bf16
        qb: fx.Array[fx.Float32, 16 * QS, 16]  # the 16 B columns' q, packed bf16
        knb: fx.Array[fx.Float32, SPEC * ATT_D // 2, 16]  # the new k, packed bf16
        vn: fx.Array[fx.Float32, SPEC * ATT_D, 16]  # the new v
        sp: fx.Array[fx.Float32, 4 * 8 * 16 * SPS, 16]  # QK partials [block][K step][column][16]
        pb: fx.Array[fx.Float32, 16 * PS, 16]  # P, packed bf16 [column][64 positions]
        alph: fx.Array[fx.Float32, 16, 16]

    return _Lds


def _st_bf16(res, elem, v):
    """Store one bf16 element at bf16 offset ``elem``."""
    bo.buffer_store(fx.Float32(v).to(fx.BFloat16).ir_value(), res, fx.Int32(elem))


def _pick(cond, a, b):
    return a if cond else b


def build_layer_mtp(kind: str, groups: int, max_ctx: int = 16384, timing: bool = False, vmajor: bool = False, variant: str = ""):
    """The launcher of the ``kind`` ("gdn" | "att") MTP layer kernel for up to 16
    ``groups`` tokens: ``launch(*table, *peers, rank, nreq, cnt_gdn, cnt_att, positions,
    m0, m1, m2, bt_stride, stream=)``, ``nreq`` requests (SPEC tokens each, request-major).
    GDN: m0 = spec_state_indices (int32 [nreq, SPEC], -1: pad), m1 = num_accepted_tokens
    (int32 [nreq]), m2 unused. Attention: m0 = block table (rows ``bt_stride`` apart),
    m1 = context lens (int32 [nreq], the new tokens included), m2 = slot mapping (int64
    [SPEC nreq])."""
    assert kind in ("gdn", "att") and groups in (1, 2, 4)
    G = groups
    TMAX = TG * G
    RMAX = TMAX // SPEC
    A = arg_index(kind)
    MAX_BLK = (max_ctx + 15) // 16
    CH_LANE = 4  # A3: chunks a lane (NCH <= 128)
    M3_MP = TMAX // 2  # M3 tasks a CTA at most: 128 / floor(256 / T)
    Lds = _lds_struct(G)
    # attention hand-offs (A1 -> A2, A3 -> A4): fine-grained LL polls up to 16 tokens (a
    # CTA needs one request's rows), plain stores behind per-CTA flags past (every CTA
    # needs most of them)
    PLAIN_A = G > 1
    build_key = (kind, G, max_ctx, THREADS, timing, vmajor, variant)

    @flyc.kernel(name=f"qwen35_mono_mtp_{kind}_g{G}_c{max_ctx}{'_vm' if vmajor else ''}{'_timing' if timing else ''}{variant}")
    def kernel(
        a00: fx.Int64,
        a01: fx.Int64,
        a02: fx.Int64,
        a03: fx.Int64,
        a04: fx.Int64,
        a05: fx.Int64,
        a06: fx.Int64,
        a07: fx.Int64,
        a08: fx.Int64,
        a09: fx.Int64,
        a10: fx.Int64,
        a11: fx.Int64,
        a12: fx.Int64,
        a13: fx.Int64,
        a14: fx.Int64,
        a15: fx.Int64,
        a16: fx.Int64,
        a17: fx.Int64,
        a18: fx.Int64,
        a19: fx.Int64,
        a20: fx.Int64,
        a21: fx.Int64,
        a22: fx.Int64,
        a23: fx.Int64,
        a24: fx.Int64,
        a25: fx.Int64,
        a26: fx.Int64,
        a27: fx.Int64,
        pr0: fx.Int64,
        pr1: fx.Int64,
        pr2: fx.Int64,
        pr3: fx.Int64,
        rank: fx.Int32,
        nreq: fx.Int32,
        cnt_gdn: fx.Pointer,
        cnt_att: fx.Pointer,
        positions: fx.Pointer,
        m0: fx.Pointer,
        m1: fx.Pointer,
        m2: fx.Pointer,
        bt_stride: fx.Int32,
    ):
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        wave = uniform(tid // WAVE)
        lane = tid % WAVE
        gw = wave * BLOCKS + bid
        h16 = lane // 16
        n16 = lane % 16
        R_ = fx.Int32(nreq)
        T_ = R_ * SPEC

        from flydsl._mlir.dialects import llvm as _llvm

        lds = fx.SharedAllocator().allocate(Lds).peek()
        L_red = lds_addr(lds.red.ptr)
        if fx.const_expr(timing):
            if wave == 0:
                c0 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))
                if lane == 0:
                    lds_sti(L_red, 254, fx.Int32(c0 & 0xFFFFFFFF))
                    lds_sti(L_red, 255, fx.Int32(c0 >> 32))
        table = [fx.Int64(v) for v in (a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27)][: len(A)]

        def arg(name):
            return table[A[name]]

        def R(name, live=None):
            return rsrc_at(arg(name), live)

        def RP(ptr, live=None):
            return rsrc_at(fx.Int64(ptrtoint(ptr)), live)

        cnt_own = _pick(kind == "gdn", cnt_gdn, cnt_att)
        launch_idx = counter_base(cnt_gdn, BLOCKS) // BLOCKS + counter_base(cnt_att, BLOCKS) // BLOCKS
        tag = launch_idx + 1
        tag0 = launch_idx * 2 + 1
        tag1 = launch_idx * 2 + 2

        L_xb = lds_addr(lds.xb.ptr)
        L_nw = lds_addr(lds.nw.ptr)
        L_gb = lds_addr(lds.gb.ptr)
        L_wr = lds_addr(lds.wr.ptr)
        L_y = lds_addr(lds.y16.ptr)
        L_part = lds_addr(lds.part.ptr)
        L_rcv = lds_addr(lds.rcv.ptr)
        L_sele = lds_addr(lds.sel_e.ptr)
        L_selw = lds_addr(lds.sel_w.ptr)
        L_sg = lds_addr(lds.sg.ptr)
        L_slot = lds_addr(lds.slot.ptr)
        L_cnt = lds_addr(lds.cnt.ptr)
        L_off = lds_addr(lds.off.ptr)
        L_pos = lds_addr(lds.pos.ptr)
        L_list = lds_addr(lds.list_.ptr)
        L_grp = lds_addr(lds.grp.ptr)
        L_ngrp = lds_addr(lds.ngrp.ptr)
        L_shb = lds_addr(lds.shb.ptr)
        L_qb = lds_addr(lds.qb.ptr)
        L_knb = lds_addr(lds.knb.ptr)
        L_vn = lds_addr(lds.vn.ptr)
        L_sp = lds_addr(lds.sp.ptr)
        L_pb = lds_addr(lds.pb.ptr)
        L_al = lds_addr(lds.alph.ptr)

        r_mb = R("mbox")
        r_sc = R("scratch")
        r_stamps = R("stamps")

        def stamp(i, words=None):
            if timing:
                if fx.const_expr(words is None):
                    if wave == 0:
                        clk = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))
                        st_dw(r_stamps, (bid * STAMPS + i) * 2, [fx.Int32(clk & 0xFFFFFFFF), fx.Int32(clk >> 32)])
                else:
                    st_dw(r_stamps, (bid * STAMPS + i) * 2, words)

        stamp(18)
        # the norm weights, first of all loads (stored to LDS once the early loads are
        # issued: a store's wait on its load waits for every load issued before it)
        nw_in = ld_dw(R("ln1_w"), 2 * tid, n=2) + ld_dw(R("ln2_w"), 2 * tid, n=2)
        peer_ptrs = [fx.Int64(v) for v in (pr0, pr1, pr2, pr3)][:TP]
        own_ptr = peer_ptrs[0]
        for p in fx.range_constexpr(1, TP):
            own_ptr = (rank == p).select(peer_ptrs[p], own_ptr)
        peer_res = [rsrc_at(p) for p in peer_ptrs]
        own_res = rsrc_at(own_ptr)

        def groups_loop(body):
            """``body(g)`` for each group g that holds a token (16 g < T): one copy of
            the body, a runtime loop (code size: the instruction cache)."""
            if fx.const_expr(G == 1):
                body(0)
            else:
                for g in range((T_ + TG - 1) // TG):
                    body(fx.Int32(g))

        def preload_frags(name, tile, live, ksteps, nsteps):
            res = R(name, live)
            out = []
            for i in fx.range_constexpr(nsteps):
                dw = ((tile * ksteps + wave * nsteps + i) * WAVE + lane) * 8
                out.append(ld_dw(res, dw) + ld_dw(res, dw + 4))
            return out

        def tile_mma(frags, xb, xstride):
            """This wave's K slice of a 16-row tile times the group's 16 token rows of
            packed bf16 LDS ``xb``: the partial [16 rows x 16 tokens] into L_wr."""
            acc = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
            for i in fx.range_constexpr(len(frags)):
                k_dw = (wave * len(frags) + i) * 16 + h16 * 8
                acc = wmma_bf16(frags[i], lds_ld8w(xb, n16 * xstride + k_dw), acc)
            lds_st4(L_wr, (wave * TG + n16) * WRS + h16 * 8, acc[:4])
            lds_st4(L_wr, (wave * TG + n16) * WRS + h16 * 8 + 4, acc[4:])

        def tile_sum(n, r, waves=WAVES):
            y = lds_ld(L_wr, n * WRS + r)
            for w in fx.range_constexpr(1, waves):
                y = y + lds_ld(L_wr, (w * TG + n) * WRS + r)
            return y

        def st_xb(xb, xstride, n, d4, vals):
            lds_sti(xb, n * xstride + d4 // 2, bf16_pack(vals[0], vals[1]))
            lds_sti(xb, n * xstride + d4 // 2 + 1, bf16_pack(vals[2], vals[3]))

        def row_dw(t, m):
            """Dword offset of dims 128 m + 4 lane .. + 3 of bf16 row t (HIDDEN wide):
            a wave reads a token, lane l its dims 128 m + 4 l (coalesced 256 B a m)."""
            return t * (HIDDEN // 2) + 64 * m + 2 * lane

        # ---------------------------------------------------------------- metadata
        if fx.const_expr(kind == "gdn"):
            r_m0 = RP(m0)
            r_m1 = RP(m1)
            # conv: thread 16 q + i (q < R): request q, channel 16 b + i
            c_q = fx.min(tid // 16, fx.Int32(RMAX - 1))
            c_live = (bid * 16 < GDN_QKV) & (wave * WAVE < 16 * R_)
            c_ch = fx.min(bid * 16 + tid % 16, fx.Int32(GDN_QKV - 1))
            c_acc = fx.Int32(ld_dw(RP(m1, c_live), c_q, n=1)[0])
            c_slot = fx.Int32(ld_dw(RP(m0, c_live), c_q * SPEC, n=1)[0])
            # G2 task (request, v head, 8 columns) of this wave
            g2_q = fx.min(gw, fx.Int32(RMAX * GDN_VH * 16 - 1))
            g2_r = g2_q // (GDN_VH * 16)
            # within a request, task i = 8 jb + hv: the 8 column blocks of a 128-byte state
            # line (jb 8m .. 8m + 7, one head) run on CTAs of one bid % 8, i.e. one XCD, so
            # the line is written whole in one L2 (split over 8 L2s, every partial line's
            # write-back is a read-modify-write: 14 us of a c16 layer)
            g2_hv = g2_q % GDN_VH
            g2_jb = (g2_q % (GDN_VH * 16)) // GDN_VH
            g2_live = gw < R_ * (GDN_VH * 16)
            g2_acc = uniform(fx.Int32(ld_dw(rsrc_at(fx.Int64(ptrtoint(m1)), g2_live), g2_r, n=1)[0]))
            g2_si = uniform(fx.Int32(ld_dw(rsrc_at(fx.Int64(ptrtoint(m0)), g2_live), g2_r * SPEC + fx.max(g2_acc - 1, fx.Int32(0)), n=1)[0]))
        else:
            # attention: CTA (request, chunk); NCH chunks a request
            NCH = fx.min(fx.Int32(128), fx.Int32(BLOCKS) // R_)
            att_ok = bid < R_ * NCH
            r_a = fx.min(bid // NCH, R_ - 1)
            c_a = att_ok.select(bid % NCH, fx.Int32(0))

            def block_of(p, b, live=None):
                g = fx.min((p * NCH + c_a) * 4 + b, fx.Int32(MAX_BLK - 1))
                res = RP(m0) if live is None else rsrc_at(fx.Int64(ptrtoint(m0)), live)
                return fx.max(fx.Int32(ld_dw(res, r_a * bt_stride + g, n=1)[0]), fx.Int32(0))

            L_a = fx.Int32(ld_dw(RP(m1), r_a, n=1)[0])
            blk0 = [block_of(0, b) for b in range(4)]

        # out_proj / o_proj at 4 groups: CTA b and b + OUT_TILES share rows 16 (b % OUT_TILES)
        # .., each the tile's even / odd groups (the whole grid, half the groups in series;
        # c16 GDN 89 -> 84 us). Fewer groups: CTAs < OUT_TILES, every group (no gain at 2,
        # a loss at 1)
        SPLIT = G == 4
        o_tile = _pick(SPLIT, bid % OUT_TILES, fx.min(bid, fx.Int32(OUT_TILES - 1)))
        o_half = _pick(SPLIT, bid // OUT_TILES, fx.Int32(0))
        o_live = _pick(SPLIT, None, bid < OUT_TILES)
        o_own = _pick(SPLIT, bid >= 0, bid < OUT_TILES)  # this CTA owns an out tile

        # ---------------------------------------------------------------- early loads
        if fx.const_expr(kind == "gdn"):
            g1_f = preload_frags("g1_w", fx.min(bid, fx.Int32(G1_TILES - 1)), bid < G1_TILES, HIDDEN // 32, 4)
            conv_w = bf16_unpack(ld_dw(R("conv_w", c_live), c_ch * 2, n=2))
            taps = []
            for k in fx.range_constexpr(3):
                e = (fx.max(c_slot, fx.Int32(0)) * CONV_ROWS + fx.max(c_acc - 1, fx.Int32(0)) + k) * GDN_QKV + c_ch
                taps.append(bf16_at(ld_dw(R("conv_state", c_live), e // 2, n=1)[0], e % 2))
            r_ss = R("ssm_state")
            g2_ok = g2_live & (g2_si >= 0)
            # this wave's state: rows k = 4 lane + i, columns v = 8 jb + c; K-major storage
            # [hv][k][v] (16 B a row), or V-major [hv][v][k] (ATOM's fused GDN decode
            # layout: 8 B a column, a wave's column one whole 256-byte row)
            g2_state = []
            r_ss0 = rsrc_at(arg("ssm_state"), g2_ok)
            sbase = (fx.max(g2_si, fx.Int32(0)) * GDN_VH + g2_hv) * GDN_D * GDN_D
            if fx.const_expr(vmajor):
                for c in fx.range_constexpr(8):
                    g2_state.append(ld_dw(r_ss0, (sbase + (8 * g2_jb + c) * GDN_D + 4 * lane) // 2, n=2))
            else:
                for i in fx.range_constexpr(4):
                    g2_state.append(ld_dw(r_ss0, (sbase + (4 * lane + i) * GDN_D + 8 * g2_jb) // 2))
            g2_alog = bf16_at(ld_dw(R("a_log", g2_live), g2_hv // 2, n=1)[0], g2_hv % 2)
            g2_dtb = bf16_at(ld_dw(R("dt_bias", g2_live), g2_hv // 2, n=1)[0], g2_hv % 2)
            op_f = preload_frags("out_proj", o_tile, o_live, 1024 // 32, 2)
        else:
            a1_f = preload_frags("qkvg", fx.min(bid, fx.Int32(A1_TILES - 1)), bid < A1_TILES, HIDDEN // 32, 4)
            G_a = (L_a + 15) // 16

            def kv_frags(blks, rk, rv):
                """A pass's fragments of this wave (kernels.py): K for QK tasks i = 0, 1
                (block wave // 8 + 2 i, K step wave % 8), V^T for PV K steps q = 0, 1."""
                ks = []
                for i in fx.range_constexpr(2):
                    blk = (wave < 8).select(blks[2 * i], blks[2 * i + 1])
                    e0 = ((blk * 32 + 4 * (wave % 8) + 2 * h16) * 16 + n16) * 8
                    ks.append([fx.Int32(w) for w in ld_dw(rk, e0 // 2, cm=CM_DEV) + ld_dw(rk, (e0 + 128) // 2, cm=CM_DEV)])
                vs = []
                for q in fx.range_constexpr(2):
                    f = []
                    for i in fx.range_constexpr(2):
                        e0 = ((blks[2 * q + i] * 2 + h16) * ATT_D + 16 * wave + n16) * 8
                        f += [fx.Int32(w) for w in ld_dw(rv, e0 // 2, cm=CM_DEV)]
                    vs.append(f)
                return ks, vs

            pre_k, pre_v = kv_frags(blk0, R("k_cache", att_ok), R("v_cache", att_ok))

        # ---------------------------------------------------------------- N1 + first projection
        def n1_group(g):
            """Wave w: token 16 g + w's N1 output (xn1: normalized by the previous layer's
            all-reduce 1, or the runner's for layer 0) -> xb row w (a padding token's row:
            zeros, no access)."""
            t = g * TG + wave
            r_x = R("scratch", t < T_)
            xw = [ld_dw(r_x, SCRATCH["xn1"] + row_dw(t, m), n=2, cm=CM_DEV) for m in fx.range_constexpr(16)]
            for m in fx.range_constexpr(16):
                lds_sti(L_xb, wave * XS + 64 * m + 2 * lane, xw[m][0])
                lds_sti(L_xb, wave * XS + 64 * m + 2 * lane + 1, xw[m][1])


        # ln1_w | ln2_w into LDS: N1 / N2 read them there (a global read after the norm's
        # sum is one more dependent round trip a group)
        lds_sti(L_nw, 2 * tid, nw_in[0])
        lds_sti(L_nw, 2 * tid + 1, nw_in[1])
        lds_sti(L_nw, HIDDEN // 2 + 2 * tid, nw_in[2])
        lds_sti(L_nw, HIDDEN // 2 + 2 * tid + 1, nw_in[3])
        lds_barrier()

        if fx.const_expr(kind == "gdn"):
            r_cs = R("conv_state")

            def g1_group(g):
                if fx.const_expr(False):
                    if g == 1:
                        stamp(16)
                n1_group(g)
                lds_barrier()
                if fx.const_expr(False):
                    if g == 1:
                        stamp(17)
                if g == 0:
                    stamp(0, [lds_ldi(L_red, 254), lds_ldi(L_red, 255)])
                    stamp(1)
                if bid < G1_TILES:
                    tile_mma(g1_f, L_xb, XS)
                lds_barrier()
                if fx.const_expr(False):
                    if g == 1:
                        stamp(20)
                if (bid < G1_TILES) & (tid < 256):
                    n = tid // 16
                    r = tid % 16
                    t = g * TG + n
                    y = tile_sum(n, r)
                    row = bid * 16 + r
                    if t < T_:
                        if row >= GDN_QKVZ:
                            rr = row - GDN_QKVZ
                            slot = (rr < GDN_VH).select(2 * rr, 2 * (rr - GDN_VH) + 1)
                            st_f32(r_sc, SCRATCH["gbap"] + t * 16 + slot, [bf16r(y)], cm=CM_DEV)
                        else:
                            if row >= GDN_QKV:
                                _st_bf16(r_sc, 2 * SCRATCH["gzp"] + t * GDN_Z + row - GDN_QKV, bf16r(y))
                            else:
                                lds_st(L_y, n * 16 + r, bf16r(y))
                lds_barrier()
                if fx.const_expr(False):
                    if g == 1:
                        stamp(21)
                # conv of the group's requests: thread 16 q + i (q = 4 g .. 4 g + 3), the
                # SPEC tokens in order through the window [taps, x0 .. x3]
                if (bid * 16 < GDN_QKV) & (tid // (16 * RG) == g) & (tid // 16 < R_):
                    q = tid // 16
                    ql = q % RG
                    c = c_ch
                    win = list(taps) + [lds_ld(L_y, (ql * SPEC + j) * 16 + tid % 16) for j in fx.range_constexpr(SPEC)]
                    if c_slot >= 0:
                        for j in fx.range_constexpr(SPEC):
                            acc = conv_w[0] * win[j] + conv_w[1] * win[j + 1] + conv_w[2] * win[j + 2] + conv_w[3] * win[j + 3]
                            _st_bf16(r_sc, 2 * SCRATCH["gqkvp"] + (q * SPEC + j) * GDN_QKV + c, bf16r(silu(acc)))
                        for k in fx.range_constexpr(CONV_ROWS):
                            _st_bf16(r_cs, (c_slot * CONV_ROWS + k) * GDN_QKV + c, win[k + 1])
                    else:
                        for j in fx.range_constexpr(SPEC):
                            _st_bf16(r_sc, 2 * SCRATCH["gqkvp"] + (q * SPEC + j) * GDN_QKV + c, 0.0)
                lds_barrier()
                if fx.const_expr(False):
                    if g == 1:
                        stamp(23)

            groups_loop(g1_group)
            # this CTA's G1 / conv stores have landed: its flag (G2 waits on every CTA's)
            rocdl.s_wait_storecnt(0)
            lds_barrier()
            if tid == 0:
                put1(r_mb, MBOX["g1done"] + bid, 1.0, tag)
            stamp(2)
            stamp(3)

            # ------------------------------------------------------------ G2
            # (request, v head, 8 columns) a wave: the request's SPEC tokens in order on
            # one state (f32 in registers), the state after token j to slot j
            if g2_live:
                kh = g2_hv // (GDN_VH // GDN_KH)
                ok = g2_si >= 0
                unp = [bf16_unpack(w) for w in g2_state]  # V-major: a column's 4 rows; K-major: a row's 8 columns
                st = [[(unp[c][i] if vmajor else unp[i][c]) for c in range(8)] for i in range(4)]
                r_m0g = rsrc_at(fx.Int64(ptrtoint(m0)))
                # every CTA's G1 stores have landed (lane l reads flags 8 l .. 8 l + 7)
                _ = poll2(r_mb, [MBOX["g1done"] + 8 * lane + 2 * u for u in range(4)], tag)
                for j in range(SPEC):  # dynamic: one copy of the step, the state carried
                    t = g2_r * SPEC + j
                    so = uniform(fx.Int32(ld_dw(r_m0g, t, n=1)[0]))
                    qkb = SCRATCH["gqkvp"] + t * (GDN_QKV // 2)
                    qv = bf16_unpack(ld_dw(r_sc, qkb + (kh * GDN_D + 4 * lane) // 2, n=2, cm=CM_DEV))
                    kv = bf16_unpack(ld_dw(r_sc, qkb + (GDN_KH * GDN_D + kh * GDN_D + 4 * lane) // 2, n=2, cm=CM_DEV))
                    vv = bf16_unpack(ld_dw(r_sc, qkb + (2 * GDN_KH * GDN_D + g2_hv * GDN_D + 8 * g2_jb) // 2, n=4, cm=CM_DEV))
                    ba = ld_f32(r_sc, SCRATCH["gbap"] + t * 16 + 2 * g2_hv, n=2, cm=CM_DEV)
                    bv = ba[0]
                    av = ba[1]
                    qn = fmath.rsqrt(wave_sum(dot(qv, qv)) + 1e-6) * (GDN_D**-0.5)
                    kn_ = fmath.rsqrt(wave_sum(dot(kv, kv)) + 1e-6)
                    qv = [v * qn for v in qv]
                    kv = [v * kn_ for v in kv]
                    gg = -fmath.exp(g2_alog) * softplus(av + g2_dtb)
                    beta = bf16r(sigmoid(bv))
                    decay = fmath.exp(gg)
                    st = [[v * decay for v in st[i]] for i in fx.range_constexpr(4)]
                    # the 8 columns' k . S and q . S reduced together (col8_sum), read back
                    # per column with v_readlane (lane 4 c holds column c)
                    kvs = col8_sum([st[0][c] * kv[0] + st[1][c] * kv[1] + st[2][c] * kv[2] + st[3][c] * kv[3] for c in fx.range_constexpr(8)], lane)
                    dl = [(vv[c] - lane_val(kvs, 4 * c)) * beta for c in fx.range_constexpr(8)]
                    st = [[st[i][c] + kv[i] * dl[c] for c in fx.range_constexpr(8)] for i in fx.range_constexpr(4)]
                    os_ = col8_sum([st[0][c] * qv[0] + st[1][c] * qv[1] + st[2][c] * qv[2] + st[3][c] * qv[3] for c in fx.range_constexpr(8)], lane)
                    o = [lane_val(os_, 4 * c) for c in fx.range_constexpr(8)]
                    if (so >= 0) & ok & fx.const_expr("nostore" not in _ABL):
                        obase = (so * GDN_VH + g2_hv) * GDN_D * GDN_D
                        if fx.const_expr(vmajor):
                            for c in fx.range_constexpr(8):
                                st_dw(r_ss, (obase + (8 * g2_jb + c) * GDN_D + 4 * lane) // 2, [bf16_pack(st[0][c], st[1][c]), bf16_pack(st[2][c], st[3][c])])
                        else:
                            for i in fx.range_constexpr(4):
                                st_dw(r_ss, (obase + (4 * lane + i) * GDN_D + 8 * g2_jb) // 2, [bf16_pack(st[i][2 * n], st[i][2 * n + 1]) for n in fx.range_constexpr(4)])
                    if lane == 0:
                        st_dw(r_sc, SCRATCH["gop"] + (t * GDN_Z + g2_hv * GDN_D + 8 * g2_jb) // 2,
                              [bf16_pack(ok.select(o[2 * i], fx.Float32(0.0)), ok.select(o[2 * i + 1], fx.Float32(0.0))) for i in range(4)], cm=CM_DEV)
            # this CTA's G2 stores (and its G1 ones) have landed: its flag (G3 waits on all)
            rocdl.s_wait_storecnt(0)
            lds_barrier()
            if tid == 0:
                put1(r_mb, MBOX["g2done"] + bid, 1.0, tag)
            stamp(4)
            stamp(5)

            # ------------------------------------------------------------ G3 + out_proj
            gnw = bf16_unpack(ld_dw(R("gnorm_w"), 2 * lane, n=2))

            def out_load(g):
                """Wave w: head w % 8 of the group's tokens 2 i + w // 8: their o and z."""
                h = wave % GDN_VH
                oz = []
                for i in fx.range_constexpr(TG * GDN_VH // WAVES):
                    n = 2 * i + wave // GDN_VH
                    off = (fx.min(g * TG + n, T_ - 1) * GDN_Z + h * GDN_D + 4 * lane) // 2
                    oz.append((ld_dw(r_sc, SCRATCH["gop"] + off, n=2, cm=CM_DEV), ld_dw(r_sc, SCRATCH["gzp"] + off, n=2, cm=CM_DEV)))
                return oz

            def out_input(g, oz=None):
                """The group's gated RMSNorm (o, z loaded by out_load) into packed LDS."""
                if fx.const_expr(oz is None):
                    oz = out_load(g)
                h = wave % GDN_VH
                for i in fx.range_constexpr(TG * GDN_VH // WAVES):
                    n = 2 * i + wave // GDN_VH
                    o = bf16_unpack(oz[i][0])
                    z = bf16_unpack(oz[i][1])
                    rs = fmath.rsqrt(wave_sum(dot(o, o)) / float(GDN_D) + EPS)
                    if g * TG + n < T_:
                        st_xb(L_gb, GS, n, h * GDN_D + 4 * lane, [bf16r(o[i_] * rs * gnw[i_] * silu(z[i_])) for i_ in fx.range_constexpr(4)])
        else:
            # ------------------------------------------------------------ A1
            def a1_group(g):
                n1_group(g)
                lds_barrier()
                if g == 0:
                    stamp(0, [lds_ldi(L_red, 254), lds_ldi(L_red, 255)])
                    stamp(1)
                if bid < A1_TILES:
                    tile_mma(a1_f, L_xb, XS)
                lds_barrier()
                if (bid < A1_TILES) & (tid < 256):
                    n = tid // 16
                    t = g * TG + n
                    y = tile_sum(n, tid % 16)
                    if t < T_:
                        if fx.const_expr(PLAIN_A):
                            _st_bf16(r_sc, 2 * SCRATCH["qkvgp"] + t * ATT_QKVG + bid * 16 + tid % 16, bf16r(y))
                        else:
                            put1(r_mb, MBOX["qkvg"] + t * 2560 + bid * 16 + tid % 16, bf16r(y), tag)
                lds_barrier()

            groups_loop(a1_group)
            if fx.const_expr(PLAIN_A):  # this CTA's qkvg stores have landed: its flag
                rocdl.s_wait_storecnt(0)
                lds_barrier()
                if tid == 0:
                    put1(r_mb, MBOX["a1done"] + bid, 1.0, tag)
            stamp(2)
            stamp(3)

            # ------------------------------------------------------------ A2
            # CTA (r, c): the request's SPEC tokens' 4 q heads (B column n = 4 j + head),
            # new k and v, normed and roped, into LDS; then its passes (kernels.py) with
            # 16 columns; token j's scores masked past position L - SPEC + j
            if fx.const_expr(True):
                L = L_a
                c = c_a
                r_k = R("k_cache")
                r_v = R("v_cache")
                r_cos = R("cos")
                r_sin = R("sin")
                qnw = bf16_unpack(ld_dw(R("q_norm_w"), 4 * lane, n=4))
                knw = bf16_unpack(ld_dw(R("k_norm_w"), 4 * lane, n=4))

                if fx.const_expr(PLAIN_A):  # every CTA's qkvg stores have landed
                    _ = poll2(r_mb, [MBOX["a1done"] + 8 * lane + 2 * u for u in range(4)], tag)

                def norm_rope(src, w_, t):
                    """Head vector at qkvg element ``src`` (+ 8 lane): Gemma RMSNorm (w_), NeoX
                    RoPE on dims < 64 at token t's position: 4 packed bf16 dwords."""
                    if fx.const_expr(PLAIN_A):
                        x = bf16_unpack(ld_dw(r_sc, SCRATCH["qkvgp"] + src // 2, n=4, cm=CM_DEV))
                    else:
                        x = poll2(r_mb, [MBOX["qkvg"] + src + u for u in (0, 2, 4, 6)], tag)
                    rs = fmath.rsqrt(wave_sum(dot(x, x)) / float(ATT_D) + EPS)
                    xn = [bf16r(x[n] * rs * (w_[n] + 1.0)) for n in fx.range_constexpr(8)]
                    pos = fx.Int32(ld_dw(RP(positions), 2 * t, n=1)[0])
                    f0 = (8 * lane) % 32
                    cs = bf16_unpack(ld_dw(r_cos, (pos * 32 + f0) // 2, n=4))
                    sn = bf16_unpack(ld_dw(r_sin, (pos * 32 + f0) // 2, n=4))
                    other = [v.shuffle_xor(4, WAVE) for v in xn]
                    lo = lane < 4
                    rot = lane < 8
                    words = []
                    for n in fx.range_constexpr(4):
                        pair = []
                        for u in fx.range_constexpr(2):
                            k_ = 2 * n + u
                            r_ = lo.select(xn[k_] * cs[k_] - other[k_] * sn[k_], xn[k_] * cs[k_] + other[k_] * sn[k_])
                            pair.append(rot.select(bf16r(r_), xn[k_]))
                        words.append(bf16_pack(pair[0], pair[1]))
                    return words

                tq0 = r_a * SPEC
                # wave w: column w = (token w // 4, head w % 4); waves < SPEC also the new k
                qw = norm_rope((tq0 + wave // ATT_QH) * 2560 + ATT_QH * ATT_D + (wave % ATT_QH) * ATT_D + 8 * lane, qnw, tq0 + wave // ATT_QH)
                for n in fx.range_constexpr(4):
                    lds_sti(L_qb, wave * QS + 4 * lane + n, qw[n])
                if wave < SPEC:
                    kw = norm_rope((tq0 + wave) * 2560 + 2 * ATT_QH * ATT_D + 8 * lane, knw, tq0 + wave)
                    for n in fx.range_constexpr(4):
                        lds_sti(L_knb, wave * (ATT_D // 2) + 4 * lane + n, kw[n])
                if tid < SPEC * ATT_D // 2:
                    j = tid // (ATT_D // 2)
                    d2 = tid % (ATT_D // 2)
                    if fx.const_expr(PLAIN_A):
                        vv2 = bf16_unpack(ld_dw(r_sc, SCRATCH["qkvgp"] + ((tq0 + j) * 2560 + 2 * ATT_QH * ATT_D + ATT_D + 2 * d2) // 2, n=1, cm=CM_DEV))
                    else:
                        vv2 = poll2(r_mb, [MBOX["qkvg"] + (tq0 + j) * 2560 + 2 * ATT_QH * ATT_D + ATT_D + 2 * d2], tag)
                    lds_st(L_vn, j * ATT_D + 2 * d2, vv2[0])
                    lds_st(L_vn, j * ATT_D + 2 * d2 + 1, vv2[1])
                lds_barrier()
                stamp(19)

                # the new tokens: token j at position L - SPEC + j, block gn_j, offset o_j,
                # in chunk (gn_j // 4) % NCH's pass (gn_j // 4) // NCH. Its owner writes
                # K / V to the cache and patches them into that pass's fragments
                new = []
                for j in fx.range_constexpr(SPEC):
                    pj = L - SPEC + j
                    gn = fx.max(pj, fx.Int32(0)) // 16
                    own = ((gn // 4) % NCH == c) & (pj >= 0) & att_ok
                    new.append((own, (gn // 4) // NCH, gn % 4, fx.max(pj, fx.Int32(0)) % 16))
                if wave < SPEC:
                    for j in fx.range_constexpr(SPEC):
                        if fx.const_expr(True):
                            own, _, _, _ = new[j]
                            slot = fx.Int32(ld_dw(RP(m2), 2 * (tq0 + j), n=1)[0])
                            if (wave == j) & own & (slot >= 0):
                                blk = slot // 16
                                off = slot % 16
                                st_dw(r_k, (((blk * 32 + lane) * 16 + off) * 8) // 2, lds_ld4w(L_knb, j * (ATT_D // 2) + 4 * lane))
                                for n in fx.range_constexpr(8):
                                    _st_bf16(r_v, ((blk * 2 + off // 8) * ATT_D + 8 * lane + n) * 8 + off % 8, lds_ld(L_vn, j * ATT_D + 8 * lane + n))

                def patch(p, ks, vs):
                    out_k = [list(ks[i]) for i in range(2)]
                    out_v = [list(vs[q]) for q in range(2)]
                    for j in fx.range_constexpr(SPEC):
                        own, p_j, b_j, o_j = new[j]
                        here = own & (p == p_j)
                        kn = lds_ld8w(L_knb, j * (ATT_D // 2) + 16 * (wave % 8) + 8 * h16)
                        for i in fx.range_constexpr(2):
                            hit = here & (wave // 8 + 2 * i == b_j) & (n16 == o_j)
                            out_k[i] = [hit.select(kn[x], out_k[i][x]) for x in range(8)]
                        vb = fx.Int32(bf16_pack(lds_ld(L_vn, j * ATT_D + 16 * wave + n16), fx.Float32(0.0))) & 0xFFFF
                        e = o_j % 8
                        for q in fx.range_constexpr(2):
                            f = out_v[q]
                            for i in fx.range_constexpr(2):
                                hit = here & (b_j == 2 * q + i) & (h16 == o_j // 8)
                                for x in fx.range_constexpr(4):
                                    w = f[4 * i + x]
                                    nw_ = (e % 2 == 0).select((w & fx.Int32(-65536)) | vb, (w & 0xFFFF) | (vb << 16))
                                    f[4 * i + x] = (hit & (e // 2 == x)).select(nw_, w)
                            out_v[q] = f
                    return out_k, out_v

                # softmax column of this wave: token sj, its last visible position
                sj = wave // ATT_QH
                lim = L - SPEC + sj  # positions <= lim
                o_acc = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
                m_r = fx.Float32(NEG)
                l_r = fx.Float32(0.0)
                # software-pipelined: pass p's K / V fragments are in registers when it
                # starts, it issues pass p + 1's loads (their block ids came one pass
                # earlier) and pass p + 2's block ids, then computes: the loads' latency
                # hides under the compute instead of following it
                kc = pre_k
                vc = pre_v
                bnx = [block_of(1, b, (NCH + c) * 4 < G_a) for b in range(4)]
                p = fx.Int32(0)
                while att_ok & ((p * NCH + c) * 4 < G_a):
                    kp, vp_ = patch(p, kc, vc)
                    live1 = ((p + 1) * NCH + c) * 4 < G_a
                    kn, vn = kv_frags(bnx, rsrc_at(arg("k_cache"), live1), rsrc_at(arg("v_cache"), live1))
                    bnx2 = [block_of(p + 2, b, ((p + 2) * NCH + c) * 4 < G_a) for b in range(4)]
                    qf = lds_ld8w(L_qb, n16 * QS + 16 * (wave % 8) + 8 * h16)
                    for i in fx.range_constexpr(2):
                        d = wmma_bf16(kp[i], qf, [fx.Float32(0.0)] * 8)
                        base = (((wave // 8 + 2 * i) * 8 + wave % 8) * 16 + n16) * SPS + 8 * h16
                        lds_st4(L_sp, base, d[:4])
                        lds_st4(L_sp, base + 4, d[4:])
                    lds_barrier()
                    # softmax: wave w = column w, lanes over positions lane, lane + 32
                    sc = []
                    for half in fx.range_constexpr(2):
                        x_ = lane + 32 * half
                        b = x_ // 16
                        r = x_ % 16
                        acc = lds_ld(L_sp, ((b * 8) * 16 + wave) * SPS + r)
                        for st_ in fx.range_constexpr(1, 8):
                            acc = acc + lds_ld(L_sp, ((b * 8 + st_) * 16 + wave) * SPS + r)
                        ok = ((p * NCH + c) * 4 + b) * 16 + r <= lim
                        sc.append(ok.select(acc * ATT_SCALE, fx.Float32(NEG)))
                    m_new = fx.max(m_r, wave_max(fx.max(sc[0], sc[1])))
                    alpha = fmath.exp(m_r - m_new)
                    pr = [(sc[u] > NEG * 0.5).select(fmath.exp(sc[u] - m_new), fx.Float32(0.0)) for u in fx.range_constexpr(2)]
                    l_r = l_r * alpha + wave_sum(pr[0] + pr[1])
                    m_r = m_new
                    for half in fx.range_constexpr(2):
                        nxt = pr[half].shuffle_xor(1, WAVE)
                        if lane % 2 == 0:
                            lds_sti(L_pb, wave * PS + (lane + 32 * half) // 2, bf16_pack(pr[half], nxt))
                    if lane == 0:
                        lds_st(L_al, wave, alpha)
                    lds_barrier()
                    # PV: dims 16 wave .., the 16 columns
                    al = lds_ld(L_al, n16)
                    o_acc = [v * al for v in o_acc]
                    for q in fx.range_constexpr(2):
                        pf = lds_ld4w(L_pb, n16 * PS + (2 * q) * 8 + 4 * h16) + lds_ld4w(L_pb, n16 * PS + (2 * q + 1) * 8 + 4 * h16)
                        o_acc = wmma_bf16(vp_[q], pf, o_acc)
                    p = p + 1
                    kc = kn
                    vc = vn
                    bnx = bnx2
                stamp(20)
                # the partial (unnormalized O, running max / sum) of (CTA, column)
                if att_ok:
                    pidx = bid * 16 + n16
                    st_f32(r_sc, SCRATCH["aop"] + pidx * ATT_D + 16 * wave + 8 * h16, o_acc[:4], cm=CM_DEV)
                    st_f32(r_sc, SCRATCH["aop"] + pidx * ATT_D + 16 * wave + 8 * h16 + 4, o_acc[4:], cm=CM_DEV)
                    if lane == 0:
                        st_f32(r_sc, SCRATCH["aml"] + (bid * 16 + wave) * 2, [m_r, l_r], cm=CM_DEV)
                rocdl.s_wait_storecnt(0)
                stamp(21)
                lds_barrier()
                if (tid == 0) & att_ok:
                    put1(r_mb, MBOX["aflag"] + bid, 1.0, tag)
            stamp(4)
            stamp(5)

            # ------------------------------------------------------------ A3
            # (token, head, 32 dims) a wave; lanes over the request's chunks
            if gw < T_ * ATT_QH * 8:
                t = gw // (ATT_QH * 8)
                h = (gw // 8) % ATT_QH
                db = gw % 8
                rq = t // SPEC
                col = (t % SPEC) * ATT_QH + h
                flags = []
                for k in fx.range_constexpr(CH_LANE):
                    cc = lane + k * WAVE
                    flags.append(MBOX["aflag"] + rq * NCH + (cc < NCH).select(cc, 0))
                _ = poll1(r_mb, flags, tag)
                ms = []
                ls = []
                for k in fx.range_constexpr(CH_LANE):
                    cc = lane + k * WAVE
                    ok = cc < NCH
                    ml = ld_f32(r_sc, SCRATCH["aml"] + ((rq * NCH + ok.select(cc, 0)) * 16 + col) * 2, n=2, cm=CM_DEV)
                    ms.append(ok.select(ml[0], fx.Float32(NEG)))
                    ls.append(ok.select(ml[1], fx.Float32(0.0)))
                M = ms[0]
                for k in fx.range_constexpr(1, CH_LANE):
                    M = fx.max(M, ms[k])
                M = wave_max(M)
                wts = [fmath.exp(ms[k] - M) for k in fx.range_constexpr(CH_LANE)]
                Ls = wts[0] * ls[0]
                for k in fx.range_constexpr(1, CH_LANE):
                    Ls = Ls + wts[k] * ls[k]
                Ls = wave_sum(Ls)
                acc = [fx.Float32(0.0) for _ in fx.range_constexpr(32)]
                for k in fx.range_constexpr(CH_LANE):
                    cc = lane + k * WAVE
                    ok = cc < NCH
                    ob = SCRATCH["aop"] + ((rq * NCH + ok.select(cc, 0)) * 16 + col) * ATT_D + 32 * db
                    for v4 in fx.range_constexpr(8):
                        vals = ld_f32(r_sc, ob + 4 * v4, cm=CM_DEV)
                        for n in fx.range_constexpr(4):
                            acc[4 * v4 + n] = acc[4 * v4 + n] + wts[k] * vals[n]
                mine = fx.Float32(0.0)
                for n in fx.range_constexpr(32):
                    s_ = wave_sum(acc[n])
                    mine = (lane == n).select(s_, mine)
                Lsafe = (Ls > 0.0).select(Ls, fx.Float32(1.0))
                d = 32 * db + lane
                if fx.const_expr(PLAIN_A):
                    gate = bf16_at(ld_dw(r_sc, SCRATCH["qkvgp"] + (t * 2560 + h * ATT_D + d) // 2, n=1, cm=CM_DEV)[0], d % 2)
                else:
                    gate = poll1(r_mb, [MBOX["qkvg"] + t * 2560 + h * ATT_D + d], tag)[0]
                ao = bf16r(bf16r(mine / Lsafe) * bf16r(sigmoid(gate)))
                if fx.const_expr(PLAIN_A):
                    _st_bf16(r_sc, 2 * SCRATCH["aop2"] + t * 1024 + h * ATT_D + d, ao)
                else:
                    put1(r_mb, MBOX["ao"] + t * 1024 + h * ATT_D + d, ao, tag)
            if fx.const_expr(PLAIN_A):  # this CTA's gated outputs have landed: its flag
                rocdl.s_wait_storecnt(0)
                lds_barrier()
                if tid == 0:
                    put1(r_mb, MBOX["a3done"] + bid, 1.0, tag)
            stamp(6)

            def out_input(g):
                # A4 input: the group's gated attention outputs into packed LDS
                # thread: 4 dims of tokens 2 i + tid // 256, all polled in one batch
                NI = TG * 256 // THREADS
                if fx.const_expr(PLAIN_A):
                    vals = []
                    for i in fx.range_constexpr(NI):
                        e4 = tid + i * THREADS
                        t = fx.min(g * TG + e4 // 256, T_ - 1)
                        vals.append(ld_dw(r_sc, SCRATCH["aop2"] + (t * 1024 + 4 * (e4 % 256)) // 2, n=2, cm=CM_DEV))
                    for i in fx.range_constexpr(NI):
                        e4 = tid + i * THREADS
                        if g * TG + e4 // 256 < T_:
                            lds_sti(L_gb, (e4 // 256) * GS + (e4 % 256) * 2, vals[i][0])
                            lds_sti(L_gb, (e4 // 256) * GS + (e4 % 256) * 2 + 1, vals[i][1])
                else:
                    pr = []
                    for i in fx.range_constexpr(NI):
                        e4 = tid + i * THREADS
                        t = fx.min(g * TG + e4 // 256, T_ - 1)
                        pr += [MBOX["ao"] + t * 1024 + 4 * (e4 % 256), MBOX["ao"] + t * 1024 + 4 * (e4 % 256) + 2]
                    pv = poll2(r_mb, pr, tag)
                    for i in fx.range_constexpr(NI):
                        e4 = tid + i * THREADS
                        if g * TG + e4 // 256 < T_:
                            st_xb(L_gb, GS, e4 // 256, (e4 % 256) * 4, pv[4 * i: 4 * i + 4])

            op_f = preload_frags("o_proj", o_tile, o_live, 1024 // 32, 2)

        # ---------------------------------------------------------------- out proj
        # every CTA's G2 (and G1) / A3 stores have landed (lane l reads flags 8 l .. + 7)
        if fx.const_expr(kind == "gdn" or PLAIN_A):
            _ = poll2(r_mb, [MBOX[_pick(kind == "gdn", "g2done", "a3done")] + 8 * lane + 2 * u for u in range(4)], tag)
        stamp(24)

        def out_group(g, pre=None):
            if fx.const_expr(pre is None):
                out_input(g)
            else:
                out_input(g, pre)
            if fx.const_expr(timing):
                if g == 0:
                    stamp(25)
            lds_barrier()
            if o_own:
                tile_mma(op_f, L_gb, GS)
            lds_barrier()
            if o_own & (tid < 256):
                n = tid // 16
                r = tid % 16
                t = g * TG + n
                y = tile_sum(n, r)
                if t < T_:
                    lds_st(L_part, t * 16 + r, bf16r(y))
            lds_barrier()
            if fx.const_expr(timing):
                for k in fx.range_constexpr(G):
                    if g == k:
                        stamp(26 + k)

        if fx.const_expr(SPLIT):
            for k in range(((T_ + TG - 1) // TG - o_half + 1) // 2):
                out_group(o_half + 2 * fx.Int32(k))
        else:
            groups_loop(out_group)
        stamp(_pick(kind == "gdn", 6, 8))

        nx_rows = fx.Int32(arg("nx_rows"))
        if nx_rows > 0:
            if bid < nx_rows:
                prefetch_line(arg("nx_first") + fx.Int64(bid * 65536 + tid * 128))
            if (bid < OUT_TILES) & (tid < 256):
                prefetch_line(arg("nx_out") + fx.Int64(bid * 32768 + tid * 128))
            if bid < M1_TILES:
                prefetch_line(arg("nx_m1") + fx.Int64(bid * 65536 + tid * 128))

        m1_f = preload_frags("m1_w", fx.min(bid, fx.Int32(M1_TILES - 1)), bid < M1_TILES, HIDDEN // 32, 4)
        sh_f = preload_frags("sh_w2", bid % OUT_TILES, (bid < OUT_TILES) & (wave < 4), SH_INTER // 32, 1)

        # ---------------------------------------------------------------- all-reduces
        AR_K = TMAX // 8  # 16-byte lines a lane: 16 dims x T tokens = 4 T lines a peer

        def all_reduce(slot, tg, tile, push, recv_slots, half=None):
            """Push this CTA's 16 dims (rows 16 ``tile`` ..) of every token (L_part; with
            ``half``, the tokens of its groups g = half mod 2) to every rank's peer slot
            ``slot`` (wave W - TP + p to rank p); a receiving CTA polls ``recv_slots`` from
            every rank into L_rcv [slot index][src][t][16]."""
            if (wave >= WAVES - TP) & push:
                src = wave - (WAVES - TP)
                for k in fx.range_constexpr(AR_K):
                    i = lane + WAVE * k
                    t = fx.min(i // 4, T_ - 1)
                    h = i % 4
                    mine = (i // 4 < T_) if half is None else (i // 4 < T_) & ((i // 4) // TG % 2 == half)
                    if mine:
                        v = lds_ld4(L_part, t * 16 + 4 * h)
                        words = [bf16_pack(v[0], v[1]), bf16_pack(v[2], v[3])]
                        dst = pair_index(slot, rank, t, tile * 8 + 2 * h)
                        for pp in fx.range_constexpr(TP):
                            if src == pp:
                                st_dw(peer_res[pp], dst * 2, [words[0], tg, words[1], tg], cm=CM_SYS)
            if fx.const_expr(len(recv_slots) > 0):
                if (wave >= WAVES - TP) & (bid < OUT_TILES):
                    src = wave - (WAVES - TP)
                    pairs = []
                    for rs_ in recv_slots:
                        for k in fx.range_constexpr(AR_K):
                            i = lane + WAVE * k
                            t = fx.min(i // 4, T_ - 1)
                            pairs.append(pair_index(rs_, src, t, tile * 8 + 2 * (i % 4)))
                    got = poll_words_sys(own_res, pairs, tg)
                    for si in fx.range_constexpr(len(recv_slots)):
                        for k in fx.range_constexpr(AR_K):
                            i = lane + WAVE * k
                            t = fx.min(i // 4, T_ - 1)
                            x = si * AR_K + k
                            vals = bf16_unpack([fx.Uint32(w) for w in got[2 * x : 2 * x + 2]])
                            lds_st4(L_rcv, ((si * TP + src) * TMAX + t) * 16 + 4 * (i % 4), vals)
            lds_barrier()

        all_reduce(0, tag0, o_tile, o_own, [], half=_pick(SPLIT, o_half, None))
        # all-reduce 0 + N2, token-major: CTA OUT_TILES + t (idle through out_proj, so it
        # is already polling) receives all of token t (thread: dims 4 tid .. + 3 from
        # every rank), s2 = attn + bf16(h + res), xn = RMSNorm(s2) (1 + w2); stores
        # res_out, xn (packed bf16, M1's B) and its MXFP8 copy (M3's B), then the token's
        # flag. N2 runs once a token, not in every M1 CTA.
        t_ar = bid - OUT_TILES
        if (bid >= OUT_TILES) & (t_ar < T_):
            got = poll_words_sys(own_res, [pair_index(0, src, t_ar, 2 * tid) for src in range(TP)], tag0)
            hv = bf16_unpack(ld_dw(R("h_in"), row_dw(t_ar, wave), n=2))
            rv = bf16_unpack(ld_dw(R("res_in"), row_dw(t_ar, wave), n=2))
            parts = [bf16_unpack([fx.Uint32(w) for w in got[2 * s_: 2 * s_ + 2]]) for s_ in range(TP)]
            s2 = []
            for i in fx.range_constexpr(4):
                acc = parts[0][i]
                for s_ in fx.range_constexpr(1, TP):
                    acc = acc + parts[s_][i]
                s2.append(bf16r(acc) + bf16r(hv[i] + rv[i]))
            ss = wave_sum(dot(s2, s2))
            if lane == 0:
                lds_st(L_red, wave, ss)
            lds_barrier()
            tot = lds_ld(L_red, 0)
            for w in fx.range_constexpr(1, WAVES):
                tot = tot + lds_ld(L_red, w)
            rstd = fmath.rsqrt(tot / float(HIDDEN) + EPS)
            st_dw(R("res_out"), row_dw(t_ar, wave), [bf16_pack(s2[0], s2[1]), bf16_pack(s2[2], s2[3])])
            nw = bf16_unpack([lds_ldi(L_nw, HIDDEN // 2 + 64 * wave + 2 * lane), lds_ldi(L_nw, HIDDEN // 2 + 64 * wave + 2 * lane + 1)])
            xn = [bf16r(s2[i] * rstd * (nw[i] + 1.0)) for i in fx.range_constexpr(4)]
            st_dw(r_sc, SCRATCH["agp"] + row_dw(t_ar, wave), [bf16_pack(xn[0], xn[1]), bf16_pack(xn[2], xn[3])], cm=CM_DEV)
            q8, s8 = mx8_quant(xn, lane)
            st_dw(r_sc, SCRATCH["xq"] + t_ar * (HIDDEN // 4) + 32 * wave + lane, [q8], cm=CM_DEV)
            if lane == 0:
                st_dw(r_sc, SCRATCH["xs"] + t_ar * (HIDDEN // 128) + wave, [s8], cm=CM_DEV)
            rocdl.s_wait_storecnt(0)
            lds_barrier()
            if tid == 0:
                put1(r_mb, MBOX["xqf"] + t_ar, 1.0, tag)
        stamp(_pick(kind == "gdn", 7, 9))
        stamp(_pick(kind == "gdn", 8, 10))

        # ---------------------------------------------------------------- M1
        # M1 CTAs: a group at a time (wave w: token 16 g + w's normalized row, once its
        # flag is up) through their M1 tile
        def n2m1_group(g):
            t = g * TG + wave
            if t < T_:
                _ = poll1(r_mb, [MBOX["xqf"] + t], tag)
                for m in fx.range_constexpr(16):
                    xw = ld_dw(r_sc, SCRATCH["agp"] + row_dw(t, m), n=2, cm=CM_DEV)
                    lds_sti(L_xb, wave * XS + 64 * m + 2 * lane, xw[0])
                    lds_sti(L_xb, wave * XS + 64 * m + 2 * lane + 1, xw[1])
            lds_barrier()
            if fx.const_expr(kind == "gdn"):
                if g == 0:
                    stamp(9)
            tile_mma(m1_f, L_xb, XS)
            lds_barrier()
            if tid < 256:
                n = tid // 16
                r = tid % 16
                t2 = g * TG + n
                y = tile_sum(n, r)
                row = bid * 16 + r
                if t2 < T_:
                    if bid < M1_RT_TILES:
                        if row < ROUTER_ROWS:
                            put1(r_mb, MBOX["logits"] + t2 * LOGIT_STRIDE + row, bf16r(y), tag)
                    else:
                        lds_st(L_y, tid, y)
            lds_barrier()
            if (bid >= M1_RT_TILES) & (tid < 8 * TG):
                n = tid // 8
                r = tid % 8
                t2 = g * TG + n
                if t2 < T_:
                    j_sh = (bid - M1_RT_TILES) * 8 + r
                    g_ = lds_ld(L_y, n * 16 + r)
                    u_ = lds_ld(L_y, n * 16 + r + 8)
                    st_f32(r_sc, SCRATCH["shact"] + t2 * INTER + j_sh, [bf16r(silu(bf16r(g_)) * bf16r(u_))], cm=CM_DEV)
            lds_barrier()

        if bid < M1_TILES:
            groups_loop(n2m1_group)
        stamp(_pick(kind == "gdn", 10, 11))
        stamp(_pick(kind == "gdn", 11, 12))

        # ---------------------------------------------------------------- M2 top-8 + sort
        def topk_token(t, pub):
            lo_ = MBOX["logits"] + t * LOGIT_STRIDE + lane * 8
            vals = poll2(r_mb, [lo_, lo_ + 2, lo_ + 4, lo_ + 6], tag)
            picked_v = []
            picked_i = []
            for k in fx.range_constexpr(TOP_K):
                bv = vals[0]
                bi = lane * 8
                for i in fx.range_constexpr(1, 8):
                    take = vals[i] > bv
                    bv = take.select(vals[i], bv)
                    bi = take.select(lane * 8 + i, bi)
                for o_ in (16, 8, 4, 2, 1):
                    ov = bv.shuffle_xor(o_, WAVE)
                    oi = bi.shuffle_xor(o_, WAVE)
                    take = (ov > bv) | ((ov == bv) & (oi < bi))
                    bv = take.select(ov, bv)
                    bi = take.select(oi, bi)
                picked_v.append(bv)
                picked_i.append(bi)
                mine_ = lane == bi // 8
                for i in fx.range_constexpr(8):
                    vals[i] = (mine_ & (bi % 8 == i)).select(fx.Float32(-3.0e38), vals[i])
            es = [fmath.exp(v - picked_v[0]) for v in picked_v]
            tot = es[0]
            for k in fx.range_constexpr(1, TOP_K):
                tot = tot + es[k]
            if lane == 0:
                g = poll1(r_mb, [MBOX["logits"] + t * LOGIT_STRIDE + N_EXPERTS], tag)[0]
                if pub:  # one line a (expert, weight) pair, then the shared gate
                    for k in fx.range_constexpr(TOP_K):
                        put2(r_mb, MBOX["sel"] + (t * (TOP_K + 1) + k) * 2, picked_i[k].bitcast(fx.Float32), es[k] / tot, tag)
                    put2(r_mb, MBOX["sel"] + (t * (TOP_K + 1) + TOP_K) * 2, sigmoid(g), 0.0, tag)
                else:
                    for k in fx.range_constexpr(TOP_K):
                        lds_sti(L_sele, t * TOP_K + k, picked_i[k])
                        lds_st(L_selw, t * TOP_K + k, es[k] / tot)
                    lds_st(L_sg, t, sigmoid(g))

        # every CTA needs every token's experts (M3 / M4 run expert-major). Up to 16
        # tokens: each CTA computes them all, a wave a token. More: CTA t computes token
        # t once and publishes it, every CTA reads them all (one hop, not 2-4 rounds)
        pub = T_ > WAVES
        if pub.select((wave == 0) & (bid < T_), wave < T_):
            topk_token(pub.select(bid, wave), pub)
        if pub:
            for k in fx.range_constexpr((TMAX * (TOP_K + 1) + THREADS - 1) // THREADS):
                x = tid + THREADS * k
                t = x // (TOP_K + 1)
                kk = x % (TOP_K + 1)
                if t < T_:
                    v = poll2(r_mb, [MBOX["sel"] + x * 2], tag)
                    if kk < TOP_K:
                        lds_sti(L_sele, t * TOP_K + kk, v[0].bitcast(fx.Int32))
                        lds_st(L_selw, t * TOP_K + kk, v[1])
                    else:
                        lds_st(L_sg, t, v[0])
        if tid < 2 * N_EXPERTS:
            lds_sti(L_cnt, tid, fx.Int32(0))
        lds_barrier()
        stamp(16)
        # the slots (token t, k-th expert: slot 8 t + k) sorted by expert, the same in
        # every CTA (the groups must agree: an M3 task publishes its group's slots for
        # another CTA's M4): expert e's token bitmap (atomic OR: order-free), a slot's
        # rank = its expert's tokens below it. Expert e's slots in groups of up to 16
        NSL = T_ * TOP_K
        for k in fx.range_constexpr((TMAX * TOP_K + THREADS - 1) // THREADS):
            sl = tid + THREADS * k
            if sl < NSL:
                t = sl // TOP_K
                lds_atomic_or(L_cnt, 2 * lds_ldi(L_sele, sl) + t // 32, fx.Int32(1) << (t % 32))
        lds_barrier()
        for k in fx.range_constexpr((TMAX * TOP_K + THREADS - 1) // THREADS):
            sl = tid + THREADS * k
            if sl < NSL:
                t = sl // TOP_K
                e = lds_ldi(L_sele, sl)
                lo = lds_ldi(L_cnt, 2 * e)
                hi = lds_ldi(L_cnt, 2 * e + 1)
                below_lo = (t >= 32).select(lo, lo & ((fx.Int32(1) << (t % 32)) - 1))
                below_hi = (t >= 32).select(hi & ((fx.Int32(1) << (t % 32)) - 1), fx.Int32(0))
                lds_sti(L_pos, sl, popc(below_lo) + popc(below_hi))
        if wave == 0:  # lane l: experts 8 l .. 8 l + 7
            cn = [popc(lds_ldi(L_cnt, 2 * (8 * lane + i))) + popc(lds_ldi(L_cnt, 2 * (8 * lane + i) + 1)) for i in fx.range_constexpr(8)]
            gn = [(cv + 15) // 16 for cv in cn]
            cs = cn[0]
            gs = gn[0]
            for i in fx.range_constexpr(1, 8):
                cs = cs + cn[i]
                gs = gs + gn[i]
            ci = cs
            gi = gs
            for o_ in (1, 2, 4, 8, 16):
                pc = fx.Int32(gpu.shuffle_up(ci, o_, WAVE))
                pg = fx.Int32(gpu.shuffle_up(gi, o_, WAVE))
                ci = (lane >= o_).select(ci + pc, ci)
                gi = (lane >= o_).select(gi + pg, gi)
            cb = ci - cs
            gb = gi - gs
            for i in fx.range_constexpr(8):
                lds_sti(L_off, 8 * lane + i, cb)
                for q in fx.range_constexpr(TMAX // 16):
                    if q < gn[i]:
                        lds_sti(L_grp, gb + q, (8 * lane + i) | ((cb + 16 * q) << 8) | (fx.min(cn[i] - 16 * q, fx.Int32(16)) << 20))
                cb = cb + cn[i]
                gb = gb + gn[i]
            if lane == WAVE - 1:
                lds_sti(L_ngrp, 0, gi)
        lds_barrier()
        for k in fx.range_constexpr((TMAX * TOP_K + THREADS - 1) // THREADS):
            sl = tid + THREADS * k
            if sl < NSL:
                lds_sti(L_list, lds_ldi(L_off, lds_ldi(L_sele, sl)) + lds_ldi(L_pos, sl), sl)
        lds_barrier()
        NG = uniform(lds_ldi(L_ngrp, 0))
        stamp(17)

        def group(gi):
            gp = uniform(lds_ldi(L_grp, gi))
            return gp & 0xFF, (gp >> 8) & 0x3FF, gp >> 20

        # ---------------------------------------------------------------- M3 routed gate / up
        # task = a group (CTA gi % 256): its tokens' MXFP8 x staged in LDS (the B columns);
        # wave w = w13 tile w (gate / up of j = 8 w .. + 7) over the whole K, accumulated
        # in its WMMA C (no cross-wave reduction), the expert's rows streamed 4 K steps
        # ahead. The 128 activations a token are quantized a 32-block a wave and stored.
        arg_w13, arg_w13s, arg_w2, arg_w2s = arg("w13"), arg("w13_s"), arg("w2"), arg("w2_s")
        XQS = HIDDEN // 4 + 4  # LDS stride (dwords) of a staged token's MXFP8 row
        KS = HIDDEN // 128  # K steps

        def m3_task(i):
            gi = bid + BLOCKS * i
            ok3 = gi < NG
            e, start, cnt = group(fx.min(gi, NG - 1))
            # staging loads first, then every K step's w13 fragments (loads retire in
            # order: the staging stores wait only for theirs, the weights stream under them)
            tn = tid // 32
            sn = lds_ldi(L_list, start + fx.min(tn, cnt - 1))
            src = SCRATCH["xq"] + (sn // TOP_K) * (HIDDEN // 4) + 16 * (tid % 32)
            r_scq = rsrc_at(arg("scratch"), ok3)
            stg = [ld_dw(r_scq, src + 4 * v, cm=CM_DEV) for v in range(4)]
            sn2 = lds_ldi(L_list, start + fx.min(tid // KS, cnt - 1))
            stg_s = ld_dw(r_scq, SCRATCH["xs"] + (sn2 // TOP_K) * KS + tid % KS, n=1, cm=CM_DEV)[0]
            rw = rsrc_at(arg_w13, ok3)
            rws = rsrc_at(arg_w13s, ok3)
            n = wave * 16 + n16
            afr = []
            for ks in fx.range_constexpr(KS):
                ob = e * W13_EXPERT_BYTES + (wave * 32 + 2 * ks) * 512 + lane * 16
                sa = ld_dw(rws, e * (W13S_EXPERT_BYTES // 4) + (n // 32) * 512 + ks * 32 + n % 32, n=1)[0]
                afr.append((ld_dw(rw, ob // 4) + ld_dw(rw, (ob + 512) // 4), sa))
            if ok3:
                for v in fx.range_constexpr(4):
                    for x in fx.range_constexpr(4):
                        lds_sti(L_xb, tn * XQS + 16 * (tid % 32) + 4 * v + x, stg[v][x])
                if tid < 16 * KS:
                    lds_sti(L_xb, 16 * XQS + tid, stg_s)
            lds_barrier()
            stamp(23)
            acc = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
            for ks in fx.range_constexpr(KS):
                bq = []
                for j in fx.range_constexpr(4):
                    bq += lds_ld4w(L_xb, n16 * XQS + 32 * ks + 8 * j + 4 * h16)
                acc = wmma_fp4_fp8(afr[ks][0], bq, afr[ks][1], lds_ldi(L_xb, 16 * XQS + n16 * KS + ks), acc)
            if ok3:
                # rows 8 h16 + 2 i (gate) / + 1 (up): j = 8 wave + 4 h16 + i, column n16
                for i in fx.range_constexpr(4):
                    lds_st(L_gb, n16 * INTER + 8 * wave + 4 * h16 + i, silu(acc[2 * i]) * acc[2 * i + 1])
            lds_barrier()
            stamp(19)
            if ok3:
                for uu in fx.range_constexpr(4):  # unit 4 wave + uu: token unit // 4, block unit % 4
                    nn = (4 * wave + uu) // 4
                    bb = (4 * wave + uu) % 4
                    if nn < cnt:
                        s_ = lds_ldi(L_list, start + nn)
                        av = lds_ld(L_gb, nn * INTER + 32 * bb + lane)
                        am = wave_max(abs_f(av))
                        ex = fx.min(fx.max((am.bitcast(fx.Int32) >> 23) & 0xFF, fx.Int32(7)), fx.Int32(254))
                        mult = ((261 - ex) << 23).bitcast(fx.Float32)
                        qb = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(av * mult).ir_value(), fx.Float32(0.0).ir_value(), fx.Int32(0).ir_value(), False)) & 0xFF
                        dw = qb | (qb.shuffle_xor(1, WAVE) << 8) | (qb.shuffle_xor(2, WAVE) << 16) | (qb.shuffle_xor(3, WAVE) << 24)
                        if lane % 4 == 0:
                            st_dw(r_sc, SCRATCH["actd"] + s_ * (INTER // 4) + bb * 8 + lane // 4, [dw], cm=CM_DEV)
                        if lane == 0:
                            st_dw(r_sc, SCRATCH["acts"] + s_ * 4 + bb, [ex - 7], cm=CM_DEV)
            lds_barrier()

        def m3_task_quad(i):
            """Up to 16 tokens (one group of builds): task q = (group q // 4, quad q % 4:
            tiles 4 quad .. + 3, j = 32 quad .. + 31), wave w = K step w, the 16 K steps
            summed in LDS: 4 x the tasks of m3_task, for parallelism at few groups."""
            q = bid + BLOCKS * i
            ok3 = q < NG * 4
            e, start, cnt = group(fx.min(q, NG * 4 - 1) // 4)
            quad = q % 4
            s_n = lds_ldi(L_list, start + fx.min(n16, cnt - 1))
            t_n = s_n // TOP_K
            pre = []
            rw = rsrc_at(arg_w13, ok3)
            rws = rsrc_at(arg_w13s, ok3)
            for u in fx.range_constexpr(4):
                tile = quad * 4 + u
                ob = e * W13_EXPERT_BYTES + (tile * 32 + 2 * wave) * 512 + lane * 16
                n = tile * 16 + n16
                sa = ld_dw(rws, e * (W13S_EXPERT_BYTES // 4) + (n // 32) * 512 + wave * 32 + n % 32, n=1)[0]
                pre.append((ld_dw(rw, ob // 4) + ld_dw(rw, (ob + 512) // 4), sa))
            r_scq = rsrc_at(arg("scratch"), ok3)
            bq = []
            for j in fx.range_constexpr(4):
                bq += ld_dw(r_scq, SCRATCH["xq"] + t_n * (HIDDEN // 4) + 32 * wave + 8 * j + 4 * h16, cm=CM_DEV)
            sbq = ld_dw(r_scq, SCRATCH["xs"] + t_n * KS + wave, n=1, cm=CM_DEV)[0]
            for u in fx.range_constexpr(4):
                d = wmma_fp4_fp8(pre[u][0], bq, pre[u][1], sbq, [fx.Float32(0.0)] * 8)
                lds_st4(L_xb, ((u * WAVES + wave) * 16 + n16) * 16 + 8 * h16, d[:4])
                lds_st4(L_xb, ((u * WAVES + wave) * 16 + n16) * 16 + 8 * h16 + 4, d[4:])
            lds_barrier()
            if ok3:
                for x in fx.range_constexpr(2):  # (tile u, column, row): 1024 sums over the K steps
                    o = tid + THREADS * x
                    u = o // 256
                    y = lds_ld(L_xb, u * 4096 + o % 256)
                    for w in fx.range_constexpr(1, WAVES):
                        y = y + lds_ld(L_xb, u * 4096 + w * 256 + o % 256)
                    lds_st(L_gb, o, y)
            lds_barrier()
            if ok3 & (wave < cnt):
                # wave = column: lane l = j 32 quad + l (tile l // 8, gate row 2 (l % 8), up + 1)
                s_ = lds_ldi(L_list, start + wave)
                yb = ((lane // 8) * 16 + wave) * 16 + 2 * (lane % 8)
                av = silu(lds_ld(L_gb, yb)) * lds_ld(L_gb, yb + 1)
                am = wave_max(abs_f(av))
                ex = fx.min(fx.max((am.bitcast(fx.Int32) >> 23) & 0xFF, fx.Int32(7)), fx.Int32(254))
                mult = ((261 - ex) << 23).bitcast(fx.Float32)
                qb = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(av * mult).ir_value(), fx.Float32(0.0).ir_value(), fx.Int32(0).ir_value(), False)) & 0xFF
                dw = qb | (qb.shuffle_xor(1, WAVE) << 8) | (qb.shuffle_xor(2, WAVE) << 16) | (qb.shuffle_xor(3, WAVE) << 24)
                if lane % 4 == 0:
                    st_dw(r_sc, SCRATCH["actd"] + s_ * (INTER // 4) + quad * 8 + lane // 4, [dw], cm=CM_DEV)
                if lane == 0:
                    st_dw(r_sc, SCRATCH["acts"] + s_ * 4 + quad, [ex - 7], cm=CM_DEV)
            lds_barrier()

        # every token's MXFP8 x is published (one check a wave, not a poll a task)
        xf = MBOX["xqf"] + fx.min(2 * lane, T_ - 2)
        _ = poll1(r_mb, [xf, xf + 1], tag)
        if fx.const_expr("nom3" not in _ABL):
            if fx.const_expr(G == 1):  # few groups: 4 tasks a group (more CTAs streaming)
                for i in range((NG * 4 + BLOCKS - 1) // BLOCKS):
                    m3_task_quad(fx.Int32(i))
            else:  # a task a group: the expert's rows read once, no K reduction
                for i in range((NG + BLOCKS - 1) // BLOCKS):
                    m3_task(fx.Int32(i))
        # this CTA's M3 activations (and an M1 CTA's shared ones) have landed: its flag
        rocdl.s_wait_storecnt(0)
        lds_barrier()
        if tid == 0:
            put1(r_mb, MBOX["m3done"] + bid, 1.0, tag)
        stamp(20)
        stamp(_pick(kind == "gdn", 12, 13))
        if fx.const_expr(kind == "gdn"):
            stamp(13)

        # ---------------------------------------------------------------- M4 down + all-reduce 1
        m4t = bid
        if bid < OUT_TILES:
            # every CTA's M3 stores have landed (one wait: lane l reads flags 8 l .. 8 l + 7)
            _ = poll2(r_mb, [MBOX["m3done"] + 8 * lane + 2 * u for u in range(4)], tag)
            # the shared expert's activations: wave a token, packed bf16
            def sh_in(k):
                t = wave + WAVES * k
                if t < T_:
                    st_xb(L_shb, SHS, t, 4 * lane, ld_f32(r_sc, SCRATCH["shact"] + t * INTER + 4 * lane, cm=CM_DEV))

            for k in range((T_ + WAVES - 1) // WAVES):
                sh_in(fx.Int32(k))

            # routed: wave a group (one WMMA: rows 16 b .. of w2[e], the group's slots as
            # columns, their MXFP8 activations loaded straight into the B fragment), two
            # groups an iteration with every load issued first
            def m4_load(gi):
                ok = gi < NG
                e, start, cnt = group(fx.min(gi, NG - 1))
                s_n = lds_ldi(L_list, start + fx.min(n16, cnt - 1))
                ob = e * W2_EXPERT_BYTES + m4t * 4 * 512 + lane * 16
                n = m4t * 16 + n16
                sa = ld_dw(rsrc_at(arg_w2s, ok), e * (W2S_EXPERT_BYTES // 4) + (n // 32) * 64 + n % 32, n=1)[0]
                rw = rsrc_at(arg_w2, ok)
                af = ld_dw(rw, ob // 4) + ld_dw(rw, (ob + 512) // 4)
                bq = []
                for j in fx.range_constexpr(4):
                    bq += ld_dw(r_sc, SCRATCH["actd"] + s_n * (INTER // 4) + 8 * j + 4 * h16, cm=CM_DEV)
                sc4 = ld_dw(r_sc, SCRATCH["acts"] + s_n * 4, cm=CM_DEV)
                sb = (fx.Int32(sc4[0]) & 0xFF) | ((fx.Int32(sc4[1]) & 0xFF) << 8) | ((fx.Int32(sc4[2]) & 0xFF) << 16) | ((fx.Int32(sc4[3]) & 0xFF) << 24)
                return ok, cnt, s_n, af, sa, bq, sb

            def m4_grp(m):
                ld = [m4_load(wave + WAVES * (2 * m + u)) for u in range(2)]
                for u in fx.range_constexpr(2):
                    ok, cnt, s_n, af, sa, bq, sb = ld[u]
                    d = wmma_fp4_fp8(af, bq, sa, sb, [fx.Float32(0.0)] * 8)
                    if ok & (n16 < cnt):
                        lds_st4(L_slot, s_n * 16 + 8 * h16, d[:4])
                        lds_st4(L_slot, s_n * 16 + 8 * h16 + 4, d[4:])

            if fx.const_expr("nom4" not in _ABL):
                for m in range((NG + 2 * WAVES - 1) // (2 * WAVES)):
                    m4_grp(fx.Int32(m))
            lds_barrier()
            stamp(21)
            if fx.const_expr(kind == "gdn"):
                stamp(22)

            # shared: the tokens as columns, K step = wave < 4; then the routed slots
            # (fixed expert order) + gate x shared
            def m4_group(g):
                if wave < 4:
                    d = wmma_bf16(sh_f[0], lds_ld8w(L_shb, (g * TG + n16) * SHS + 16 * wave + 8 * h16), [fx.Float32(0.0)] * 8)
                    lds_st4(L_wr, (wave * TG + n16) * WRS + h16 * 8, d[:4])
                    lds_st4(L_wr, (wave * TG + n16) * WRS + h16 * 8 + 4, d[4:])
                lds_barrier()
                if tid < 256:
                    n = tid // 16
                    r = tid % 16
                    t = g * TG + n
                    if t < T_:
                        shared = tile_sum(n, r, 4)
                        routed = fx.Float32(0.0)
                        for kk in fx.range_constexpr(TOP_K):
                            routed = routed + lds_ld(L_selw, t * TOP_K + kk) * lds_ld(L_slot, (t * TOP_K + kk) * 16 + r)
                        lds_st(L_part, t * 16 + r, bf16r(routed + lds_ld(L_sg, t) * shared))
                lds_barrier()

            groups_loop(m4_group)
        stamp(14)
        all_reduce(1, tag1, bid, bid < OUT_TILES, [])
        # all-reduce 1, token-major as all-reduce 0 (CTA OUT_TILES + t): h_out, and unless
        # this is the last layer the next layer's N1, xn = RMSNorm(h_out + res_out) (1 +
        # its ln1_w, 2 HIDDEN elements on in the packed norm weights) into xn1
        if (bid >= OUT_TILES) & (t_ar < T_):
            got1 = poll_words_sys(own_res, [pair_index(1, src, t_ar, 2 * tid) for src in range(TP)], tag1)
            parts1 = [bf16_unpack([fx.Uint32(w1) for w1 in got1[2 * s1_: 2 * s1_ + 2]]) for s1_ in range(TP)]
            hs1 = []
            for i in fx.range_constexpr(4):
                acc1 = parts1[0][i]
                for s1_ in fx.range_constexpr(1, TP):
                    acc1 = acc1 + parts1[s1_][i]
                hs1.append(bf16r(acc1))
            st_dw(R("h_out"), row_dw(t_ar, wave), [bf16_pack(hs1[0], hs1[1]), bf16_pack(hs1[2], hs1[3])])
            if arg("nx_first") != 0:
                rv1 = bf16_unpack(ld_dw(R("res_out"), row_dw(t_ar, wave), n=2, cm=CM_DEV))
                nw1 = bf16_unpack(ld_dw(rsrc_at(arg("ln1_w") + 4 * HIDDEN), 64 * wave + 2 * lane, n=2))
                sx1 = [hs1[i] + rv1[i] for i in fx.range_constexpr(4)]
                ss1 = wave_sum(dot(sx1, sx1))
                if lane == 0:
                    lds_st(L_red, wave, ss1)
                lds_barrier()
                tot1 = lds_ld(L_red, 0)
                for w1 in fx.range_constexpr(1, WAVES):
                    tot1 = tot1 + lds_ld(L_red, w1)
                rstd1 = fmath.rsqrt(tot1 / float(HIDDEN) + EPS)
                xn1 = [bf16r(sx1[i] * rstd1 * (nw1[i] + 1.0)) for i in fx.range_constexpr(4)]
                st_dw(r_sc, SCRATCH["xn1"] + row_dw(t_ar, wave), [bf16_pack(xn1[0], xn1[1]), bf16_pack(xn1[2], xn1[3])], cm=CM_DEV)
        lds_barrier()
        stamp(15)
        if tid == 0:
            fx.atomic_add(cnt_own, fx.Int32(1), syncscope=AGENT)

    @flyc.jit
    def launch(
        a00: fx.Int64,
        a01: fx.Int64,
        a02: fx.Int64,
        a03: fx.Int64,
        a04: fx.Int64,
        a05: fx.Int64,
        a06: fx.Int64,
        a07: fx.Int64,
        a08: fx.Int64,
        a09: fx.Int64,
        a10: fx.Int64,
        a11: fx.Int64,
        a12: fx.Int64,
        a13: fx.Int64,
        a14: fx.Int64,
        a15: fx.Int64,
        a16: fx.Int64,
        a17: fx.Int64,
        a18: fx.Int64,
        a19: fx.Int64,
        a20: fx.Int64,
        a21: fx.Int64,
        a22: fx.Int64,
        a23: fx.Int64,
        a24: fx.Int64,
        a25: fx.Int64,
        a26: fx.Int64,
        a27: fx.Int64,
        pr0: fx.Int64,
        pr1: fx.Int64,
        pr2: fx.Int64,
        pr3: fx.Int64,
        rank: fx.Int32,
        nreq: fx.Int32,
        cnt_gdn: fx.Pointer,
        cnt_att: fx.Pointer,
        positions: fx.Pointer,
        m0: fx.Pointer,
        m1: fx.Pointer,
        m2: fx.Pointer,
        bt_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = build_key
        kernel(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, pr0, pr1, pr2, pr3, rank, nreq, cnt_gdn, cnt_att, positions, m0, m1, m2, bt_stride).launch(
            grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch
