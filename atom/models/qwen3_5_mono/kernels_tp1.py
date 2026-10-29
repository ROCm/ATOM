# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5-35B-A3B fused decoder layer for MTP verify steps on ONE gfx1250 GPU (TP1),
GDN state on ReplaySSM: R requests of S tokens each (the anchor and its drafts), T = S R
tokens, up to 32 (two groups of 16, the columns of one WMMA B tile). One persistent
FlyDSL kernel a layer, 256 CTAs of 16 waves, stages handed off through flags in a
tagged mailbox (kernels_mtp.py is the TP4 kernel it follows).

What changes from the TP4 kernel:

* Widths x4 and no all-reduce. A projection has more 16-row tiles than CTAs: CTA b
  streams tiles b, b + 256, .. (the next tile's weights in flight under the current
  one), every group of tokens against a tile while its weights are in registers.
  out_proj / o_proj (K = 4096) run on CTAs < 128 with the whole K in registers, their
  B fragments loaded straight from the bf16 input rows (GDN's gated RMSNorm applied in
  place once, a (token, v head) a wave, before). Where TP4 all-reduces, CTA 128 + t
  gathers token t from the out tiles through the mailbox, then norms it.
* GDN on ReplaySSM (fla_ops/replayssm.py, aiter's fused_gdn_verify_qkvz): a wave's
  slice of the state (request, v head, 8 columns) is rebuilt from the checkpoint and
  the h committed records, S = exp(sum g) S_0 + sum_j exp(the gates after j) u_j k_j^T;
  on a flush (h + 2 S > CAP) it becomes the new checkpoint and the window's records
  start at 0. The tokens update it in order, each storing its u record; the state
  after the window is not stored. The k and g records are shared by a head's 16 column
  waves (other CTAs, which may still be reading them), so they are stashed and written
  once every CTA's G2 is done.
* Attention: 2 kv heads, 8 q heads a kv head. CTA (request, kv head, chunk), a
  request's chunks in proportion to its context; the B columns are the request's S
  tokens x 8 q heads (up to 3 tiles of 16). A pass walks 64 positions of the chunk's
  blocks, any context length.
* MoE: an expert's gate / up are 1024 rows (64 tiles), its down projection K = 512 (4
  scaled-WMMA K steps); the shared expert's 512 x 2 rows are 64 M1 tiles.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, ptrtoint, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen3_5_mono.config_tp1 import (
    ATT_D,
    ATT_G_OFF,
    ATT_GQA,
    ATT_K_OFF,
    ATT_KVH,
    ATT_O,
    ATT_Q_OFF,
    ATT_QH,
    ATT_QKVG,
    ATT_SCALE,
    ATT_V_OFF,
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
    TG,
    THREADS,
    TOP_K,
    W13_EXPERT_BYTES,
    W13S_EXPERT_BYTES,
    W2_EXPERT_BYTES,
    W2S_EXPERT_BYTES,
    WAVE,
    WAVES,
)
from atom.models.qwen3_5_mono.layout_tp1 import (
    A1_TILES,
    ATT_COLS,
    G1_TILES,
    LOGIT_STRIDE,
    M1_RT_TILES,
    M1_TILES,
    MBOX,
    OUT_K,
    OUT_TILES,
    SCRATCH,
    STAMPS,
    arg_index,
)
from atom.models.qwen3_5_mono.prims import (
    AGENT,
    CM_DEV,
    abs_f,
    bf16_at,
    bf16_pack,
    bf16_unpack,
    bf16r,
    col8_sum,
    counter_base,
    dot,
    lane_val,
    lds_addr,
    lds_atomic_or,
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
    mx8_quant,
    poll1,
    poll2,
    popc,
    prefetch_line,
    put1,
    put2,
    rsrc_at,
    sigmoid,
    silu,
    softplus,
    st_dw,
    st_f32,
    uniform,
    wave_max,
    wave_sum,
    wave_sum_scatter,
    wmma_bf16,
    wmma_fp4_fp8,
)

NEG = -1.0e30
XS = HIDDEN // 2 + 4  # LDS stride (dwords) of a token's 2048 bf16 (N1 / M1 B)
SHS = SH_INTER // 2 + 4  # a token's shared activations
XQS = HIDDEN // 4 + 4  # a staged token's MXFP8 row (M3)
QS = ATT_D // 2 + 4  # an attention B column's q head
PS = 32 + 4  # a column's 64 P values
WRS = 16 + 4  # a (wave, column)'s 16 rows
SPS = 16 + 4  # a (block, K quarter, column)'s 16 positions
WR_G = WAVES * TG * WRS  # a group's per-wave partials
KS_X = HIDDEN // 128  # K steps of an MXFP8 token row (M3)
KS_W2 = INTER // 128  # K steps of w2 (M4)
OUT_KS = OUT_K // 32  # out_proj / o_proj K steps: 128, a wave's WAVES apart
CH_LANE = 4  # A3: chunks a lane (NCH <= 128)
# attention scratch inside the xb region (free from A1's end to A4's start), f32 offsets
_A_QB = 0
_A_KNB = _A_QB + ATT_COLS * QS
_A_VN = _A_KNB + 8 * ATT_D // 2
_A_SP = _A_VN + 8 * ATT_D
_A_PB = _A_SP + 16 * ATT_COLS * SPS
_A_AL = _A_PB + ATT_COLS * PS
_A_END = _A_AL + ATT_COLS


@functools.lru_cache(maxsize=None)
def _lds_struct(groups: int):
    tmax = TG * groups
    xb_f32 = max(tmax * XS, 16 * XQS + 16 * KS_X, _A_END)

    @fx.struct
    class _Lds:
        red: fx.Array[fx.Float32, 256, 16]
        nw: fx.Array[fx.Float32, HIDDEN, 16]  # ln1_w | ln2_w, packed bf16
        xb: fx.Array[fx.Float32, xb_f32, 16]  # N1 / M1 B, M3 staging, attention
        wr: fx.Array[fx.Float32, groups * WR_G, 16]  # a tile's per-wave partials, a group each
        y16: fx.Array[fx.Float32, tmax * 16, 16]  # a G1 tile's 16 channels of every token
        part: fx.Array[fx.Float32, tmax * 16, 16]  # the CTA's 16 dims of every token
        act: fx.Array[fx.Float32, TG * 128, 16]  # M3: a task's activations
        sel_e: fx.Array[fx.Float32, tmax * TOP_K, 16]  # i32 bits
        sel_w: fx.Array[fx.Float32, tmax * TOP_K, 16]
        sg: fx.Array[fx.Float32, tmax, 16]
        slot: fx.Array[fx.Float32, tmax * TOP_K * 16, 16]  # M4: a slot's 16 rows
        cnt: fx.Array[fx.Float32, 2 * N_EXPERTS, 16]
        off: fx.Array[fx.Float32, N_EXPERTS, 16]
        pos: fx.Array[fx.Float32, tmax * TOP_K, 16]
        list_: fx.Array[fx.Float32, tmax * TOP_K, 16]
        grp: fx.Array[fx.Float32, tmax * TOP_K // 16 + N_EXPERTS, 16]
        ngrp: fx.Array[fx.Float32, 16, 16]
        shb: fx.Array[fx.Float32, tmax * SHS, 16]

    return _Lds


def _st_bf16(res, elem, v):
    """Store one bf16 element at bf16 offset ``elem``."""
    bo.buffer_store(fx.Float32(v).to(fx.BFloat16).ir_value(), res, fx.Int32(elem))


def _pick(cond, a, b):
    return a if cond else b


def build_layer_tp1(kind: str, groups: int, spec: int, cap: int = 16, max_ctx: int = 262144, timing: bool = False):
    """The launcher of the ``kind`` ("gdn" | "att") TP1 layer kernel for up to 16
    ``groups`` tokens of ``spec`` tokens a request: ``launch(*table, nreq, cnt_gdn,
    cnt_att, positions, m0, m1, m2, m3, mstride, stream=)``, ``nreq`` requests
    (request-major). GDN: m0 = the conv slots (int32, request r's at r mstride), m1 =
    num_accepted_tokens (int32 [nreq]), m2 = the ReplaySSM slots (int32 [nreq], -1: pad),
    m3 = write_pos (int32 [slots], read only), ``cap`` the record capacity. Attention:
    m0 = block table (rows ``mstride`` apart), m1 = context lens (int32 [nreq], the new
    tokens included), m2 = slot mapping (int64 [spec nreq])."""
    assert kind in ("gdn", "att") and groups in (1, 2) and spec in (4, 5)
    G = groups
    S = spec
    TMAX = TG * G
    RMAX = TMAX // S
    CONV_ROWS = S + 2  # the conv window: 3 + the drafts
    NCOL = ATT_GQA * S  # attention B columns: token j, q head h -> 8 j + h
    CT = (NCOL + 15) // 16
    NCOLP = 16 * CT
    assert NCOLP <= ATT_COLS and S * ATT_GQA <= 48
    CAP = cap
    assert 2 * S <= CAP <= 32
    MAX_BLK = (max_ctx + 15) // 16
    A = arg_index(kind)
    Lds = _lds_struct(G)
    build_key = (kind, G, S, CAP, max_ctx, THREADS, timing)

    @flyc.kernel(name=f"qwen35_mono_tp1_{kind}_g{G}_s{S}{'_timing' if timing else ''}")
    def kernel(
        a00: fx.Int64, a01: fx.Int64, a02: fx.Int64, a03: fx.Int64,
        a04: fx.Int64, a05: fx.Int64, a06: fx.Int64, a07: fx.Int64,
        a08: fx.Int64, a09: fx.Int64, a10: fx.Int64, a11: fx.Int64,
        a12: fx.Int64, a13: fx.Int64, a14: fx.Int64, a15: fx.Int64,
        a16: fx.Int64, a17: fx.Int64, a18: fx.Int64, a19: fx.Int64,
        a20: fx.Int64, a21: fx.Int64, a22: fx.Int64, a23: fx.Int64,
        a24: fx.Int64, a25: fx.Int64, a26: fx.Int64, a27: fx.Int64,
        a28: fx.Int64, a29: fx.Int64, a30: fx.Int64, a31: fx.Int64,
        nreq: fx.Int32,
        cnt_gdn: fx.Pointer,
        cnt_att: fx.Pointer,
        positions: fx.Pointer,
        m0: fx.Pointer,
        m1: fx.Pointer,
        m2: fx.Pointer,
        m3: fx.Pointer,
        mstride: fx.Int32,
    ):  # fmt: skip
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        wave = uniform(tid // WAVE)
        lane = tid % WAVE
        gw = wave * BLOCKS + bid
        h16 = lane // 16
        n16 = lane % 16
        R_ = fx.Int32(nreq)
        T_ = R_ * S

        from flydsl._mlir.dialects import llvm as _llvm

        lds = fx.SharedAllocator().allocate(Lds).peek()
        L_red = lds_addr(lds.red.ptr)
        if fx.const_expr(timing):
            if wave == 0:
                c0 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))
                if lane == 0:
                    lds_sti(L_red, 254, fx.Int32(c0 & 0xFFFFFFFF))
                    lds_sti(L_red, 255, fx.Int32(c0 >> 32))
        table = [fx.Int64(v) for v in (a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31)][: len(A)]

        def arg(name):
            return table[A[name]]

        def R(name, live=None):
            return rsrc_at(arg(name), live)

        def RP(ptr, live=None):
            return rsrc_at(fx.Int64(ptrtoint(ptr)), live)

        cnt_own = _pick(kind == "gdn", cnt_gdn, cnt_att)
        launch_idx = counter_base(cnt_gdn, BLOCKS) // BLOCKS + counter_base(cnt_att, BLOCKS) // BLOCKS
        tag = launch_idx + 1

        L_xb = lds_addr(lds.xb.ptr)
        L_nw = lds_addr(lds.nw.ptr)
        L_wr = lds_addr(lds.wr.ptr)
        L_y = lds_addr(lds.y16.ptr)
        L_part = lds_addr(lds.part.ptr)
        L_act = lds_addr(lds.act.ptr)
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
        # attention's LDS, inside the xb region (byte addresses)
        L_qb = L_xb + 4 * _A_QB
        L_knb = L_xb + 4 * _A_KNB
        L_vn = L_xb + 4 * _A_VN
        L_sp = L_xb + 4 * _A_SP
        L_pb = L_xb + 4 * _A_PB
        L_al = L_xb + 4 * _A_AL

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

        def flag_all(name, one_wave=True):
            """Every CTA's ``name`` flag is up (lane l reads flags 8 l .. 8 l + 7). Wave 0
            polls and the CTA barrier releases the rest: 16 waves polling the same 256
            flags multiply every poll's L2 traffic by 16 while producers store. Under a
            wave-dependent branch (``one_wave=False``) the wave polls itself."""
            if fx.const_expr(one_wave):
                if wave == 0:
                    _ = poll2(r_mb, [MBOX[name] + 8 * lane + 2 * u for u in range(4)], tag)
                lds_barrier()
            else:
                _ = poll2(r_mb, [MBOX[name] + 8 * lane + 2 * u for u in range(4)], tag)

        def flag_mine(name):
            """This CTA's stores have landed: its ``name`` flag."""
            rocdl.s_wait_storecnt(0)
            lds_barrier()
            if tid == 0:
                put1(r_mb, MBOX[name] + bid, 1.0, tag)

        nw_in = ld_dw(R("ln1_w"), 2 * tid, n=2) + ld_dw(R("ln2_w"), 2 * tid, n=2)

        def preload_frags(name, tile, live, ksteps, nsteps):
            """A 16-row tile's fragments of this wave: K steps wave nsteps .. + nsteps."""
            res = R(name, live)
            out = []
            for i in fx.range_constexpr(nsteps):
                dw = ((tile * ksteps + wave * nsteps + i) * WAVE + lane) * 8
                out.append(ld_dw(res, dw) + ld_dw(res, dw + 4))
            return out

        def preload_frags_il(name, tile, live, ksteps, nsteps):
            """As preload_frags, K steps wave + WAVES m (m < nsteps): every wave has two K
            steps in each 1024-wide chunk of out_proj / o_proj."""
            res = R(name, live)
            out = []
            for m in fx.range_constexpr(nsteps):
                dw = ((tile * ksteps + wave + WAVES * m) * WAVE + lane) * 8
                out.append(ld_dw(res, dw) + ld_dw(res, dw + 4))
            return out

        def tile_mma(frags, row0, xstride, wr0):
            """This wave's K slice of a 16-row tile times token rows row0 .. + 15 of the
            packed bf16 xb: the partial [16 rows x 16 tokens] into L_wr at wr0."""
            acc = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
            for i in fx.range_constexpr(len(frags)):
                k_dw = (wave * len(frags) + i) * 16 + h16 * 8
                acc = wmma_bf16(frags[i], lds_ld8w(L_xb, (row0 + n16) * xstride + k_dw), acc)
            lds_st4(L_wr, wr0 + (wave * TG + n16) * WRS + h16 * 8, acc[:4])
            lds_st4(L_wr, wr0 + (wave * TG + n16) * WRS + h16 * 8 + 4, acc[4:])

        def tile_sum(n, r, wr0, waves=WAVES):
            y = lds_ld(L_wr, wr0 + n * WRS + r)
            for w in fx.range_constexpr(1, waves):
                y = y + lds_ld(L_wr, wr0 + (w * TG + n) * WRS + r)
            return y

        def st_xb(base, xstride, n, d4, vals):
            lds_sti(base, n * xstride + d4 // 2, bf16_pack(vals[0], vals[1]))
            lds_sti(base, n * xstride + d4 // 2 + 1, bf16_pack(vals[2], vals[3]))

        def row_dw(t, m):
            """Dword offset of dims 128 m + 4 lane .. + 3 of bf16 row t (HIDDEN wide)."""
            return t * (HIDDEN // 2) + 64 * m + 2 * lane

        # the N1 rows first: a load wait retires in issue order, so issued after the
        # weight / state preloads below, N1's wait would cover every one of them
        xn1_w = []
        for g in fx.range_constexpr(G):
            t_x = g * TG + wave
            xn1_w.append([ld_dw(R("scratch", t_x < T_), SCRATCH["xn1"] + row_dw(t_x, m), n=2, cm=CM_DEV) for m in fx.range_constexpr(16)])

        # ---------------------------------------------------------------- metadata
        if fx.const_expr(kind == "gdn"):
            # conv: thread 16 q + i (q < R): request q, channel i of the tile
            c_q = fx.min(tid // 16, fx.Int32(RMAX - 1))
            c_i = tid % 16
            c_live = wave * WAVE < 16 * R_
            c_acc = fx.Int32(ld_dw(RP(m1, c_live), c_q, n=1)[0])
            c_slot = fx.Int32(ld_dw(RP(m0, c_live), c_q * mstride, n=1)[0])
            # G2 task (request, v head, 8 columns) of this wave; a head's 16 column blocks
            # on CTAs of one bid % 8 (one XCD), as the TP4 kernel
            NT2 = GDN_VH * 16
            g2_q = fx.min(gw, fx.Int32(RMAX * NT2 - 1))
            g2_r = g2_q // NT2
            g2_hv = g2_q % GDN_VH
            g2_jb = (g2_q % NT2) // GDN_VH
            g2_live = gw < R_ * NT2
            g2_slot = uniform(fx.Int32(ld_dw(RP(m2, g2_live), g2_r, n=1)[0]))
            g2_ok = g2_live & (g2_slot >= 0)
            g2_sl = fx.max(g2_slot, fx.Int32(0))
            g2_h = uniform(fx.max(fx.Int32(ld_dw(RP(m3, g2_ok), g2_sl, n=1)[0]), fx.Int32(0)))
            g2_flush = g2_h + 2 * S > CAP
            g2_base = g2_flush.select(fx.Int32(0), g2_h)
            g2_rec = (g2_sl * GDN_VH + g2_hv) * CAP  # the head's first record row
        else:
            # attention: CTA (request, kv head, chunk). A request's kv heads get chunks in
            # proportion to its context, n_r = max(1, (128 - RMAX) blocks_r / all blocks)
            # each (an even split leaves the CTAs of short contexts idle behind the
            # longest): request r's CTAs 2 (n_0 + .. + n_{r-1}) + n_r kvh + chunk
            a_len = [uniform(fx.Int32(ld_dw(RP(m1), fx.min(fx.Int32(q), R_ - 1), n=1)[0])) for q in range(RMAX)]
            a_blk = [(R_ > q).select((a_len[q] + 15) // 16, fx.Int32(0)) for q in range(RMAX)]
            a_tot = a_blk[0]
            for q in fx.range_constexpr(1, RMAX):
                a_tot = a_tot + a_blk[q]
            a_tot = fx.max(a_tot, fx.Int32(1))
            a_nch = [(R_ > q).select(fx.max(fx.Int32(1), (128 - RMAX) * a_blk[q] // a_tot), fx.Int32(0)) for q in range(RMAX)]
            a_base = []
            a_end = fx.Int32(0)
            for q in fx.range_constexpr(RMAX):
                a_base.append(a_end)
                a_end = a_end + ATT_KVH * a_nch[q]

            def a_pick(vals, q):
                """vals[q] for a request index q (uniform or per lane)."""
                v = vals[0]
                for q_ in fx.range_constexpr(1, RMAX):
                    v = (q == q_).select(vals[q_], v)
                return v

            att_ok = bid < a_end
            r_a = fx.Int32(0)
            for q in fx.range_constexpr(1, RMAX):
                r_a = (bid >= a_base[q]).select(fx.Int32(q), r_a)
            r_a = fx.min(r_a, R_ - 1)
            NCH = fx.max(a_pick(a_nch, r_a), fx.Int32(1))
            a_off = att_ok.select(bid - a_pick(a_base, r_a), fx.Int32(0))
            kvh = fx.min(a_off // NCH, fx.Int32(ATT_KVH - 1))
            c_a = a_off % NCH

            def block_of(p, b, live=None):
                g = fx.min((p * NCH + c_a) * 4 + b, fx.Int32(MAX_BLK - 1))
                res = RP(m0) if live is None else rsrc_at(fx.Int64(ptrtoint(m0)), live)
                return fx.max(fx.Int32(ld_dw(res, r_a * mstride + g, n=1)[0]), fx.Int32(0))

            L_a = a_pick(a_len, r_a)
            blk0 = [block_of(0, b) for b in range(4)]

        # ---------------------------------------------------------------- early loads
        if fx.const_expr(kind == "gdn"):
            g1_f0 = preload_frags("g1_w", bid, bid < G1_TILES, HIDDEN // 32, 4)
            conv_w = []
            taps = []
            for k in fx.range_constexpr(2):  # CTA b's conv tiles: b and b + 256
                ch = (bid + BLOCKS * k) * 16 + c_i
                conv_w.append(bf16_unpack(ld_dw(R("conv_w", c_live), ch * 2, n=2)))
                tk = []
                for j in fx.range_constexpr(3):
                    e = (fx.max(c_slot, fx.Int32(0)) * CONV_ROWS + fx.max(c_acc - 1, fx.Int32(0)) + j) * GDN_QKV + ch
                    tk.append(bf16_at(ld_dw(R("conv_state", c_live), e // 2, n=1)[0], e % 2))
                taps.append(tk)
            # this wave's checkpoint slice: rows k = 4 lane + i, columns v = 8 jb + c,
            # [slot][hv][k][v] (V contiguous: a row's 8 columns one 16-byte load)
            r_ck = rsrc_at(arg("ckpt"), g2_ok)
            sbase = (g2_sl * GDN_VH + g2_hv) * GDN_D * GDN_D
            g2_ck = [ld_dw(r_ck, (sbase + (4 * lane + i) * GDN_D + 8 * g2_jb) // 2) for i in range(4)]
            # the committed records' gates, lane j: record j (0 past the cursor)
            g_rec = ld_f32(rsrc_at(arg("rec_g"), g2_ok & (lane < g2_h)), g2_rec + fx.min(lane, fx.Int32(CAP - 1)), n=1)[0]
            g_rec = (lane < g2_h).select(g_rec, fx.Float32(0.0))
            g2_alog = bf16_at(ld_dw(R("a_log", g2_live), g2_hv // 2, n=1)[0], g2_hv % 2)
            g2_dtb = bf16_at(ld_dw(R("dt_bias", g2_live), g2_hv // 2, n=1)[0], g2_hv % 2)
        else:
            a1_f0 = preload_frags("qkvg", bid, bid < A1_TILES, HIDDEN // 32, 4)

            def kv_frags(blks, rk, rv):
                """A pass's fragments of this wave. K: block wave // 4, K steps 2 (wave % 4)
                + i (A rows: the block's 16 positions). V^T for PV K steps q = 0, 1 (A rows:
                dims 16 wave ..; positions: blocks 2 q, 2 q + 1)."""
                bw = wave // 4
                blk = blks[0]
                for b in fx.range_constexpr(1, 4):
                    blk = (bw == b).select(blks[b], blk)
                ks = []
                for i in fx.range_constexpr(2):
                    kst = 2 * (wave % 4) + i
                    e0 = (((blk * ATT_KVH + kvh) * 32 + 4 * kst + 2 * h16) * 16 + n16) * 8
                    ks.append([fx.Int32(w) for w in ld_dw(rk, e0 // 2, cm=CM_DEV) + ld_dw(rk, (e0 + 128) // 2, cm=CM_DEV)])
                vs = []
                for q in fx.range_constexpr(2):
                    f = []
                    for i in fx.range_constexpr(2):
                        e0 = ((((blks[2 * q + i] * ATT_KVH + kvh) * 2 + h16) * ATT_D) + 16 * wave + n16) * 8
                        f += [fx.Int32(w) for w in ld_dw(rv, e0 // 2, cm=CM_DEV)]
                    vs.append(f)
                return ks, vs

            pre_k, pre_v = kv_frags(blk0, R("k_cache", att_ok), R("v_cache", att_ok))

        # ---------------------------------------------------------------- N1
        def n1_all():
            """Wave w: tokens 16 g + w's N1 outputs (xn1: normalized by the previous
            layer's gather 1, or the runner's for layer 0) -> xb rows (padding: zeros)."""
            for g in fx.range_constexpr(G):
                t = g * TG + wave
                xw = xn1_w[g]
                for m in fx.range_constexpr(16):
                    lds_sti(L_xb, t * XS + 64 * m + 2 * lane, xw[m][0])
                    lds_sti(L_xb, t * XS + 64 * m + 2 * lane + 1, xw[m][1])

        lds_sti(L_nw, 2 * tid, nw_in[0])
        lds_sti(L_nw, 2 * tid + 1, nw_in[1])
        lds_sti(L_nw, HIDDEN // 2 + 2 * tid, nw_in[2])
        lds_sti(L_nw, HIDDEN // 2 + 2 * tid + 1, nw_in[3])
        n1_all()
        lds_barrier()
        stamp(0, [lds_ldi(L_red, 254), lds_ldi(L_red, 255)])
        stamp(1)

        def g_rows_out(tile, k, y, t, r):
            """G1 output row 16 tile + r of token t (k: the tile's slot in CTA b's list)."""
            if fx.const_expr(k == 3):  # b | a: pairs a v head
                rr = tile * 16 + r - GDN_QKVZ
                slot = (rr < GDN_VH).select(2 * rr, 2 * (rr - GDN_VH) + 1)
                st_f32(r_sc, SCRATCH["gbap"] + t * (2 * GDN_VH) + slot, [bf16r(y)], cm=CM_DEV)
            elif fx.const_expr(k == 2):  # z
                _st_bf16(r_sc, 2 * SCRATCH["gzp"] + t * GDN_Z + tile * 16 + r - GDN_QKV, bf16r(y))
            else:  # q | k | v: the conv input
                lds_st(L_y, t * 16 + r, bf16r(y))

        if fx.const_expr(kind == "gdn"):
            r_cs = R("conv_state")
            # ------------------------------------------------------------ G1 + conv
            frags = g1_f0
            for k in fx.range_constexpr(4):  # tiles b, b + 256 (conv), b + 512 (z), b + 768 (b | a)
                tile = bid + BLOCKS * k
                live = tile < G1_TILES
                if fx.const_expr(k + 1 < 4):
                    tn = bid + BLOCKS * (k + 1)
                    nxt = preload_frags("g1_w", fx.min(tn, fx.Int32(G1_TILES - 1)), tn < G1_TILES, HIDDEN // 32, 4)
                if live:
                    for g in fx.range_constexpr(G):
                        tile_mma(frags, g * TG, XS, g * WR_G)
                lds_barrier()
                if live & (tid < 256 * G):
                    g_ = tid // 256
                    n = (tid // 16) % 16
                    r = tid % 16
                    t = g_ * TG + n
                    y = tile_sum(n, r, g_ * WR_G)
                    if t < T_:
                        g_rows_out(tile, k, y, t, r)
                lds_barrier()
                if fx.const_expr(k < 2):
                    # conv of channel 16 tile + i for request q (thread 16 q + i): the S
                    # tokens in order through the window [taps, x0 ..]
                    if tid // 16 < R_:
                        c = tile * 16 + c_i
                        cw = conv_w[k]
                        win = list(taps[k]) + [lds_ld(L_y, (c_q * S + j) * 16 + c_i) for j in fx.range_constexpr(S)]
                        if c_slot >= 0:
                            for j in fx.range_constexpr(S):
                                acc = cw[0] * win[j] + cw[1] * win[j + 1] + cw[2] * win[j + 2] + cw[3] * win[j + 3]
                                _st_bf16(r_sc, 2 * SCRATCH["gqkvp"] + (c_q * S + j) * GDN_QKV + c, bf16r(silu(acc)))
                            for j in fx.range_constexpr(CONV_ROWS):
                                _st_bf16(r_cs, (c_slot * CONV_ROWS + j) * GDN_QKV + c, win[j + 1])
                        else:
                            for j in fx.range_constexpr(S):
                                _st_bf16(r_sc, 2 * SCRATCH["gqkvp"] + (c_q * S + j) * GDN_QKV + c, 0.0)
                    lds_barrier()
                if fx.const_expr(k + 1 < 4):
                    frags = nxt
            flag_mine("g1done")
            stamp(2)
            stamp(3)

            # ------------------------------------------------------------ G2 (ReplaySSM)
            if g2_live:
                kh = g2_hv // (GDN_VH // GDN_KH)
                r_rk = rsrc_at(arg("rec_k"), g2_ok)
                r_ru = rsrc_at(arg("rec_u"), g2_ok)
                # replay weights: lane j's record j weighs exp(the gates after it); the
                # checkpoint decays by exp(all of them)
                cs = g_rec
                for o_ in (1, 2, 4, 8, 16):
                    pc = fx.Float32(gpu.shuffle_up(cs, o_, WAVE))
                    cs = (lane >= o_).select(cs + pc, cs)
                tot = wave_sum(g_rec)
                w_rec = (lane < g2_h).select(fmath.exp(tot - cs), fx.Float32(0.0))
                start = fmath.exp(tot)
                unp = [bf16_unpack(w) for w in g2_ck]  # a row's 8 columns
                st = [[unp[i][c] * start for c in range(8)] for i in range(4)]
                for j in fx.range_constexpr(CAP):  # the committed records, unrolled (j < h)
                    lj = j < g2_h
                    kj = bf16_unpack(ld_dw(rsrc_at(arg("rec_k"), g2_ok & lj), ((g2_rec + j) * GDN_D + 4 * lane) // 2, n=2))
                    uj = bf16_unpack(ld_dw(rsrc_at(arg("rec_u"), g2_ok & lj), ((g2_rec + j) * GDN_D + 8 * g2_jb) // 2, n=4))
                    wj = lane_val(w_rec, j)
                    wu = [uj[c] * wj for c in range(8)]
                    st = [[st[i][c] + wu[c] * kj[i] for c in range(8)] for i in range(4)]
                flag_all("g1done", one_wave=False)
                if g2_flush & g2_ok:  # the rebuilt state is the new checkpoint
                    for i in fx.range_constexpr(4):
                        st_dw(r_ck, (sbase + (4 * lane + i) * GDN_D + 8 * g2_jb) // 2, [bf16_pack(st[i][2 * n], st[i][2 * n + 1]) for n in fx.range_constexpr(4)])
                def g2_inputs(t):
                    """Token t's q, k (this k head), v (this wave's 8 columns) and b, a, raw."""
                    qkb = SCRATCH["gqkvp"] + t * (GDN_QKV // 2)
                    return (ld_dw(r_sc, qkb + (kh * GDN_D + 4 * lane) // 2, n=2, cm=CM_DEV)
                            + ld_dw(r_sc, qkb + (GDN_KH * GDN_D + kh * GDN_D + 4 * lane) // 2, n=2, cm=CM_DEV)
                            + ld_dw(r_sc, qkb + (2 * GDN_KH * GDN_D + g2_hv * GDN_D + 8 * g2_jb) // 2, n=4, cm=CM_DEV)
                            + [fx.Uint32(v.bitcast(fx.Int32)) for v in ld_f32(r_sc, SCRATCH["gbap"] + t * (2 * GDN_VH) + 2 * g2_hv, n=2, cm=CM_DEV)])  # fmt: skip

                nxt = g2_inputs(g2_r * S)
                for j in range(S):  # dynamic: one copy of the step, the state and the next inputs carried
                    t = g2_r * S + j
                    cur = nxt
                    nxt = g2_inputs(g2_r * S + fx.min(j + 1, fx.Int32(S - 1)))  # in flight under this token
                    qv = bf16_unpack(cur[0:2])
                    kv = bf16_unpack(cur[2:4])
                    vv = bf16_unpack(cur[4:8])
                    ba = [fx.Int32(w).bitcast(fx.Float32) for w in cur[8:10]]
                    qn = fmath.rsqrt(wave_sum(dot(qv, qv)) + 1e-6) * (GDN_D**-0.5)
                    kn_ = fmath.rsqrt(wave_sum(dot(kv, kv)) + 1e-6)
                    qv = [v * qn for v in qv]
                    kv = [v * kn_ for v in kv]
                    gg = -fmath.exp(g2_alog) * softplus(ba[1] + g2_dtb)
                    beta = bf16r(sigmoid(ba[0]))
                    decay = fmath.exp(gg)
                    st = [[v * decay for v in st[i]] for i in fx.range_constexpr(4)]
                    kvs = col8_sum([st[0][c] * kv[0] + st[1][c] * kv[1] + st[2][c] * kv[2] + st[3][c] * kv[3] for c in fx.range_constexpr(8)], lane)
                    dl = [(vv[c] - lane_val(kvs, 4 * c)) * beta for c in fx.range_constexpr(8)]
                    st = [[st[i][c] + kv[i] * dl[c] for c in fx.range_constexpr(8)] for i in fx.range_constexpr(4)]
                    os_ = col8_sum([st[0][c] * qv[0] + st[1][c] * qv[1] + st[2][c] * qv[2] + st[3][c] * qv[3] for c in fx.range_constexpr(8)], lane)
                    o = [lane_val(os_, 4 * c) for c in fx.range_constexpr(8)]
                    if lane == 0:
                        if g2_ok:  # this token's u record, this wave's 8 columns
                            st_dw(r_ru, ((g2_rec + g2_base + j) * GDN_D + 8 * g2_jb) // 2, [bf16_pack(dl[2 * i], dl[2 * i + 1]) for i in range(4)])
                        st_dw(r_sc, SCRATCH["gop"] + (t * GDN_Z + g2_hv * GDN_D + 8 * g2_jb) // 2,
                              [bf16_pack(g2_ok.select(o[2 * i], fx.Float32(0.0)), g2_ok.select(o[2 * i + 1], fx.Float32(0.0))) for i in range(4)], cm=CM_DEV)
                    if g2_jb == 0:  # stash the shared records: k (a k head's, from its even v head) and g
                        if g2_hv % 2 == 0:
                            st_dw(r_sc, SCRATCH["rks"] + (t * GDN_KH + kh) * (GDN_D // 2) + 2 * lane, [bf16_pack(kv[0], kv[1]), bf16_pack(kv[2], kv[3])], cm=CM_DEV)
                        if lane == 0:
                            st_f32(r_sc, SCRATCH["rgs"] + t * GDN_VH + g2_hv, [gg], cm=CM_DEV)
            flag_mine("g2done")
            stamp(4)
            # out_proj's weights in flight under the barriers (held across G2 they spilled)
            op_f = preload_frags_il("out_proj", fx.min(bid, fx.Int32(OUT_TILES - 1)), bid < OUT_TILES, OUT_KS, OUT_KS // WAVES)
            gnw = bf16_unpack(ld_dw(R("gnorm_w"), 2 * lane, n=2))
            flag_all("g2done")
            # a (token, v head) a wave: the gated RMSNorm of the recurrence output, in place
            # (out_proj's input), and the window's k / g records, now that no G2 wave reads
            # the records
            if gw < T_ * GDN_VH:
                t = gw // GDN_VH
                hv = gw % GDN_VH
                off = SCRATCH["gop"] + (t * GDN_Z + hv * GDN_D + 4 * lane) // 2
                o = bf16_unpack(ld_dw(r_sc, off, n=2, cm=CM_DEV))
                z = bf16_unpack(ld_dw(r_sc, SCRATCH["gzp"] - SCRATCH["gop"] + off, n=2, cm=CM_DEV))
                rq = t // S
                slot = fx.Int32(ld_dw(RP(m2), rq, n=1)[0])
                if slot >= 0:
                    h = fx.max(fx.Int32(ld_dw(RP(m3), slot, n=1)[0]), fx.Int32(0))
                    pos = (h + 2 * S > CAP).select(fx.Int32(0), h) + t % S
                    row = (slot * GDN_VH + hv) * CAP + pos
                    kw = ld_dw(r_sc, SCRATCH["rks"] + (t * GDN_KH + hv // 2) * (GDN_D // 2) + 2 * lane, n=2, cm=CM_DEV)
                    st_dw(R("rec_k"), row * (GDN_D // 2) + 2 * lane, kw)
                    if lane == 0:
                        st_f32(R("rec_g"), row, ld_f32(r_sc, SCRATCH["rgs"] + t * GDN_VH + hv, n=1, cm=CM_DEV))
                rs = fmath.rsqrt(wave_sum(dot(o, o)) / float(GDN_D) + EPS)
                y = [bf16r(o[i] * rs * gnw[i] * silu(z[i])) for i in fx.range_constexpr(4)]
                st_dw(r_sc, off, [bf16_pack(y[0], y[1]), bf16_pack(y[2], y[3])], cm=CM_DEV)
            flag_mine("gndone")
            stamp(5)
            flag_all("gndone")
        else:
            # ------------------------------------------------------------ A1
            frags = a1_f0
            for k in fx.range_constexpr(3):  # tiles b, b + 256, b + 512 (b < 64)
                tile = bid + BLOCKS * k
                live = tile < A1_TILES
                if fx.const_expr(k + 1 < 3):
                    tn = bid + BLOCKS * (k + 1)
                    nxt = preload_frags("qkvg", fx.min(tn, fx.Int32(A1_TILES - 1)), tn < A1_TILES, HIDDEN // 32, 4)
                if live:
                    for g in fx.range_constexpr(G):
                        tile_mma(frags, g * TG, XS, g * WR_G)
                lds_barrier()
                if live & (tid < 256 * G):
                    g_ = tid // 256
                    n = (tid // 16) % 16
                    t = g_ * TG + n
                    y = tile_sum(n, tid % 16, g_ * WR_G)
                    if t < T_:
                        _st_bf16(r_sc, 2 * SCRATCH["qkvgp"] + t * ATT_QKVG + tile * 16 + tid % 16, bf16r(y))
                lds_barrier()
                if fx.const_expr(k + 1 < 3):
                    frags = nxt
            flag_mine("a1done")
            stamp(2)
            stamp(3)

            # ------------------------------------------------------------ A2
            L = L_a
            c = c_a
            r_k = R("k_cache")
            r_v = R("v_cache")
            r_cos = R("cos")
            r_sin = R("sin")
            qnw = bf16_unpack(ld_dw(R("q_norm_w"), 4 * lane, n=4))
            knw = bf16_unpack(ld_dw(R("k_norm_w"), 4 * lane, n=4))
            tq0 = r_a * S

            def rope_tab(t):
                """Token t's cos / sin at dims 8 lane .. (mod 32): positions -> table, loaded
                before the qkvg wait (the chain is off it)."""
                pos = fx.Int32(ld_dw(RP(positions), 2 * t, n=1)[0])
                f0 = (8 * lane) % 32
                return bf16_unpack(ld_dw(r_cos, (pos * 32 + f0) // 2, n=4)), bf16_unpack(ld_dw(r_sin, (pos * 32 + f0) // 2, n=4))

            q_tab = [rope_tab(tq0 + fx.min((wave + 16 * u) // ATT_GQA, fx.Int32(S - 1))) for u in range(CT)]
            k_tab = rope_tab(tq0 + fx.min(wave, fx.Int32(S - 1)))
            flag_all("a1done")

            def norm_rope(src, w_, tab):
                """Head vector at qkvg element ``src`` (+ 8 lane): Gemma RMSNorm (w_), NeoX
                RoPE on dims < 64 with the token's cos / sin ``tab``: 4 packed bf16 dwords."""
                x = bf16_unpack(ld_dw(r_sc, SCRATCH["qkvgp"] + src // 2, n=4, cm=CM_DEV))
                rs = fmath.rsqrt(wave_sum(dot(x, x)) / float(ATT_D) + EPS)
                xn = [bf16r(x[n] * rs * (w_[n] + 1.0)) for n in fx.range_constexpr(8)]
                cs, sn = tab
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

            # B column w + 16 u = (token col // 8, q head 8 kvh + col % 8) of this wave
            for u in fx.range_constexpr(CT):
                col = wave + 16 * u
                if col < NCOL:
                    qw = norm_rope((tq0 + col // ATT_GQA) * ATT_QKVG + ATT_Q_OFF + (kvh * ATT_GQA + col % ATT_GQA) * ATT_D + 8 * lane, qnw, q_tab[u])
                    for n in fx.range_constexpr(4):
                        lds_sti(L_qb, col * QS + 4 * lane + n, qw[n])
                else:
                    for n in fx.range_constexpr(4):
                        lds_sti(L_qb, col * QS + 4 * lane + n, fx.Int32(0))
            if wave < S:  # the new k of token wave (this kv head)
                kw = norm_rope((tq0 + wave) * ATT_QKVG + ATT_K_OFF + kvh * ATT_D + 8 * lane, knw, k_tab)
                for n in fx.range_constexpr(4):
                    lds_sti(L_knb, wave * (ATT_D // 2) + 4 * lane + n, kw[n])
            for x in fx.range_constexpr((S * ATT_D // 2 + THREADS - 1) // THREADS):  # the new v (S x 128 pairs)
                e2 = tid + THREADS * x
                if e2 < S * ATT_D // 2:
                    j = e2 // (ATT_D // 2)
                    d2 = e2 % (ATT_D // 2)
                    vv2 = bf16_unpack(ld_dw(r_sc, SCRATCH["qkvgp"] + ((tq0 + j) * ATT_QKVG + ATT_V_OFF + kvh * ATT_D + 2 * d2) // 2, n=1, cm=CM_DEV))
                    lds_st(L_vn, j * ATT_D + 2 * d2, vv2[0])
                    lds_st(L_vn, j * ATT_D + 2 * d2 + 1, vv2[1])
            lds_barrier()
            stamp(19)

            # the new tokens: token j at position L - S + j, block gn_j, offset o_j, in
            # chunk (gn_j // 4) % NCH's pass (gn_j // 4) // NCH. Its owner writes K / V to
            # the cache and patches them into that pass's fragments
            new = []
            for j in fx.range_constexpr(S):
                pj = L - S + j
                gn = fx.max(pj, fx.Int32(0)) // 16
                own = ((gn // 4) % NCH == c) & (pj >= 0) & att_ok
                new.append((own, (gn // 4) // NCH, gn % 4, fx.max(pj, fx.Int32(0)) % 16))
            if wave < S:
                for j in fx.range_constexpr(S):
                    own = new[j][0]
                    slot = fx.Int32(ld_dw(RP(m2), 2 * (tq0 + j), n=1)[0])
                    if (wave == j) & own & (slot >= 0):
                        blk = slot // 16
                        off = slot % 16
                        st_dw(r_k, ((((blk * ATT_KVH + kvh) * 32 + lane) * 16 + off) * 8) // 2, lds_ld4w(L_knb, j * (ATT_D // 2) + 4 * lane))
                        for n in fx.range_constexpr(8):
                            _st_bf16(r_v, (((blk * ATT_KVH + kvh) * 2 + off // 8) * ATT_D + 8 * lane + n) * 8 + off % 8, lds_ld(L_vn, j * ATT_D + 8 * lane + n))

            stamp(22)

            def patch(p, ks, vs):
                out_k = [list(ks[i]) for i in range(2)]
                out_v = [list(vs[q]) for q in range(2)]
                for j in fx.range_constexpr(S):
                    own, p_j, b_j, o_j = new[j]
                    here = own & (p == p_j)
                    for i in fx.range_constexpr(2):
                        kst = 2 * (wave % 4) + i
                        kn = lds_ld8w(L_knb, j * (ATT_D // 2) + 16 * kst + 8 * h16)
                        hit = here & (wave // 4 == b_j) & (n16 == o_j)
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

            G_a = (L + 15) // 16
            o_acc = [[fx.Float32(0.0) for _ in fx.range_constexpr(8)] for _ in range(CT)]
            m_r = [fx.Float32(NEG) for _ in range(CT)]
            l_r = [fx.Float32(0.0) for _ in range(CT)]
            kc = pre_k
            vc = pre_v
            bnx = [block_of(1, b, (NCH + c) * 4 < G_a) for b in range(4)]
            p = fx.Int32(0)
            while att_ok & ((p * NCH + c) * 4 < G_a):
                stamp(25)  # timing builds: the last pass's start
                kp, vp_ = patch(p, kc, vc)
                live1 = ((p + 1) * NCH + c) * 4 < G_a
                kn, vn = kv_frags(bnx, rsrc_at(arg("k_cache"), live1), rsrc_at(arg("v_cache"), live1))
                bnx2 = [block_of(p + 2, b, ((p + 2) * NCH + c) * 4 < G_a) for b in range(4)]
                # QK: wave (block wave // 4, K quarter wave % 4), every column tile
                for ct in fx.range_constexpr(CT):
                    d = [fx.Float32(0.0)] * 8
                    for i in fx.range_constexpr(2):
                        kst = 2 * (wave % 4) + i
                        qf = lds_ld8w(L_qb, (16 * ct + n16) * QS + 16 * kst + 8 * h16)
                        d = wmma_bf16(kp[i], qf, d)
                    base = ((wave // 4 * 4 + wave % 4) * NCOLP + 16 * ct + n16) * SPS + 8 * h16
                    lds_st4(L_sp, base, d[:4])
                    lds_st4(L_sp, base + 4, d[4:])
                lds_barrier()
                stamp(23)
                # softmax: wave w = columns w + 16 u, lanes over positions lane, lane + 32
                m_nx = []
                l_nx = []
                for u in fx.range_constexpr(CT):
                    col = wave + 16 * u
                    lim = L - S + col // ATT_GQA
                    sc = []
                    for half in fx.range_constexpr(2):
                        x_ = lane + 32 * half
                        b_ = x_ // 16
                        r_ = x_ % 16
                        acc = lds_ld(L_sp, ((b_ * 4) * NCOLP + col) * SPS + r_)
                        for kq in fx.range_constexpr(1, 4):
                            acc = acc + lds_ld(L_sp, ((b_ * 4 + kq) * NCOLP + col) * SPS + r_)
                        ok = ((((p * NCH + c) * 4 + b_) * 16 + r_) <= lim) & (col < NCOL)
                        sc.append(ok.select(acc * ATT_SCALE, fx.Float32(NEG)))
                    m_new = fx.max(m_r[u], wave_max(fx.max(sc[0], sc[1])))
                    alpha = fmath.exp(m_r[u] - m_new)
                    pr = [(sc[h_] > NEG * 0.5).select(fmath.exp(sc[h_] - m_new), fx.Float32(0.0)) for h_ in fx.range_constexpr(2)]
                    l_nx.append(l_r[u] * alpha + wave_sum(pr[0] + pr[1]))
                    m_nx.append(m_new)
                    for half in fx.range_constexpr(2):
                        nxt_p = pr[half].shuffle_xor(1, WAVE)
                        if lane % 2 == 0:
                            lds_sti(L_pb, col * PS + (lane + 32 * half) // 2, bf16_pack(pr[half], nxt_p))
                    if lane == 0:
                        lds_st(L_al, col, alpha)
                m_r = m_nx
                l_r = l_nx
                lds_barrier()
                stamp(24)
                # PV: wave w = dims 16 w .., every column tile
                o_nx = []
                for ct in fx.range_constexpr(CT):
                    al = lds_ld(L_al, 16 * ct + n16)
                    acc_o = [v * al for v in o_acc[ct]]
                    for q in fx.range_constexpr(2):
                        pf = lds_ld4w(L_pb, (16 * ct + n16) * PS + (2 * q) * 8 + 4 * h16) + lds_ld4w(L_pb, (16 * ct + n16) * PS + (2 * q + 1) * 8 + 4 * h16)
                        acc_o = wmma_bf16(vp_[q], pf, acc_o)
                    o_nx.append(acc_o)
                o_acc = o_nx
                lds_barrier()
                p = p + 1
                kc = kn
                vc = vn
                bnx = bnx2
            stamp(20)
            # the partials (unnormalized O, running max / sum) of (CTA, column)
            if att_ok:
                for ct in fx.range_constexpr(CT):
                    pidx = bid * ATT_COLS + 16 * ct + n16
                    st_f32(r_sc, SCRATCH["aop"] + pidx * ATT_D + 16 * wave + 8 * h16, o_acc[ct][:4], cm=CM_DEV)
                    st_f32(r_sc, SCRATCH["aop"] + pidx * ATT_D + 16 * wave + 8 * h16 + 4, o_acc[ct][4:], cm=CM_DEV)
                if lane == 0:
                    for u in fx.range_constexpr(CT):
                        st_f32(r_sc, SCRATCH["aml"] + (bid * ATT_COLS + wave + 16 * u) * 2, [m_r[u], l_r[u]], cm=CM_DEV)
            rocdl.s_wait_storecnt(0)
            lds_barrier()
            if (tid == 0) & att_ok:
                put1(r_mb, MBOX["aflag"] + bid, 1.0, tag)
            stamp(4)
            stamp(5)
            op_f = preload_frags_il("o_proj", fx.min(bid, fx.Int32(OUT_TILES - 1)), bid < OUT_TILES, OUT_KS, OUT_KS // WAVES)

            # ------------------------------------------------------------ A3
            # (token, q head, 32 dims) a wave; lanes over its (request, kv head)'s chunks
            if gw < T_ * ATT_QH * 8:
                t = gw // (ATT_QH * 8)
                h = (gw // 8) % ATT_QH
                db = gw % 8
                rq = t // S
                col = (t % S) * ATT_GQA + h % ATT_GQA
                NCH = fx.max(a_pick(a_nch, rq), fx.Int32(1))  # this token's request's
                b0 = a_pick(a_base, rq) + (h // ATT_GQA) * NCH
                flags = []
                for k in fx.range_constexpr(CH_LANE):
                    cc = lane + k * WAVE
                    flags.append(MBOX["aflag"] + b0 + (cc < NCH).select(cc, 0))
                _ = poll1(r_mb, flags, tag)
                ms = []
                ls = []
                for k in fx.range_constexpr(CH_LANE):
                    cc = lane + k * WAVE
                    ok = cc < NCH
                    ml = ld_f32(r_sc, SCRATCH["aml"] + ((b0 + ok.select(cc, 0)) * ATT_COLS + col) * 2, n=2, cm=CM_DEV)
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
                    ob = SCRATCH["aop"] + ((b0 + ok.select(cc, 0)) * ATT_COLS + col) * ATT_D + 32 * db
                    for v4 in fx.range_constexpr(8):
                        vals = ld_f32(r_sc, ob + 4 * v4, cm=CM_DEV)
                        for n in fx.range_constexpr(4):
                            acc[4 * v4 + n] = acc[4 * v4 + n] + wts[k] * vals[n]
                mine = wave_sum_scatter(acc, lane)  # lane n: the wave sum of acc[n]
                Lsafe = (Ls > 0.0).select(Ls, fx.Float32(1.0))
                d = 32 * db + lane
                gate = bf16_at(ld_dw(r_sc, SCRATCH["qkvgp"] + (t * ATT_QKVG + ATT_G_OFF + h * ATT_D + d) // 2, n=1, cm=CM_DEV)[0], d % 2)
                ao = bf16r(bf16r(mine / Lsafe) * bf16r(sigmoid(gate)))
                _st_bf16(r_sc, 2 * SCRATCH["aop2"] + t * ATT_O + h * ATT_D + d, ao)
            flag_mine("a3done")
            stamp(6)
            stamp(7)
            flag_all("a3done")

        # ---------------------------------------------------------------- out_proj / o_proj
        # CTA b < 128: rows 16 b .., the whole K in registers (wave w: K steps w + 16 m);
        # the B fragments (token n16, 16 dims) straight from the bf16 input rows (a CTA
        # >= 128 loads and multiplies zeros: no value assigned under a dynamic branch)
        r_in = rsrc_at(arg("scratch"), bid < OUT_TILES)
        src = SCRATCH[_pick(kind == "gdn", "gop", "aop2")]
        acc_g = []
        for g in fx.range_constexpr(G):
            x0 = src + fx.min(g * TG + n16, T_ - 1) * (OUT_K // 2) + h16 * 8
            xf = [ld_dw(r_in, x0 + (wave + WAVES * m) * 16, cm=CM_DEV) + ld_dw(r_in, x0 + (wave + WAVES * m) * 16 + 4, cm=CM_DEV) for m in fx.range_constexpr(OUT_KS // WAVES)]
            a_ = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
            for m in fx.range_constexpr(OUT_KS // WAVES):
                a_ = wmma_bf16(op_f[m], xf[m], a_)
            acc_g.append(a_)
        if bid < OUT_TILES:
            for g in fx.range_constexpr(G):
                lds_st4(L_wr, g * WR_G + (wave * TG + n16) * WRS + h16 * 8, acc_g[g][:4])
                lds_st4(L_wr, g * WR_G + (wave * TG + n16) * WRS + h16 * 8 + 4, acc_g[g][4:])
        lds_barrier()
        if (bid < OUT_TILES) & (tid < 256 * G):
            g_ = tid // 256
            n = (tid // 16) % 16
            r = tid % 16
            t = g_ * TG + n
            y = tile_sum(n, r, g_ * WR_G)
            if t < T_:
                lds_st(L_part, t * 16 + r, bf16r(y))
        lds_barrier()
        stamp(_pick(kind == "gdn", 6, 8))

        nx_rows = fx.Int32(arg("nx_rows"))
        if nx_rows > 0:
            if bid < nx_rows:
                prefetch_line(arg("nx_first") + fx.Int64(bid * 65536 + tid * 128))
            if bid < OUT_TILES:
                prefetch_line(arg("nx_out") + fx.Int64(bid * 131072 + tid * 128))
                prefetch_line(arg("nx_out") + fx.Int64(bid * 131072 + 65536 + tid * 128))
            if bid < M1_TILES:
                prefetch_line(arg("nx_m1") + fx.Int64(bid * 65536 + tid * 128))

        m1_f = preload_frags("m1_w", fx.min(bid, fx.Int32(M1_TILES - 1)), bid < M1_TILES, HIDDEN // 32, 4)
        sh_f = preload_frags("sh_w2", fx.min(bid, fx.Int32(OUT_TILES - 1)), bid < OUT_TILES, SH_INTER // 32, 1)

        # ---------------------------------------------------------------- gathers
        def push(name):
            """CTA b < 128: its 16 dims of every token (L_part) into the mailbox region
            ``name``: line (t, h) = dims 16 b + 4 h .. + 3 of token t, two bf16 a pair."""
            if (bid < OUT_TILES) & (tid < T_ * 4):
                t = tid // 4
                hh = tid % 4
                v = lds_ld4(L_part, t * 16 + 4 * hh)
                pair = MBOX[name] + t * (HIDDEN // 2) + bid * 8 + 2 * hh
                st_dw(r_mb, pair * 2, [bf16_pack(v[0], v[1]), tag, bf16_pack(v[2], v[3]), tag], cm=CM_DEV)

        def pull(name, t):
            """Token t's dims 4 tid .. + 3 from the mailbox region ``name``."""
            got = poll2(r_mb, [MBOX[name] + t * (HIDDEN // 2) + 2 * tid], tag)
            return bf16_unpack([fx.Uint32(got[0].bitcast(fx.Int32)), fx.Uint32(got[1].bitcast(fx.Int32))])

        push("ar0")
        # gather 0 + N2, token-major: CTA 128 + t receives all of token t (thread: dims
        # 4 tid .. + 3), s2 = attn + bf16(h + res), xn = RMSNorm(s2) (1 + w2); stores
        # res_out, xn (packed bf16, M1's B) and its MXFP8 copy (M3's B), then the token's
        # flag
        t_ar = bid - OUT_TILES
        if (bid >= OUT_TILES) & (t_ar < T_):
            part = pull("ar0", t_ar)
            hv_ = bf16_unpack(ld_dw(R("h_in"), row_dw(t_ar, wave), n=2))
            rv_ = bf16_unpack(ld_dw(R("res_in"), row_dw(t_ar, wave), n=2))
            s2 = [bf16r(part[i]) + bf16r(hv_[i] + rv_[i]) for i in fx.range_constexpr(4)]
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
        # CTAs < 81: router (17 tiles, row 256 the shared gate logit) and the shared
        # expert's gate / up (64 tiles, 8 + 8 rows); every token's normalized row once its
        # flag is up
        if bid < M1_TILES:
            for g in fx.range_constexpr(G):
                t = g * TG + wave
                if t < T_:
                    _ = poll1(r_mb, [MBOX["xqf"] + t], tag)
                    for m in fx.range_constexpr(16):
                        xw = ld_dw(r_sc, SCRATCH["agp"] + row_dw(t, m), n=2, cm=CM_DEV)
                        lds_sti(L_xb, t * XS + 64 * m + 2 * lane, xw[0])
                        lds_sti(L_xb, t * XS + 64 * m + 2 * lane + 1, xw[1])
            lds_barrier()
            if fx.const_expr(kind == "gdn"):
                stamp(9)
            for g in fx.range_constexpr(G):
                tile_mma(m1_f, g * TG, XS, g * WR_G)
            lds_barrier()
            if tid < 256 * G:
                g_ = tid // 256
                n = (tid // 16) % 16
                r = tid % 16
                t2 = g_ * TG + n
                y = tile_sum(n, r, g_ * WR_G)
                row = bid * 16 + r
                if t2 < T_:
                    if bid < M1_RT_TILES:
                        if row < ROUTER_ROWS:
                            put1(r_mb, MBOX["logits"] + t2 * LOGIT_STRIDE + row, bf16r(y), tag)
                    else:
                        lds_st(L_y, t2 * 16 + r, y)
            lds_barrier()
            if (bid >= M1_RT_TILES) & (tid < 8 * TMAX):
                n = tid // 8
                r = tid % 8
                if n < T_:
                    j_sh = (bid - M1_RT_TILES) * 8 + r
                    g_ = lds_ld(L_y, n * 16 + r)
                    u_ = lds_ld(L_y, n * 16 + r + 8)
                    st_f32(r_sc, SCRATCH["shact"] + n * SH_INTER + j_sh, [bf16r(silu(bf16r(g_)) * bf16r(u_))], cm=CM_DEV)
            lds_barrier()
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
                if pub:
                    for k in fx.range_constexpr(TOP_K):
                        put2(r_mb, MBOX["sel"] + (t * (TOP_K + 1) + k) * 2, picked_i[k].bitcast(fx.Float32), es[k] / tot, tag)
                    put2(r_mb, MBOX["sel"] + (t * (TOP_K + 1) + TOP_K) * 2, sigmoid(g), 0.0, tag)
                else:
                    for k in fx.range_constexpr(TOP_K):
                        lds_sti(L_sele, t * TOP_K + k, picked_i[k])
                        lds_st(L_selw, t * TOP_K + k, es[k] / tot)
                    lds_st(L_sg, t, sigmoid(g))

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
            cs_ = cn[0]
            gs_ = gn[0]
            for i in fx.range_constexpr(1, 8):
                cs_ = cs_ + cn[i]
                gs_ = gs_ + gn[i]
            ci = cs_
            gi = gs_
            for o_ in (1, 2, 4, 8, 16):
                pc = fx.Int32(gpu.shuffle_up(ci, o_, WAVE))
                pg = fx.Int32(gpu.shuffle_up(gi, o_, WAVE))
                ci = (lane >= o_).select(ci + pc, ci)
                gi = (lane >= o_).select(gi + pg, gi)
            cb = ci - cs_
            gb = gi - gs_
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

        # the step's routed weights into L2 once the groups are known, before M3's input
        # is: this CTA's M3 tasks' w13 tiles and scales and (CTAs < 128) its w2 tile of
        # every group, so M3's later rounds and M4's groups load from L2
        pf_w13, pf_w13s, pf_w2, pf_w2s = arg("w13"), arg("w13_s"), arg("w2"), arg("w2_s")
        if fx.const_expr(G == 1):
            for i in range((NG * 16 + BLOCKS - 1) // BLOCKS):  # task q: 4 tiles, 64 KB
                q = bid + BLOCKS * fx.Int32(i)
                if q < NG * 16:
                    e = uniform(lds_ldi(L_grp, q // 16)) & 0xFF
                    prefetch_line(pf_w13 + fx.Int64(e * W13_EXPERT_BYTES + (q % 16) * 65536 + tid * 128))
                    if tid < 32:
                        prefetch_line(pf_w13s + fx.Int64(e * W13S_EXPERT_BYTES + (q % 16) * 4096 + tid * 128))
        else:
            for i in range((NG * 4 + BLOCKS - 1) // BLOCKS):  # task: 16 tiles, 256 KB
                gq = bid + BLOCKS * fx.Int32(i)
                if gq < NG * 4:
                    e = uniform(lds_ldi(L_grp, gq // 4)) & 0xFF
                    for x in fx.range_constexpr(4):
                        prefetch_line(pf_w13 + fx.Int64(e * W13_EXPERT_BYTES + (gq % 4) * 262144 + (tid + THREADS * x) * 128))
                    if tid < 128:
                        prefetch_line(pf_w13s + fx.Int64(e * W13S_EXPERT_BYTES + (gq % 4) * 16384 + tid * 128))
        if bid < OUT_TILES:
            for k in range((NG + WAVES - 1) // WAVES):  # wave w: group w + 16 k's 4 KB tile
                gi = wave + WAVES * fx.Int32(k)
                if gi < NG:
                    e = uniform(lds_ldi(L_grp, gi)) & 0xFF
                    prefetch_line(pf_w2 + fx.Int64(e * W2_EXPERT_BYTES + bid * (KS_W2 * 1024) + lane * 128))
                    if lane < 4:
                        prefetch_line(pf_w2s + fx.Int64(e * W2S_EXPERT_BYTES + (bid // 2) * (KS_W2 * 128) + lane * 128))

        def group(gi):
            gp = uniform(lds_ldi(L_grp, gi))
            return gp & 0xFF, (gp >> 8) & 0x3FF, gp >> 20

        # ---------------------------------------------------------------- M3 routed gate / up
        arg_w13, arg_w13s, arg_w2, arg_w2s = arg("w13"), arg("w13_s"), arg("w2"), arg("w2_s")

        def quant_store(av, s_, blk):
            """A wave's 32 activations (lane l: activation 32 blk + l of slot s_) as MXFP8."""
            am = wave_max(abs_f(av))
            ex = fx.min(fx.max((am.bitcast(fx.Int32) >> 23) & 0xFF, fx.Int32(7)), fx.Int32(254))
            mult = ((261 - ex) << 23).bitcast(fx.Float32)
            qb = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(av * mult).ir_value(), fx.Float32(0.0).ir_value(), fx.Int32(0).ir_value(), False)) & 0xFF
            dw = qb | (qb.shuffle_xor(1, WAVE) << 8) | (qb.shuffle_xor(2, WAVE) << 16) | (qb.shuffle_xor(3, WAVE) << 24)
            if lane % 4 == 0:
                st_dw(r_sc, SCRATCH["actd"] + s_ * (INTER // 4) + blk * 8 + lane // 4, [dw], cm=CM_DEV)
            if lane == 0:
                st_dw(r_sc, SCRATCH["acts"] + s_ * (INTER // 32) + blk, [ex - 7], cm=CM_DEV)

        def m3_task(i):
            """More than 16 tokens: task = (group, quarter: tiles 16 quarter .. + 15), wave
            w = tile 16 quarter + w over the whole K, the group's MXFP8 x staged in LDS."""
            gi = bid + BLOCKS * i
            ok3 = gi < NG * 4
            e, start, cnt = group(fx.min(gi, NG * 4 - 1) // 4)
            qr = gi % 4
            tn = tid // 32
            sn = lds_ldi(L_list, start + fx.min(tn, cnt - 1))
            src = SCRATCH["xq"] + (sn // TOP_K) * (HIDDEN // 4) + 16 * (tid % 32)
            r_scq = rsrc_at(arg("scratch"), ok3)
            stg = [ld_dw(r_scq, src + 4 * v, cm=CM_DEV) for v in range(4)]
            sn2 = lds_ldi(L_list, start + fx.min(tid // KS_X, cnt - 1))
            stg_s = ld_dw(r_scq, SCRATCH["xs"] + (sn2 // TOP_K) * KS_X + tid % KS_X, n=1, cm=CM_DEV)[0]
            rw = rsrc_at(arg_w13, ok3)
            rws = rsrc_at(arg_w13s, ok3)
            tile = 16 * qr + wave
            n = tile * 16 + n16
            afr = []
            for ks in fx.range_constexpr(KS_X):
                ob = e * W13_EXPERT_BYTES + (tile * 32 + 2 * ks) * 512 + lane * 16
                sa = ld_dw(rws, e * (W13S_EXPERT_BYTES // 4) + (n // 32) * 512 + ks * 32 + n % 32, n=1)[0]
                afr.append((ld_dw(rw, ob // 4) + ld_dw(rw, (ob + 512) // 4), sa))
            if ok3:
                for v in fx.range_constexpr(4):
                    for x in fx.range_constexpr(4):
                        lds_sti(L_xb, tn * XQS + 16 * (tid % 32) + 4 * v + x, stg[v][x])
                if tid < 16 * KS_X:
                    lds_sti(L_xb, 16 * XQS + tid, stg_s)
            lds_barrier()
            acc = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
            for ks in fx.range_constexpr(KS_X):
                bq = []
                for j in fx.range_constexpr(4):
                    bq += lds_ld4w(L_xb, n16 * XQS + 32 * ks + 8 * j + 4 * h16)
                acc = wmma_fp4_fp8(afr[ks][0], bq, afr[ks][1], lds_ldi(L_xb, 16 * XQS + n16 * KS_X + ks), acc)
            if ok3:
                # rows 8 h16 + 2 i (gate) / + 1 (up): activation 8 tile + 4 h16 + i (local
                # 8 wave + 4 h16 + i of the quarter's 128), column n16
                for i in fx.range_constexpr(4):
                    lds_st(L_act, n16 * 128 + 8 * wave + 4 * h16 + i, silu(acc[2 * i]) * acc[2 * i + 1])
            lds_barrier()
            if ok3:
                for uu in fx.range_constexpr(4):  # unit 4 wave + uu: token unit // 4, block unit % 4
                    nn = (4 * wave + uu) // 4
                    bb = (4 * wave + uu) % 4
                    if nn < cnt:
                        s_ = lds_ldi(L_list, start + nn)
                        quant_store(lds_ld(L_act, nn * 128 + 32 * bb + lane), s_, 4 * qr + bb)
            lds_barrier()

        def m3_task_quad(i):
            """Up to 16 tokens: task q = (group q // 16, quad q % 16: tiles 4 quad .. + 3,
            activations 32 quad .. + 31), wave w = K step w, the 16 K steps summed in LDS."""
            q = bid + BLOCKS * i
            ok3 = q < NG * 16
            e, start, cnt = group(fx.min(q, NG * 16 - 1) // 16)
            quad = q % 16
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
            sbq = ld_dw(r_scq, SCRATCH["xs"] + t_n * KS_X + wave, n=1, cm=CM_DEV)[0]
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
                    lds_st(L_act, o, y)
            lds_barrier()
            if ok3 & (wave < cnt):
                # wave = column: lane l = activation 32 quad + l (tile l // 8, gate row 2 (l % 8), up + 1)
                s_ = lds_ldi(L_list, start + wave)
                yb = ((lane // 8) * 16 + wave) * 16 + 2 * (lane % 8)
                quant_store(silu(lds_ld(L_act, yb)) * lds_ld(L_act, yb + 1), s_, quad)
            lds_barrier()

        xf = MBOX["xqf"] + fx.min(2 * lane, T_ - 2)
        _ = poll1(r_mb, [xf, xf + 1], tag)
        if fx.const_expr(G == 1):
            for i in range((NG * 16 + BLOCKS - 1) // BLOCKS):
                m3_task_quad(fx.Int32(i))
        else:
            for i in range((NG * 4 + BLOCKS - 1) // BLOCKS):
                m3_task(fx.Int32(i))
        flag_mine("m3done")
        stamp(_pick(kind == "gdn", 12, 13))
        if fx.const_expr(kind == "gdn"):
            stamp(13)

        # ---------------------------------------------------------------- M4 down + gather 1
        m4t = bid

        def m4_wload(gi):
            """Group gi's w2 tile (this CTA's 16 rows) and its scales: known since M2."""
            ok = gi < NG
            e = group(fx.min(gi, NG - 1))[0]
            n = m4t * 16 + n16
            rw = rsrc_at(arg_w2, ok)
            rws_ = rsrc_at(arg_w2s, ok)
            out = []
            for ks in fx.range_constexpr(KS_W2):
                ob = e * W2_EXPERT_BYTES + m4t * (KS_W2 * 1024) + ks * 1024 + lane * 16
                sa = ld_dw(rws_, e * (W2S_EXPERT_BYTES // 4) + (n // 32) * (KS_W2 * 32) + ks * 32 + n % 32, n=1)[0]
                out.append((ld_dw(rw, ob // 4) + ld_dw(rw, (ob + 512) // 4), sa))
            return out

        if bid < OUT_TILES:
            w_first = [m4_wload(wave + WAVES * u) for u in range(2)]  # in flight under the M3 barrier
            flag_all("m3done")
            for k in range((T_ + WAVES - 1) // WAVES):  # the shared expert's activations
                t = wave + WAVES * fx.Int32(k)
                if t < T_:
                    for x in fx.range_constexpr(SH_INTER // 128):
                        st_xb(L_shb, SHS, t, 128 * x + 4 * lane, ld_f32(r_sc, SCRATCH["shact"] + t * SH_INTER + 128 * x + 4 * lane, cm=CM_DEV))

            def m4_load(gi, w=None):
                ok = gi < NG
                e, start, cnt = group(fx.min(gi, NG - 1))
                s_n = lds_ldi(L_list, start + fx.min(n16, cnt - 1))
                if fx.const_expr(w is None):
                    w = m4_wload(gi)
                ld = []
                for ks in fx.range_constexpr(KS_W2):
                    af, sa = w[ks]
                    bq = []
                    for j in fx.range_constexpr(4):
                        bq += ld_dw(r_sc, SCRATCH["actd"] + s_n * (INTER // 4) + 32 * ks + 8 * j + 4 * h16, cm=CM_DEV)
                    sc4 = ld_dw(r_sc, SCRATCH["acts"] + s_n * (INTER // 32) + 4 * ks, cm=CM_DEV)
                    sb = (fx.Int32(sc4[0]) & 0xFF) | ((fx.Int32(sc4[1]) & 0xFF) << 8) | ((fx.Int32(sc4[2]) & 0xFF) << 16) | ((fx.Int32(sc4[3]) & 0xFF) << 24)
                    ld.append((af, sa, bq, sb))
                return ok, cnt, s_n, ld

            def m4_grp(m, w=None):
                lds_ = [m4_load(wave + WAVES * (2 * m + u), None if w is None else w[u]) for u in range(2)]
                for u in fx.range_constexpr(2):
                    ok, cnt, s_n, ld = lds_[u]
                    d = [fx.Float32(0.0)] * 8
                    for ks in fx.range_constexpr(KS_W2):
                        af, sa, bq, sb = ld[ks]
                        d = wmma_fp4_fp8(af, bq, sa, sb, d)
                    if ok & (n16 < cnt):
                        lds_st4(L_slot, s_n * 16 + 8 * h16, d[:4])
                        lds_st4(L_slot, s_n * 16 + 8 * h16 + 4, d[4:])

            m4_grp(fx.Int32(0), w_first)
            for m in range(1, (NG + 2 * WAVES - 1) // (2 * WAVES)):
                m4_grp(fx.Int32(m))
            lds_barrier()
            stamp(21)

            # shared: the tokens as columns, K step = wave; then the routed slots (fixed
            # expert order) + gate x shared
            for g in fx.range_constexpr(G):
                d = wmma_bf16(sh_f[0], lds_ld8w(L_shb, (g * TG + n16) * SHS + 16 * wave + 8 * h16), [fx.Float32(0.0)] * 8)
                lds_st4(L_wr, g * WR_G + (wave * TG + n16) * WRS + h16 * 8, d[:4])
                lds_st4(L_wr, g * WR_G + (wave * TG + n16) * WRS + h16 * 8 + 4, d[4:])
            lds_barrier()
            if tid < 256 * G:
                g_ = tid // 256
                n = (tid // 16) % 16
                r = tid % 16
                t = g_ * TG + n
                if t < T_:
                    shared = tile_sum(n, r, g_ * WR_G)
                    routed = fx.Float32(0.0)
                    for kk in fx.range_constexpr(TOP_K):
                        routed = routed + lds_ld(L_selw, t * TOP_K + kk) * lds_ld(L_slot, (t * TOP_K + kk) * 16 + r)
                    lds_st(L_part, t * 16 + r, bf16r(routed + lds_ld(L_sg, t) * shared))
            lds_barrier()
        stamp(14)
        push("ar1")
        # gather 1, token-major (CTA 128 + t): h_out, and unless this is the last layer
        # the next layer's N1, xn = RMSNorm(h_out + res_out) (1 + its ln1_w, 2 HIDDEN
        # elements on in the packed norm weights) into xn1
        if (bid >= OUT_TILES) & (t_ar < T_):
            hs1 = [bf16r(v) for v in pull("ar1", t_ar)]
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
        a00: fx.Int64, a01: fx.Int64, a02: fx.Int64, a03: fx.Int64,
        a04: fx.Int64, a05: fx.Int64, a06: fx.Int64, a07: fx.Int64,
        a08: fx.Int64, a09: fx.Int64, a10: fx.Int64, a11: fx.Int64,
        a12: fx.Int64, a13: fx.Int64, a14: fx.Int64, a15: fx.Int64,
        a16: fx.Int64, a17: fx.Int64, a18: fx.Int64, a19: fx.Int64,
        a20: fx.Int64, a21: fx.Int64, a22: fx.Int64, a23: fx.Int64,
        a24: fx.Int64, a25: fx.Int64, a26: fx.Int64, a27: fx.Int64,
        a28: fx.Int64, a29: fx.Int64, a30: fx.Int64, a31: fx.Int64,
        nreq: fx.Int32,
        cnt_gdn: fx.Pointer,
        cnt_att: fx.Pointer,
        positions: fx.Pointer,
        m0: fx.Pointer,
        m1: fx.Pointer,
        m2: fx.Pointer,
        m3: fx.Pointer,
        mstride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):  # fmt: skip
        _ = build_key
        kernel(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, nreq, cnt_gdn, cnt_att, positions, m0, m1, m2, m3, mstride).launch(
            grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch
