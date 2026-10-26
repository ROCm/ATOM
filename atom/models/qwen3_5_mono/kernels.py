# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5-35B-A3B fused decoder layer, one persistent gfx1250 FlyDSL kernel a layer
(TP4 rank, 1..MAX_TOKENS decode tokens). BLOCKS CTAs of THREADS threads, all
co-resident; CTA b owns hidden dims [8b, 8b + 8) in both all-reduces.

GDN layer (``kind="gdn"``)::

    N1   every CTA   s = h + res; xn = RMSNorm(s) (1 + w)            -> LDS
    G1   row a wave  in_proj_qkvz | in_proj_ba; conv update fused on q|k|v rows
    ---  barrier
    G2   (token, v head, 8 columns) a wave: l2norm(q, k), gating, delta-rule step
    ---  barrier
    G3   every CTA gated RMSNorm; CTA b: out_proj rows 8b..8b+7; all-reduce slot 0
    ---  barrier (the reduced rows, all-gathered through scratch)
    N2   every CTA   s2 = attn + bf16(s); xn = RMSNorm(s2) (1 + w2)
    M1   row a wave  router (+ shared gate) and the shared expert's gate / up
    ---  barrier
    M2   every CTA top-8;  M3 routed experts' gate / up (MXFP4)
    ---  barrier
    M4   CTA b: down rows 8b..8b+7 (8 experts + shared); all-reduce slot 1 -> h_out

Full-attention layer (``kind="att"``): G1..G3 become A1 qkvg rows | barrier | A2 q/k
norm + RoPE, the new token's K / V into the cache, split-context attention partials
| barrier | A3 combine + output gate | barrier | A4 o_proj rows + all-reduce.

Numerics follow ATOM's bf16 roundings (projection outputs, norms, conv / ssm state,
beta, the gated outputs); the MoE activations stay f32 (ATOM quantizes them to MXFP4).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, ptrtoint, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen3_5_mono.config import (
    ATT_D,
    ATT_QH,
    ATT_SCALE,
    BLOCKS,
    EPS,
    GDN_D,
    GDN_KH,
    GDN_QKV,
    GDN_QKVZ,
    GDN_VH,
    HIDDEN,
    INTER,
    MAX_TOKENS,
    N_EXPERTS,
    ROUTER_ROWS,
    SH_INTER,
    THREADS,
    TOP_K,
    TP,
    WAVE,
    WAVES,
    GRID_WAVES,
)
from atom.models.qwen3_5_mono.layout import (
    A1_TILES,
    G1_TILES,
    M1_RT_TILES,
    M1_TILES,
    OUT_TILES,
    STAMPS,
    ACT_SLOTS,
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
    grid_barrier,
    lds_addr,
    compiler_fence,
    lds_barrier,
    lds_ld,
    lds_ld4,
    lds_ld8,
    lds_ld4w,
    lds_ld8w,
    lds_ldi,
    lds_st,
    lds_st4,
    lds_sti,
    ld_dw,
    ld_f32,
    ll_poll,
    ll_put,
    prefetch_line,
    opaque,
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
)

NEG = -1.0e30
W13_EXPERT_BYTES = 512 * 1024  # [512 rows (gate / up interleaved, padded), 1024 B]
W13S_EXPERT_BYTES = 16 * 2048
W2_EXPERT_BYTES = 2048 * 128
W2S_EXPERT_BYTES = 64 * 256


@fx.struct
class _Lds:
    s: fx.Array[fx.Float32, MAX_TOKENS * HIDDEN, 16]  # bf16(h + res)
    act: fx.Array[fx.Float32, MAX_TOKENS * ACT_SLOTS * INTER, 16]
    red: fx.Array[fx.Float32, 256, 16]
    sel_e: fx.Array[fx.Float32, MAX_TOKENS * TOP_K, 16]  # i32 bits
    sel_w: fx.Array[fx.Float32, MAX_TOKENS * TOP_K, 16]
    sg: fx.Array[fx.Float32, 16, 16]
    part: fx.Array[fx.Float32, MAX_TOKENS * 16, 16]
    rcv: fx.Array[fx.Float32, TP * MAX_TOKENS * 16, 16]
    xb: fx.Array[fx.Float32, MAX_TOKENS * HIDDEN // 2, 16]  # xn as packed bf16 (WMMA B)
    gb: fx.Array[fx.Float32, MAX_TOKENS * 512, 16]  # go as packed bf16
    wr: fx.Array[fx.Float32, WAVES * MAX_TOKENS * 16, 16]  # a tile's per-wave partials
    y16: fx.Array[fx.Float32, 2 * MAX_TOKENS * 16, 16]
    xq: fx.Array[fx.Float32, MAX_TOKENS * HIDDEN // 4, 16]  # N2's xn as MXFP8 (M3's B)
    xs: fx.Array[fx.Float32, MAX_TOKENS * HIDDEN // 128, 16]  # its scales, a dword a K step
    aq: fx.Array[fx.Float32, MAX_TOKENS * TOP_K * INTER // 4, 16]  # routed activations, MXFP8
    as_: fx.Array[fx.Float32, MAX_TOKENS * TOP_K, 16]
    shb: fx.Array[fx.Float32, MAX_TOKENS * SH_INTER // 2, 16]  # shared activation, bf16
    qb: fx.Array[fx.Float32, ATT_QH * ATT_D // 2, 16]  # q heads, packed bf16 (QK's B)
    knb: fx.Array[fx.Float32, ATT_D // 2, 16]  # the new k, packed bf16
    sp: fx.Array[fx.Float32, 4 * 8 * ATT_QH * 16, 16]  # QK partials [block][K step][head][16]
    pb: fx.Array[fx.Float32, ATT_QH * 32, 16]  # P, packed bf16 [head][64 positions]
    alph: fx.Array[fx.Float32, 16, 16]
    vn: fx.Array[fx.Float32, MAX_TOKENS * ATT_D, 16]


def _st_bf16(res, elem, v):
    """Store one bf16 element at bf16 offset ``elem``."""
    bo.buffer_store(fx.Float32(v).to(fx.BFloat16).ir_value(), res, fx.Int32(elem))


def _pick(cond, a, b):
    return a if cond else b


def build_layer(kind: str, tokens: int, max_ctx: int = 16384, timing: bool = False):
    """The launcher of the ``kind`` ("gdn" | "att") layer kernel for ``tokens`` decode
    tokens: ``launch(*table, *peers, rank, cnt_gdn, cnt_att, positions, m0, m1, m2,
    bt_stride, stream=)``: ``table`` the NARGS int64 entries of ``kind``'s argument
    table (layout.py, zero-padded), ``peers`` the TP peer-buffer bases. ``positions``: int64 [3, T] (row 0 read). GDN: m0 / m1 = the int32
    write / read state slots (-1: pad), m2 unused. Attention: m0 = block table (int32,
    rows ``bt_stride`` apart), m1 = context lens (int32), m2 = slot mapping (int64)."""
    assert kind in ("gdn", "att") and 1 <= tokens <= MAX_TOKENS
    T_ = tokens
    A = arg_index(kind)
    # attention: CTA (token t, context chunk c) for all ATT_QH q heads (they share the KV
    # head: the WMMA B columns). A pass covers 4 KV-cache blocks (64 positions); chunk
    # c's pass p is blocks 4 (p NCH + c) .. + 3: addresses independent of the length.
    NCH = min(128, BLOCKS // T_)  # chunks (CTAs) a token
    PASSES = (max_ctx + 64 * NCH - 1) // (64 * NCH)
    CH_LANE = (NCH + WAVE - 1) // WAVE  # combine: chunks a lane
    MAX_BLK = (max_ctx + 15) // 16
    build_key = (kind, tokens, max_ctx, THREADS, timing)

    @flyc.kernel(name=f"qwen35_mono_{kind}_t{tokens}_c{max_ctx}{'_timing' if timing else ''}")
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
        wave = uniform(tid // WAVE)  # scalar: per-wave rows / branches stay in SGPRs
        lane = tid % WAVE
        # a stage's task q runs on CTA q % BLOCKS: small stages spread over every CU
        gw = wave * BLOCKS + bid

        from flydsl._mlir.dialects import llvm as _llvm

        lds = fx.SharedAllocator().allocate(_Lds).peek()
        L_red = lds_addr(lds.red.ptr)
        if fx.const_expr(timing):
            # wave 0 alone reads the entry clock, into LDS until stamp 0: a clock read
            # is a message to the dispatcher, and one from every wave at entry costs
            # ~0.7 us a launch and skews the entries it measures
            if wave == 0:
                c0 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))
                if lane == 0:
                    lds_sti(L_red, 254, fx.Int32(c0 & 0xFFFFFFFF))
                    lds_sti(L_red, 255, fx.Int32(c0 >> 32))
        # the argument table's entries and the peer bases are kernel arguments (kernarg
        # SGPRs): a table in memory would put a dependent load ahead of every pointer
        table = [fx.Int64(v) for v in (a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27)][: len(A)]

        def arg(name):
            return table[A[name]]

        def R(name, live=None):
            return rsrc_at(arg(name), live)

        def RP(ptr):
            return rsrc_at(fx.Int64(ptrtoint(ptr)))

        cnt_own = _pick(kind == "gdn", cnt_gdn, cnt_att)
        # every CTA arrives once, at the end; no CTA can finish before every CTA has
        # read the counters (the all-reduce all-gather waits on every CTA's slice)
        launch_idx = counter_base(cnt_gdn, BLOCKS) // BLOCKS + counter_base(cnt_att, BLOCKS) // BLOCKS
        tag = launch_idx + 1  # this launch's hand-offs
        tag0 = launch_idx * 2 + 1  # the all-reduces
        tag1 = launch_idx * 2 + 2

        L_s = lds_addr(lds.s.ptr)
        L_act = lds_addr(lds.act.ptr)
        L_sele = lds_addr(lds.sel_e.ptr)
        L_selw = lds_addr(lds.sel_w.ptr)
        L_sg = lds_addr(lds.sg.ptr)
        L_part = lds_addr(lds.part.ptr)
        L_rcv = lds_addr(lds.rcv.ptr)
        L_vn = lds_addr(lds.vn.ptr)
        L_xb = lds_addr(lds.xb.ptr)
        L_gb = lds_addr(lds.gb.ptr)
        L_wr = lds_addr(lds.wr.ptr)
        L_y = lds_addr(lds.y16.ptr)
        L_xq = lds_addr(lds.xq.ptr)
        L_xs = lds_addr(lds.xs.ptr)
        L_aq = lds_addr(lds.aq.ptr)
        L_as = lds_addr(lds.as_.ptr)
        L_shb = lds_addr(lds.shb.ptr)
        L_qb = lds_addr(lds.qb.ptr)
        L_knb = lds_addr(lds.knb.ptr)
        L_sp = lds_addr(lds.sp.ptr)
        L_pb = lds_addr(lds.pb.ptr)
        L_al = lds_addr(lds.alph.ptr)

        r_sc = R("scratch")
        r_mb = R("mbox")
        r_stamps = R("stamps")

        def stamp(i, words=None):
            # timing builds: wave 0 stores its CTA's slot (one clock read a CTA: every
            # wave's would queue 4096 dispatcher messages, ~0.7 us, into what it times)
            if timing:
                if fx.const_expr(words is None):
                    if wave == 0:
                        clk = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.readsteadycounter", [], [], []))
                        st_dw(r_stamps, (bid * STAMPS + i) * 2, [fx.Int32(clk & 0xFFFFFFFF), fx.Int32(clk >> 32)])
                else:
                    st_dw(r_stamps, (bid * STAMPS + i) * 2, words)

        stamp(18)  # the kernarg is in (this store's address came from it)
        peer_ptrs = [fx.Int64(v) for v in (pr0, pr1, pr2, pr3)][:TP]
        own_ptr = peer_ptrs[0]
        for p in fx.range_constexpr(1, TP):
            own_ptr = (rank == p).select(peer_ptrs[p], own_ptr)
        peer_res = [rsrc_at(p) for p in peer_ptrs]
        own_res = rsrc_at(own_ptr)

        def preload_frags(name, tile, live, ksteps, nsteps):
            """This wave's A fragments of a WMMA-packed weight (layout.py): K steps
            wave nsteps .. + nsteps of 16-row tile ``tile``, loaded now and used later.
            Not ``live``: an empty resource (zeros, no memory access)."""
            res = R(name, live)
            out = []
            for i in fx.range_constexpr(nsteps):
                dw = ((tile * ksteps + wave * nsteps + i) * WAVE + lane) * 8
                out.append(ld_dw(res, dw) + ld_dw(res, dw + 4))
            return out

        def tile_mma(frags, xb, xstride):
            """This wave's K slice of a 16-row tile times the T token rows of packed bf16
            LDS ``xb`` (``xstride`` dwords apart): its partial [16 rows x T] into L_wr.
            Lanes past the tokens read the last token's row; their columns are unused."""
            n = lane % 16
            xt = fx.min(n, fx.Int32(T_ - 1))
            h = lane // 16
            acc = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
            for i in fx.range_constexpr(len(frags)):
                k_dw = (wave * len(frags) + i) * 16 + h * 8
                acc = wmma_bf16(frags[i], lds_ld8w(xb, xt * xstride + k_dw), acc)
            if n < T_:
                lds_st4(L_wr, (wave * T_ + n) * 16 + h * 8, acc[:4])
                lds_st4(L_wr, (wave * T_ + n) * 16 + h * 8 + 4, acc[4:])

        def tile_sum():
            """After a barrier, thread tid < 16 T: row tid % 16, token tid // 16 of the
            tile, summed over the waves in a fixed order."""
            r = tid % 16
            t = fx.min(tid // 16, fx.Int32(T_ - 1))
            y = lds_ld(L_wr, t * 16 + r)
            for w in fx.range_constexpr(1, WAVES):
                y = y + lds_ld(L_wr, (w * T_ + t) * 16 + r)
            return y

        def st_xb(xb, xstride, t, d4, vals):
            """4 values (dims d4 .. d4 + 3 of token t) into packed bf16 LDS ``xb``."""
            lds_sti(xb, t * xstride + d4 // 2, bf16_pack(vals[0], vals[1]))
            lds_sti(xb, t * xstride + d4 // 2 + 1, bf16_pack(vals[2], vals[3]))

        # ---------------------------------------------------------------- N1 loads
        # issued before the early loads: loads complete in order, so N1 would otherwise
        # wait for every weight / state load issued ahead of its own
        # the per-step metadata the early loads' addresses depend on go first: they are
        # in once N1's inputs are, so the early loads issue without a dependent wait
        if fx.const_expr(kind == "gdn"):
            slot_out = [fx.Int32(ld_dw(RP(m0), t, n=1)[0]) for t in fx.range_constexpr(T_)]
            slot_in = [fx.Int32(ld_dw(RP(m1), t, n=1)[0]) for t in fx.range_constexpr(T_)]
            # G2 task (token, v head, 8 columns) of this wave
            g2_q = fx.min(gw, fx.Int32(T_ * GDN_VH * 16 - 1))
            g2_t = g2_q // (GDN_VH * 16)
            g2_hv = (g2_q // 16) % GDN_VH
            g2_jb = g2_q % 16
            g2_si = fx.Int32(ld_dw(RP(m1), g2_t, n=1)[0])
            g2_so = fx.Int32(ld_dw(RP(m0), g2_t, n=1)[0])
        else:
            # attention: this CTA's (token, chunk) and its first pass's cache blocks
            att_ok = bid < T_ * NCH
            t_a = fx.min(bid // NCH, fx.Int32(T_ - 1))
            c_a = att_ok.select(bid % NCH, fx.Int32(0))

            def block_of(p, b, live=None):
                """Cache block of pass p's block b (past the sequence: its zeroed table
                entry); not ``live``: none loaded."""
                g = fx.min((p * NCH + c_a) * 4 + b, fx.Int32(MAX_BLK - 1))
                res = RP(m0) if live is None else rsrc_at(fx.Int64(ptrtoint(m0)), live)
                return fx.max(fx.Int32(ld_dw(res, t_a * bt_stride + g, n=1)[0]), fx.Int32(0))

            L_a = fx.Int32(ld_dw(RP(m1), t_a, n=1)[0])
            blk0 = [block_of(0, b) for b in range(4)]
        r_h = R("h_in")
        r_r = R("res_in")
        nw = bf16_unpack(ld_dw(R("ln1_w"), tid * 2, n=2))
        n1_h = [ld_dw(r_h, t * (HIDDEN // 2) + tid * 2, n=2) for t in fx.range_constexpr(T_)]
        n1_r = [ld_dw(r_r, t * (HIDDEN // 2) + tid * 2, n=2) for t in fx.range_constexpr(T_)]

        # ---------------------------------------------------------------- N1
        s1 = []
        for t in fx.range_constexpr(T_):
            hs = bf16_unpack(n1_h[t])
            rs = bf16_unpack(n1_r[t])
            s = [hs[n] + rs[n] for n in fx.range_constexpr(4)]
            s1.append(s)
            if fx.const_expr(t == 0):
                stamp(16)
            ss = wave_sum(dot(s, s))
            if lane == 0:
                lds_st(L_red, t * WAVES + wave, ss)
            lds_st4(L_s, t * HIDDEN + tid * 4, [bf16r(v) for v in s])

        # ---------------------------------------------------------------- early loads
        # issued once N1's inputs are in: every CTA reads the same few KB of h / res,
        # and a burst of weight loads ahead of them delays those hot lines (N1 ~6 us)
        # everything a stage reads that does not depend on the stages before it is
        # loaded here, unconditionally (an idle wave loads a clamped, cached row: a
        # value cannot leave a dynamic branch); the stages consume the registers
        if fx.const_expr(kind == "gdn"):
            # G1: CTA b < G1_TILES the 16-row tile b, its waves K steps 4 w .. 4 w + 3
            g1_f = preload_frags("g1_w", fx.min(bid, fx.Int32(G1_TILES - 1)), bid < G1_TILES, HIDDEN // 32, 4)
            # conv (thread tid < 16 T: channel c = 16 b + tid % 16 of token tid // 16, if
            # a q | k | v row): weights, slots, the 3 state taps
            r_cs = R("conv_state")
            c_t = fx.min(tid // 16, fx.Int32(T_ - 1))
            conv_live = (bid * 16 < GDN_QKV) & (tid < 16 * T_)
            c_ch = fx.min(bid * 16 + tid % 16, fx.Int32(GDN_QKV - 1))
            conv_w = bf16_unpack(ld_dw(R("conv_w", conv_live), c_ch * 2, n=2))
            c_si = slot_in[0]
            c_so = slot_out[0]
            for t in fx.range_constexpr(1, T_):
                c_si = (c_t == t).select(slot_in[t], c_si)
                c_so = (c_t == t).select(slot_out[t], c_so)
            taps = []
            for k in fx.range_constexpr(3):
                e = (fx.max(c_si, fx.Int32(0)) * 3 + k) * GDN_QKV + c_ch
                taps.append(bf16_at(ld_dw(R("conv_state", conv_live), e // 2, n=1)[0], e % 2))
            # G2 (token, v head, 8 columns): the state rows and the gating parameters
            r_ss = R("ssm_state")
            g2_live = gw < T_ * GDN_VH * 16
            g2_state = []
            for i in fx.range_constexpr(4):
                e = ((fx.max(g2_si, fx.Int32(0)) * GDN_VH + g2_hv) * GDN_D + 4 * lane + i) * GDN_D + 8 * g2_jb
                g2_state.append(ld_dw(rsrc_at(arg("ssm_state"), g2_live), e // 2))
            g2_alog = bf16_at(ld_dw(R("a_log", g2_live), g2_hv // 2, n=1)[0], g2_hv % 2)
            g2_dtb = bf16_at(ld_dw(R("dt_bias", g2_live), g2_hv // 2, n=1)[0], g2_hv % 2)
            # G3 out_proj: CTA b < OUT_TILES the tile b (K = 1024: 2 steps a wave)
            op_f = preload_frags("out_proj", fx.min(bid, fx.Int32(OUT_TILES - 1)), bid < OUT_TILES, 1024 // 32, 2)
        else:
            a1_f = preload_frags("qkvg", fx.min(bid, fx.Int32(A1_TILES - 1)), bid < A1_TILES, HIDDEN // 32, 4)
            # attention: this wave's first-pass K / V fragments
            G_a = (L_a + 15) // 16  # the sequence's cache blocks

            def kv_frags(blks, rk, rv):
                """A pass's fragments of this wave: K for its QK tasks i = 0, 1 (block
                wave // 8 + 2 i, K step wave % 8: lane row = position lane % 16, dims
                32 s + 16 (lane // 16) ..); V^T for PV K steps q = 0, 1 (blocks 2q, 2q + 1:
                lane row = dim 16 wave + lane % 16, positions 8 (lane // 16) .. of each)."""
                h = lane // 16
                r = lane % 16
                ks = []
                for i in fx.range_constexpr(2):
                    blk = (wave < 8).select(blks[2 * i], blks[2 * i + 1])
                    e0 = ((blk * 32 + 4 * (wave % 8) + 2 * h) * 16 + r) * 8
                    ks.append([fx.Int32(w) for w in ld_dw(rk, e0 // 2, cm=CM_DEV) + ld_dw(rk, (e0 + 128) // 2, cm=CM_DEV)])
                vs = []
                for q in fx.range_constexpr(2):
                    f = []
                    for i in fx.range_constexpr(2):
                        e0 = ((blks[2 * q + i] * 2 + h) * ATT_D + 16 * wave + r) * 8
                        f += [fx.Int32(w) for w in ld_dw(rv, e0 // 2, cm=CM_DEV)]
                    vs.append(f)
                return ks, vs

            pre_k, pre_v = kv_frags(blk0, R("k_cache"), R("v_cache"))
        lds_barrier()
        stamp(17)
        for t in fx.range_constexpr(T_):
            tot = lds_ld(L_red, t * WAVES)
            for w in fx.range_constexpr(1, WAVES):
                tot = tot + lds_ld(L_red, t * WAVES + w)
            rstd = fmath.rsqrt(tot / float(HIDDEN) + EPS)
            xn1 = [bf16r(s1[t][n] * rstd * (nw[n] + 1.0)) for n in fx.range_constexpr(4)]
            st_xb(L_xb, HIDDEN // 2, t, tid * 4, xn1)
        lds_barrier()
        stamp(0, [lds_ldi(L_red, 254), lds_ldi(L_red, 255)])  # the kernel entry (wave 0)
        stamp(1)

        if fx.const_expr(kind == "gdn"):
            # ------------------------------------------------------------ G1
            if bid < G1_TILES:
                tile_mma(g1_f, L_xb, HIDDEN // 2)
            lds_barrier()
            if (bid < G1_TILES) & (tid < 16 * T_):
                y = tile_sum()
                t = tid // 16
                row = bid * 16 + tid % 16
                if row >= GDN_QKVZ:
                    # b_h at pair 2h, a_h at 2h + 1: a G2 wave reads its (b, a) as one line
                    r = row - GDN_QKVZ
                    slot = (r < GDN_VH).select(2 * r, 2 * (r - GDN_VH) + 1)
                    put1(r_mb, MBOX["gba"] + t * 16 + slot, bf16r(y), tag)
                else:
                    if row >= GDN_QKV:
                        put1(r_mb, MBOX["gz"] + t * 1024 + row - GDN_QKV, bf16r(y), tag)
                    else:
                        # conv update on channel c = row: [s0 s1 s2] x -> silu(w . [s0 s1 s2 x])
                        c = row
                        cw = conv_w
                        x = bf16r(y)
                        tp = taps
                        if c_so >= 0:
                            acc = cw[0] * tp[0] + cw[1] * tp[1] + cw[2] * tp[2] + cw[3] * x
                            put1(r_mb, MBOX["gqkv"] + t * GDN_QKV + c, bf16r(silu(acc)), tag)
                            _st_bf16(r_cs, (c_so * 3 + 0) * GDN_QKV + c, tp[1])
                            _st_bf16(r_cs, (c_so * 3 + 1) * GDN_QKV + c, tp[2])
                            _st_bf16(r_cs, (c_so * 3 + 2) * GDN_QKV + c, x)
                        else:  # a pad row still hands off (its consumers wait)
                            put1(r_mb, MBOX["gqkv"] + t * GDN_QKV + c, 0.0, tag)
            stamp(2)
            stamp(3)

            # ------------------------------------------------------------ G2
            if gw < T_ * GDN_VH * 16:
                t = g2_t
                hv = g2_hv
                jb = g2_jb
                kh = hv // (GDN_VH // GDN_KH)
                qk_base = MBOX["gqkv"] + t * GDN_QKV
                qp = qk_base + kh * GDN_D + 4 * lane
                kp = qk_base + GDN_KH * GDN_D + kh * GDN_D + 4 * lane
                vp = qk_base + 2 * GDN_KH * GDN_D + hv * GDN_D + 8 * jb
                got = poll2(r_mb, [qp, qp + 2, kp, kp + 2, vp, vp + 2, vp + 4, vp + 6, MBOX["gba"] + t * 16 + 2 * hv], tag)
                qv = got[0:4]
                kv = got[4:8]
                vv = got[8:16]
                bv = got[16]
                av = got[17]
                qn = fmath.rsqrt(wave_sum(dot(qv, qv)) + 1e-6) * (GDN_D**-0.5)
                kn_ = fmath.rsqrt(wave_sum(dot(kv, kv)) + 1e-6)
                qv = [v * qn for v in qv]
                kv = [v * kn_ for v in kv]
                g = -fmath.exp(g2_alog) * softplus(av + g2_dtb)
                beta = bf16r(sigmoid(bv))
                decay = fmath.exp(g)
                so = g2_so
                ob = MBOX["go"] + t * 1024 + hv * GDN_D + 8 * jb
                if so >= 0:
                    # st[i][j]: rows k = 4 lane + i, columns 8 jb + j (loaded at the start)
                    st = [[v * decay for v in bf16_unpack(g2_state[i])] for i in fx.range_constexpr(4)]
                    kvm = [wave_sum(st[0][j] * kv[0] + st[1][j] * kv[1] + st[2][j] * kv[2] + st[3][j] * kv[3]) for j in fx.range_constexpr(8)]
                    dl = [(vv[j] - kvm[j]) * beta for j in fx.range_constexpr(8)]
                    st = [[st[i][j] + kv[i] * dl[j] for j in fx.range_constexpr(8)] for i in fx.range_constexpr(4)]
                    o = [wave_sum(st[0][j] * qv[0] + st[1][j] * qv[1] + st[2][j] * qv[2] + st[3][j] * qv[3]) for j in fx.range_constexpr(8)]
                    for i in fx.range_constexpr(4):
                        e = ((so * GDN_VH + hv) * GDN_D + 4 * lane + i) * GDN_D + 8 * jb
                        st_dw(r_ss, e // 2, [bf16_pack(st[i][2 * n], st[i][2 * n + 1]) for n in fx.range_constexpr(4)])
                    if lane == 0:
                        for i in fx.range_constexpr(4):
                            put2(r_mb, ob + 2 * i, bf16r(o[2 * i]), bf16r(o[2 * i + 1]), tag)
                else:
                    if lane == 0:
                        for i in fx.range_constexpr(4):
                            put2(r_mb, ob + 2 * i, 0.0, 0.0, tag)
            stamp(4)
            stamp(5)

            # ------------------------------------------------------------ G3
            gnw = bf16_unpack(ld_dw(R("gnorm_w"), 2 * lane, n=2))
            for it in fx.range_constexpr((T_ * GDN_VH + WAVES - 1) // WAVES):
                task = wave + it * WAVES
                if task < T_ * GDN_VH:
                    t = task // GDN_VH
                    h = task % GDN_VH
                    off = t * 1024 + h * GDN_D + 4 * lane
                    oz = poll2(r_mb, [MBOX["go"] + off, MBOX["go"] + off + 2, MBOX["gz"] + off, MBOX["gz"] + off + 2], tag)
                    o = oz[0:4]
                    z = oz[4:8]
                    rs = fmath.rsqrt(wave_sum(dot(o, o)) / float(GDN_D) + EPS)
                    st_xb(L_gb, 512, t, h * GDN_D + 4 * lane, [bf16r(o[n] * rs * gnw[n] * silu(z[n])) for n in fx.range_constexpr(4)])
            lds_barrier()
        else:
            # ------------------------------------------------------------ A1
            if bid < A1_TILES:
                tile_mma(a1_f, L_xb, HIDDEN // 2)
            lds_barrier()
            if (bid < A1_TILES) & (tid < 16 * T_):
                put1(r_mb, MBOX["qkvg"] + (tid // 16) * 2560 + bid * 16 + tid % 16, bf16r(tile_sum()), tag)
            stamp(2)
            stamp(3)

            # ------------------------------------------------------------ A2
            # CTA (t, c): its token's 4 q heads and new k (Gemma RMSNorm per head, NeoX
            # RoPE on dims < 64) and v into LDS; then its passes, 64 positions each: QK
            # (2 WMMAs a wave, K steps reduced in LDS) | online softmax (a wave a head) |
            # PV (wave w: dims 16 w .., 2 WMMAs). Unconditional, stores masked by att_ok:
            # inside a dynamic if, LLVM sinks the early K / V loads down into it.
            if fx.const_expr(True):
                t = t_a
                c = c_a
                L = L_a
                r_k = R("k_cache")
                r_v = R("v_cache")
                r_cos = R("cos")
                r_sin = R("sin")
                qnw = bf16_unpack(ld_dw(R("q_norm_w"), 4 * lane, n=4))
                knw = bf16_unpack(ld_dw(R("k_norm_w"), 4 * lane, n=4))
                if wave < ATT_QH + 1:
                    is_k = wave == ATT_QH
                    src = MBOX["qkvg"] + t * 2560 + is_k.select(fx.Int32(2 * ATT_QH * ATT_D), ATT_QH * ATT_D + wave * ATT_D) + 8 * lane
                    x = poll2(r_mb, [src, src + 2, src + 4, src + 6], tag)
                    rs = fmath.rsqrt(wave_sum(dot(x, x)) / float(ATT_D) + EPS)
                    xn = [bf16r(x[n] * rs * (is_k.select(knw[n], qnw[n]) + 1.0)) for n in fx.range_constexpr(8)]
                    # lanes 0-3 hold dims 0..31, lanes 4-7 dims 32..63: partner lane ^ 4
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
                    base = is_k.select(fx.Int32(0), wave * (ATT_D // 2)) + 4 * lane
                    for n in fx.range_constexpr(4):
                        if is_k:
                            lds_sti(L_knb, 4 * lane + n, words[n])
                        else:
                            lds_sti(L_qb, base + n, words[n])
                if tid < ATT_D // 2:
                    vp = MBOX["qkvg"] + t * 2560 + 2 * ATT_QH * ATT_D + ATT_D + 2 * tid
                    vv2 = poll2(r_mb, [vp], tag)
                    lds_st(L_vn, 2 * tid, vv2[0])
                    lds_st(L_vn, 2 * tid + 1, vv2[1])
                lds_barrier()
                stamp(19)  # q / k / v in, normed

                # the new token (position L - 1): block gn at offset o_new, in chunk
                # (gn // 4) % NCH's pass (gn // 4) // NCH, block gn % 4 of it. Its CTA
                # writes K / V to the cache, and patches them into that pass's fragments
                # (the early loads read the cache before the write)
                gn = (L - 1) // 16
                o_new = (L - 1) % 16
                p_new = (gn // 4) // NCH
                b_new = gn % 4
                is_owner = ((gn // 4) % NCH == c) & (L > 0) & att_ok
                slot = fx.Int32(ld_dw(RP(m2), 2 * t, n=1)[0])
                if is_owner & (slot >= 0) & (tid < WAVE):
                    blk = slot // 16
                    off = slot % 16
                    st_dw(r_k, (((blk * 32 + lane) * 16 + off) * 8) // 2, lds_ld4w(L_knb, 4 * lane))
                    vv = lds_ld8(L_vn, 8 * lane)
                    for n in fx.range_constexpr(8):
                        _st_bf16(r_v, ((blk * 2 + off // 8) * ATT_D + 8 * lane + n) * 8 + off % 8, vv[n])

                h16 = lane // 16
                n16 = lane % 16
                hq = fx.min(n16, fx.Int32(ATT_QH - 1))  # B columns past the heads: unused

                def patch(p, ks, vs):
                    here = is_owner & (p == p_new)
                    kn = lds_ld8w(L_knb, 16 * (wave % 8) + 8 * h16)
                    out_k = []
                    for i in fx.range_constexpr(2):
                        hit = here & (wave // 8 + 2 * i == b_new) & (n16 == o_new)
                        out_k.append([hit.select(kn[x], ks[i][x]) for x in range(8)])
                    vb = fx.Int32(bf16_pack(lds_ld(L_vn, 16 * wave + n16), fx.Float32(0.0))) & 0xFFFF
                    e = o_new % 8
                    out_v = []
                    for q in fx.range_constexpr(2):
                        f = list(vs[q])
                        for i in fx.range_constexpr(2):
                            hit = here & (b_new == 2 * q + i) & (h16 == o_new // 8)
                            for x in fx.range_constexpr(4):
                                w = f[4 * i + x]
                                nw_ = (e % 2 == 0).select((w & fx.Int32(-65536)) | vb, (w & 0xFFFF) | (vb << 16))
                                f[4 * i + x] = (hit & (e // 2 == x)).select(nw_, w)
                        out_v.append(f)
                    return out_k, out_v

                o_acc = [fx.Float32(0.0) for _ in fx.range_constexpr(8)]
                m_r = fx.Float32(NEG)
                l_r = fx.Float32(0.0)
                kc = pre_k
                vc = pre_v
                p = fx.Int32(0)
                while (p < PASSES) & ((p * NCH + c) * 4 < G_a):
                    kp, vp_ = patch(p, kc, vc)
                    # QK: task i = (block wave // 8 + 2 i, K step wave % 8)
                    qf = lds_ld8w(L_qb, hq * (ATT_D // 2) + 16 * (wave % 8) + 8 * h16)
                    for i in fx.range_constexpr(2):
                        d = wmma_bf16(kp[i], qf, [fx.Float32(0.0)] * 8)
                        if n16 < ATT_QH:
                            base = (((wave // 8 + 2 * i) * 8 + wave % 8) * ATT_QH + n16) * 16 + 8 * h16
                            lds_st4(L_sp, base, d[:4])
                            lds_st4(L_sp, base + 4, d[4:])
                    lds_barrier()
                    # softmax: wave w < ATT_QH = head w, lanes over positions lane, lane + 32
                    if wave < ATT_QH:
                        sc = []
                        for half in fx.range_constexpr(2):
                            x_ = lane + 32 * half
                            b = x_ // 16
                            r = x_ % 16
                            acc = lds_ld(L_sp, ((b * 8) * ATT_QH + wave) * 16 + r)
                            for st in fx.range_constexpr(1, 8):
                                acc = acc + lds_ld(L_sp, ((b * 8 + st) * ATT_QH + wave) * 16 + r)
                            ok = ((p * NCH + c) * 4 + b) * 16 + r < L
                            sc.append(ok.select(acc * ATT_SCALE, fx.Float32(NEG)))
                        m_new = fx.max(m_r, wave_max(fx.max(sc[0], sc[1])))
                        alpha = fmath.exp(m_r - m_new)
                        pr = [(sc[u] > NEG * 0.5).select(fmath.exp(sc[u] - m_new), fx.Float32(0.0)) for u in fx.range_constexpr(2)]
                        l_r = l_r * alpha + wave_sum(pr[0] + pr[1])
                        m_r = m_new
                        for half in fx.range_constexpr(2):
                            nxt = pr[half].shuffle_xor(1, WAVE)
                            if lane % 2 == 0:
                                lds_sti(L_pb, wave * 32 + (lane + 32 * half) // 2, bf16_pack(pr[half], nxt))
                        if lane == 0:
                            lds_st(L_al, wave, alpha)
                    lds_barrier()
                    # PV: dims 16 wave .., heads as columns
                    al = lds_ld(L_al, hq)
                    o_acc = [v * al for v in o_acc]
                    for q in fx.range_constexpr(2):
                        pf = lds_ld4w(L_pb, hq * 32 + (2 * q) * 8 + 4 * h16) + lds_ld4w(L_pb, hq * 32 + (2 * q + 1) * 8 + 4 * h16)
                        o_acc = wmma_bf16(vp_[q], pf, o_acc)
                    p = p + 1
                    live = (p < PASSES) & ((p * NCH + c) * 4 < G_a)
                    kc, vc = kv_frags([block_of(p, b, live) for b in range(4)], rsrc_at(arg("k_cache"), live), rsrc_at(arg("v_cache"), live))
                stamp(20)  # the passes are done
                # the partial (unnormalized O, running max / sum) of (t, c, head)
                if (n16 < ATT_QH) & att_ok:
                    pidx = (t * NCH + c) * ATT_QH + n16
                    st_f32(r_sc, SCRATCH["aop"] + pidx * ATT_D + 16 * wave + 8 * h16, o_acc[:4], cm=CM_DEV)
                    st_f32(r_sc, SCRATCH["aop"] + pidx * ATT_D + 16 * wave + 8 * h16 + 4, o_acc[4:], cm=CM_DEV)
                if (wave < ATT_QH) & (lane == 0) & att_ok:
                    st_f32(r_sc, SCRATCH["aml"] + ((t * NCH + c) * ATT_QH + wave) * 2, [m_r, l_r], cm=CM_DEV)
                # publish: the partials' plain stores land, then the flag pair
                rocdl.s_wait_storecnt(0)
                stamp(21)  # the CTA's partials have landed
                lds_barrier()
                if (tid == 0) & att_ok:
                    put1(r_mb, MBOX["aflag"] + t * NCH + c, 1.0, tag)
            stamp(4)
            stamp(5)

            # ------------------------------------------------------------ A3
            # (token, head, 32 dims) a wave; lanes over the chunks
            if gw < T_ * ATT_QH * 8:
                t = gw // (ATT_QH * 8)
                h = (gw // 8) % ATT_QH
                db = gw % 8
                flags = []
                for k in fx.range_constexpr(CH_LANE):
                    c = lane + k * WAVE
                    flags.append(MBOX["aflag"] + t * NCH + (c < NCH).select(c, 0))
                _ = poll1(r_mb, flags, tag)
                ms = []
                ls = []
                for k in fx.range_constexpr(CH_LANE):
                    c = lane + k * WAVE
                    ok = c < NCH
                    ml = ld_f32(r_sc, SCRATCH["aml"] + ((t * NCH + ok.select(c, 0)) * ATT_QH + h) * 2, n=2, cm=CM_DEV)
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
                    c = lane + k * WAVE
                    ok = c < NCH
                    ob = SCRATCH["aop"] + ((t * NCH + ok.select(c, 0)) * ATT_QH + h) * ATT_D + 32 * db
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
                gate = poll1(r_mb, [MBOX["qkvg"] + t * 2560 + h * ATT_D + d], tag)[0]
                ao = bf16r(bf16r(mine / Lsafe) * bf16r(sigmoid(gate)))
                put1(r_mb, MBOX["ao"] + t * 1024 + h * ATT_D + d, ao, tag)
            stamp(6)

            # ------------------------------------------------------------ A4 (input)
            for i in fx.range_constexpr((T_ * 256 + THREADS - 1) // THREADS):
                e4 = tid + i * THREADS
                if e4 < T_ * 256:
                    st_xb(L_gb, 512, e4 // 256, (e4 % 256) * 4, poll2(r_mb, [MBOX["ao"] + 4 * e4, MBOX["ao"] + 4 * e4 + 2], tag))
            lds_barrier()
            stamp(7)

        # ---------------------------------------------------------------- out proj
        # CTA b < OUT_TILES: rows 16b .. 16b + 15, K = 1024 from packed LDS go
        if fx.const_expr(kind == "att"):
            op_f = preload_frags("o_proj", fx.min(bid, fx.Int32(OUT_TILES - 1)), bid < OUT_TILES, 1024 // 32, 2)
        if bid < OUT_TILES:
            tile_mma(op_f, L_gb, 512)
        lds_barrier()
        if (bid < OUT_TILES) & (tid < 16 * T_):
            lds_st(L_part, (tid // 16) * 16 + tid % 16, bf16r(tile_sum()))
        lds_barrier()
        stamp(_pick(kind == "gdn", 6, 8))

        # L2 prefetch of the next layer's rows this CTA will read (see layout), under
        # the all-reduce wait
        nx_rows = fx.Int32(arg("nx_rows"))
        if nx_rows > 0:  # nx_rows: the next first projection's tiles (64 KB each)
            if bid < nx_rows:
                prefetch_line(arg("nx_first") + fx.Int64(bid * 65536 + tid * 128))
            if (bid < OUT_TILES) & (tid < 256):  # its out tile: 32 KB
                prefetch_line(arg("nx_out") + fx.Int64(bid * 32768 + tid * 128))
            if bid < M1_TILES:
                prefetch_line(arg("nx_m1") + fx.Int64(bid * 65536 + tid * 128))

        # M1's tile loads here, under the all-reduce wait: CTA b < M1_RT_TILES the router
        # rows 16b.. (row 256 = the shared gate logit), b < M1_TILES a shared tile
        m1_f = preload_frags("m1_w", fx.min(bid, fx.Int32(M1_TILES - 1)), bid < M1_TILES, HIDDEN // 32, 4)
        # M4's shared down tile (K = 128: step = wave < 4)
        sh_f = preload_frags("sh_w2", fx.min(bid, fx.Int32(OUT_TILES - 1)), (bid < OUT_TILES) & (wave < 4), SH_INTER // 32, 1)

        # ---------------------------------------------------------------- all-reduce 0
        # wave W - TP + p exchanges with rank p (its peer pointer stays scalar, and no
        # weight load is in flight ahead of its poll): lane l < 2T pushes dims
        # 8b + 4h .. + 3 of token t = l // 2 (h = l % 2) as two bf16-pair words in one
        # 16-byte LL store, then polls the same line from rank p on this rank
        # (CTA b < OUT_TILES owns dims 16b .. 16b + 15: lane l < 4T dims 16b + 4h .. + 3
        # of token t = l // 4, h = l % 4)
        if (wave >= WAVES - TP) & (bid < OUT_TILES):
            src = wave - (WAVES - TP)
            if lane < 4 * T_:
                t = lane // 4
                h = lane % 4
                v = lds_ld4(L_part, t * 16 + 4 * h)
                words = [bf16_pack(v[0], v[1]), bf16_pack(v[2], v[3])]
                dst = pair_index(0, rank, t, bid * 8 + 2 * h)
                for p in fx.range_constexpr(TP):
                    if src == p:
                        st_dw(peer_res[p], dst * 2, [words[0], tag0, words[1], tag0], cm=CM_SYS)
                got = poll_words_sys(own_res, [pair_index(0, src, t, bid * 8 + 2 * h)], tag0)
                vals = bf16_unpack([fx.Uint32(w) for w in got])
                lds_st4(L_rcv, src * 64 + t * 16 + 4 * h, vals)
        lds_barrier()
        if (tid < 16 * T_) & (bid < OUT_TILES):
            acc = lds_ld(L_rcv, tid)
            for q in fx.range_constexpr(1, TP):
                acc = acc + lds_ld(L_rcv, q * 64 + tid)
            t = tid // 16
            put1(r_mb, MBOX["ag"] + t * HIDDEN + bid * 16 + tid % 16, bf16r(acc), tag)
        stamp(_pick(kind == "gdn", 7, 9))
        stamp(_pick(kind == "gdn", 8, 10))

        # ---------------------------------------------------------------- N2
        nw2 = bf16_unpack(ld_dw(R("ln2_w"), tid * 2, n=2))
        r_ro = R("res_out")
        s2 = []
        ag = poll2(r_mb, [MBOX["ag"] + t * HIDDEN + tid * 4 + h for t in fx.range_constexpr(T_) for h in (0, 2)], tag)
        if fx.const_expr(kind == "gdn"):
            stamp(19)  # the all-gathered attention output is in
        for t in fx.range_constexpr(T_):
            a = ag[4 * t : 4 * t + 4]
            r0 = lds_ld4(L_s, t * HIDDEN + tid * 4)
            s = [a[n] + r0[n] for n in fx.range_constexpr(4)]
            s2.append(s)
            ss = wave_sum(dot(s, s))
            if lane == 0:
                lds_st(L_red, 64 + t * WAVES + wave, ss)
            if bid == 0:
                st_dw(r_ro, t * (HIDDEN // 2) + tid * 2, [bf16_pack(s[0], s[1]), bf16_pack(s[2], s[3])])
        lds_barrier()
        for t in fx.range_constexpr(T_):
            tot = lds_ld(L_red, 64 + t * WAVES)
            for w in fx.range_constexpr(1, WAVES):
                tot = tot + lds_ld(L_red, 64 + t * WAVES + w)
            rstd = fmath.rsqrt(tot / float(HIDDEN) + EPS)
            xn2 = [bf16r(s2[t][n] * rstd * (nw2[n] + 1.0)) for n in fx.range_constexpr(4)]
            st_xb(L_xb, HIDDEN // 2, t, tid * 4, xn2)
            # M3's B: MXFP8, wave w's 128 dims = K step w's 4 blocks
            q8, s8 = mx8_quant(xn2, lane)
            lds_sti(L_xq, t * (HIDDEN // 4) + tid, q8)
            if lane == 0:
                lds_sti(L_xs, t * (HIDDEN // 128) + wave, s8)
        lds_barrier()
        if fx.const_expr(kind == "gdn"):
            stamp(9)

        # ---------------------------------------------------------------- M1
        if bid < M1_TILES:
            tile_mma(m1_f, L_xb, HIDDEN // 2)
        lds_barrier()
        if (bid < M1_TILES) & (tid < 16 * T_):
            y = tile_sum()
            t = tid // 16
            row = bid * 16 + tid % 16
            if bid < M1_RT_TILES:
                if row < ROUTER_ROWS:
                    put1(r_mb, MBOX["logits"] + t * LOGIT_STRIDE + row, bf16r(y), tag)
            else:
                lds_st(L_y, tid, y)
        lds_barrier()
        if (bid >= M1_RT_TILES) & (bid < M1_TILES) & (tid < 8 * T_):
            # shared tile s = b - M1_RT_TILES: rows r < 8 the gate of j = 8 s + r, r + 8 its up
            t = tid // 8
            r = tid % 8
            j_sh = (bid - M1_RT_TILES) * 8 + r
            g_ = lds_ld(L_y, t * 16 + r)
            u_ = lds_ld(L_y, t * 16 + r + 8)
            put1(r_mb, MBOX["act"] + (t * ACT_SLOTS + TOP_K) * INTER + j_sh, bf16r(silu(bf16r(g_)) * bf16r(u_)), tag)
        stamp(_pick(kind == "gdn", 10, 11))
        stamp(_pick(kind == "gdn", 11, 12))

        # ---------------------------------------------------------------- M2 top-8
        if wave < T_:
            t = wave
            lo_ = MBOX["logits"] + t * LOGIT_STRIDE + lane * 8
            vals = poll2(r_mb, [lo_, lo_ + 2, lo_ + 4, lo_ + 6], tag)
            if fx.const_expr(kind == "gdn"):
                stamp(20)  # the router logits are in
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
                for k in fx.range_constexpr(TOP_K):
                    lds_sti(L_sele, t * TOP_K + k, picked_i[k])
                    lds_st(L_selw, t * TOP_K + k, es[k] / tot)
                g = poll1(r_mb, [MBOX["logits"] + t * LOGIT_STRIDE + N_EXPERTS], tag)[0]
                lds_st(L_sg, t, sigmoid(g))
        lds_barrier()

        # The experts are known: M3's and M4's expert tiles load now. M3 task q (CTA
        # q % BLOCKS) = (slot q // 16, its 16-row tile q % 16 of w13: gate / up of j =
        # 8 tile ..); wave w = K step w. M4: CTA b < OUT_TILES, output rows 16b ..;
        # slot w + WAVES i (K = 128 real: one step).
        M3_TASKS = T_ * TOP_K * 16
        M3_PER = (M3_TASKS + BLOCKS - 1) // BLOCKS
        NSLOT = T_ * TOP_K
        M4_PER = (NSLOT + WAVES - 1) // WAVES
        arg_w13, arg_w13s, arg_w2, arg_w2s = arg("w13"), arg("w13_s"), arg("w2"), arg("w2_s")
        m3_pre = []
        for i in fx.range_constexpr(M3_PER):
            q = bid + BLOCKS * i
            live = q < M3_TASKS
            qq = fx.min(q, fx.Int32(M3_TASKS - 1))
            tile = qq % 16
            e = lds_ldi(L_sele, qq // 16)
            ob = e * W13_EXPERT_BYTES + (tile * 32 + 2 * wave) * 512 + lane * 16
            rw = rsrc_at(arg_w13, live)
            n = tile * 16 + lane % 16
            sa = ld_dw(rsrc_at(arg_w13s, live), e * (W13S_EXPERT_BYTES // 4) + (n // 32) * 512 + wave * 32 + n % 32, n=1)[0]
            m3_pre.append((ld_dw(rw, ob // 4) + ld_dw(rw, (ob + 512) // 4), sa))
        m4_pre = []
        for i in fx.range_constexpr(M4_PER):
            sl = fx.min(wave + WAVES * i, fx.Int32(NSLOT - 1))
            live = (wave + WAVES * i < NSLOT) & (bid < OUT_TILES)
            e = lds_ldi(L_sele, sl)
            ob = e * W2_EXPERT_BYTES + bid * 4 * 512 + lane * 16
            rw = rsrc_at(arg_w2, live)
            n = bid * 16 + lane % 16
            sa = ld_dw(rsrc_at(arg_w2s, live), e * (W2S_EXPERT_BYTES // 4) + (n // 32) * 64 + n % 32, n=1)[0]
            m4_pre.append((ld_dw(rw, ob // 4) + ld_dw(rw, (ob + 512) // 4), sa))

        # ---------------------------------------------------------------- M3 routed gate / up
        h16 = lane // 16
        for i in fx.range_constexpr(M3_PER):
            q = bid + BLOCKS * i
            qq = fx.min(q, fx.Int32(M3_TASKS - 1))
            t = qq // (16 * TOP_K)
            bq = []
            for j in fx.range_constexpr(4):
                bq += lds_ld4w(L_xq, t * (HIDDEN // 4) + 32 * wave + 8 * j + 4 * h16)
            d = wmma_fp4_fp8(m3_pre[i][0], bq, m3_pre[i][1], lds_ldi(L_xs, t * (HIDDEN // 128) + wave), [fx.Float32(0.0)] * 8)
            if lane % 16 == 0:  # column 0 (every column is the token)
                lds_st4(L_wr, (i * WAVES + wave) * 16 + h16 * 8, d[:4])
                lds_st4(L_wr, (i * WAVES + wave) * 16 + h16 * 8 + 4, d[4:])
        lds_barrier()
        if tid < 16 * M3_PER:
            i = tid // 16
            r = tid % 16
            y = lds_ld(L_wr, i * WAVES * 16 + r)
            for w in fx.range_constexpr(1, WAVES):
                y = y + lds_ld(L_wr, (i * WAVES + w) * 16 + r)
            lds_st(L_y, tid, y)
        lds_barrier()
        if tid < 8 * M3_PER:
            i = tid // 8
            q = bid + BLOCKS * i
            if q < M3_TASKS:
                sl = q // 16
                t = sl // TOP_K
                k = sl % TOP_K
                j = (q % 16) * 8 + tid % 8
                g = lds_ld(L_y, i * 16 + 2 * (tid % 8))
                u = lds_ld(L_y, i * 16 + 2 * (tid % 8) + 1)
                if fx.const_expr(kind == "gdn"):
                    stamp(21)  # the expert rows are done
                put1(r_mb, MBOX["act"] + (t * ACT_SLOTS + k) * INTER + j, silu(g) * u, tag)
        stamp(_pick(kind == "gdn", 12, 13))
        if fx.const_expr(kind == "gdn"):
            stamp(13)

        # ---------------------------------------------------------------- M4 down + all-reduce 1
        for i in fx.range_constexpr((T_ * ACT_SLOTS * INTER // 4 + THREADS - 1) // THREADS):
            e4 = tid + i * THREADS
            if e4 < T_ * ACT_SLOTS * INTER // 4:
                lds_st4(L_act, e4 * 4, poll2(r_mb, [MBOX["act"] + 4 * e4, MBOX["act"] + 4 * e4 + 2], tag))
        lds_barrier()
        if fx.const_expr(kind == "gdn"):
            stamp(22)  # the activations are in
        # the routed slots' activations to MXFP8 (a wave a slot: its 128 = one K step),
        # the shared expert's to packed bf16
        for i in fx.range_constexpr(M4_PER):
            sl = wave + WAVES * i
            if sl < NSLOT:
                v = lds_ld4(L_act, ((sl // TOP_K) * ACT_SLOTS + sl % TOP_K) * INTER + 4 * lane)
                q8, s8 = mx8_quant(v, lane)
                lds_sti(L_aq, sl * (INTER // 4) + lane, q8)
                if lane == 0:
                    lds_sti(L_as, sl, s8)
        if wave < T_:
            st_xb(L_shb, SH_INTER // 2, wave, 4 * lane, lds_ld4(L_act, (wave * ACT_SLOTS + TOP_K) * INTER + 4 * lane))
        lds_barrier()
        if bid < OUT_TILES:
            for i in fx.range_constexpr(M4_PER):
                sl = fx.min(wave + WAVES * i, fx.Int32(NSLOT - 1))
                bq = []
                for j in fx.range_constexpr(4):
                    bq += lds_ld4w(L_aq, sl * (INTER // 4) + 8 * j + 4 * h16)
                d = wmma_fp4_fp8(m4_pre[i][0], bq, m4_pre[i][1], lds_ldi(L_as, sl), [fx.Float32(0.0)] * 8)
                wk = lds_ld(L_selw, sl)
                if (lane % 16 == 0) & (wave + WAVES * i < NSLOT):
                    lds_st4(L_wr, sl * 16 + h16 * 8, [wk * v for v in d[:4]])
                    lds_st4(L_wr, sl * 16 + h16 * 8 + 4, [wk * v for v in d[4:]])
            if wave < 4:  # the shared expert: K step wave, the tokens as columns
                n = lane % 16
                d = wmma_bf16(sh_f[0], lds_ld8w(L_shb, fx.min(n, fx.Int32(T_ - 1)) * (SH_INTER // 2) + 16 * wave + 8 * h16), [fx.Float32(0.0)] * 8)
                if n < T_:
                    lds_st4(L_wr, 512 + (wave * T_ + n) * 16 + h16 * 8, d[:4])
                    lds_st4(L_wr, 512 + (wave * T_ + n) * 16 + h16 * 8 + 4, d[4:])
        lds_barrier()
        if (bid < OUT_TILES) & (tid < 16 * T_):
            t = tid // 16
            r = tid % 16
            routed = lds_ld(L_wr, (t * TOP_K) * 16 + r)
            for k in fx.range_constexpr(1, TOP_K):
                routed = routed + lds_ld(L_wr, (t * TOP_K + k) * 16 + r)
            shared = lds_ld(L_wr, 512 + t * 16 + r)
            for w in fx.range_constexpr(1, 4):
                shared = shared + lds_ld(L_wr, 512 + (w * T_ + t) * 16 + r)
            lds_st(L_part, t * 16 + r, bf16r(routed + lds_ld(L_sg, t) * shared))
        lds_barrier()
        stamp(14)
        # all-reduce 1 as all-reduce 0 (CTA b < OUT_TILES, dims 16b ..), then h_out
        if (wave >= WAVES - TP) & (bid < OUT_TILES):
            src = wave - (WAVES - TP)
            if lane < 4 * T_:
                t = lane // 4
                h = lane % 4
                v = lds_ld4(L_part, t * 16 + 4 * h)
                words = [bf16_pack(v[0], v[1]), bf16_pack(v[2], v[3])]
                dst = pair_index(1, rank, t, bid * 8 + 2 * h)
                for p in fx.range_constexpr(TP):
                    if src == p:
                        st_dw(peer_res[p], dst * 2, [words[0], tag1, words[1], tag1], cm=CM_SYS)
                got = poll_words_sys(own_res, [pair_index(1, src, t, bid * 8 + 2 * h)], tag1)
                vals = bf16_unpack([fx.Uint32(w) for w in got])
                lds_st4(L_rcv, src * 64 + t * 16 + 4 * h, vals)
        lds_barrier()
        if (tid < 8 * T_) & (bid < OUT_TILES):
            t = tid // 8
            pr = tid % 8
            sums = []
            for n in fx.range_constexpr(2):
                idx = t * 16 + 2 * pr + n
                acc = lds_ld(L_rcv, idx)
                for q in fx.range_constexpr(1, TP):
                    acc = acc + lds_ld(L_rcv, q * 64 + idx)
                sums.append(acc)
            st_dw(R("h_out"), t * (HIDDEN // 2) + bid * 8 + pr, [bf16_pack(sums[0], sums[1])])
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
        kernel(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, pr0, pr1, pr2, pr3, rank, cnt_gdn, cnt_att, positions, m0, m1, m2, bt_stride).launch(
            grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch
