# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  (traced stage closures read the loop's values on purpose: the tracer unrolls)
"""The gated full-attention mixer of the Qwen3.5 MonoKernel: q | gate, k, v projection; q / k Gemma
norm + partial RoPE and the K / V write into ATOM's paged FP8 cache; split (flash-decoding) attention;
the merge with the sigmoid output gate, publishing ``gated`` for the shared out_proj.

The cache is ATOM's MHA pool layout (block size BS = 16): K ``[blocks, kv_heads, head_dim / 16, BS, 16]``,
V ``[blocks, kv_heads, BS / 16, head_dim, 16]``, one fp32 scale per (kv head, token) for each. A tile
keeps the FP8 codes (exact in bf16) and applies a key's scale to its score / softmax weight.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen3_5.mono.config import FP8_MAX
from atom.models.qwen3_5.mono.kernels.attn_flydsl import flydsl_split_merge
from atom.models.qwen3_5.mono.kernels.common import sigmoid, sigmoid_precise
from atom.models.qwen3_5.mono.kernels.ops import i64_of, ld_f32, lds_ld, lds_st
from atom.models.qwen3_5.mono.kernels.ops import lds_i64 as lds_i64_at
from atom.models.qwen3_5.mono.layout import BS, PROJ_TILE
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    block_max,
    exp,
    fp8_pack4,
    rcp,
    read_lane_i32,
    rsq,
    rsrc,
    traced,
    uniform,
    wave_max,
    wave_sum,
    xshfl,
)
from atom.mono.device.sync import Addr
from atom.mono.plan.execution import THREADS, WAVES
from atom.mono.plan.trace import Space


@traced
def attn_stages(ctx):
    """1-4. q | gate, k, v projection; norm + RoPE + cache write; split attention; merge + gate."""
    EPS = ctx["EPS"]
    QKF = ctx["QKF"]
    QKF_O = ctx["QKF_O"]
    G = ctx["G"]
    H = ctx["H"]
    HD = ctx["HD"]
    KVD = ctx["KVD"]
    NSPLIT = ctx["NSPLIT"]
    CPS = ctx["CPS"]
    OUT_K = ctx["OUT_K"]
    QG = ctx["QG"]
    ROPE = ctx["ROPE"]
    S = ctx["S"]
    T_SEQ = ctx["T_SEQ"]
    SCALE = ctx["SCALE"]
    block_sum = ctx["block_sum"]
    block_tables = ctx["block_tables"]
    slot_mapping = ctx["slot_mapping"]
    bt_stride = ctx["bt_stride"]
    g_kn1 = ctx["g_kn1"]
    g_qn1 = ctx["g_qn1"]
    k_cache = ctx["k_cache"]
    k_scale = ctx["k_scale"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    positions = ctx["positions"]
    context_lens = ctx["context_lens"]
    proj_stage = ctx["proj_stage"]
    put = ctx["put"]
    put2 = ctx["put2"]
    put_bf = ctx["put_bf"]
    red = ctx["red"]
    rope_cos = ctx["rope_cos"]
    rope_sin = ctx["rope_sin"]
    s_qkv = ctx["s_qkv"]
    start = ctx["start"]
    tid = ctx["tid"]
    v_cache = ctx["v_cache"]
    v_scale = ctx["v_scale"]
    w_qkv = ctx["w_qkv"]
    wave = ctx["wave"]
    xs = ctx["xs"]
    r_pos, r_bt, r_slot = rsrc(positions), rsrc(block_tables), rsrc(slot_mapping)
    r_ctx = rsrc(context_lens)
    r_ks, r_vs = rsrc(k_scale), rsrc(v_scale)
    # A layer's K / V pool passes 4 GiB, the reach of a buffer offset (about 5 GB at TP4): a cache
    # block's bytes are addressed from that block's own 64-bit base, never as one offset from the
    # pool's (the scales, 64 B a block, stay within reach)
    KVB = HD * BS  # bytes of a cache block (this rank's one KV head), K and V alike

    def kv_block(pool, blk):
        """A buffer resource on cache block ``blk`` (wave-uniform) of ``pool``."""
        return rsrc(pool + fx.Int64(blk) * KVB)

    def v_words(blk, byte):
        """Two words at ``byte`` of V cache block ``blk`` (a lane's own block): a 64-bit address."""
        at = v_cache + fx.Int64(blk) * KVB + fx.Int64(byte)
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 8), at)
        return fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(2, fx.Int32)))

    def ld_pos(s):
        return uniform(bo.buffer_load(r_pos, s, vec_width=1, dtype=T.i32))

    def ld_block(s, i):
        return fx.Int32(
            bo.buffer_load(r_bt, s * bt_stride + i, vec_width=1, dtype=T.i32)
        )

    # ================================ 1. q | gate, k, v projection (FP8)
    def qkv_epilogue(s, row, v):
        # ATOM's row order: every local head's gate, then every head's q; mailbox qg holds [q, gate] per head
        if row < QG:
            hq = row % (QG // 2)
            put(
                mb("qg"),
                s * QG
                + (hq // HD) * 2 * HD
                + (row < QG // 2).select(fx.Int32(HD), fx.Int32(0))
                + hq % HD,
                bf16_round(v),
            )
        else:
            put(mb("kvr"), s * 2 * KVD + row - QG, bf16_round(v))

    proj_stage("proj", (QG + 2 * KVD) // PROJ_TILE, w_qkv, s_qkv, False, qkv_epilogue)

    # ================================ 2. Gemma q / k norm, partial RoPE; K / V -> the paged FP8 cache
    # task j < H: query head j; j == H: this rank's K / V head (one: local_kv_heads == 1)
    r_cos, r_sin, r_gq, r_gk = rsrc(rope_cos), rsrc(rope_sin), rsrc(g_qn1), rsrc(g_kn1)
    HALF = ROPE // 2

    def kv_quant(x, live):
        """Per-token FP8 E4M3 over the head as the original path writes it -> (FP8 value as f32, scale,
        the FP8 word). aiter's ``reshape_and_cache``: the scale ``amax / 448`` (stored as is: 0 for an
        all-zero head), x times ``1 / max(scale, 1e-12)``; ATOM's fused q / k norm + RoPE + cache kernel
        (QKF): ``amax / 448`` (1 for an all-zero head), x divided by it."""
        amax = block_max(live.select(fmath.absf(x), fx.Float32(0.0)), lane, wave, red)
        sc = amax / FP8_MAX
        if const_expr(QKF):
            sc = (sc == 0.0).select(fx.Float32(1.0), sc)
            q = x / sc
        else:
            q = x * (1.0 / fx.max(sc, fx.Float32(1.0e-12)))
        word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False))
        v2 = fx.Vector.make_type(2, fx.Float32)
        q8 = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))[0]
        return q8, sc, word

    for tt in range(start("qk"), S * (H + 1), G):
        tt = fx.Int32(tt)
        s = tt // (H + 1)
        j = tt % (H + 1)
        is_kv = j == H
        gpu.barrier()
        pos = ld_pos(s)
        live = tid < HD
        d = fx.min(tid, HD - 1)
        # q / gate rows (qg) or the K / V rows (kvr): one address, the region named by the q tasks
        src = Addr("qg", Space.SCRATCH, is_kv.select(mb("kvr").value, mb("qg").value))
        idx = is_kv.select(s * 2 * KVD + d, s * QG + j * 2 * HD + d)
        gam = is_kv.select(ld_f32(r_gk, d), ld_f32(r_gq, d))
        got = poll([(src, idx, 1), (mb("kvr"), s * 2 * KVD + KVD + d, 1)])
        x = got[0][0].bitcast(fx.Float32)
        vraw = got[1][0].bitcast(fx.Float32)
        rstd = rsq(block_sum(live.select(x * x, fx.Float32(0.0))) * (1.0 / HD) + EPS)
        # the Gemma norm, then RoPE: rounded to bf16 between (the original's two kernels), or once after
        # (QKF: its fused kernel, which also quantizes K for the cache from the unrounded value)
        xn = x * rstd * gam
        if const_expr(not QKF):
            xn = bf16_round(xn)
        lds_st(outs, d, xn)
        gpu.barrier()
        partner = lds_ld(outs, (d < HALF).select(d + HALF, d - HALF))
        cs = ld_f32(r_cos, pos * HALF + d % HALF)
        sn = ld_f32(r_sin, pos * HALF + d % HALF)
        rot = (d < HALF).select(xn * cs - partner * sn, xn * cs + partner * sn)
        y_f = (d < ROPE).select(rot, xn)
        y = bf16_round(y_f)
        if live & (j < H):
            put(mb("q"), (s * H + j) * HD + d, y)
        if is_kv:
            k8, ksc, kword = kv_quant(y_f if QKF else y, live)
            v8, vsc, vword = kv_quant(vraw, live)
            # the token's cache slot (block * BS + offset) is the engine's; -1 = a padding row: no write
            slot = uniform(bo.buffer_load(r_slot, s, vec_width=1, dtype=T.i32))
            blk = slot // BS
            t = slot % BS
            if live & (slot >= 0):
                kb = arith.trunci(T.i8, fx.Int32(kword).ir_value())
                vb = arith.trunci(T.i8, fx.Int32(vword).ir_value())
                bo.buffer_store(
                    kb, kv_block(k_cache, blk), ((d // 16) * BS + t) * 16 + d % 16
                )
                bo.buffer_store(
                    vb, kv_block(v_cache, blk), ((t // 16) * HD + d) * 16 + t % 16
                )
            if live:
                # this launch's row for the split tasks: other CTAs may not see the cache stores yet
                put(mb("kvnew"), s * (2 * HD + 2) + d, k8)
                put(mb("kvnew"), s * (2 * HD + 2) + HD + d, v8)
            if (tid == 0) & (slot >= 0):
                bo.buffer_store(ksc, r_ks, blk * BS + t)
                bo.buffer_store(vsc, r_vs, blk * BS + t)
            if tid == 0:
                put(mb("kvnew"), s * (2 * HD + 2) + 2 * HD, ksc)
                put(mb("kvnew"), s * (2 * HD + 2) + 2 * HD + 1, vsc)

    if const_expr(ctx["PA_FLYDSL"]):
        # ATOM_PA_FLYDSL: baseline's decode attention is aiter's FlyDSL paged decode
        flydsl_split_merge(ctx)
        return {}

    # ================================ 3. split attention (aiter's pa_decode_gluon, FP8 compute)
    # Baseline ATOM's decode attention, rounding point for rounding point (spec: aiter's
    # paged_attention_decode_sliding_window_head_1 + the pa_ps reduce): NSPLIT splits a sequence's
    # tokens evenly (page = ceil(ctx / NSPLIT)); a split walks the 256-key partitions it touches in order,
    # keeping only its own keys. Per partition: Q (FP8, unscaled) x K on FP8 MFMAs (head dim in order),
    # the score (scale * k_scale) * dot, an online softmax in exp2, P' = FP8(v_scale * (448 / vmax) * p)
    # with vmax the partition's largest V scale, and P' x V on FP8 MFMAs (keys in order) times vmax / 448.
    # A split publishes its bf16 output and its (max, exp sum). Task = (row, split): the row's 8 heads are
    # the MFMA's rows. This launch's keys (positions pos0 ..) come from ``kvnew``: other CTAs may not see
    # their cache stores yet.
    MASKV = -3.4e38
    # rows padded off the 64-word stride: the 16 lanes of an MFMA operand / score row group would hit one
    # LDS bank (an even pad keeps the 8-byte operand loads aligned)
    SXS, SPP, SQ8 = CPS + 1, CPS // 4 + 2, HD // 4 + 2
    XS = 0  # scores [16 heads][CPS] f32
    XPP = XS + 16 * SXS  # P' FP8 [16 heads][CPS / 4 words]
    XQ8 = XPP + 16 * SPP  # Q FP8 [16 heads][HD / 4 words]
    XA = XQ8 + 16 * SQ8  # per head: softmax rescale a | max | exp sum
    XKN = (
        XA + 64
    )  # this launch's keys: K FP8 [T_SEQ][HD / 4 words] | V FP8 | their k, v scales
    XVN = XKN + T_SEQ * HD // 4
    XSN = XVN + T_SEQ * HD // 4
    NG = HD // 16 // WAVES  # PV: 16-dim groups per wave
    assert NG == 2 and CPS // 16 == 2 * WAVES and H == WAVES
    g4 = lane // 16

    def lds_i64(idx):
        return lds_i64_at(xs, idx)

    def fp8_byte(word_f32, byte):
        """Byte ``byte`` (runtime) of an LDS word as an E4M3FN value."""
        w = word_f32.bitcast(fx.Int32) >> (8 * byte)
        return fx.Vector(
            rocdl.cvt_pk_f32_fp8(
                res=fx.Vector.make_type(2, fx.Float32), src=w, word_sel=False
            )
        )[0]

    def mfma8(a, b, c):
        return fx.Vector(rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a, b, c, 0, 0, 0]))

    for tt in range(start("split"), S * NSPLIT, G):
        tt = fx.Int32(tt)
        s = tt // NSPLIT
        js = tt % NSPLIT
        gpu.barrier()
        # the sequence's KV length (cached + this launch's T_SEQ keys, as ATOM's attention takes it: an
        # MRoPE position is not the cache index); rows s0 .. s0 + T_SEQ - 1, keys pos0 .. this launch's.
        # A padded row's length is raised to T_SEQ (its keys masked as before) and any length kept within
        # what its block-table row addresses (a capture's synthetic lengths)
        ctx_len = fx.min(
            fx.max(
                uniform(bo.buffer_load(r_ctx, s // T_SEQ, vec_width=1, dtype=T.i32)),
                fx.Int32(T_SEQ),
            ),
            bt_stride * BS,
        )
        s0 = s - s % T_SEQ
        pos0 = ctx_len - T_SEQ
        pos = pos0 + s % T_SEQ  # this row's own key: the causal bound
        page = (ctx_len + NSPLIT - 1) // NSPLIT
        k0 = page * js
        k1 = fx.min(ctx_len, k0 + page)
        p0 = k0 // CPS
        npart = (k0 < ctx_len).select((k1 + CPS - 1) // CPS - p0, fx.Int32(0))
        last_blk = (ctx_len - 1) // BS

        def raw_loads(tb):
            """Partition ``tb``'s global loads, none waiting on this launch: per QK block (2) its 8 K words
            and 4 k scales, the softmax's 4 v scales, P' V's 16 V words. A block past the context reads the
            last one (masked at use)."""
            out = []
            for gi in range_constexpr(2):
                bi = tb // BS + wave * 2 + gi
                blk = uniform(ld_block(s0, fx.min(bi, last_blk)))  # this wave's block
                r_kb = kv_block(k_cache, blk)
                for c in range_constexpr(HD // 32):
                    d = 32 * c + 8 * g4
                    off = ((d // 16) * BS + lane % 16) * 16 + d % 16
                    out.append(
                        fx.Vector(
                            bo.buffer_load(r_kb, off // 4, vec_width=2, dtype=T.i32)
                        )
                    )
                for e in range_constexpr(4):
                    out.append(ld_f32(r_ks, blk * BS + 4 * g4 + e))
            for j in range_constexpr(4):
                t = tb + 4 * lane + j
                blk = ld_block(s0, fx.min(t // BS, last_blk))
                out.append(ld_f32(r_vs, blk * BS + t % BS))
            for n in range_constexpr(NG):
                dim = (wave * NG + n) * 16 + lane % 16
                for c in range_constexpr(CPS // 32):
                    tk = 32 * c + 8 * g4  # this lane's 8 keys (one cache block's half)
                    blk = ld_block(s0, fx.min((tb + tk) // BS, last_blk))
                    out.append(v_words(blk, dim * 16 + tk % 16))
            return out

        # the first partition's loads before this launch's inputs: their latency hides under the wait
        # MTP verify (T_SEQ > 1) loads each partition at its top: carrying the loads across the loop next to
        # the verify path's live values spills SGPRs into a scratch slot
        PIPE = T_SEQ == 1
        raw0 = raw_loads(p0 * CPS) if PIPE else []
        # Q -> FP8 bytes (unscaled), heads >= H zero; P' rows >= H zero
        qv = poll([(mb("q"), s * H * HD + 4 * tid + 2 * i, 2) for i in range(2)])
        lds_st(
            xs,
            XQ8 + (tid // (HD // 4)) * SQ8 + tid % (HD // 4),
            fp8_pack4(
                *[qv[i][e].bitcast(fx.Float32) for i in range(2) for e in range(2)]
            ).bitcast(fx.Float32),
        )
        lds_st(
            xs, XQ8 + (H + tid // (HD // 4)) * SQ8 + tid % (HD // 4), fx.Float32(0.0)
        )
        if tid < (16 - H) * CPS // 4:
            lds_st(
                xs,
                XPP + (H + tid // (CPS // 4)) * SPP + tid % (CPS // 4),
                fx.Float32(0.0),
            )
        # this launch's keys, for every split with a tile: K / V codes back to bytes, a word a thread
        # (in rounds: past 4 tokens a request the words outnumber the threads)
        NKW = T_SEQ * 2 * HD // 4
        for rnd in range_constexpr(-(-NKW // THREADS)):
            wk = tid if rnd == 0 else tid + rnd * THREADS
            if (npart > 0) & (wk < NKW):
                i_ = wk // (2 * HD // 4)
                kv_ = (wk // (HD // 4)) % 2
                d4 = (wk % (HD // 4)) * 4
                base = (s0 + i_) * (2 * HD + 2) + kv_ * HD + d4
                got = poll([(mb("kvnew"), base, 2), (mb("kvnew"), base + 2, 2)])
                w = fp8_pack4(
                    *[got[i][e].bitcast(fx.Float32) for i in range(2) for e in range(2)]
                )
                lds_st(
                    xs,
                    XKN + kv_ * (T_SEQ * HD // 4) + i_ * (HD // 4) + d4 // 4,
                    w.bitcast(fx.Float32),
                )
        if (npart > 0) & (tid < 2 * T_SEQ):
            sc2 = poll(
                [(mb("kvnew"), (s0 + tid // 2) * (2 * HD + 2) + 2 * HD + tid % 2, 1)]
            )
            lds_st(xs, XSN + tid, sc2[0][0].bitcast(fx.Float32))
        gpu.barrier()

        def key_patch(t, cached, odd):
            """Key ``t``'s cached scale, or this launch's from LDS (k scale: odd 0, v: odd 1)."""
            new = (t >= pos0) & (t < ctx_len)
            i_ = fx.min(fx.max(t - pos0, 0), T_SEQ - 1)
            return new.select(lds_ld(xs, XSN + 2 * i_ + odd), cached)

        # raw_loads' layout: per QK block gi, 8 K words then 4 k scales; 4 v scales; 16 V words
        RAW_K, RAW_VS, RAW_V = 0, 24, 28
        RAW_VEC = [isinstance(x, fx.Vector) for x in raw0]
        # the online softmax's (max, exp sum), the P V accumulators (heads 4 g4 + e, this wave's two 16-dim
        # groups) and the partition's loads, carried; a partition issues the next one's loads
        init = [fx.Float32(float("-inf")), fx.Float32(0.0)]
        init += [fx.Float32(0.0) for _ in range(8)] + raw0
        for pi, carry in range(0, npart, fx.Int32(1), init=init):
            tb = (p0 + fx.Int32(pi)) * CPS
            m = fx.Float32(carry[0])
            lsum = fx.Float32(carry[1])
            av = [fx.Float32(carry[2 + i]) for i in range(8)]
            if const_expr(PIPE):
                raw = [
                    (
                        fx.Vector(carry[10 + i])
                        if RAW_VEC[i]
                        else fx.Float32(carry[10 + i])
                    )
                    for i in range(len(RAW_VEC))
                ]
            else:
                raw = raw_loads(tb)
            # the softmax's V scales (lane: keys 4 lane ..)
            vss = []
            for j in range_constexpr(4):
                t = tb + 4 * lane + j
                vs_ = key_patch(t, raw[RAW_VS + j], 1)
                vss.append((t < ctx_len).select(vs_, fx.Float32(0.0)))
            # QK: keys of this wave's two cache blocks (rows) x heads (columns), head dim in order
            for gi in range_constexpr(2):
                bi = tb // BS + wave * 2 + gi
                trow = bi * BS + lane % 16  # this lane's A-row key
                newk = (trow >= pos0) & (trow < ctx_len)
                inew = fx.min(fx.max(trow - pos0, 0), T_SEQ - 1)
                c4 = fx.Vector.filled(4, 0.0, fx.Float32)
                for c in range_constexpr(HD // 32):
                    d = 32 * c + 8 * g4
                    w2 = raw[RAW_K + gi * 12 + c]
                    kn = lds_i64(XKN + inew * (HD // 4) + d // 4)
                    ka = newk.select(kn, i64_of(w2[0], w2[1]))
                    c4 = mfma8(ka, lds_i64(XQ8 + (lane % 16) * SQ8 + d // 4), c4)
                for e in range_constexpr(4):
                    t = bi * BS + 4 * g4 + e
                    ks_ = key_patch(t, raw[RAW_K + gi * 12 + 8 + e], 0)
                    ok = (t >= k0) & (t < k1) & (t <= pos)
                    sc_ = (SCALE * ks_) * c4[e]
                    lds_st(
                        xs,
                        XS + (lane % 16) * SXS + (t - tb),
                        ok.select(sc_, fx.Float32(MASKV)),
                    )
            gpu.barrier()
            # online softmax of head `wave` over the partition (lane: keys 4 lane ..)
            scs = [lds_ld(xs, XS + wave * SXS + 4 * lane + j) for j in range(4)]
            vmax = wave_max(fx.max(fx.max(vss[0], vss[1]), fx.max(vss[2], vss[3])))
            m_new = fx.max(
                m, wave_max(fx.max(fx.max(scs[0], scs[1]), fx.max(scs[2], scs[3])))
            )
            alpha = exp(m - m_new)
            ps_ = [exp(scs[j] - m_new) for j in range(4)]
            lsum = fmath.fma(
                alpha, lsum, wave_sum(((ps_[0] + ps_[1]) + ps_[2]) + ps_[3])
            )
            m = m_new
            f8i = 448.0 * rcp(vmax + 1.0e-8)
            pq = [(vss[j] * f8i) * ps_[j] for j in range(4)]
            lds_st(xs, XPP + wave * SPP + lane, fp8_pack4(*pq).bitcast(fx.Float32))
            pscale = vmax * (1.0 / 448.0)
            if lane == 0:
                lds_st(xs, XA + wave, alpha)
            gpu.barrier()
            # P' V: heads (rows 4 g4 + e) x this wave's 16-dim groups, keys in order
            al = fx.Vector.from_elements(
                [lds_ld(xs, XA + 4 * g4 + e) for e in range(4)], fx.Float32
            )
            accs = [
                fx.Vector.from_elements(av[0:4], fx.Float32),
                fx.Vector.from_elements(av[4:8], fx.Float32),
            ]
            for n in range_constexpr(NG):
                dim = (wave * NG + n) * 16 + lane % 16
                pv = fx.Vector.filled(4, 0.0, fx.Float32)
                for c in range_constexpr(CPS // 32):
                    tk = 32 * c + 8 * g4  # this lane's 8 keys (one cache block's half)
                    w2 = raw[RAW_V + n * (CPS // 32) + c]
                    # this launch's keys (the suffix from pos0 on) are not in the cache yet for every
                    # CTA: their bytes zeroed here, their products added below
                    nk = fx.min(
                        fx.max(pos0 - (tb + tk), 0), 8
                    )  # this lane's cached keys
                    m0 = (nk >= 4).select(fx.Int32(-1), (fx.Int32(1) << (8 * nk)) - 1)
                    m1 = (nk >= 8).select(
                        fx.Int32(-1),
                        (nk <= 4).select(
                            fx.Int32(0), (fx.Int32(1) << (8 * (nk - 4))) - 1
                        ),
                    )
                    pv = mfma8(
                        lds_i64(XPP + (lane % 16) * SPP + tk // 4),
                        i64_of(w2[0] & m0, w2[1] & m1),
                        pv,
                    )
                # this launch's keys in the partition: P' x V from LDS (exact FP8 products)
                vdim = dim // 4
                for i in range_constexpr(T_SEQ):
                    t = pos0 + i
                    inside = (t >= tb) & (t < tb + CPS)
                    tl = fx.min(fx.max(t - tb, 0), CPS - 1)
                    vn = fp8_byte(lds_ld(xs, XVN + i * (HD // 4) + vdim), dim % 4)
                    for e in range_constexpr(4):
                        pq = fp8_byte(
                            lds_ld(xs, XPP + (4 * g4 + e) * SPP + tl // 4),
                            tl % 4,
                        )
                        pv_e = pv[e] + inside.select(pq * vn, fx.Float32(0.0))
                        pv = fx.Vector.from_elements(
                            [pv_e if j == e else pv[j] for j in range(4)], fx.Float32
                        )
                accs[n] = fx.Vector.from_elements(
                    [fmath.fma(accs[n][e], al[e], pscale * pv[e]) for e in range(4)],
                    fx.Float32,
                )
            # the next partition's loads, when there is one (a split's last would cost a wasted round trip)
            nxt = raw if PIPE else []
            # (two ifs: a build-time choice around a runtime test)
            if const_expr(PIPE):  # noqa: SIM102
                if fx.Int32(pi) + 1 < npart:
                    nxt = raw_loads(tb + CPS)
            gpu.barrier()
            res = (
                yield [m, lsum] + [accs[n][e] for n in range(2) for e in range(4)] + nxt
            )
        m = fx.Float32(res[0])
        lsum = fx.Float32(res[1])
        a0, a1, a2, a3, a4, a5, a6, a7 = [fx.Float32(res[2 + i]) for i in range(8)]
        # the split's output: acc * (1 / l) to bf16; (max, exp sum) per head
        if lane == 0:
            lds_st(xs, XA + 16 + wave, m)
            lds_st(xs, XA + 32 + wave, lsum)
        gpu.barrier()
        accs_ = [a0, a1, a2, a3, a4, a5, a6, a7]
        for e in range_constexpr(4):
            head = 4 * g4 + e
            le = lds_ld(xs, XA + 32 + fx.min(head, H - 1))
            inv_l = 1.0 / (le > 0.0).select(le, fx.Float32(1.0))
            for n in range_constexpr(NG):
                if head < H:
                    dim = (wave * NG + n) * 16 + lane % 16
                    lds_st(outs, head * HD + dim, bf16_round(accs_[4 * n + e] * inv_l))
        gpu.barrier()
        for i in range_constexpr(H * HD // 2 // 512):
            pr_ = tid + i * 512
            put_bf(
                mb("sp_out"),
                ((s * NSPLIT + js) * H) * HD + 2 * pr_,
                [lds_ld(outs, 2 * pr_), lds_ld(outs, 2 * pr_ + 1)],
            )
        if lane == 0:
            put2(
                mb("sp_ml"),
                ((s * NSPLIT + js) * H + wave) * 2,
                lds_ld(xs, XA + 16 + wave),
                lds_ld(xs, XA + 32 + wave),
            )

    # ================================ 4. merge the splits (aiter's pa_ps reduce), output gate -> gated
    NSP2 = 1 << (NSPLIT - 1).bit_length()
    BUTTERFLY = tuple(NSP2 >> i for i in range(1, NSP2.bit_length()))  # NSP2 / 2 .. 1
    for tt in range(start("merge"), S * H, G):
        tt = fx.Int32(tt)
        s = tt // H
        h = tt % H
        gpu.barrier()
        # every wave reduces the (at most 64) splits alike, wave 0 stores the weights: no wave-divergent
        # region around the lane shuffles
        jt = fx.min(lane, NSPLIT - 1)
        ml = poll([(mb("sp_ml"), ((s * NSPLIT + jt) * H + h) * 2, 2)])[0]
        in_ = lane < NSPLIT
        mj = in_.select(ml[0].bitcast(fx.Float32), fx.Float32(float("-inf")))
        lj = in_.select(ml[1].bitcast(fx.Float32), fx.Float32(0.0))
        gmax = wave_max(mj)
        sgmax = (gmax > -3.4028234663852886e38).select(gmax, fx.Float32(0.0))
        valid = mj > -3.4028234663852886e38
        x = lj * valid.select(exp(mj - sgmax), fx.Float32(0.0))
        # the exp sum by xor butterfly over the next power of two of lanes
        tot = (lane < NSP2).select(x, fx.Float32(0.0))
        for o_ in BUTTERFLY:
            tot = tot + xshfl(tot, o_)
        tot0 = read_lane_i32(tot.bitcast(fx.Int32), 0).bitcast(fx.Float32)
        wj = x / (tot0 > 0.0).select(tot0, fx.Float32(1.0))
        if (wave == 0) & (lane < NSPLIT):
            lds_st(misc, lane, wj)
        gpu.barrier()
        # every thread polls (a dim pair, clamped); the first HD / 2 store: no divergent poll loop
        tq = tid % (HD // 2)
        parts = poll(
            [
                (mb("sp_out"), (((s * NSPLIT + j) * H + h) * HD) // 2 + tq, 1)
                for j in range(NSPLIT)
            ]
        )
        o0 = fx.Float32(0.0)
        o1 = fx.Float32(0.0)
        for j in range_constexpr(NSPLIT):
            v0, v1 = bf2_f32(parts[j][0])
            wj = lds_ld(misc, j)
            o0 = fmath.fma(v0, wj, o0)
            o1 = fmath.fma(v1, wj, o1)
        gates = poll([(mb("qg"), s * QG + h * 2 * HD + HD + 2 * tq, 2)])[0]
        # ATOM's torch.sigmoid (accurate expf), or under QKF its Triton kernel's tl.sigmoid
        sig = sigmoid if const_expr(QKF_O) else sigmoid_precise
        sg = [sig(gates[e].bitcast(fx.Float32)) for e in range(2)]
        if tid < HD // 2:
            if const_expr(
                QKF_O
            ):  # quantized for o_proj from f32 (ATOM's sigmoid * mul + quant)
                put2(
                    mb("gated"),
                    s * OUT_K + h * HD + 2 * tq,
                    sg[0] * bf16_round(o0),
                    sg[1] * bf16_round(o1),
                )
            else:
                # ATOM's torch.sigmoid then * on bf16 tensors: the sigmoid and the product each bf16
                put_bf(
                    mb("gated"),
                    s * OUT_K + h * HD + 2 * tq,
                    [
                        bf16_round(bf16_round(sg[e]) * bf16_round(o))
                        for e, o in enumerate((o0, o1))
                    ],
                )
    return {}
