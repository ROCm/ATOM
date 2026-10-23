# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  (traced stage closures read the loop's values on purpose: the tracer unrolls)
"""The Gated DeltaNet mixer of the Qwen3.5 MonoKernel: projections (+ causal conv) and the delta-rule
recurrence with its gated norm (publishing ``gated`` for the shared out_proj)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen3_5.mono.kernels.common import (
    div_rn,
    fp8_word,
    quant_rule_q3,
    sigmoid,
    sigmoid_fast,
    silu,
)
from atom.models.qwen3_5.mono.kernels.ops import (
    ld_bf16,
    ld_f32,
    lds_ld,
    lds_st,
    log,
    sqrt,
)
from atom.models.qwen3_5.mono.layout import PROJ_GROUPS, PROJ_TILE
from atom.mono.device.ops import bf16_round, exp, rsq, rsrc, traced, uniform, xshfl
from atom.mono.plan.execution import WAVES

# recurrence thread mapping over ATOM's K-major state [K][V]: lane % 8 takes 16 consecutive k, lane // 8
# and the wave take 2 consecutive v (one 4-byte vector per k); k-reductions span 8 lanes (3 DPP levels)
KPT = 16  # k per thread
VPT = 2  # v per thread
KL = 8  # lanes sharing a v pair (over k)


def _sum8(v):
    for off in (1, 2, 4):
        v = v + xshfl(v, off)
    return v


@traced
def gdn_stages(ctx):
    """1-3. qkv|z and b|a projections (conv fused), recurrence + gated norm."""
    CAP = ctx["CAP"]
    CONV_DIM = ctx["CONV_DIM"]
    EPS = ctx["EPS"]
    G = ctx["G"]
    HV = ctx["HV"]
    KEY_DIM = ctx["KEY_DIM"]
    K_DIM = ctx["K_DIM"]
    MX = ctx["MX"]
    S = ctx["S"]
    REPLAY = ctx["REPLAY"]
    T_SEQ = ctx["T_SEQ"]
    VALUE_DIM = ctx["VALUE_DIM"]
    VPK = ctx["VPK"]
    V_DIM = ctx["V_DIM"]
    a_log = ctx["a_log"]
    block_sum = ctx["block_sum"]
    conv_state = ctx["conv_state"]
    dt_bias = ctx["dt_bias"]
    g_norm = ctx["g_norm"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    n_accepted = ctx["n_accepted"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    proj_stage = ctx["proj_stage"]
    put = ctx["put"]
    put_bf = ctx["put_bf"]
    rec_g = ctx["rec_g"]
    rec_k = ctx["rec_k"]
    rec_u = ctx["rec_u"]
    s_qkvz = ctx["s_qkvz"]
    slots_in = ctx["slots_in"]
    slots_out = ctx["slots_out"]
    ssm_state = ctx["ssm_state"]
    start = ctx["start"]
    tid = ctx["tid"]
    w_ba = ctx["w_ba"]
    w_conv = ctx["w_conv"]
    w_qkvz = ctx["w_qkvz"]
    wave = ctx["wave"]
    write_pos = ctx["write_pos"]
    xs = ctx["xs"]

    r_si, r_so, r_na = rsrc(slots_in), rsrc(slots_out), rsrc(n_accepted)
    r_conv, r_wc = rsrc(conv_state), rsrc(w_conv)
    # MTP verify (T_SEQ > 1): rows q * T_SEQ + j are token j of sequence q; slots_out[row] = the state slot
    # after the row's token (ATOM's spec_state_indices), n_accepted[q] = the tokens the last step accepted.
    # The sequence's conv window lives in token 0's slot (CW = T_SEQ + 2 inputs, read from n_accepted - 1).
    CW = T_SEQ + 2
    # ReplaySSM (REPLAY, a verify step): the request's state is a checkpoint (slot slots_in[q], the
    # ssm_state pool) plus the first h = write_pos[slot] of CAP records (k, u, g) a v-head, the state
    # S_h = e^{C} S_0 + sum_j e^{C - C_j} k_j u_j^T (C_j the running sum of the log-decays g). A step
    # whose window might not fit (h + 2 T_SEQ > CAP) first folds them into the checkpoint and writes its
    # records from 0; its tokens append theirs, the cursor advanced by ATOM (``replayssm_commit``).
    r_wp = rsrc(write_pos)
    ROWS = PROJ_TILE * PROJ_GROUPS

    # ================================ 1. qkv | z (FP8) and b | a (BF16) projections
    def conv_prefetch(s, row):
        """``conv_epilogue``'s loads for sample ``s``, row ``row`` (the GEMM's z rows clamp to a
        channel; their values go unused): the slots, the 3 stored inputs and the channel's taps.
        """
        ch = fx.min(row, CONV_DIM - 1)
        si = fx.Int32(bo.buffer_load(r_si, s, vec_width=1, dtype=T.i32))
        so = fx.Int32(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
        cs = [
            ld_bf16(r_conv, (fx.max(si, 0) * 3 + j) * CONV_DIM + ch) for j in range(3)
        ]
        wv = fx.Vector(bo.buffer_load(r_wc, ch * 4, vec_width=4, dtype=T.bf16)).to(
            fx.Float32
        )
        return si, so, cs, wv

    def conv_epilogue(s, ch, v, pf):
        """Channel ``ch`` of sample ``s``: causal conv over the 3 stored inputs and ``v``, SiLU;
        the state shifts to slots_out. A padded sample (negative slot) publishes 0. ``pf``:
        ``conv_prefetch``'s loads, issued before the GEMM."""
        si, so, cs, wv = pf
        ok = (si >= 0) & (so >= 0)
        x = bf16_round(v)
        # ATOM's causal_conv1d_update: each bf16 x bf16 tap product rounded to bf16, summed in f32, oldest first
        acc = (
            bf16_round(cs[0] * wv[0])
            + bf16_round(cs[1] * wv[1])
            + bf16_round(cs[2] * wv[2])
            + bf16_round(x * wv[3])
        )
        if ok:
            for j, val in ((0, cs[1]), (1, cs[2]), (2, x)):
                bo.buffer_store(
                    val.to(fx.BFloat16), r_conv, (so * 3 + j) * CONV_DIM + ch
                )
        put(
            mb("conv"),
            s * CONV_DIM + ch,
            ok.select(bf16_round(silu(acc)), fx.Float32(0.0)),
        )

    def conv_epilogue_mtp(s, ch, v):
        """Channel ``ch`` of every token of sequence ``s // T_SEQ`` (on token 0's thread): the window
        [3 stored inputs from n_accepted - 1, this launch's T_SEQ] -> T_SEQ outputs; the stored
        inputs shift by one (ATOM's speculative causal_conv1d_update)."""
        if s % T_SEQ == 0:
            so = fx.Int32(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
            na = fx.Int32(bo.buffer_load(r_na, s // T_SEQ, vec_width=1, dtype=T.i32))
            ok = (so >= 0) & (na >= 1)
            base = fx.max(so, 0) * CW + fx.max(na - 1, 0)
            win = [ld_bf16(r_conv, (base + j) * CONV_DIM + ch) for j in range(3)]
            win += [
                bf16_round(lds_ld(outs, (s + j) * ROWS + ch % ROWS))
                for j in range(T_SEQ)
            ]
            wv = fx.Vector(bo.buffer_load(r_wc, ch * 4, vec_width=4, dtype=T.bf16)).to(
                fx.Float32
            )
            for j in range_constexpr(T_SEQ):
                acc = (
                    bf16_round(win[j] * wv[0])
                    + bf16_round(win[j + 1] * wv[1])
                    + bf16_round(win[j + 2] * wv[2])
                    + bf16_round(win[j + 3] * wv[3])
                )
                put(
                    mb("conv"),
                    (s + j) * CONV_DIM + ch,
                    ok.select(bf16_round(silu(acc)), fx.Float32(0.0)),
                )
            if ok:
                for i in range_constexpr(CW):
                    bo.buffer_store(
                        win[i + 1].to(fx.BFloat16),
                        r_conv,
                        (fx.max(so, 0) * CW + i) * CONV_DIM + ch,
                    )

    def qkvz_epilogue(s, row, v, pf=None):
        if row < CONV_DIM:
            if const_expr(T_SEQ == 1):
                conv_epilogue(s, row, v, pf)
            else:
                conv_epilogue_mtp(s, row, v)
        else:
            put(mb("z"), s * VALUE_DIM + row - CONV_DIM, bf16_round(v))

    proj_stage(
        "proj",
        (CONV_DIM + VALUE_DIM) // PROJ_TILE,
        w_qkvz,
        s_qkvz,
        False,
        qkvz_epilogue,
        prefetch=conv_prefetch if T_SEQ == 1 else None,
    )
    proj_stage(
        "proj_ba",
        2 * HV // PROJ_TILE,
        w_ba,
        None,
        True,
        lambda s, row, v: put(mb("ba"), s * 2 * HV + row, bf16_round(v)),
    )

    # ================================ 2. gated delta rule + gated RMSNorm (one task per sample x v-head)
    assert (
        K_DIM == KL * KPT and V_DIM == WAVES * (64 // KL) * VPT
    ), "one 128 x 128 head over 512 threads"
    HEAD_BYTES = V_DIM * K_DIM * 2
    kg = lane % KL
    k0_ = kg * KPT
    v0_ = (wave * (64 // KL) + lane // KL) * VPT
    r_alog, r_dtb, r_gn = rsrc(a_log), rsrc(dt_bias), rsrc(g_norm)
    v4f = fx.Vector.make_type(4, fx.Float32)
    vvf = fx.Vector.make_type(VPT, fx.Float32)
    # LDS (f32 words of xs): q 0, k K_DIM, v 2*K_DIM, z 2*K_DIM + V_DIM, b / a after
    XQ, XK, XV = 0, K_DIM, 2 * K_DIM
    XZ = XV + V_DIM
    XB = XZ + V_DIM
    for tt in range(start("gdn"), S // T_SEQ * HV, G):
        tt = fx.Int32(tt)
        sq = tt // HV
        h = tt % HV
        kh = h // VPK
        gpu.barrier()
        if const_expr(REPLAY):
            si = uniform(bo.buffer_load(r_si, sq, vec_width=1, dtype=T.i32))
            live = si >= 0
            # the prefill sentinel -1 (commit not yet run): no records, write from 0
            hp = fx.max(
                uniform(bo.buffer_load(r_wp, fx.max(si, 0), vec_width=1, dtype=T.i32)),
                0,
            )
            flush = hp + 2 * T_SEQ > CAP
            rbase = flush.select(fx.Int32(0), hp)
            rec0 = (
                fx.Int64(fx.max(si, 0)) * HV + fx.Int64(h)
            ) * CAP  # this v-head's record 0
        elif const_expr(T_SEQ == 1):
            si = uniform(bo.buffer_load(r_si, sq, vec_width=1, dtype=T.i32))
            live = si >= 0
        else:
            # MTP verify: the state after the last accepted token of the previous step
            na = uniform(bo.buffer_load(r_na, sq, vec_width=1, dtype=T.i32))
            si = uniform(
                bo.buffer_load(
                    r_so, sq * T_SEQ + fx.max(na - 1, 0), vec_width=1, dtype=T.i32
                )
            )
            live = (si >= 0) & (na >= 1)
        # the state does not depend on this launch: issue its loads before waiting for the inputs
        r_in = rsrc(
            ssm_state + (fx.Int64(fx.max(si, 0)) * HV + fx.Int64(h)) * HEAD_BYTES
        )
        sv = [
            fx.Vector(
                bo.buffer_load(
                    r_in, (k0_ + j) * V_DIM + v0_, vec_width=VPT, dtype=T.bf16
                )
            ).to(fx.Float32)
            for j in range(KPT)
        ]
        if const_expr(REPLAY):
            # S_h: the checkpoint decayed by e^{C}, plus each record weighted by e^{C - C_j}, as ATOM's
            # replay. The gates are wave-uniform (scalar registers); the records run as a loop over the h live
            # ones, the state carried: unrolled over all CAP, their loads hoisted together spill at S = 8.
            r_g0 = rsrc(rec_g + rec0 * 4)

            def gate(j):
                return uniform(
                    bo.buffer_load(r_g0, j, vec_width=1, dtype=T.i32)
                ).bitcast(fx.Float32)

            c = fx.Float32(0.0)
            for j in range_constexpr(CAP):
                c = c + (fx.Int32(j) < hp).select(gate(j), fx.Float32(0.0))
            dec0 = fx.Vector.filled(VPT, exp(c), fx.Float32)
            for j in range_constexpr(KPT):
                sv[j] = sv[j] * dec0
            r_k0 = rsrc(rec_k + rec0 * (K_DIM * 2))
            r_u0 = rsrc(rec_u + rec0 * (V_DIM * 2))
            for jr, acc in range(0, hp, fx.Int32(1), init=sv + [fx.Float32(0.0)]):
                jj = fx.Int32(jr)
                cj = fx.Float32(acc[KPT]) + gate(jj)
                w = exp(c - cj)
                kr = [
                    fx.Vector(
                        bo.buffer_load(
                            r_k0, jj * K_DIM + k0_ + 8 * p, vec_width=8, dtype=T.bf16
                        )
                    ).to(fx.Float32)
                    for p in range(KPT // 8)
                ]
                ur = fx.Vector(
                    bo.buffer_load(r_u0, jj * V_DIM + v0_, vec_width=VPT, dtype=T.bf16)
                ).to(fx.Float32)
                # ATOM's replay dot (DOT_MODE 2): k e^{C - C_j} split into bf16 hi + lo, each times the bf16 u
                st = []
                for i in range_constexpr(KPT):
                    kw = kr[i // 8][i % 8] * w
                    hi = bf16_round(kw)
                    lo = bf16_round(kw - hi)
                    part = fx.math.fma(
                        fx.Vector.filled(VPT, hi, fx.Float32), ur, fx.Vector(acc[i])
                    )
                    st.append(
                        fx.math.fma(fx.Vector.filled(VPT, lo, fx.Float32), ur, part)
                    )
                res = yield st + [cj]
            sv = [fx.Vector(res[i]) for i in range(KPT)]
            if flush & live:
                r_ck = rsrc(
                    ssm_state
                    + (fx.Int64(fx.max(si, 0)) * HV + fx.Int64(h)) * HEAD_BYTES
                )
                for j in range_constexpr(KPT):
                    bo.buffer_store(
                        sv[j].to(fx.BFloat16), r_ck, (k0_ + j) * V_DIM + v0_
                    )
        alog = ld_f32(r_alog, h)
        dtb = ld_f32(r_dtb, h)
        for tok in range_constexpr(
            T_SEQ
        ):  # the sequence's tokens in order, the state in registers
            s = sq * T_SEQ + tok
            so = uniform(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
            ok = live & (so >= 0)
            if const_expr(REPLAY):
                ok = live
            if const_expr(tok > 0):
                gpu.barrier()  # the previous token's LDS inputs are read
            if tid < K_DIM:
                got = poll(
                    [
                        (mb("conv"), s * CONV_DIM + kh * K_DIM + tid, 1),
                        (mb("conv"), s * CONV_DIM + KEY_DIM + kh * K_DIM + tid, 1),
                        (mb("conv"), s * CONV_DIM + 2 * KEY_DIM + h * V_DIM + tid, 1),
                        (mb("z"), s * VALUE_DIM + h * V_DIM + tid, 1),
                    ]
                )
                for j, off in ((0, XQ), (1, XK), (2, XV), (3, XZ)):
                    lds_st(xs, off + tid, got[j][0].bitcast(fx.Float32))
            if tid == K_DIM:
                got = poll(
                    [(mb("ba"), s * 2 * HV + h, 1), (mb("ba"), s * 2 * HV + HV + h, 1)]
                )
                lds_st(xs, XB, got[0][0].bitcast(fx.Float32))
                lds_st(xs, XB + 1, got[1][0].bitcast(fx.Float32))
            gpu.barrier()
            b_in = lds_ld(xs, XB)
            a_in = lds_ld(xs, XB + 1)
            # ATOM's fused_gdn_gating: beta stored as bf16, softplus log(1 + exp(x)) up to 20
            beta = bf16_round(sigmoid(b_in))
            x_sp = a_in + dtb
            softplus = (x_sp > 20.0).select(x_sp, log(1.0 + exp(x_sp)))
            g_log = -exp(alog) * softplus
            decay = exp(g_log)
            # this thread's KPT q / k (L2-normalized over the head: sums over the KL lanes)
            qx = [
                fx.Vector(fx.ptr_load(xs + (XQ + k0_ + 4 * i), result_type=v4f))
                for i in range(KPT // 4)
            ]
            kx = [
                fx.Vector(fx.ptr_load(xs + (XK + k0_ + 4 * i), result_type=v4f))
                for i in range(KPT // 4)
            ]
            qq = [qx[j // 4][j % 4] for j in range(KPT)]
            kk = [kx[j // 4][j % 4] for j in range(KPT)]
            qsq = fx.Float32(0.0)
            ksq = fx.Float32(0.0)
            for j in range_constexpr(KPT):
                qsq = qsq + qq[j] * qq[j]
                ksq = ksq + kk[j] * kk[j]
            # ATOM's recurrence: x / sqrt(sum x^2 + 1e-6), then q * K^-0.5
            q_den = sqrt(_sum8(qsq) + 1.0e-6)
            k_den = sqrt(_sum8(ksq) + 1.0e-6)
            # x / den as the IEEE divide rounds it, at a reciprocal's cost (common.div_rn)
            rq, rk = 1.0 / q_den, 1.0 / k_den
            qq = [div_rn(x, q_den, rq) * (K_DIM**-0.5) for x in qq]
            kk = [div_rn(x, k_den, rk) for x in kk]
            # decay and S^T k over this thread's k (VPT v at once), summed over the KL lanes
            dvec = fx.Vector.filled(VPT, decay, fx.Float32)
            skp = fx.Vector.filled(VPT, 0.0, fx.Float32)
            for j in range_constexpr(KPT):
                sv[j] = sv[j] * dvec
                skp = fx.math.fma(sv[j], fx.Vector.filled(VPT, kk[j], fx.Float32), skp)
            vin = fx.Vector(fx.ptr_load(xs + (XV + v0_), result_type=vvf))
            delta = [(vin[e] - _sum8(skp[e])) * beta for e in range(VPT)]
            dv = fx.Vector.from_elements(delta, fx.Float32)
            sqp = fx.Vector.filled(VPT, 0.0, fx.Float32)
            for j in range_constexpr(KPT):
                sv[j] = fx.math.fma(fx.Vector.filled(VPT, kk[j], fx.Float32), dv, sv[j])
                sqp = fx.math.fma(sv[j], fx.Vector.filled(VPT, qq[j], fx.Float32), sqp)
            # o = S_new^T q, stored bf16
            outs_v = [bf16_round(_sum8(sqp[e])) for e in range(VPT)]
            # gated RMSNorm over the head's V_DIM outputs (lane kg 0 of each group holds its VPT once)
            ssq = fx.Float32(0.0)
            for e in range_constexpr(VPT):
                o2 = outs_v[e] * outs_v[e]
                ssq = ssq + (bf16_round(o2) if const_expr(MX) else o2)
            var = block_sum((kg == 0).select(ssq, fx.Float32(0.0))) * (1.0 / V_DIM)
            if const_expr(MX):
                # ATOM's RMSNormGated.forward_native on bf16 tensors: x.pow(2), the mean, + eps and
                # rsqrt each round to bf16
                rstd = bf16_round(rsq(bf16_round(bf16_round(var) + EPS)))
            else:
                rstd = rsq(var + EPS)
            if kg == 0:
                for e in range_constexpr(VPT):
                    v_idx = v0_ + e
                    z = lds_ld(xs, XZ + v_idx)
                    if const_expr(MX):
                        nrm = bf16_round(outs_v[e] * rstd)
                        # torch's SiLU: z / (1 + expf(-z)), accurate expf, bf16
                        sz = bf16_round(z / (1.0 + fmath.exp(-z)))
                        gated = bf16_round(bf16_round(nrm * ld_f32(r_gn, v_idx)) * sz)
                    else:
                        # (x * w) * inv_std * silu(z) in f32 (-ffast-math: v_rcp in the sigmoid)
                        gated = (outs_v[e] * ld_f32(r_gn, v_idx) * rstd) * (
                            z * sigmoid_fast(z)
                        )
                    lds_st(outs, v_idx, ok.select(gated, fx.Float32(0.0)))
            gpu.barrier()
            if const_expr(MX):
                if tid < V_DIM // 2:
                    put_bf(
                        mb("gated"),
                        s * VALUE_DIM + h * V_DIM + 2 * tid,
                        [lds_ld(outs, 2 * tid), lds_ld(outs, 2 * tid + 1)],
                    )
            elif tid < V_DIM // 4:  # the head is one 128-block: Q3 FP8 for out_proj
                g4 = [lds_ld(outs, 4 * tid + e) for e in range(4)]
                amax = fx.max(
                    fx.max(fmath.absf(g4[0]), fmath.absf(g4[1])),
                    fx.max(fmath.absf(g4[2]), fmath.absf(g4[3])),
                )
                for o in (1, 2, 4, 8, 16):
                    amax = fx.max(amax, xshfl(amax, o))
                sc, q = quant_rule_q3(amax)
                put(
                    mb("gq"),
                    (s * VALUE_DIM + h * V_DIM) // 4 + tid,
                    fp8_word(q(g4[0]), q(g4[1]), q(g4[2]), q(g4[3])),
                )
                if tid == 0:
                    put(mb("gqs"), s * (VALUE_DIM // 128) + h * (V_DIM // 128), sc)
            # the new state after the publish: out_proj waits on the gated row, not on this store
            if const_expr(REPLAY):
                # this token's record (k normalized, u = beta (v - S^T k), the log-decay g) at rbase + tok
                rp = rbase + tok
                if ok & (v0_ == 0):  # the KL lanes of v pair 0 hold every k once
                    for p in range_constexpr(KPT // 8):
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                kk[8 * p : 8 * p + 8], fx.Float32
                            ).to(fx.BFloat16),
                            rsrc(rec_k + rec0 * (K_DIM * 2)),
                            rp * K_DIM + k0_ + 8 * p,
                        )
                if ok & (kg == 0):
                    bo.buffer_store(
                        dv.to(fx.BFloat16),
                        rsrc(rec_u + rec0 * (V_DIM * 2)),
                        rp * V_DIM + v0_,
                    )
                if ok & (tid == 0):
                    bo.buffer_store(g_log, rsrc(rec_g + rec0 * 4), rp)
            elif ok:
                r_o = rsrc(ssm_state + (fx.Int64(so) * HV + fx.Int64(h)) * HEAD_BYTES)
                for j in range_constexpr(KPT):
                    bo.buffer_store(sv[j].to(fx.BFloat16), r_o, (k0_ + j) * V_DIM + v0_)

    return {}
