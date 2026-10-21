# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  (traced stage closures read the loop's values on purpose: the tracer unrolls)
"""What both token mixers of the Qwen3.5 MonoKernel share: the input-RMSNorm projection GEMV and
out_proj with the attention-side TP reduce."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen3_5.mono.kernels.common import (
    quant_rule_q1,
    quant_rule_q1_norm,
    quant_rule_q2,
    quant_rule_q4,
)
from atom.models.qwen3_5.mono.kernels.ops import lds_ld
from atom.models.qwen3_5.mono.layout import (
    MOE_MISC,
    PROJ_GROUPS,
    PROJ_TILE,
    ROW_TILE,
    bf16_gemm_rule,
)
from atom.mono.device.ops import bf16_round, rsrc, traced
from atom.mono.plan.execution import WAVES


@traced
def mixer_defs(ctx):
    """``proj_stage``: a GEMV over ``rmsnorm(h) * (1 + g_in)`` whose rows go to an epilogue."""
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    MX = ctx["MX"]
    S = ctx["S"]
    XIN = ctx["XIN"]
    XQ_BLOCKS = ctx["XQ_BLOCKS"]
    layer = ctx["layer"]
    misc = ctx["misc"]
    emit_out = ctx["emit_out"]
    g_in1 = ctx["g_in1"]
    h_in = ctx["h_in"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    stage_x_gemma = ctx["stage_x_gemma"]
    stage_x8 = ctx["stage_x8"]
    put = ctx["put"]
    mb = ctx["mb"]
    xs = ctx["xs"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    unit_bf16_rows = ctx["st"].unit_bf16_rows
    unit_f8f8 = ctx["unit_f8f8"]
    wave = ctx["wave"]
    r_h = rsrc(h_in)
    NKC = HIDDEN // 64
    WPR = WAVES // PROJ_GROUPS
    UPW = NKC // WPR
    UPW8 = HIDDEN // 128 // WPR  # 128-K units of an FP8 matrix per wave
    ROWS = PROJ_TILE * PROJ_GROUPS
    first = layer == 0

    def input_quant(amax):
        """The input norm's FP8 quant: the first layer (and the MTP draft, both unfused) its own norm
        kernel's, every later layer the fused all-reduce norm's."""
        sc1, q1 = quant_rule_q1_norm(amax)
        sc2, q2 = quant_rule_q2(amax)
        return first.select(sc1, sc2), lambda x: first.select(q1(x), q2(x))

    def ld_h(sks):
        """The f32 residual stream (the input norm's operand: aiter's fused all-reduce norm normalizes
        ``residual + bf16(rank sum)`` before rounding it)."""
        res = []
        for s, k in sks:
            v = fx.Vector(bo.buffer_load(r_h, s * HIDDEN + k, vec_width=4, dtype=T.f32))
            res.append([v[j] for j in range(4)])
        return res

    def xin_stage():
        """Stage ``xin`` (past XIN_MIN rows): a task per row normalizes it and quantizes it to FP8 once, as each
        projection task otherwise would (the same values and rounding), publishing the bytes (``xinq``) and
        the per-128 scales (``xins``) the FP8 projections stage from."""
        NQ, NB = HIDDEN // 4, HIDDEN // 128
        for tt in range(start("xin"), S, G):
            s = fx.Int32(tt)
            gpu.barrier()
            stage_x_gemma(
                lambda sks: ld_h([(s, k) for _r, k in sks]),
                HIDDEN,
                g_in1,
                count=1,
                fp8_rule=input_quant,
            )
            gpu.barrier()
            for i in range_constexpr(NQ // 512):
                w = tid + i * 512
                put(mb("xinq"), s * NQ + w, lds_ld(xs, w))
            if tid < NB:
                put(mb("xins"), s * NB + tid, lds_ld(misc, MOE_MISC + tid))

    def proj_stage(name, n_groups, w, sc, bf16_w, epilogue, prefetch=None):
        """Stage ``name``: ``n_groups`` 16-row groups of FP8 (scales ``sc``) or BF16 matrix ``w``,
        PROJ_GROUPS per task; ``epilogue(s, row, value)`` takes each output (``tid`` < S * rows). An FP8
        matrix runs W8A8 (aiter's blockscale GEMM): the input FP8 per 1x128, each 128-K block's partial
        times ``a_scale * w_scale``. An FP8 matrix of the MXFP4 checkpoint (``MX``) is BF16 there,
        row-major as ATOM holds it. ``prefetch(s, row)`` (every thread, s clamped) issues the epilogue's
        loads before the GEMM; its result is the epilogue's fourth argument.
        """
        r_w, r_s = rsrc(w), (None if bf16_w else rsrc(sc))
        a8 = not bf16_w and not MX
        if const_expr(name == "proj" and XIN):
            xin_stage()
        batch = max(1, NKC // WAVES // (4 if bf16_w else 1))  # units in flight per wave
        if const_expr(S > 8):  # past 8 samples the staged activations need the room
            batch = max(1, batch // 2)
        if const_expr(a8):  # a 128-K unit loads what two 64-K units did
            batch = max(1, batch // 2)
        for t in range(start(name), n_groups // PROJ_GROUPS, G):
            t = fx.Int32(t)
            gpu.barrier()
            rg = t * PROJ_GROUPS + wave // WPR

            def u(c):
                if const_expr(a8):
                    kb = (wave % WPR) * UPW8 + c
                    sn = n_sel()
                    return unit_f8f8(
                        r_w,
                        r_s,
                        rg,
                        kb * 2,
                        NKC,
                        HIDDEN,
                        sn * (HIDDEN // 4) + kb * 32,
                        lambda: lds_ld(misc, MOE_MISC + sn * XQ_BLOCKS + kb),
                    )
                kc = (wave % WPR) * UPW + c
                bw = (n_sel() * HIDDEN + kc * 64) // 2
                if const_expr(bf16_w):
                    return unit_bf16(r_w, rg, kc, NKC, bw)
                return unit_bf16_rows(r_w, rg, kc, HIDDEN, bw)  # MX: BF16 row-major

            pre = [u(c) for c in range(batch)]
            # the epilogue's loads, ahead of the GEMM
            if const_expr(prefetch is not None):
                pf = prefetch(fx.min(tid // ROWS, S - 1), t * ROWS + tid % ROWS)
            if const_expr(a8 and XIN):
                stage_x8("xinq", "xins", HIDDEN, list(range(S)))
            else:
                stage_x_gemma(ld_h, HIDDEN, g_in1, fp8_rule=input_quant if a8 else None)
            gpu.barrier()
            acc = run_units(u, UPW8 if a8 else UPW, batch, pre)
            reduce_rows(PROJ_GROUPS, acc, emit_out(ROWS))
            gpu.barrier()
            if tid < S * ROWS:
                if const_expr(prefetch is not None):
                    epilogue(tid // ROWS, t * ROWS + tid % ROWS, lds_ld(outs, tid), pf)
                else:
                    epilogue(tid // ROWS, t * ROWS + tid % ROWS, lds_ld(outs, tid))

    return {"proj_stage": proj_stage}


@traced
def out_stages(ctx):
    """out_proj over the mixer's [S, OUT_K] output, the attention-side TP peer reduce and the residual
    ``h`` -> mailbox ``a``. The FP8 checkpoint runs it W8A8: attention's bf16 ``gated`` output quantized
    as aiter's per-1x128 quant does (Q1), the GDN output FP8 already (``gq``, its gated norm's Q3).
    """
    FULL = ctx["FULL"]
    QKF_O = ctx["QKF_O"]
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    MX = ctx["MX"]
    OUT_K = ctx["OUT_K"]
    S = ctx["S"]
    misc = ctx["misc"]
    emit_out = ctx["emit_out"]
    h_in = ctx["h_in"]
    mb = ctx["mb"]
    n_sel = ctx["n_sel"]
    peer_reduce = ctx["peer_reduce"]
    put2 = ctx["put2"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    s_out = ctx["s_out"]
    stage_f32_q8 = ctx["stage_f32_q8"]
    stage_pairs_q8 = ctx["stage_pairs_q8"]
    stage_x8 = ctx["stage_x8"]
    stage_x_pairs = ctx["stage_x_pairs"]
    start = ctx["start"]
    unit_f8f8 = ctx["unit_f8f8"]
    w_out = ctx["w_out"]
    wave = ctx["wave"]
    r_h = rsrc(h_in)
    OUT_NKC = OUT_K // 64
    OUT_R = ROW_TILE // 16
    OUT_WPR = WAVES // OUT_R
    OUT_UPW = OUT_NKC // OUT_WPR if MX else OUT_K // 128 // OUT_WPR
    # the MXFP4 checkpoint's BF16 out_proj rounds as baseline ATOM's GEMM: with K halves, wave j of a
    # row group takes the first (j even) or second (j odd) 128 of every 256-wide K tile
    OUT_RULE = bf16_gemm_rule("out", S, ctx["T_SEQ"]) if MX else None
    HALVES = OUT_RULE is not None and OUT_RULE[0] == "halves"
    if HALVES:
        assert OUT_WPR % 2 == 0 and OUT_K % 256 == 0 and OUT_WPR * OUT_UPW == OUT_NKC
    r_wo, r_so = rsrc(w_out), rsrc(s_out)

    def residual(s, row):
        """The stored (bf16) residual: the f32 stream's rounding."""
        v = fx.Vector(bo.buffer_load(r_h, s * HIDDEN + row, vec_width=2, dtype=T.f32))
        return bf16_round(v[0]), bf16_round(v[1])

    for t in range(start("out"), HIDDEN // ROW_TILE, G):
        t = fx.Int32(t)
        rg = t * OUT_R + wave // OUT_WPR

        def u_o(c):
            if const_expr(MX):
                j = wave % OUT_WPR
                if const_expr(HALVES):
                    q = (j // 2) * OUT_UPW + c  # this half's q-th 64-wide chunk
                    kc = (q // 2) * 4 + q % 2 + 2 * (j % 2)
                else:
                    kc = j * OUT_UPW + c
                bw = (n_sel() * OUT_K + kc * 64) // 2
                return ctx["st"].unit_bf16_rows(r_wo, rg, kc, OUT_K, bw)
            kb = (wave % OUT_WPR) * OUT_UPW + c
            sn = n_sel()
            return unit_f8f8(
                r_wo,
                r_so,
                rg,
                kb * 2,
                OUT_NKC,
                OUT_K,
                sn * (OUT_K // 4) + kb * 32,
                lambda: lds_ld(misc, MOE_MISC + sn * (OUT_K // 128) + kb),
            )

        pre = [u_o(c) for c in range(OUT_UPW)]
        gpu.barrier()
        if const_expr(MX):
            stage_x_pairs("gated", S * OUT_K, lambda k: k)
        elif const_expr(QKF_O):
            stage_f32_q8("gated", OUT_K, quant_rule_q4)
        elif const_expr(FULL):
            stage_pairs_q8("gated", OUT_K, quant_rule_q1)
        else:
            stage_x8("gq", "gqs", OUT_K, list(range(S)))
        gpu.barrier()
        acc = run_units(u_o, OUT_UPW, OUT_UPW, pre)
        reduce_rows(OUT_R, acc, emit_out(ROW_TILE), rule=OUT_RULE)
        gpu.barrier()
        peer_reduce(
            "attn",
            t,
            residual,
            lambda s, row, v0, v1: put2(mb("a"), s * HIDDEN + row, v0, v1),
            tile=ROW_TILE,
        )
        gpu.barrier()
    return {}
