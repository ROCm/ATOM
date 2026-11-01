# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K4 stage: the router GEMV and the operand helpers the experts share
(``kernels.stages.router`` on wave32: a task's 8 experts each on two waves, one
K half a wave, the halves summed in that order; a lane routes four experts,
``lane + 32 i``)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.minimax_m3.mono.config import HIDDEN, N_ROUTED, SHARED_EXPERT, TOP_K
from atom.models.minimax_m3.mono.gfx1250.layout import N_ROUTER, SCORE_COPIES
from atom.models.minimax_m3.mono.kernels.common import ld_bf16x12
from atom.mono.device.gfx1250 import (
    CM_DEV,
    THREADS,
    WAVE,
    WAVES,
    butterfly,
    publish,
    wave_sum,
)
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, pow2
from atom.mono.device.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    rsrc,
    traced,
)
from atom.mono.device.sync import preg

XN_SLICE = HIDDEN // N_ROUTER  # normalized-input elements a router task publishes
EPL = N_ROUTED // WAVE  # experts a lane routes
GATE_HALF = HIDDEN // 2  # a wave's K half of its expert's gate row
GATE_ROUNDS = GATE_HALF // (WAVE * 8)  # 8 bf16 a lane a round


@traced
def router_defs(k4ctx):
    """``kernels.stages.router.router_defs`` on gfx1250."""
    AG_RS = k4ctx["AG_RS"]
    CM_K1 = k4ctx["CM_K1"]
    G = k4ctx["G"]
    SY = k4ctx["SY"]
    WIDE = k4ctx["WIDE"]
    XN8_WORDS = k4ctx["XN8_WORDS"]
    XSC_WORDS = k4ctx["XSC_WORDS"]
    acquire = k4ctx["acquire"]
    bias = k4ctx["bias"]
    eps = k4ctx["eps"]
    fuse_k1 = k4ctx["fuse_k1"]
    x8_in = k4ctx["x8_in"]
    g4 = k4ctx["g4"]
    g_post = k4ctx["g_post"]
    h_in = k4ctx["h_in"]
    h_mid = k4ctx["h_mid"]
    k1_poll = k4ctx["k1_poll"]
    lane = k4ctx["lane"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    mb_put_words = k4ctx["mb_put_words"]
    mr = k4ctx["mr"]
    rdone_mb = k4ctx["rdone_mb"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    route_key = k4ctx["route_key"]
    route_scale = k4ctx["route_scale"]
    rwt = k4ctx["rwt"]
    scores_copy = k4ctx["scores_copy"]
    shared_weight = k4ctx["shared_weight"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    sym = k4ctx["sym"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    w_gate = k4ctx["w_gate"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]
    v4i = fx.Vector.make_type(4, fx.Int32)

    # ======================================================= MoE helpers
    def route_top4(k, keys, sigs, sig, wave_mask=None, mask_bit=0):
        """``router.route_top4`` with this lane's experts lane + 32 i (``keys`` /
        ``sigs``, i < EPL)."""
        if wave == k % WAVES:
            for i in range_constexpr(EPL):
                fx.ptr_store(sigs[i], sig + (k * N_ROUTED + WAVE * i + lane))
            ks = list(keys)
            pid = fx.Int32(0)
            for p in range_constexpr(TOP_K):
                m = ks[0]
                for i in range_constexpr(1, EPL):
                    m = fx.max(m, ks[i])
                m = butterfly(m, (16, 8, 4, 2, 1), fx.max)
                ks = [(kk == m).select(fx.Int32(-(2**31)), kk) for kk in ks]
                pid = (lane == p).select(127 - (m & 127), pid)
            if lane < TOP_K:
                w = fx.ptr_load(sig + (k * N_ROUTED + pid))
                tot = butterfly(w, (1, 2))
                fx.ptr_store(pid, route + (8 * k + lane))
                fx.ptr_store(
                    w * (route_scale / fx.max(tot, 1e-20)), rwt + (8 * k + lane)
                )
                if const_expr(wave_mask is not None):
                    was = fx.Int32(fx.ptr_load(wave_mask + pid))
                    fx.ptr_store(was | mask_bit, wave_mask + pid)
            if lane == 0:
                fx.ptr_store(fx.Int32(SHARED_EXPERT), route + (8 * k + TOP_K))
                fx.ptr_store(fx.Float32(shared_weight), rwt + (8 * k + TOP_K))

    def poll_scores(k, copy):
        """This lane's experts' (routing key, sigmoid) of token k (copy ``copy``)."""
        got = mb_poll(
            [(scores_copy(copy, k), (lane + WAVE * i) * 2, 2) for i in range(EPL)]
        )
        return [g[0] for g in got], [g[1].bitcast(fx.Float32) for g in got]

    def mx8_scale(vals, offsets):
        amax = fx.max(vals[0], -vals[0])
        for v in vals[1:]:
            amax = fx.max(amax, fx.max(v, -v))
        amax = butterfly(amax, offsets, fx.max)
        e = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        return e, hw_rcp(pow2(e))

    def mx8_q(v, inv):
        return clamp_fp8(v * inv)

    def lds_b8(chunk_word):
        """This lane's WMMA B operand of the 128-k FP8 chunk at LDS word
        ``chunk_word``: k 32 j + 16 g .. + 16, j < 4 (g = lane // 16)."""
        h = [
            fx.Vector(fx.ptr_load(xs + (chunk_word + 8 * j + 4 * g4), result_type=v4i))
            for j in range(4)
        ]
        return fx.Vector.from_elements(
            [h[j][e] for j in range(4) for e in range(4)], fx.Int32
        )

    def lds_i32(word):
        return fx.ptr_load(xs + word).bitcast(fx.Int32)

    def lds_scale4(word):
        """The WMMA scale dword of the four E8M0 codes at LDS words ``word`` ..
        ``word`` + 3 (one code a word, as the stages stage them), byte j code j."""
        v = fx.Vector(fx.ptr_load(xs + word, result_type=v4i))
        return (
            (v[0] & 0xFF) | ((v[1] & 0xFF) << 8) | ((v[2] & 0xFF) << 16) | (v[3] << 24)
        )

    # ============================================================ 4. router
    def stage_router():
        r_g = rsrc(g_post)
        r_gate = rsrc(w_gate)
        RE = HIDDEN // THREADS
        rw0 = tid * (RE // 2)
        for tt in range(start("router"), N_ROUTER * tokens, G):
            tt = fx.Int32(tt)
            k = tt // N_ROUTER
            t = tt % N_ROUTER
            e_r = t * (WAVES // 2) + wave // 2
            kh = wave % 2
            bias_r = fx.Float32(
                bo.buffer_load(rsrc(bias), e_r, vec_width=1, dtype=T.f32)
            )
            gw = [
                fx.Vector(
                    bo.buffer_load(
                        r_gate,
                        (e_r * HIDDEN + kh * GATE_HALF + (i * WAVE + lane) * 8) // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for i in range(GATE_ROUNDS)
            ]
            gp = ld_bf16x12(r_g, rw0)
            # x8_in: h_in is the previous launch's
            if const_expr(fuse_k1 and not x8_in):
                if wave == 0:
                    k1_poll([(rdone_mb, k, 1)])
                gpu.barrier()
                acquire()
            hres = ld_bf16x12(rsrc(h_in), k * (HIDDEN // 2) + rw0, CM_K1)
            a_src = preg(sym, SY["a_ag"], "a_ag") if AG_RS else mr("a")
            arow = mb_poll(
                [
                    (a_src, (k * HIDDEN + tid * RE) // 2 + 2 * h, 2)
                    for h in range(RE // 4)
                ]
            )
            stamp(22)
            v = [
                f + fx.Float32(hres[4 * h + 2 * qq + i])
                for h in range(RE // 4)
                for qq in range(2)
                for i, f in enumerate(bf2_f32(arow[h][qq]))
            ]
            if const_expr(AG_RS):  # noqa: SIM102
                if tid // (XN_SLICE // RE) == t:
                    hw = (k * HIDDEN + tid * RE) // 2
                    hi = (
                        fx.Vector.from_elements(v, fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32)
                    )
                    bo.buffer_store(
                        fx.Vector.from_elements([hi[e] for e in range(4)], fx.Int32),
                        rsrc(h_mid),
                        hw,
                    )
                    bo.buffer_store(
                        fx.Vector.from_elements([hi[4], hi[5]], fx.Int32),
                        rsrc(h_mid),
                        hw + 4,
                    )
            ssw = fx.Float32(0.0)
            for e in range_constexpr(RE):
                ssw = fmath.fma(v[e], v[e], ssw)
            ssw = wave_sum(ssw)
            if lane == 0:
                fx.ptr_store(ssw, red + wave)
            gpu.barrier()
            ss = fx.ptr_load(red)
            for w in range_constexpr(1, WAVES):
                ss = ss + fx.ptr_load(red + w)
            stamp(24)
            rstd = hw_rsq(ss / float(HIDDEN) + eps)
            for j in range_constexpr(RE // 2):
                fx.ptr_store(
                    bf16_pair(
                        v[2 * j] * rstd * (fx.Float32(gp[2 * j]) + 1.0),
                        v[2 * j + 1] * rstd * (fx.Float32(gp[2 * j + 1]) + 1.0),
                    ),
                    xs + (rw0 + j),
                )
            gpu.barrier()
            stamp(25)
            if tid < XN_SLICE // 4:  # 4 elements a lane, 8 lanes a 32-block
                w0 = t * (XN_SLICE // 2) + tid * 2
                f = [
                    vv
                    for j in range(2)
                    for vv in bf2_f32(fx.ptr_load(xs + (w0 + j)).bitcast(fx.Int32))
                ]
                e, inv = mx8_scale(f, (1, 2, 4))
                word = fp8_pack4(*[mx8_q(vv, inv) for vv in f])
                xw = k * XN8_WORDS + t * (XN_SLICE // 4) + tid
                xsw = k * XSC_WORDS + t * (XN_SLICE // 32) + tid // 8
                if const_expr(WIDE):
                    bo.buffer_store(word, rsrc(mb("xn8p")), xw, cache_modifier=CM_DEV)
                    if tid % 8 == 0:
                        bo.buffer_store(e, rsrc(mb("xscp")), xsw, cache_modifier=CM_DEV)
                else:
                    mb_put_words(mr("xn8"), xw, [word])
                    if tid % 8 == 0:
                        mb_put_words(mr("xsc"), xsw, [e])
            stamp(26)
            # gfx1250 has no v_dot2_f32_bf16: the pairs' halves as two FMA chains
            accs = [fx.Float32(0.0) for _ in range(4)]
            for i in range_constexpr(GATE_ROUNDS):
                xv = fx.Vector(
                    fx.ptr_load(
                        xs + (kh * GATE_HALF // 2 + (i * WAVE + lane) * 4),
                        result_type=v4i,
                    )
                )
                for j in range_constexpr(4):
                    g_lo, g_hi = bf2_f32(gw[i][j])
                    x_lo, x_hi = bf2_f32(xv[j])
                    accs[j] = fmath.fma(g_hi, x_hi, fmath.fma(g_lo, x_lo, accs[j]))
            part = wave_sum((accs[0] + accs[1]) + (accs[2] + accs[3]))
            if lane == 0:
                fx.ptr_store(part, red + (WAVES + wave))
            gpu.barrier()
            stamp(23)
            if kh == 0:
                logit = fx.ptr_load(red + (WAVES + wave)) + fx.ptr_load(
                    red + (WAVES + wave + 1)
                )
                sc = hw_rcp(1.0 + hw_exp2(-bf16_round(logit) * LOG2E))
                rk = route_key(sc + bias_r, e_r)
                if lane < SCORE_COPIES:
                    mb_put_words(
                        scores_copy(lane, k), e_r * 2, [rk, sc.bitcast(fx.Int32)]
                    )
            if const_expr(WIDE):
                publish(mb_put, mr("xndone"), tt, fx.Int32(1), tid == 0)

    return {
        "route_top4": route_top4,
        "poll_scores": poll_scores,
        "mx8_scale": mx8_scale,
        "mx8_q": mx8_q,
        "lds_b8": lds_b8,
        "lds_i32": lds_i32,
        "lds_scale4": lds_scale4,
        "stage_router": stage_router,
    }
