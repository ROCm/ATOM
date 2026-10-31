# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K4 stages: the per-(token, slot) MoE, S <= 4
(``kernels.stages.moe_narrow`` on wave32 and the a8w4 WMMA)."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from atom.models.minimax_m3.mono.config import INTER, MOE_SLOTS, TOP_K
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.moe_common import DN_GW
from atom.models.minimax_m3.mono.gfx1250.kernels.stages.router import XN_SLICE
from atom.models.minimax_m3.mono.gfx1250.layout import (
    ACC_WORDS,
    DN_ROWS,
    N_DN,
    N_ROUTER,
    SCORE_COPIES,
    SH_TASKS,
    XN8_ROW,
    XSC_ROW,
    shared_after_routing,
)
from atom.mono.device.gfx1250 import THREADS, WAVE, acc_row_sum, wmma_a8w4
from atom.mono.device.ops import bf2_f32, bf16_pair, bf16_round, fp8_pack4, traced
from atom.mono.plan.trace import enter_stage


@traced
def moe_narrow_defs(k4ctx):
    """``kernels.stages.moe_narrow.moe_narrow_defs`` on gfx1250."""
    BASE = k4ctx["BASE"]
    CPT = k4ctx["CPT"]
    MID16_WORDS = k4ctx["MID16_WORDS"]
    MID8_LDS = k4ctx["MID8_LDS"]
    MID8_WORDS = k4ctx["MID8_WORDS"]
    MIDSC_LDS = k4ctx["MIDSC_LDS"]
    MIDSC_WORDS = k4ctx["MIDSC_WORDS"]
    UG_ITERS = k4ctx["UG_ITERS"]
    XN8_WORDS = k4ctx["XN8_WORDS"]
    XSC_LDS = k4ctx["XSC_LDS"]
    XSC_WORDS = k4ctx["XSC_WORDS"]
    bid = k4ctx["bid"]
    down_load = k4ctx["down_load"]
    down_load_shared = k4ctx["down_load_shared"]
    ffn_finish = k4ctx["ffn_finish"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_scale4 = k4ctx["lds_scale4"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    misc = k4ctx["misc"]
    mr = k4ctx["mr"]
    mx8_q = k4ctx["mx8_q"]
    mx8_scale = k4ctx["mx8_scale"]
    opart = k4ctx["opart"]
    poll_scores = k4ctx["poll_scores"]
    route = k4ctx["route"]
    route_top4 = k4ctx["route_top4"]
    rwt = k4ctx["rwt"]
    sh_compute = k4ctx["sh_compute"]
    sh_load = k4ctx["sh_load"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    ug_compute = k4ctx["ug_compute"]
    ug_load = k4ctx["ug_load"]
    ug_task_id = k4ctx["ug_task_id"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]
    v4i = fx.Vector.make_type(4, fx.Int32)

    def stage_moe():
        td = bid - BASE["down"]
        has_dn = (td >= 0) & (td < N_DN)
        td = fx.min(fx.max(td, 0), N_DN - 1)
        has_sh = start("shared") < SH_TASKS
        n_xw = XN8_WORDS // THREADS

        def xn_specs(k):
            return [
                (mr("xn8"), k * XN8_WORDS + tid + i * THREADS, 1) for i in range(n_xw)
            ] + [(mr("xsc"), k * XSC_WORDS + fx.min(tid, XSC_WORDS - 1), 1)]

        def stage_xn(k, got):
            for i in range_constexpr(n_xw):
                fx.ptr_store(got[i][0], xs + (k * XN8_ROW + tid + i * THREADS))
            if tid < XSC_WORDS:
                fx.ptr_store(got[n_xw][0], xs + (XSC_LDS + k * XSC_ROW + tid))

        def fetch_xn(tok=None):
            # every router task's slice, lane + WAVE c: (token, slice) of N_ROUTER
            if wave == 0:
                specs = []
                for c in range_constexpr((N_ROUTER * tokens + WAVE - 1) // WAVE):
                    ix = lane + WAVE * c
                    gate_tok = (
                        fx.min(ix // N_ROUTER, tokens - 1) if tok is None else tok
                    )
                    specs.append(
                        (
                            mr("xn8"),
                            gate_tok * XN8_WORDS
                            + (ix % N_ROUTER + 1) * (XN_SLICE // 4)
                            - 1,
                            1,
                        )
                    )
                mb_poll(specs)
            gpu.barrier()
            if tok is None:
                got = mb_poll([sp for k in range(tokens) for sp in xn_specs(k)])
                for k in range_constexpr(tokens):
                    stage_xn(k, got[k * (n_xw + 1) : (k + 1) * (n_xw + 1)])
            else:
                stage_xn(tok, mb_poll(xn_specs(tok)))

        sh_late = shared_after_routing(tokens)
        if const_expr(sh_late):
            sh_j = fx.min(start("shared"), SH_TASKS - 1)
            sh_w = sh_load(sh_j, has_sh)
            if has_sh:
                fetch_xn()
        elif has_sh:
            sh_j0 = start("shared")
            sh_w0 = sh_load(sh_j0)
            fetch_xn()
            gpu.barrier()
            sh_compute(sh_j0, sh_w0[0], sh_w0[1])
        if ug_task_id(0)[1] | has_dn:
            if start("shared") >= SH_TASKS:
                fetch_xn(bid // CPT if CPT is not None else None)
            mine = bid % SCORE_COPIES
            for k in range_constexpr(tokens):
                if wave == k:
                    keys, sigs = poll_scores(k, mine)
                    stamp(21)
                    route_top4(k, keys, sigs, opart)
            stamp(2)
            gpu.barrier()
            stamp(6)
            if (bid == BASE["down"]) & (tid < MOE_SLOTS * tokens):
                sk = tid // MOE_SLOTS
                sj = tid % MOE_SLOTS
                mb_put(
                    mr("sel"),
                    sk * 2 * MOE_SLOTS + sj,
                    fx.ptr_load(route + (8 * sk + sj)),
                )
                mb_put(
                    mr("sel"),
                    sk * 2 * MOE_SLOTS + MOE_SLOTS + sj,
                    fx.ptr_load(rwt + (8 * sk + sj)),
                )
            cur = ug_load(ug_task_id(0)[0])
            stamp(7)
            if const_expr(sh_late):  # noqa: SIM102
                if has_sh:
                    sh_compute(sh_j, sh_w[0], sh_w[1])
            for it in range_constexpr(UG_ITERS):
                t, has_t = ug_task_id(it)
                if const_expr(it + 1 < UG_ITERS):
                    nxt = ug_load(ug_task_id(it + 1)[0])
                if has_t:
                    ug_compute(t, cur[0], cur[1])
                if const_expr(it == 0):
                    dn_units = {0: down_load(0, td, has_dn)}
                    sh_units = down_load_shared(td, has_dn)
                if const_expr(tokens > 1 and it == UG_ITERS - 1):
                    dn_units[1] = down_load(1, td, has_dn)
                if const_expr(it + 1 < UG_ITERS):
                    cur = nxt
            stamp(8)
            if has_dn:
                moe_down(td, has_dn, dn_units, sh_units)
                stamp(11)

    def moe_down(t, has_dn, dn_units, sh_units):
        """Every token's route-weighted down GEMV of row groups 2 t, 2 t + 1 ->
        ffn_finish."""
        enter_stage("down")

        def stage_mids(k):
            mids = mb_poll(
                [
                    (
                        mr("mid"),
                        k * MOE_SLOTS * INTER
                        + fx.min(tid * 2 + i * 2 * THREADS, MOE_SLOTS * INTER - 2),
                        2,
                    )
                    for i in range(4)
                ]
            )
            for i in range_constexpr(4):
                e = tid * 2 + i * 2 * THREADS
                if e < MOE_SLOTS * INTER:
                    fx.ptr_store(
                        bf16_pair(
                            mids[i][0].bitcast(fx.Float32),
                            mids[i][1].bitcast(fx.Float32),
                        ),
                        xs + (k * (MOE_SLOTS * INTER // 2) + e // 2),
                    )

        for k in range_constexpr(tokens):
            stage_mids(k)
        stamp(10)
        gpu.barrier()
        for k in range_constexpr(tokens):
            if tid < MID8_WORDS // 2:  # 8 elements a lane, 4 lanes a 32-block
                w = fx.Vector(
                    fx.ptr_load(xs + (k * MID16_WORDS + tid * 4), result_type=v4i)
                )
                f = [v for j in range(4) for v in bf2_f32(w[j])]
                sc, inv = mx8_scale(f, (1, 2))
                qv = [mx8_q(v, inv) for v in f]
                fx.ptr_store(
                    fp8_pack4(*qv[:4]), xs + (MID8_LDS + k * MID8_WORDS + tid * 2)
                )
                fx.ptr_store(
                    fp8_pack4(*qv[4:]), xs + (MID8_LDS + k * MID8_WORDS + tid * 2 + 1)
                )
                if tid % 4 == 0:
                    fx.ptr_store(sc, xs + (MIDSC_LDS + k * MIDSC_WORDS + tid // 4))
        gpu.barrier()
        # the shared expert once, B column l % 16 = token l % 16 -> opart[0]
        col_tok = fx.min(l16, tokens - 1)
        sh = fx.Vector.filled(8, 0.0, fx.Float32)
        for wv, sc, kc, wgt in sh_units:
            off = TOP_K * INTER + kc * 128
            cu = wmma_a8w4(
                wv,
                lds_b8(MID8_LDS + col_tok * MID8_WORDS + off // 4),
                fx.Vector.filled(8, 0.0, fx.Float32),
                sc,
                lds_scale4(MIDSC_LDS + col_tok * MIDSC_WORDS + off // 32),
            )
            sh = fx.Vector.from_elements(
                [sh[j] + cu[j] * wgt for j in range(8)], fx.Float32
            )
        fx.ptr_store(sh, opart + (wave * WAVE + lane) * 8)
        for k in range_constexpr(tokens):
            if const_expr(k + 2 < tokens):
                dn_units[k + 2] = down_load(k + 2, t, has_dn)
            acc = [fx.Float32(0.0) for _ in range(8)]
            for wv, sc, sl, kc, wgt in dn_units.pop(k):
                off = sl * INTER + kc * 128
                cu = wmma_a8w4(
                    wv,
                    lds_b8(MID8_LDS + k * MID8_WORDS + off // 4),
                    fx.Vector.filled(8, 0.0, fx.Float32),
                    sc,
                    lds_scale4(MIDSC_LDS + k * MIDSC_WORDS + off // 32),
                )
                acc = [acc[j] + cu[j] * wgt for j in range(8)]
            fx.ptr_store(
                fx.Vector.from_elements(acc, fx.Float32),
                opart + ((k + 1) * ACC_WORDS + (wave * WAVE + lane) * 8),
            )
        gpu.barrier()
        if tid < DN_ROWS * tokens:
            k = tid // DN_ROWS
            grp = (tid % DN_ROWS) // 16
            rr = tid % 16
            waves = [grp * DN_GW + w for w in range(DN_GW)]
            tot = acc_row_sum(opart + (k + 1) * ACC_WORDS, rr, 0, waves) + acc_row_sum(
                opart, rr, k, waves
            )
            fx.ptr_store(bf16_round(tot), misc + tid)
        gpu.barrier()
        ffn_finish(t)

    return {"stage_moe": stage_moe, "moe_down": moe_down}
