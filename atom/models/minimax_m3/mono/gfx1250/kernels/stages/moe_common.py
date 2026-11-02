# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K4 stages: the narrow MoE's up / gate / down task operands and GEMVs
(``kernels.stages.moe_common`` on the gfx1250 expert layout, see the package).

w13 rows are GUGU: a task's 16 mid rows 16 cg .. are interleaved rows 32 cg ..,
row groups 2 cg and 2 cg + 1, each 8 (gate, up) pairs, so swiglu meets its
operands in one accumulator tile. The shared expert's half task j is row group
j. Every GEMV is the scaled 16x16x128 FP4 x FP8 WMMA; activation scales are staged
in LDS a code a word, as on gfx950, and read four at a time (``lds_scale4``).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from flydsl.expr import const_expr, range_constexpr

from atom.models.minimax_m3.mono.config import (
    HIDDEN,
    INTER,
    MOE_SLOTS,
    SHARED_EXPERT,
    TOP_K,
)
from atom.models.minimax_m3.mono.gfx1250.layout import (
    SH_ROWS,
    UG_PER_SLOT,
    UG_ROWS,
    XN8_ROW,
    XSC_ROW,
)
from atom.models.minimax_m3.mono.kernels.stages.shared import (
    S2_BYTES,
    S13_BYTES,
    W2_BYTES,
    W13_BYTES,
    W13_ROWS,
)
from atom.mono.device.gfx1250 import (
    CM_NT,
    WAVE,
    acc_row_sum,
    lds_barrier,
    n32k4_word,
    shuffled_rows,
    wmma_a8w4,
)
from atom.mono.device.ops import bf16_round, rsrc, traced, uniform

DN_GW = 8  # waves a down row group (two groups a task)


@traced
def moe_common_defs(k4ctx):
    """``kernels.stages.moe_common.moe_common_defs`` on gfx1250."""
    CPT = k4ctx["CPT"]
    DN_KC = k4ctx["DN_KC"]
    DN_UNITS = k4ctx["DN_UNITS"]
    DN_UPW = k4ctx["DN_UPW"]
    G = k4ctx["G"]
    ROUTED_TASKS = k4ctx["ROUTED_TASKS"]
    SH_UPW = k4ctx["SH_UPW"]
    TPT = k4ctx["TPT"]
    UG_CPW = k4ctx["UG_CPW"]
    XSC_LDS = k4ctx["XSC_LDS"]
    bid = k4ctx["bid"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_scale4 = k4ctx["lds_scale4"]
    mb_put = k4ctx["mb_put"]
    mr = k4ctx["mr"]
    opart = k4ctx["opart"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    rwt = k4ctx["rwt"]
    s13 = k4ctx["s13"]
    s2 = k4ctx["s2"]
    shared_weight = k4ctx["shared_weight"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    swiglu_limit = k4ctx["swiglu_limit"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    w13 = k4ctx["w13"]
    w2 = k4ctx["w2"]
    wave = k4ctx["wave"]

    def ug_task_id(it):
        if const_expr(CPT is not None):
            u = bid % CPT + it * CPT
            return (bid // CPT) * TPT + fx.min(u, TPT - 1), u < TPT
        t = start("ug") + it * G
        return fx.min(t, ROUTED_TASKS - 1), t < ROUTED_TASKS

    def ug_task(t):
        shared = t >= ROUTED_TASKS
        r = fx.min(t, ROUTED_TASKS - 1)
        tk = r // (TOP_K * UG_PER_SLOT)
        slot = shared.select(
            fx.Int32(TOP_K), (r % (TOP_K * UG_PER_SLOT)) // UG_PER_SLOT
        )
        return shared, tk, slot, t % UG_PER_SLOT

    def ug_load(t):
        """Up / gate weights and scales of task t: [row group 2 cg + h][chunk]."""
        shared, tk, slot, cg = ug_task(t)
        e_u = shared.select(
            fx.Int32(SHARED_EXPERT),
            fx.max(
                fx.min(uniform(fx.ptr_load(route + (8 * tk + slot))), SHARED_EXPERT), 0
            ),
        )
        r_w = rsrc(w13 + fx.Int64(e_u) * W13_BYTES)
        r_s = rsrc(s13)
        wts, scs = [], []
        for h in range_constexpr(2):
            rg = 2 * cg + h
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[h].append(shuffled_rows(r_w, rg, HIDDEN // 2, kc, lane, 64, CM_NT))
                scs[h].append(
                    n32k4_word(r_s, e_u * W13_ROWS + 16 * rg + l16, kc, HIDDEN)
                )
        return wts, scs

    def ug_compute(t, wts, scs):
        """Up / gate GEMV of task t -> swiglu -> its 16 mid values."""
        shared, tk, slot, cg = ug_task(t)
        col = shared.select(fx.min(l16, tokens - 1), tk)
        acc = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(2)]
        sbs = [
            lds_scale4(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4)
            for i in range(UG_CPW)
        ]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            xb = lds_b8(col * XN8_ROW + kc * 32)
            for h in range_constexpr(2):
                acc[h] = wmma_a8w4(wts[h][i], xb, acc[h], scs[h][i], sbs[i])
        stamp(9)
        fx.ptr_store(acc[0], red + (wave * WAVE + lane) * 8)
        fx.ptr_store(acc[1], opart + (wave * WAVE + lane) * 8)
        lds_barrier()
        if tid < shared.select(fx.Int32(UG_ROWS * tokens), fx.Int32(UG_ROWS)):
            m = tid % UG_ROWS
            c = tid // UG_ROWS
            kk = shared.select(c, tk)
            # mid m is pair m % 8 of tile m // 8 (gate row 2 p, up row 2 p + 1)
            p = m % 8
            first = m < 8
            g = first.select(acc_row_sum(red, 2 * p, c), acc_row_sum(opart, 2 * p, c))
            u = first.select(
                acc_row_sum(red, 2 * p + 1, c), acc_row_sum(opart, 2 * p + 1, c)
            )
            y = swiglu_mul_batch([g], [u], fx.Float32(-swiglu_limit))[0]
            mb_put(
                mr("mid"),
                (kk * MOE_SLOTS + slot) * INTER + cg * UG_ROWS + m,
                bf16_round(y),
            )
        lds_barrier()

    def sh_load(j, has=None):
        """The shared expert's half task j: interleaved row group j (8 pairs)."""
        if const_expr(has is None):
            r_w = rsrc(w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES)
            r_s = rsrc(s13)
        else:
            r_w = rsrc(
                w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES,
                has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
            )
            r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        wts, scs = [], []
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            wts.append(shuffled_rows(r_w, j, HIDDEN // 2, kc, lane, 64, CM_NT))
            scs.append(
                n32k4_word(r_s, SHARED_EXPERT * W13_ROWS + 16 * j + l16, kc, HIDDEN)
            )
        return wts, scs

    def sh_compute(j, wts, scs):
        col = fx.min(l16, tokens - 1)
        c = fx.Vector.filled(8, 0.0, fx.Float32)
        sbs = [
            lds_scale4(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4)
            for i in range(UG_CPW)
        ]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            c = wmma_a8w4(wts[i], lds_b8(col * XN8_ROW + kc * 32), c, scs[i], sbs[i])
        fx.ptr_store(c, red + (wave * WAVE + lane) * 8)
        lds_barrier()
        if tid < SH_ROWS * tokens:
            r = tid % SH_ROWS
            cl = tid // SH_ROWS
            g = acc_row_sum(red, 2 * r, cl)
            u = acc_row_sum(red, 2 * r + 1, cl)
            y = swiglu_mul_batch([g], [u], fx.Float32(-swiglu_limit))[0]
            mb_put(
                mr("mid"),
                (cl * MOE_SLOTS + TOP_K) * INTER + j * SH_ROWS + r,
                bf16_round(y),
            )
        lds_barrier()

    def down_load(k, td, has_dn):
        """Token k's route-weighted down units of row group td * 2 + wave // 8."""
        dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
        dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
        rg_d = td * 2 + wave // DN_GW
        units = []
        for i in range_constexpr(DN_UPW):
            j = (wave % DN_GW) + DN_GW * i
            jj = fx.min(j, DN_UNITS - 1)
            sl = jj // DN_KC
            kc = jj % DN_KC
            e_d = fx.max(
                fx.min(uniform(fx.ptr_load(route + (8 * k + sl))), SHARED_EXPERT), 0
            )
            wv = shuffled_rows(
                rsrc(w2 + fx.Int64(e_d) * W2_BYTES, dn_bytes),
                rg_d,
                INTER // 2,
                kc,
                lane,
                64,
                CM_NT,
            )
            sc = n32k4_word(
                rsrc(s2, dn_scale_bytes), e_d * HIDDEN + 16 * rg_d + l16, kc, INTER
            )
            wgt = (j < DN_UNITS).select(
                fx.ptr_load(rwt + (8 * k + sl)), fx.Float32(0.0)
            )
            units.append((wv, sc, sl, kc, wgt))
        return units

    def down_load_shared(td, has_dn):
        dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
        dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
        rg_d = td * 2 + wave // DN_GW
        units = []
        for i in range_constexpr(SH_UPW):
            j = (wave % DN_GW) + DN_GW * i
            kc = fx.min(j, DN_KC - 1)
            wv = shuffled_rows(
                rsrc(w2 + fx.Int64(SHARED_EXPERT) * W2_BYTES, dn_bytes),
                rg_d,
                INTER // 2,
                kc,
                lane,
                64,
                CM_NT,
            )
            sc = n32k4_word(
                rsrc(s2, dn_scale_bytes),
                SHARED_EXPERT * HIDDEN + 16 * rg_d + l16,
                kc,
                INTER,
            )
            wgt = (j < DN_KC).select(fx.Float32(shared_weight), fx.Float32(0.0))
            units.append((wv, sc, kc, wgt))
        return units

    return {
        "ug_task_id": ug_task_id,
        "ug_task": ug_task,
        "ug_load": ug_load,
        "ug_compute": ug_compute,
        "sh_load": sh_load,
        "sh_compute": sh_compute,
        "down_load": down_load,
        "down_load_shared": down_load_shared,
    }
