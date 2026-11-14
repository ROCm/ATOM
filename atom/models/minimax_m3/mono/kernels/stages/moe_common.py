# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the narrow MoE's up / gate / down tasks (the FFN all-reduce:
``ffn_reduce``)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.minimax_m3.mono.config import (
    HIDDEN,
    INTER,
    MOE_SLOTS,
    SHARED_EXPERT,
    TOP_K,
)
from atom.models.minimax_m3.mono.kernels.stages.shared import (
    S2_BYTES,
    S2S_BYTES,
    S13_BYTES,
    S13S_BYTES,
    W2_BYTES,
    W2_SCALE_COLS,
    W2S_BYTES,
    W13_BYTES,
    W13_ROWS,
    W13_SCALE_COLS,
    W13S_BYTES,
)
from atom.models.minimax_m3.mono.layout import (
    SH_ROWS,
    UG_PER_SLOT,
    UG_ROWS,
    XN4_ROW,
    XN8_ROW,
    XSC_ROW,
    mxfp8_tile_dword,
)
from atom.mono.device.mx import (
    FP4,
    FP8,
    fp4_tile_load,
    fp4x8_bf16,
    mfma_scaled,
    scale_index,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_pair,
    bf16_round,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from atom.mono.device.sync import publish


@traced
def moe_common_defs(k4ctx):
    """The functions of the narrow MoE's up / gate / down tasks."""
    A16N = k4ctx["A16N"]
    A4N = k4ctx["A4N"]
    X4S_LDS = k4ctx["X4S_LDS"]
    X4_LDS = k4ctx["X4_LDS"]
    CPT = k4ctx["CPT"]
    DN_KC = k4ctx["DN_KC"]
    DN_UNITS = k4ctx["DN_UNITS"]
    DN_UPW = k4ctx["DN_UPW"]
    G = k4ctx["G"]
    ROUTED_TASKS = k4ctx["ROUTED_TASKS"]
    MID_BF = k4ctx["MID_BF"]
    MID_FLAG = k4ctx["MID_FLAG"]
    SH8 = k4ctx["shared_mxfp8"]
    SH_UPW = k4ctx["SH_UPW"]
    TPT = k4ctx["TPT"]
    UG_CPW = k4ctx["UG_CPW"]
    XB_LDS = k4ctx["XB_LDS"]
    XB_ROW = k4ctx["XB_ROW"]
    XSC_LDS = k4ctx["XSC_LDS"]
    bid = k4ctx["bid"]
    e8m0_load = k4ctx["e8m0_load"]
    g4 = k4ctx["g4"]
    group_scale_byte = k4ctx["group_scale_byte"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b4 = k4ctx["lds_b4"]
    lds_b8 = k4ctx["lds_b8"]
    lds_bf8 = k4ctx["lds_bf8"]
    lds_i32 = k4ctx["lds_i32"]
    mb = k4ctx["mb"]
    mb_put = k4ctx["mb_put"]
    mb_put_bf = k4ctx["mb_put_bf"]
    mr = k4ctx["mr"]
    opart = k4ctx["opart"]
    prod_round = k4ctx["prod_round"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    rwt = k4ctx["rwt"]
    s13 = k4ctx["s13"]
    s2 = k4ctx["s2"]
    s2s = k4ctx["s2s"]
    s13s = k4ctx["s13s"]
    shared_weight = k4ctx["shared_weight"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    swiglu_limit = k4ctx["swiglu_limit"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    w13 = k4ctx["w13"]
    wrsrc = k4ctx["wrsrc"]
    w2 = k4ctx["w2"]
    w2s = k4ctx["w2s"]
    w13s = k4ctx["w13s"]
    wave = k4ctx["wave"]

    def put_mid(idx, y):
        """Mid element ``idx`` (even for a lane r % 2 == 0) := ``y``: a tagged pair, or
        (``MID_FLAG``) a bf16 pair word from every even lane (its neighbor's value
        beside its own), plain (the task's flag follows the CTA's stores)."""
        if const_expr(not (MID_FLAG or MID_BF)):
            mb_put(mr("mid"), idx, bf16_round(y))
            return
        y = bf16_round(y)
        nb = xshfl(y, 1)
        if const_expr(MID_BF):
            if idx % 2 == 0:
                mb_put_bf(mr("mid"), idx, [y, nb])
            return
        if idx % 2 == 0:
            bo.buffer_store(
                bf16_pair(y, nb).bitcast(fx.Int32),
                rsrc(mb("mid")),
                idx // 2,
                cache_modifier=CM_DEV,
            )

    def ug_task_id(it):
        """(this CTA's it-th routed task, clamped; whether it exists)."""
        if const_expr(CPT is not None):
            u = bid % CPT + it * CPT
            return (bid // CPT) * TPT + fx.min(u, TPT - 1), u < TPT
        t = start("ug") + it * G
        return fx.min(t, ROUTED_TASKS - 1), t < ROUTED_TASKS

    def ug_task(t):
        """(shared expert?, token, slot, row group) of up / gate task t."""
        shared = t >= ROUTED_TASKS
        r = fx.min(t, ROUTED_TASKS - 1)
        tk = r // (TOP_K * UG_PER_SLOT)
        slot = shared.select(
            fx.Int32(TOP_K), (r % (TOP_K * UG_PER_SLOT)) // UG_PER_SLOT
        )
        return shared, tk, slot, t % UG_PER_SLOT

    def ug_load(t):
        """Up / gate weights and scales of task t."""
        shared, tk, slot, cg = ug_task(t)
        # the bound keeps a corrupt route from addressing past the expert table
        e_u = shared.select(
            fx.Int32(SHARED_EXPERT),
            fx.min(uniform(fx.ptr_load(route + (8 * tk + slot))), SHARED_EXPERT),
        )
        r_w = wrsrc("moeu", w13 + fx.Int64(e_u) * W13_BYTES)
        r_s = wrsrc("moeu", s13)
        wts = []  # [gate, up][chunk]
        scs = []
        for gu in range_constexpr(2):
            rg = gu * UG_PER_SLOT + cg
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[gu].append(fp4_tile_load(r_w, 0, rg, HIDDEN, kc, lane * 4))
                scs[gu].append(
                    e8m0_load(
                        r_s,
                        group_scale_byte(e_u, W13_ROWS, rg, kc, W13_SCALE_COLS),
                    )
                )
        return wts, scs

    def ug_compute(t, wts, scs):
        """Up / gate GEMV of task t -> swiglu -> its 16 mid values (of every token
        for the shared expert: B column l % 16 is token l % 16)."""
        shared, tk, slot, cg = ug_task(t)
        col = shared.select(fx.min(l16, tokens - 1), tk)
        cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
        cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(A4N):
            # a4w4: the token's MXFP4 xn row and codes (the router's words, after the MXFP8 rows)
            # against the weights, one fp4 x fp4 MFMA a 128-k chunk
            xrow = X4_LDS + (0 if CPT is not None else tk * XN4_ROW)
            srow = X4S_LDS + (0 if CPT is not None else tk * XSC_ROW)
            sbs = [lds_i32(srow + (wave * UG_CPW + i) * 4 + g4) for i in range(UG_CPW)]
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                xb = lds_b4(xrow + kc * 16)
                cg_ = mfma_scaled(
                    wts[0][i], xb, cg_, scs[0][i], sbs[i], a_fmt=FP4, b_fmt=FP4
                )
                cu_ = mfma_scaled(
                    wts[1][i], xb, cu_, scs[1][i], sbs[i], a_fmt=FP4, b_fmt=FP4
                )
        elif const_expr(A16N):
            # a16w4: the token's bf16 xn row (the router's words, after the MXFP8 rows)
            # against the weights up-converted to bf16, four K = 32 MFMAs a 128-k chunk
            xrow = XB_LDS + (0 if CPT is not None else tk * XB_ROW)
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                for d in range_constexpr(4):
                    xb = lds_bf8(xrow + kc * 64 + g4 * 16 + d * 4)
                    cg_ = mfma_bf16(fp4x8_bf16(wts[0][i][d], scs[0][i]), xb, cg_)
                    cu_ = mfma_bf16(fp4x8_bf16(wts[1][i][d], scs[1][i]), xb, cu_)
        else:
            # every chunk's B scale up front: read at use, each MFMA waited out an LDS
            # latency (S = 4 up / gate 3.5 us over the unscaled floor)
            sbs = [
                lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
                for i in range(UG_CPW)
            ]
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                xb = lds_b8(col * XN8_ROW + kc * 32)
                cg_ = mfma_scaled(wts[0][i], xb, cg_, scs[0][i], sbs[i], a_fmt=FP4)
                cu_ = mfma_scaled(wts[1][i], xb, cu_, scs[1][i], sbs[i], a_fmt=FP4)
        stamp(9)
        # gate partials in red, up partials in opart (free once the split is done)
        fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
        fx.ptr_store(cu_, opart + (wave * 64 + lane) * 4)
        gpu.barrier()
        # row tid % 16; a shared task's thread tid reads token tid / 16's column
        if tid < shared.select(fx.Int32(UG_ROWS * tokens), fx.Int32(UG_ROWS)):
            r = tid % UG_ROWS
            col = tid // UG_ROWS
            kk = shared.select(col, tk)
            gsum = row_sum(red, r, col)
            usum = row_sum(opart, r, col)
            y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
            put_mid((kk * MOE_SLOTS + slot) * INTER + cg * UG_ROWS + r, y)
        if const_expr(MID_FLAG):
            publish(mb_put, mr("ugdone"), t, fx.Int32(1), tid == 0)
        gpu.barrier()

    def fp8_tile_load(r_w, row, cols, kc):
        """A lane's 32 B of an MXFP8 weight in aiter's (16, 16) preshuffle
        ([N / 16][K / 32][2][16][16] bytes: a 16-row x 32-k tile is 512 B, its two
        16-k halves apart), row ``row`` at 128-k chunk ``kc``: k 16 g .. + 16 and
        64 + 16 g .. + 16 (g = lane // 16), the f8f6f4 MFMA's operand layout (the B
        operand's, ``lds_b8``). Each 16 B piece of the 64 lanes is 1 KB contiguous."""
        dw = mxfp8_tile_dword(row, cols, kc, g4)
        halves = [
            fx.Vector(
                bo.buffer_load(
                    r_w, dw + 256 * h, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
                )
            )
            for h in range(2)
        ]
        return fx.Vector.from_elements(
            [halves[h][e] for h in range(2) for e in range(4)], fx.Int32
        )

    def fp8_scale_load(r_s, row, cols, kc):
        """The E8M0 scale dword of row ``row``'s 128-k chunk ``kc`` (the four
        32-blocks' bytes) shifted to this lane's block (g): its low byte, as the
        MFMA takes it."""
        w = fx.Int32(
            bo.buffer_load(
                r_s, row * (cols // 128) + kc, vec_width=1, dtype=T.i32,
            )
        )  # fmt: skip
        return w >> (g4 * 8)

    def sh_load(j, has=None):
        """The shared expert's half task j: gate rows 8 j .. + 8 on lanes l16 < 8,
        the up rows alike on the rest -- one MFMA tile; the waves split k."""
        row = (l16 < 8).select(SH_ROWS * j + l16, INTER + SH_ROWS * j + l16 - 8)
        if const_expr(SH8):
            sizes = (W13S_BYTES, S13S_BYTES)
            if const_expr(has is None):
                r_w, r_s = wrsrc("shu", w13s), wrsrc("shu", s13s)
            else:  # a CTA without the task reads through zero-sized buffers
                r_w = wrsrc("shu", w13s, has.select(fx.Int32(sizes[0]), fx.Int32(0)))
                r_s = wrsrc("shu", s13s, has.select(fx.Int32(sizes[1]), fx.Int32(0)))
            kcs = [wave * UG_CPW + i for i in range(UG_CPW)]
            return (
                [fp8_tile_load(r_w, row, HIDDEN, kc) for kc in kcs],
                [fp8_scale_load(r_s, row, HIDDEN, kc) for kc in kcs],
            )
        rg = row // 16
        if const_expr(has is None):
            r_w = wrsrc("moeu", w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES)
            r_s = wrsrc("moeu", s13)
        else:  # a CTA without the task reads through zero-sized buffers
            r_w = wrsrc(
                "moeu",
                w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES,
                has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
            )
            r_s = wrsrc("moeu", s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        wts = []
        scs = []
        # the row's part of scale_index, once: a chunk adds 256 (kc / 2) + 2 (kc % 2)
        sc_row = scale_index(SHARED_EXPERT * W13_ROWS + row, g4, W13_SCALE_COLS)
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            wts.append(
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        (
                            (rg * (HIDDEN // 64) + kc * 2) * 512
                            + (g4 * 16 + row % 16) * 16
                        )
                        // 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
            )
            scs.append(e8m0_load(r_s, sc_row + kc // 2 * 256 + kc % 2 * 2))
        return wts, scs

    def sh_compute(j, wts, scs):
        """Half task j's GEMV (B column l % 16 = token l % 16) -> swiglu -> the
        shared slot's mid rows 8 j .. + 8 of every token."""
        col = fx.min(l16, tokens - 1)
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        sbs = [
            lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
            for i in range(UG_CPW)
        ]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            c = mfma_scaled(
                wts[i], lds_b8(col * XN8_ROW + kc * 32), c, scs[i], sbs[i], a_fmt=SH_FMT
            )
        fx.ptr_store(c, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        # mid row r = tid % 8 of token tid / 8: gate in tile row r, up in r + 8
        if tid < SH_ROWS * tokens:
            r = tid % SH_ROWS
            col = tid // SH_ROWS
            gsum = row_sum(red, r, col)
            usum = row_sum(red, r + 8, col)
            if const_expr(prod_round):
                # production's gate_up_proj output is bf16, the swiglu reads that
                gsum = bf16_round(gsum)
                usum = bf16_round(usum)
            y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
            put_mid((col * MOE_SLOTS + TOP_K) * INTER + j * SH_ROWS + r, y)
        if const_expr(MID_FLAG):
            publish(mb_put, mr("ugdone"), ROUTED_TASKS + j, fx.Int32(1), tid == 0)
        gpu.barrier()

    def down_load(k, td, has_dn):
        """Token k's route-weighted down units of row group td; a CTA without a
        down task reads through zero-sized buffers."""
        dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
        dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
        rg_d = td * 2 + wave // 4
        units = []
        for i in range_constexpr(DN_UPW):
            if const_expr(A16N or A4N):
                # a wave of the row group takes one slot, all its chunks: the slot's sum
                # is complete in the wave, where production rounds it (bf16(acc * w))
                j = (wave % 4) * DN_KC + i
                jj = j
                sl = wave % 4
                kc = i
            else:
                j = (wave % 4) + 4 * i
                jj = fx.min(j, DN_UNITS - 1)
                sl = jj // DN_KC
                kc = jj % DN_KC
            e_d = fx.min(uniform(fx.ptr_load(route + (8 * k + sl))), SHARED_EXPERT)
            wv = fp4_tile_load(
                wrsrc("moed", w2 + fx.Int64(e_d) * W2_BYTES, dn_bytes),
                0,
                rg_d,
                INTER,
                kc,
                lane * 4,
            )
            sc = e8m0_load(
                wrsrc("moed", s2, dn_scale_bytes),
                group_scale_byte(e_d, HIDDEN, rg_d, kc, W2_SCALE_COLS),
            )
            wgt = (j < DN_UNITS).select(
                fx.ptr_load(rwt + (8 * k + sl)), fx.Float32(0.0)
            )
            units.append((wv, sc, sl, kc, wgt))
        return units

    def down_load_shared(td, has_dn):
        """The shared expert's down units of row group td (every token's)."""
        rg_d = td * 2 + wave // 4
        units = []
        if const_expr(SH8):
            r_w = wrsrc("shd", w2s, has_dn.select(fx.Int32(W2S_BYTES), fx.Int32(0)))
            r_s = wrsrc("shd", s2s, has_dn.select(fx.Int32(S2S_BYTES), fx.Int32(0)))
            for i in range_constexpr(SH_UPW):
                j = (wave % 4) + 4 * i
                kc = fx.min(j, DN_KC - 1)
                wgt = (j < DN_KC).select(fx.Float32(shared_weight), fx.Float32(0.0))
                row = rg_d * 16 + l16
                units.append(
                    (
                        fp8_tile_load(r_w, row, INTER, kc),
                        fp8_scale_load(r_s, row, INTER, kc),
                        kc,
                        wgt,
                    )
                )
            return units
        dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
        dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
        for i in range_constexpr(SH_UPW):
            j = (wave % 4) + 4 * i
            kc = fx.min(j, DN_KC - 1)
            wv = fp4_tile_load(
                wrsrc("moed", w2 + fx.Int64(SHARED_EXPERT) * W2_BYTES, dn_bytes),
                0,
                rg_d,
                INTER,
                kc,
                lane * 4,
            )
            sc = e8m0_load(
                wrsrc("moed", s2, dn_scale_bytes),
                group_scale_byte(SHARED_EXPERT, HIDDEN, rg_d, kc, W2_SCALE_COLS),
            )
            wgt = (j < DN_KC).select(fx.Float32(shared_weight), fx.Float32(0.0))
            units.append((wv, sc, kc, wgt))
        return units

    SH_FMT = FP8 if SH8 else FP4

    return {
        "SH_FMT": SH_FMT,
        "fp8_tile_load": fp8_tile_load,
        "fp8_scale_load": fp8_scale_load,
        "ug_task_id": ug_task_id,
        "ug_task": ug_task,
        "ug_load": ug_load,
        "ug_compute": ug_compute,
        "sh_load": sh_load,
        "sh_compute": sh_compute,
        "down_load": down_load,
        "down_load_shared": down_load_shared,
    }
