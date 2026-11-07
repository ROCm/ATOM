# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""q7 target UG: existing packed operands, expert-union columns.

The original router remains the sole FP8/scale producer. This candidate removes
UG operand decoding, selects each row's routes once, and computes one tile per
distinct expert. Shared UG and BF16 Down keep their arithmetic boundaries.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from atom.models.deepseek_v4.mono.kernels.helpers import traced
from atom.models.deepseek_v4.mono.kernels.primitives import (
    lds_ld,
    lds_st,
    rsrc,
    uniform,
)
from atom.mono.device.mx import FP4, FP8, mfma_scaled
from atom.mono.device.ops import ballot, lanes_below, popcount


@traced
def publish_routes(c):
    """One CTA per real row selects routes; all consumers use the same table."""
    S, tid, bid, lane = c["S"], c["tid"], c["bid"], c["lane"]
    mb, put, route_topk = c["mb"], c["put"], c["route_topk"]
    slots, topk = c["MOE_SLOTS"], c["TOP_K"]
    assert S == 7 and topk == 6
    if (bid < S) & (tid < 64):
        expert, weight = route_topk(bid)
        if lane < slots:
            slot = (lane + 1) % slots
            put(
                mb("sel"),
                bid * slots + slot,
                (lane == topk).select(fx.Int32(c["SHARED_EXPERT"]), expert),
            )
            put(
                mb("prob"),
                bid * slots + slot,
                (lane == topk).select(fx.Float32(1.0), weight),
            )
    if tid < S * slots:
        expert = c["get"](mb("sel"), tid)
        weight = c["get"](mb("prob"), tid).bitcast(fx.Float32)
        lds_st(c["keys"], tid, expert)
        lds_st(c["dnw"], tid, weight)
    gpu.barrier()


@traced
def compact_experts(c):
    """One wave builds a compact local union once, before any expert tile.

    The compact table occupies the unused tail after packed FP8 rows/scales.
    Shared UG is complete; routed input/consumer never touch this tail. Down
    reuses the arena only after the table's final consumer. No new LDS allocation.
    """
    lane, tid, keys = c["lane"], c["tid"], c["keys"]
    end = c["S"] * (c["HIDDEN"] // 4 + c["HIDDEN"] // 32)
    assert end + c["S"] * c["TOP_K"] + 1 <= c["XN"]
    table = c["xs"] + end
    slots, topk = c["MOE_SLOTS"], c["TOP_K"]
    if tid < 64:
        safe = fx.min(lane, fx.Int32(c["S"] * topk - 1))
        expert = lds_ld(keys, safe // topk * slots + safe % topk + 1)
        first = fx.Int32(1)
        for previous in range_constexpr(c["S"] * topk):
            same = (
                lds_ld(keys, previous // topk * slots + previous % topk + 1) == expert
            )
            first = ((previous < lane) & same).select(fx.Int32(0), first)
        live = (lane < c["S"] * topk) & (first != 0)
        mask = ballot(live)
        order = lanes_below(mask, lane)
        if live:
            lds_st(table, order, expert.bitcast(fx.Float32))
        if lane == 0:
            lds_st(table, c["S"] * topk, popcount(mask).bitcast(fx.Float32))
    gpu.barrier()


@traced
def stage_packed_input(c):
    """Read each existing FP8 word/scale once into this CTA's reused LDS arena.

    No FP8 decoding or requantization. The arena was sized for seven BF16 rows;
    seven FP8 rows and their E8M0 words occupy less than that existing allocation.
    """
    S, hidden, tid, xs = c["S"], c["HIDDEN"], c["tid"], c["xs"]
    mb, poll = c["mb"], c["poll"]
    words, scales = hidden // 4, hidden // 32
    pairs = -(-S * words // (2 * 512))
    specs = [
        (mb("xq"), fx.min(2 * tid + i * 1024, S * words - 2), 2) for i in range(pairs)
    ]
    values = poll(specs)
    for i in range_constexpr(pairs):
        at = 2 * tid + i * 1024
        if at < S * words:
            fx.ptr_store(
                fx.Vector.from_elements(values[i], fx.Int32).bitcast(fx.Float32),
                xs + at,
            )
    scale_items = -(-S * scales // 512)
    got_scales = poll(
        [
            (mb("xqs"), fx.min(tid + i * 512, S * scales - 1), 1)
            for i in range(scale_items)
        ]
    )
    for i in range_constexpr(scale_items):
        at = tid + i * 512
        if at < S * scales:
            code = got_scales[i][0].shrui(fx.Int32(23)) & 0xFF
            lds_st(xs, S * words + at, code.bitcast(fx.Float32))
    gpu.barrier()


def scaled_units(c, expert, channel):
    """An existing ATOM FP4 bank's eight-wave K split, with q7 FP8 columns."""
    hidden, inter = c["HIDDEN"], c["INTER"]
    wave, lane, xs = c["wave"], c["lane"], c["xs"]
    rw = rsrc(c["w_ug"] + fx.Int64(expert) * (2 * inter * hidden // 2))
    scale_bytes = 2 * inter * (-(-(hidden // 32) // 8) * 8)
    rs = rsrc(c["s_ug"] + fx.Int64(expert) * scale_bytes)
    group = channel // 2 * 2 + (lane % 16) // 8
    weight_lane = (lane & -16) | ((channel % 2) * 8 + lane % 8)
    column = fx.min(lane % 16, fx.Int32(c["S"] - 1))
    per_wave = hidden // 128 // 8
    units = []
    for i in range(per_wave):
        k = wave * per_wave + i
        # Reuse ATOM's exact FP4 weight and scale address calculation.
        unit = c["unit_mxfp4"](rw, rs, group, k, hidden, 0, None, weight_lane)
        raw, scale = unit[1]
        # Scaled MFMA's FP8 operand is two K16 halves, not one contiguous
        # K32 vector: 16*j..16*j+15 and 64+16*j..79+16*j.
        halves = [
            fx.Vector(
                fx.ptr_load(
                    xs + column * (hidden // 4) + k * 32 + 16 * half + (lane // 16) * 4,
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
            ).bitcast(fx.Int32)
            for half in range(2)
        ]
        b = fx.Vector.from_elements([v[i] for v in halves for i in range(4)], fx.Int32)
        sb = lds_ld(
            xs, c["S"] * (hidden // 4) + column * (hidden // 32) + k * 4 + lane // 16
        ).bitcast(fx.Int32)
        units.append((raw, scale, b, sb))
    return units


def scaled_dot(c, units):
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for a, sa, b, sb in units:
        acc = mfma_scaled(a, b, acc, sa, sb, a_fmt=FP4, b_fmt=FP8)
    return [acc[i] for i in range(4)]


@traced
def routed_union(c):
    """Disjoint shared and routed producers remove their serial critical chain.

    Canonical ownership is determined from the real row/slot table. Inactive
    expert/token columns neither write MID nor publish a real slot.
    """
    assert c["S"] == 7 and c["HIDDEN"] % (128 * 8) == 0
    assert c["use_mxfp8_block32"] and c["use_mxfp4_weight"]
    S, slots, topk, tid = c["S"], c["MOE_SLOTS"], c["TOP_K"], c["tid"]
    keys, outs, xs = c["keys"], c["outs"], c["xs"]
    # Shared producers finish without acquiring routed work. Other resident
    # CTAs consume the complete expert pool with a disjoint 208-CTA stride.
    if fx.Int32(c["start"]("ug")) >= c["INTER"] // 8:
        compact_experts(c)
        stage_packed_input(c)
        tiles = c["INTER"] // 8
        table = xs + S * (c["HIDDEN"] // 4 + c["HIDDEN"] // 32)
        count = uniform(lds_ld(table, S * topk).bitcast(fx.Int32))
        total = count * tiles
        workers = c["G"] - tiles
        first_task = fx.Int32(c["start"]("ug")) - tiles
        for task in range(first_task, total, workers):
            task = fx.Int32(task)
            channel = task % tiles
            expert_slot = fx.min(task // tiles, count - 1)
            expert = uniform(lds_ld(table, expert_slot).bitcast(fx.Int32))
            live = task < total
            if live:
                acc = scaled_dot(c, scaled_units(c, expert, channel))
                c["reduce_rows"](1, acc, c["emit_out"](16), count=S)
                gpu.barrier()
                if tid < S * 4:
                    row, pair = tid // 4, tid % 4
                    selected_slot = fx.Int32(0)
                    for selected in range_constexpr(1, topk + 1):
                        matches = lds_ld(keys, row * slots + selected) == expert
                        selected_slot = matches.select(
                            fx.Int32(selected), selected_slot
                        )
                    if selected_slot != 0:
                        r = pair * 2
                        g0, g1 = (
                            lds_ld(outs, row * 16 + r),
                            lds_ld(outs, row * 16 + r + 1),
                        )
                        u0, u1 = (
                            lds_ld(outs, row * 16 + 8 + r),
                            lds_ld(outs, row * 16 + 8 + r + 1),
                        )
                        m0 = c["swiglu"](g0, u0, c["swiglu_limit"])
                        m1 = c["swiglu"](g1, u1, c["swiglu_limit"])
                        c["put2"](
                            c["mb"]("mid"),
                            (row * slots + selected_slot) * c["INTER"]
                            + channel * 8
                            + r,
                            m0,
                            m1,
                        )
            gpu.barrier()
