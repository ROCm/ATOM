# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""q7 expert-union Down with ordered BF16 route reduction."""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.deepseek_v4.mono.kernels import buffer_ops as bo
from atom.models.deepseek_v4.mono.kernels.helpers import traced
from atom.models.deepseek_v4.mono.kernels.moe_ug import compact_experts
from atom.models.deepseek_v4.mono.kernels.primitives import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    lds_ld,
    lds_st,
    rsrc,
    uniform,
    xshfl,
)
from atom.mono.device.mx import FP4, FP8, mfma_scaled


def selected_route(c, row, expert):
    selected = fx.Int32(0)
    for slot in range(1, c["TOP_K"] + 1):
        match = lds_ld(c["keys"], row * c["MOE_SLOTS"] + slot) == expert
        selected = match.select(fx.Int32(slot), selected)
    return selected


@traced
def stage_expert_mid(c, expert):
    """Acquire only selected rows' MID, retaining its original MXFP8/32 codes."""
    rows, inter = c["S"], c["INTER"]
    wave, lane, xs = c["wave"], c["lane"], c["xs"]
    words, scales = inter // 4, inter // 32
    route_at = rows * (words + scales)
    if wave < rows:
        route = selected_route(c, wave, expert)
        if lane == 0:
            lds_st(xs, route_at + wave, route.bitcast(fx.Float32))
        # Unselected MFMA columns have legal zero storage, never unwritten MID.
        for block in range_constexpr(inter // 128):
            if lane < 32:
                lds_st(xs, wave * words + block * 32 + lane, fx.Float32(0.0))
            if lane < 4:
                lds_st(
                    xs,
                    rows * words + wave * scales + block * 4 + lane,
                    fx.Int32(127).bitcast(fx.Float32),
                )
        if route != 0:
            values = c["get2_many"](
                [
                    (
                        c["mb"]("mid"),
                        (wave * c["MOE_SLOTS"] + route) * inter
                        + block * 128
                        + lane * 2,
                    )
                    for block in range(inter // 128)
                ]
            )
            for block in range_constexpr(inter // 128):
                a, b, scale = c["quant_mxfp8"](*values[block])
                packed = (
                    fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
                    & 0xFFFF
                )
                neighbor = xshfl(packed, 1)
                if lane % 2 == 0:
                    lds_st(
                        xs,
                        wave * words + block * 32 + lane // 2,
                        (packed | (neighbor << 16)).bitcast(fx.Float32),
                    )
                if lane % 16 == 0:
                    code = scale.bitcast(fx.Int32).shrui(fx.Int32(23)) & 0xFF
                    lds_st(
                        xs,
                        rows * words + wave * scales + block * 4 + lane // 16,
                        code.bitcast(fx.Float32),
                    )
    gpu.barrier()


@traced
def stage_shared_mid(c):
    """Keep shared MXFP8/128 round trip and BF16 LDS exactly as before."""
    row, lane, inter = c["wave"], c["lane"], c["INTER"]
    if row < c["S"]:
        values = c["get2_many"](
            [
                (c["mb"]("mid"), row * c["MOE_SLOTS"] * inter + b * 128 + lane * 2)
                for b in range(inter // 128)
            ]
        )
        for b in range_constexpr(inter // 128):
            a, d, scale = c["quant_mxfp8"](*values[b], 128)
            payload = bf16_pair(a * scale, d * scale)
            lds_st(c["xs"], (row * (inter // 128) + b) * 64 + lane, payload)
    gpu.barrier()


def shared_units(c, tile):
    inter, tile_rows = c["INTER"], c["DN_TILE"]
    lane, wave = c["lane"], c["wave"]
    group = tile * (tile_rows // 16) + wave
    row = c["n_sel"]()
    wb, sb = rsrc(c["w_sdn"]), rsrc(c["s_sdn"])
    return [
        c["unit_fp8mx"](
            wb, sb, group, group, k, inter, (row * (inter // 128) + k) * 64, None, lane
        )
        for k in range(inter // 128)
    ]


def expert_units(c, expert, tile):
    """128 output rows share the selected expert's seven MID columns."""
    hidden, inter, tile_rows = c["HIDDEN"], c["INTER"], c["DN_TILE"]
    lane, wave, xs = c["lane"], c["wave"], c["xs"]
    rw = rsrc(c["w_dn"] + fx.Int64(expert) * (hidden * inter // 2))
    scale_bytes = hidden * (-(-(inter // 32) // 8) * 8)
    rs = rsrc(c["s_dn"] + fx.Int64(expert) * scale_bytes)
    group = tile * (tile_rows // 16) + wave
    column = fx.min(lane % 16, fx.Int32(c["S"] - 1))
    units = []
    for k in range(inter // 128):
        original = c["unit_mxfp4"](rw, rs, group, k, inter, 0, None, lane)
        raw, sa = original[1]
        halves = [
            fx.Vector(
                fx.ptr_load(
                    xs + column * (inter // 4) + k * 32 + 16 * half + (lane // 16) * 4,
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
            ).bitcast(fx.Int32)
            for half in range(2)
        ]
        b = fx.Vector.from_elements([v[i] for v in halves for i in range(4)], fx.Int32)
        sb = lds_ld(
            xs, c["S"] * (inter // 4) + column * (inter // 32) + k * 4 + lane // 16
        ).bitcast(fx.Int32)
        units.append((raw, sa, b, sb))
    return units


def expert_dot(c, units):
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for raw, sa, b, sb in units:
        acc = mfma_scaled(raw, b, acc, sa, sb, a_fmt=FP4, b_fmt=FP8)
    return [acc[i] for i in range(4)]


@traced
def publish_partial(c, tile, shared):
    rows, tile_rows = c["S"], c["DN_TILE"]
    tid = c["tid"]
    if tid < rows * tile_rows // 2:
        row, at = tid // (tile_rows // 2), (tid % (tile_rows // 2)) * 2
        route_at = rows * (c["INTER"] // 4 + c["INTER"] // 32)
        slot = (
            fx.Int32(0) if shared else lds_ld(c["xs"], route_at + row).bitcast(fx.Int32)
        )
        if shared or slot != 0:
            values = [lds_ld(c["outs"], row * tile_rows + at + j) for j in (0, 1)]
            if not shared:
                weight = lds_ld(c["dnw"], row * c["MOE_SLOTS"] + slot)
                values = [bf16_round(v * weight) for v in values]
            c["put_bf"](
                c["sym"] + fx.Int64(c["SY"]["attn"]),
                (slot * rows + row) * c["HIDDEN"] + tile * tile_rows + at,
                values,
                store_tag=c["tag"] ^ fx.Int32(-2147483648),
            )
    gpu.barrier()


@traced
def union_down(c):
    """Reuse retired local attention slots for route partials, then TP-reduce.

    Attention pushes to each destination; a rank consumes only its own arena.
    Completing local attention retires every source write to that destination.
    FFN writes each peer contribution only after local partial reads finish;
    completing FFN therefore retires all partial readers before next attention.
    XOR bit31 distinguishes these partials from ordinary attention tags.
    """
    assert c["S"] == c["TOK"] == 7 and c["npes"] == 8 and c["TOP_K"] == 6
    assert c["phase"] == "moe" and c["DN_TILE"] == 128 and c["INTER"] == 384
    assert c["SY"]["ffn"] >= c["MOE_SLOTS"] * c["S"] * c["HIDDEN"] * 4
    compact_experts(c)
    table = c["xs"] + c["S"] * (c["HIDDEN"] // 4 + c["HIDDEN"] // 32)
    count = uniform(lds_ld(table, c["S"] * c["TOP_K"]).bitcast(fx.Int32))
    tiles, tile_rows, grid = c["N_DN_TILES"], c["DN_TILE"], c["G"]
    # Partition shared jobs and expert jobs in a single 256-CTA resident pool.
    for job in range(fx.Int32(c["start"]("down")), (count + 1) * tiles, grid):
        job = fx.Int32(job)
        tile = job % tiles
        shared = job < tiles
        if shared:
            stage_shared_mid(c)
            acc = c["mma_units"](
                [fx.Float32(0.0) for _ in range(4)], shared_units(c, tile)
            )
            c["reduce_rows"](tile_rows // 16, acc, c["emit_out"](tile_rows))
            gpu.barrier()
            publish_partial(c, tile, True)
        else:
            expert = uniform(lds_ld(table, job // tiles - 1).bitcast(fx.Int32))
            stage_expert_mid(c, expert)
            acc = expert_dot(c, expert_units(c, expert, tile))
            c["reduce_rows"](tile_rows // 16, acc, c["emit_out"](tile_rows))
            gpu.barrier()
            publish_partial(c, tile, False)
    # No partial-producing job remains behind this consumer barrier.
    for tile in range(fx.Int32(c["start"]("down")), tiles, grid):
        tile = fx.Int32(tile)
        c["stamp"]("down", tile, 0)
        tid = c["tid"]
        if tid < c["S"] * tile_rows // 2:
            row, at = tid // (tile_rows // 2), (tid % (tile_rows // 2)) * 2
            pair = (row * c["HIDDEN"] + tile * tile_rows + at) // 2
            got = c["poll"](
                [
                    (
                        c["sym"] + fx.Int64(c["SY"]["attn"]),
                        slot * (c["S"] * c["HIDDEN"] // 2) + pair,
                        1,
                    )
                    for slot in range(c["MOE_SLOTS"])
                ],
                expected_tag=c["tag"] ^ fx.Int32(-2147483648),
            )
            first = bf2_f32(got[1][0])
            total = [first[0], first[1]]
            for slot in range_constexpr(2, c["MOE_SLOTS"]):
                values = bf2_f32(got[slot][0])
                total = [total[j] + values[j] for j in (0, 1)]
            shared = bf2_f32(got[0][0])
            for j in range_constexpr(2):
                value = bf16_round(bf16_round(total[j]) + shared[j])
                lds_st(c["outs"], row * tile_rows + at + j, value)
        c["stamp"]("down", tile, 2)
        gpu.barrier()
        c["stamp"]("down", tile, 3)
        c["hc_stage_coef"](1)

        def store_hc(row, stream, channel, a, b):
            bo.buffer_store(
                fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16),
                rsrc(c["x_out"]),
                (row * c["HC"] + stream) * c["HIDDEN"] + channel,
            )

        def reduce_then_hc(row, channel, a, b):
            return c["hc_post"](
                row,
                channel,
                a,
                b,
                lambda row_, stream, channel_: c["get"](
                    c["mb"]("a"),
                    ((row_ * c["HC"] + stream) * c["HIDDEN"] + channel_) // 2,
                ),
                store_hc,
            )

        c["peer_reduce"]("ffn", tile, None, reduce_then_hc, tile=tile_rows)
        gpu.barrier()
        c["stamp"]("down", tile, 4)
