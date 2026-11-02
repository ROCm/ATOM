# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared native gfx1250 projection compute for standalone and K1 tasks."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from atom.mono.device.ops import bf16_round, bf_hi, bf_lo, traced

HEAD_DIM = 512
ROPE = 64
TILE_N = 16
K_SPLIT = 8
THREADS = 256
UNIT_SCALES = 0x7F7F7F7F


def gemv_lds_bytes(rows, inner):
    return K_SPLIT * rows * TILE_N * 4


def projection_load(weight, scale, tid, bid, inner, output):
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    n0 = bid * 16
    rw = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(weight), num_records_bytes=output * inner
    )
    rs = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(scale), num_records_bytes=output * inner // 1024
    )
    steps = [wave + K_SPLIT * i for i in range((inner // 128 + K_SPLIT - 1) // K_SPLIT)]
    weights, scales = [], []
    # Prefetch all strided K128 rounds before consuming activation operands.
    # Direct FP8/scale loads remove the redundant CTA-wide LDS copy/barrier.
    for step in steps:
        st = fx.min(step, inner // 128 - 1)
        bv = [
            Vec(
                bo.buffer_load(
                    rw,
                    (
                        n0 // 16 * inner * 16
                        + st * 2048
                        + half * 256
                        + lane16 * 16
                        + j * 512
                    )
                    // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for j in range(4)
        ]
        weights.append(
            bv[0]
            .shuffle(bv[1], list(range(8)))
            .shuffle(bv[2].shuffle(bv[3], list(range(8))), list(range(16)))
        )
        scales.append(
            bo.buffer_load(
                rs, (n0 // 32) * (inner // 128) + st, vec_width=1, dtype=T.i32
            )
        )
    fx.rocdl.sched_barrier(0)

    return weights, scales, steps


@traced
def projection_ready(
    q,
    qs,
    weight,
    scale,
    cos,
    sin,
    out,
    unrotated,
    ops,
    lds,
    tid,
    bid,
    rows,
    inner,
    output,
    rope=False,
    debug=False,
    input_stride=None,
    output_stride=None,
):
    """8-wave K-strided native WMMA, with legal masked K128 tails.

    All threads enter together. Caller provides gemv_lds_bytes() LDS and a
    valid N16 tile id. The final barrier permits immediate LDS reuse.
    """
    partial_off = 0
    # A grouped projection offsets q/out to its group's first element while
    # retaining the full token stride. Buffer extents stop at the last live
    # group element, rather than claiming rows * stride bytes past that view.
    xstride = inner if input_stride is None else input_stride
    ystride = output if output_stride is None else output_stride
    lane, wave = tid % 32, tid // 32
    n0 = bid * TILE_N
    lane16, half = lane % 16, lane // 16
    ld32, st32 = make_lds_copy_ops(32)
    ld128, st128 = make_lds_copy_ops(128)
    rx = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(q), num_records_bytes=(rows - 1) * xstride + inner
    )
    rxs = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(qs), num_records_bytes=((rows - 1) * xstride + inner) // 32
    )
    rw = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(weight), num_records_bytes=output * inner
    )
    rs = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(scale), num_records_bytes=output * inner // 1024
    )
    rc = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(cos), num_records_bytes=rows * ROPE
    )
    rn = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(sin), num_records_bytes=rows * ROPE
    )
    ro = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(out), num_records_bytes=((rows - 1) * ystride + output) * 2
    )
    if const_expr(debug):
        rd = bo.create_buffer_resource_from_addr(
            fx.ptrtoint(unrotated),
            num_records_bytes=((rows - 1) * ystride + output) * 2,
        )

    weights, scales, steps = ops

    active_row = fx.min(lane16, rows - 1)
    atom = fx.make_mma_atom(
        fx.rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32)
    )
    accum = fx.make_rmem_tensor(8, fx.Float32)
    accum.store(fx.full(8, 0.0, fx.Float32))
    for i in range_constexpr((inner // 128 + K_SPLIT - 1) // K_SPLIT):
        step = steps[i]
        st = fx.min(step, inner // 128 - 1)
        live = step < inner // 128
        av = [
            Vec(
                bo.buffer_load(
                    rx,
                    (active_row * xstride + st * 128 + half * 16 + j * 32) // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for j in range_constexpr(4)
        ]
        avec = (
            av[0]
            .shuffle(av[1], list(range(8)))
            .shuffle(av[2].shuffle(av[3], list(range(8))), list(range(16)))
        )
        a, b = fx.make_rmem_tensor(16, fx.Int32), fx.make_rmem_tensor(16, fx.Int32)
        a.store(
            Vec.from_elements([live.select(avec[j], fx.Int32(0)) for j in range(16)])
        )
        b.store(
            Vec.from_elements(
                [live.select(weights[i][j], fx.Int32(0)) for j in range(16)]
            )
        )
        sa = bo.buffer_load(
            rxs, active_row * (xstride // 128) + st, vec_width=1, dtype=T.i32
        )
        fx.gemm(
            atom,
            accum,
            b,
            a,
            accum,
            scale_a=live.select(scales[i], fx.Int32(UNIT_SCALES)),
            scale_b=live.select(sa, fx.Int32(UNIT_SCALES)),
        )
    if lane16 < rows:
        values = accum.load()
        offset = partial_off + (wave * rows * TILE_N + lane16 * TILE_N + half * 8) * 4
        for j in range_constexpr(2):
            st128(
                lds,
                offset + j * 16,
                Vec.from_elements(
                    [values[j * 4 + k].bitcast(fx.Int32) for k in range(4)]
                ),
            )
    fx.gpu.barrier()

    if tid < rows * TILE_N // 2:
        row, pair = tid // (TILE_N // 2), tid % (TILE_N // 2)
        column = n0 + 2 * pair
        sums = [fx.Float32(0.0), fx.Float32(0.0)]
        for w in range_constexpr(K_SPLIT):
            for j in range_constexpr(2):
                part = fx.Int32(
                    ld32(
                        lds,
                        partial_off
                        + (w * rows * TILE_N + row * TILE_N + 2 * pair + j) * 4,
                    )[0]
                ).bitcast(fx.Float32)
                sums[j] = sums[j] + part
        values = [bf16_round(v) for v in sums]
        rotated = values
        if const_expr(rope):
            dim = column % HEAD_DIM
            table_pair = fx.max(dim - (HEAD_DIM - ROPE), 0) // 2
            at = row * (ROPE // 2) + table_pair
            cw = fx.Int32(bo.buffer_load(rc, at // 2, vec_width=1, dtype=T.i32))
            sw = fx.Int32(bo.buffer_load(rn, at // 2, vec_width=1, dtype=T.i32))
            cs = (at % 2 == 0).select(bf_lo(cw), bf_hi(cw))
            sn = (at % 2 == 0).select(bf_lo(sw), bf_hi(sw))
            live = dim >= HEAD_DIM - ROPE
            rotated = [
                live.select(values[0] * cs - values[1] * sn, values[0]),
                live.select(values[0] * sn + values[1] * cs, values[1]),
            ]
        for j in range_constexpr(2):
            bo.buffer_store(
                fx.Float32(rotated[j]).to(fx.BFloat16), ro, row * ystride + column + j
            )
            if const_expr(debug):
                bo.buffer_store(
                    values[j].to(fx.BFloat16), rd, row * ystride + column + j
                )

    fx.gpu.barrier()


@traced
def gemv_device(
    q,
    qs,
    weight,
    scale,
    cos,
    sin,
    out,
    unrotated,
    lds,
    tid,
    bid,
    rows,
    inner,
    output,
    rope=False,
    debug=False,
):
    ops = projection_load(weight, scale, tid, bid, inner, output)
    projection_ready(
        q,
        qs,
        weight,
        scale,
        cos,
        sin,
        out,
        unrotated,
        ops,
        lds,
        tid,
        bid,
        rows,
        inner,
        output,
        rope,
        debug,
    )


def grouped_lds_bytes(rows, groups):
    return rows * groups * 8 * 16 * 4


def grouped_load(weight, scale, tid, bid, output, groups):
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    split = 8 // groups
    group, part = wave // split, wave % split
    n0 = (bid * groups + group) * 16
    rw = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(weight), num_records_bytes=output * 1280
    )
    rs = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(scale), num_records_bytes=output * 1280 // 1024
    )
    weights = []
    scales = []
    steps = []
    for slot in range_constexpr(groups):
        # Ten K128 steps need a second round for the slots carrying K0/K1.
        # At groups8 these are slots0/1; groups2/4 keep their old schedule.
        rounds = 2 if slot * split < 2 else 1
        for iteration in range_constexpr(rounds):
            step = part + slot * split + iteration * 8
            st = fx.min(step, 9)
            bv = [
                Vec(
                    bo.buffer_load(
                        rw,
                        (
                            n0 // 16 * 1280 * 16
                            + st * 2048
                            + half * 256
                            + lane16 * 16
                            + j * 512
                        )
                        // 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for j in range(4)
            ]
            weights.append(
                bv[0]
                .shuffle(bv[1], list(range(8)))
                .shuffle(bv[2].shuffle(bv[3], list(range(8))), list(range(16)))
            )
            scales.append(
                bo.buffer_load(rs, (n0 // 32) * 10 + st, vec_width=1, dtype=T.i32)
            )
            steps.append(step)
    fx.rocdl.sched_barrier(0)

    return weights, scales, steps


@traced
def grouped_partials(q, qs, ops, lds, tid, bid, rows, output, groups):
    """Store eight K-strided partials per N16 tile, then synchronize LDS.

    Layout is [groups, 8, rows, 16] FP32; capacity must be
    groups*8*rows*16*4 bytes. Used by BF16 projection and IQ quant consumers.
    """
    # Eight original partials are kept independent. A physical wave computes
    # groups of those partials for one of 2/4/8 output tiles. Keep the
    # existing eight-part reduction order for all three geometries.
    split = 8 // groups
    lane, wave = tid % 32, tid // 32
    group, part = wave // split, wave % split
    lane16, half = lane % 16, lane // 16
    n0 = (bid * groups + group) * 16
    ld128, st128 = make_lds_copy_ops(128)
    rx = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(q), num_records_bytes=rows * 1280
    )
    rxs = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(qs), num_records_bytes=rows * 40
    )
    weights, scales, steps = ops
    atom = fx.make_mma_atom(
        fx.rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32)
    )
    active_row = fx.min(lane16, rows - 1)
    loaded = 0
    for slot in range_constexpr(groups):
        accum = fx.make_rmem_tensor(8, fx.Float32)
        accum.store(fx.full(8, 0.0, fx.Float32))
        rounds = 2 if slot * split < 2 else 1
        for iteration in range_constexpr(rounds):
            step = steps[loaded]
            st = fx.min(step, 9)
            live = step < 10
            av = [
                Vec(
                    bo.buffer_load(
                        rx,
                        (active_row * 1280 + st * 128 + half * 16 + j * 32) // 4,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for j in range(4)
            ]
            avec = (
                av[0]
                .shuffle(av[1], list(range(8)))
                .shuffle(av[2].shuffle(av[3], list(range(8))), list(range(16)))
            )
            a, b = fx.make_rmem_tensor(16, fx.Int32), fx.make_rmem_tensor(16, fx.Int32)
            a.store(
                Vec.from_elements(
                    [live.select(avec[j], fx.Int32(0)) for j in range(16)]
                )
            )
            b.store(
                Vec.from_elements(
                    [live.select(weights[loaded][j], fx.Int32(0)) for j in range(16)]
                )
            )
            sa = bo.buffer_load(rxs, active_row * 10 + st, vec_width=1, dtype=T.i32)
            fx.gemm(
                atom,
                accum,
                b,
                a,
                accum,
                scale_a=live.select(scales[loaded], fx.Int32(UNIT_SCALES)),
                scale_b=live.select(sa, fx.Int32(UNIT_SCALES)),
            )
            loaded += 1
        if lane16 < rows:
            values = accum.load()
            offset = (
                (group * 8 + part + slot * split) * rows * 16 + lane16 * 16 + half * 8
            ) * 4
            for j in range_constexpr(2):
                st128(
                    lds,
                    offset + j * 16,
                    Vec.from_elements(
                        [values[j * 4 + k].bitcast(fx.Int32) for k in range(4)]
                    ),
                )
    fx.gpu.barrier()


@traced
def grouped_ready(
    q,
    qs,
    weight,
    scale,
    cos,
    sin,
    out,
    ops,
    lds,
    tid,
    bid,
    rows,
    output,
    groups,
    head_dim=512,
):
    ld32, st32 = make_lds_copy_ops(32)
    rc = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(cos), num_records_bytes=rows * 64
    )
    rn = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(sin), num_records_bytes=rows * 64
    )
    ro = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(out), num_records_bytes=rows * output * 2
    )
    grouped_partials(q, qs, ops, lds, tid, bid, rows, output, groups)
    if tid < rows * groups * 8:
        row = tid // (groups * 8)
        ngroup = (tid // 8) % groups
        pair = tid % 8
        column = (bid * groups + ngroup) * 16 + pair * 2
        sums = [fx.Float32(0.0), fx.Float32(0.0)]
        for wavepart in range_constexpr(8):
            for j in range_constexpr(2):
                value = fx.Int32(
                    ld32(
                        lds,
                        ((ngroup * 8 + wavepart) * rows * 16 + row * 16 + pair * 2 + j)
                        * 4,
                    )[0]
                ).bitcast(fx.Float32)
                sums[j] = sums[j] + value
        values = [bf16_round(v) for v in sums]
        dim = column % head_dim
        at = row * 32 + fx.max(dim - (head_dim - 64), 0) // 2
        cw = fx.Int32(bo.buffer_load(rc, at // 2, vec_width=1, dtype=T.i32))
        sw = fx.Int32(bo.buffer_load(rn, at // 2, vec_width=1, dtype=T.i32))
        cs = (at % 2 == 0).select(bf_lo(cw), bf_hi(cw))
        sn = (at % 2 == 0).select(bf_lo(sw), bf_hi(sw))
        rotated = [
            (dim >= head_dim - 64).select(values[0] * cs - values[1] * sn, values[0]),
            (dim >= head_dim - 64).select(values[0] * sn + values[1] * cs, values[1]),
        ]
        for j in range_constexpr(2):
            bo.buffer_store(
                fx.Float32(rotated[j]).to(fx.BFloat16), ro, row * output + column + j
            )
    fx.gpu.barrier()


@traced
def grouped_wqb_device(
    q, qs, weight, scale, cos, sin, out, lds, tid, bid, rows, output, groups=4
):
    ops = grouped_load(weight, scale, tid, bid, output, groups)
    grouped_ready(
        q, qs, weight, scale, cos, sin, out, ops, lds, tid, bid, rows, output, groups
    )
