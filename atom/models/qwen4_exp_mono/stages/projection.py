# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 N16 projection math; schedules retain load policy and publication."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.shapes import HIDDEN
from atom.mono.device.ops import mfma_bf16, rsrc, traced
from atom.mono.plan.execution import WAVES


@traced
def projection_accumulator(
    weight, tile, lane, wave, activation, *, cache_modifier=0, nbytes=None, rows=None
):
    """Eight-wave K split with an optional mask for the TP8 GDN tail tile."""
    row = tile * 16 + lane % 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(HIDDEN // (32 * WAVES)):
        k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // 16)
        options = {} if rows is None else {"mask": row < rows}
        w = fx.Vector(
            bo.buffer_load(
                rsrc(weight, nbytes=nbytes),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cache_modifier,
                **options,
            )
        ).bitcast(fx.BFloat16)
        x = fx.Vector(
            fx.ptr_load(activation + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, x, acc)
    return acc


def output_weights(weight, tile, k_start, k_steps, lane, wave, *, width, nbytes=None):
    """Prefetch a contiguous K range while the schedule waits for its input."""
    row = tile * 16 + lane % 16
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(weight, nbytes=nbytes),
                (
                    row * width
                    + k_start
                    + (wave + WAVES * k_pass) * 32
                    + 8 * (lane // 16)
                )
                // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=0,
            )
        ).bitcast(fx.BFloat16)
        for k_pass in range_constexpr(k_steps)
    ]


def output_accumulator(
    normed, k_start, lane, wave, weight_ops, *, cache_modifier, nbytes=None
):
    """Consume the prefetched K splits in their existing MFMA order."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(len(weight_ops)):
        k0 = k_start + (wave + WAVES * k_pass) * 32 + 8 * (lane // 16)
        x = fx.Vector(
            bo.buffer_load(
                rsrc(normed, nbytes=nbytes),
                k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cache_modifier,
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(weight_ops[k_pass], x, acc)
    return acc


@traced
def publish_output_groups(poll, put, ready, output_ready, bid, tid, owner=0):
    if bid == owner:
        if tid == 0:
            for start in range_constexpr(0, 96, 8):
                poll([(ready, start + j, 1) for j in range(min(8, 96 - start))])
        gpu.barrier()
        if tid == 0:
            put(output_ready, 0, 1)
    elif bid == owner + 1:
        if tid == 0:
            for start in range_constexpr(0, HIDDEN // 16 - 96, 8):
                poll(
                    [
                        (ready, 96 + start + j, 1)
                        for j in range(min(8, HIDDEN // 16 - 96 - start))
                    ]
                )
        gpu.barrier()
        if tid == 0:
            put(output_ready, 1, 1)
