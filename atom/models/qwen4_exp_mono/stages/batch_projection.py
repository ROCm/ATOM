# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ordinary BF16 router/shared projections with one activation staging contract.

The default input is contiguous BF16. Fused HC callers supply tagged-word
loads and prefetch the first weight block before waiting for that input.
The MFMA and wave reduction order is shared by both paths.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.stages.hc import _sigmoid
from atom.mono.device.ops import bf16_round, mfma_bf16, row_sum, rsrc, traced


@fx.struct
class G1Smem:
    activation: fx.Array[fx.Int32, 4 * 1280, 16]
    reduction: fx.Array[fx.Float32, 2 * 8 * 64 * 4, 16]


@traced
def stage_activation(x, activation, tid, tokens, load=None):
    if tid < 320:
        for token in range_constexpr(tokens):
            words = (
                load(token * 1280 + tid * 4, 4)
                if load is not None
                else fx.Vector(
                    bo.buffer_load(
                        rsrc(x), token * 1280 + tid * 4, vec_width=4, dtype=T.i32
                    )
                )
            )
            fx.ptr_store(words, activation + token * 1280 + tid * 4)
    gpu.barrier()


@traced
def _router_projection(
    x, weight, out, task, tokens, sm, load=None, prefetch=False, split_k=1
):
    # A split publishes FP32 partials. The MoE preparation consumer combines
    # them before the router's BF16 rounding and top-k boundary.
    tile = task if split_k == 1 else task % 33
    split = 0 if split_k == 1 else task // 33
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    if not prefetch:
        stage_activation(x, sm.activation.ptr, tid, tokens, load)
    token = fx.min(lane % 16, tokens - 1)
    row = fx.min(tile * 16 + lane % 16, 512)
    first_at = (wave + 8 * split * (10 // split_k)) * 32 + 8 * (lane // 16)
    first_w = (
        fx.Vector(
            bo.buffer_load(
                rsrc(weight), row * 1280 + first_at // 2, vec_width=4, dtype=T.i32
            )
        ).bitcast(fx.BFloat16)
        if prefetch
        else None
    )
    if prefetch:
        stage_activation(x, sm.activation.ptr, tid, tokens, load)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k in range_constexpr(10 // split_k):
        at = (wave + 8 * (k + split * (10 // split_k))) * 32 + 8 * (lane // 16)
        w = (
            first_w
            if prefetch and k == 0
            else fx.Vector(
                bo.buffer_load(
                    rsrc(weight), row * 1280 + at // 2, vec_width=4, dtype=T.i32
                )
            ).bitcast(fx.BFloat16)
        )
        xval = fx.Vector(
            fx.ptr_load(
                sm.activation.ptr + token * 1280 + at // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, xval, acc)
    fx.ptr_store(acc, sm.reduction.ptr + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < 16 * tokens:
        t = tid // 16
        n = tile * 16 + tid % 16
        if n <= 512:
            value = row_sum(sm.reduction.ptr, tid % 16, t)
            if split_k == 1:
                bo.buffer_store(value.to(fx.BFloat16), rsrc(out), t * 513 + n)
            else:
                bo.buffer_store(value, rsrc(out), (split * tokens + t) * 513 + n)


@traced
def _shared_projection(
    x, weight, out, tile, tokens, intermediate, sm, load=None, prefetch=False
):
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    if not prefetch:
        stage_activation(x, sm.activation.ptr, tid, tokens, load)
    token = fx.min(lane % 16, tokens - 1)
    row = tile * 16 + lane % 16
    first_at = wave * 32 + 8 * (lane // 16)
    first_gw = (
        fx.Vector(
            bo.buffer_load(
                rsrc(weight), row * 1280 + first_at // 2, vec_width=4, dtype=T.i32
            )
        ).bitcast(fx.BFloat16)
        if prefetch
        else None
    )
    first_uw = (
        fx.Vector(
            bo.buffer_load(
                rsrc(weight),
                (intermediate + row) * 1280 + first_at // 2,
                vec_width=4,
                dtype=T.i32,
            )
        ).bitcast(fx.BFloat16)
        if prefetch
        else None
    )
    if prefetch:
        stage_activation(x, sm.activation.ptr, tid, tokens, load)
    gate = fx.Vector.filled(4, 0.0, fx.Float32)
    up = fx.Vector.filled(4, 0.0, fx.Float32)
    for k in range_constexpr(10):
        at = (wave + 8 * k) * 32 + 8 * (lane // 16)
        xval = fx.Vector(
            fx.ptr_load(
                sm.activation.ptr + token * 1280 + at // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        gw = (
            first_gw
            if prefetch and k == 0
            else fx.Vector(
                bo.buffer_load(
                    rsrc(weight), row * 1280 + at // 2, vec_width=4, dtype=T.i32
                )
            ).bitcast(fx.BFloat16)
        )
        uw = (
            first_uw
            if prefetch and k == 0
            else fx.Vector(
                bo.buffer_load(
                    rsrc(weight),
                    (intermediate + row) * 1280 + at // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            ).bitcast(fx.BFloat16)
        )
        gate = mfma_bf16(gw, xval, gate)
        up = mfma_bf16(uw, xval, up)
    fx.ptr_store(gate, sm.reduction.ptr + (wave * 64 + lane) * 4)
    fx.ptr_store(up, sm.reduction.ptr + 2048 + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < 16 * tokens:
        g = bf16_round(row_sum(sm.reduction.ptr, tid % 16, tid // 16))
        u = bf16_round(row_sum(sm.reduction.ptr + 2048, tid % 16, tid // 16))
        value = bf16_round(g * _sigmoid(g) * u)
        bo.buffer_store(
            value.to(fx.BFloat16),
            rsrc(out),
            (tid // 16) * intermediate + tile * 16 + tid % 16,
        )
