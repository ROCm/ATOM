# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ordinary-request GDN with rank-local geometry and shared arithmetic."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_projection import stage_activation
from atom.models.qwen4_exp_mono.stages.gdn_common import (
    conv_projection_value,
    norm_head_values,
    recurrent_views,
)
from atom.models.qwen4_exp_mono.stages.gdr_decode import create_qwen_gdr_decode
from atom.models.qwen4_exp_mono.stages.hc import _ld_bf, _sigmoid, _st_bf
from atom.mono.device.ops import (
    CM_DEV,
    bf16_round,
    kernel_symbol,
    launcher,
    mfma_bf16,
    row_sum,
    rsrc,
)

_CURRENT_STREAM = fx.Stream(None)


@fx.struct
class ProjectionSmem:
    activation: fx.Array[fx.Int32, 4 * 1280, 16]
    reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]


def build_projection(tokens, tp=8):
    dims = Dims(tp)
    rows, conv_dim = dims.gdn_proj_dim, dims.gdn_conv_dim
    name = kernel_symbol("qwen_batched_rank_projection", b=tokens, n=rows)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        x: Int64,
        w: Int64,
        out: Int64,
        conv_weight: Int64,
        state: Int64,
        read_indices: Int64,
        write_indices: Int64,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        tile = fx.block_idx.x
        sm = fx.SharedAllocator().allocate(ProjectionSmem).peek()
        stage_activation(x, sm.activation.ptr, tid, tokens)
        weight_row = fx.min(tile * 16 + lane % 16, rows - 1)
        ct = fx.min(lane % 16, tokens - 1)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for k in range_constexpr(10):
            at = (wave + 8 * k) * 32 + 8 * (lane // 16)
            weight = fx.Vector(
                bo.buffer_load(
                    rsrc(w),
                    weight_row * 1280 + at // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            ).bitcast(fx.BFloat16)
            act = fx.Vector(
                fx.ptr_load(
                    sm.activation.ptr + ct * 1280 + at // 2, result_type=T.vec(4, T.i32)
                )
            ).bitcast(fx.BFloat16)
            acc = mfma_bf16(weight, act, acc)
        fx.ptr_store(acc, sm.reduction.ptr + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < 16 * tokens:
            token = tid // 16
            row = tid % 16
            value = row_sum(sm.reduction.ptr, row, token)
            channel = tile * 16 + row
            value = bf16_round(value)
            if channel < conv_dim:
                read_slot = fx.Int32(
                    bo.buffer_load(rsrc(read_indices), token, vec_width=1, dtype=T.i32)
                )
                write_slot = fx.Int32(
                    bo.buffer_load(rsrc(write_indices), token, vec_width=1, dtype=T.i32)
                )
                if (read_slot >= 0) & (write_slot >= 0):
                    value = conv_projection_value(
                        value,
                        channel,
                        conv_weight,
                        state,
                        read_slot,
                        write_slot,
                        conv_dim,
                    )
                else:
                    value = fx.Float32(0.0)
            if channel < rows:
                bo.buffer_store(
                    value.to(fx.BFloat16), rsrc(out), token * rows + channel
                )

    @launcher(name)
    def launch(
        x: Int64,
        w: Int64,
        out: Int64,
        conv_weight: Int64,
        state: Int64,
        read_indices: Int64,
        write_indices: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = SOURCES
        kernel(x, w, out, conv_weight, state, read_indices, write_indices).launch(
            grid=((rows + 15) // 16,), block=(512,), stream=stream
        )

    return launch


_STREAM = fx.Stream(None)


@fx.struct
class OutputSmem:
    reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]


def build_output(tokens, tp=8):
    width = Dims(tp).gdn_value_dim
    name = (
        kernel_symbol("qwen_batched_gdn_output_seam", b=tokens, split=1)
        if tp == 1
        else kernel_symbol(
            "qwen_batched_rank_gdn_output", b=tokens, split=0, width=width
        )
    )

    @fx.struct
    class StagedOutputSmem:
        activation: fx.Array[fx.Int32, tokens * (width // 2), 16]
        reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(normed: Int64, weight: Int64, out: Int64):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        tile = fx.block_idx.x
        sm = (
            fx.SharedAllocator()
            .allocate(StagedOutputSmem if tp == 1 else OutputSmem)
            .peek()
        )
        if tp == 1:
            for offset in range_constexpr(0, width // 2, 512 * 4):
                word = offset + tid * 4
                if word < width // 2:
                    for row in range_constexpr(tokens):
                        act = fx.Vector(
                            bo.buffer_load(
                                rsrc(normed),
                                row * (width // 2) + word,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        fx.ptr_store(act, sm.activation.ptr + row * (width // 2) + word)
            gpu.barrier()
        token = fx.min(lane % 16, tokens - 1)

        def accumulate(k_start, passes):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for k in range_constexpr(passes):
                at = k_start + (wave + 8 * k) * 32 + 8 * (lane // 16)
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(weight),
                        (tile * 16 + lane % 16) * (width // 2) + at // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                ).bitcast(fx.BFloat16)
                x = fx.Vector(
                    fx.ptr_load(
                        sm.activation.ptr + token * (width // 2) + at // 2,
                        result_type=T.vec(4, T.i32),
                    )
                    if tp == 1
                    else bo.buffer_load(
                        rsrc(normed),
                        token * (width // 2) + at // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                ).bitcast(fx.BFloat16)
                acc = mfma_bf16(w, x, acc)
            fx.ptr_store(acc, sm.reduction.ptr + (wave * 64 + lane) * 4)
            gpu.barrier()
            value = row_sum(sm.reduction.ptr, tid % 16, tid // 16)
            gpu.barrier()
            return value

        def store(value):
            if tid < 16 * tokens:
                t = tid // 16
                row = tid % 16
                bo.buffer_store(
                    value.to(fx.BFloat16), rsrc(out), t * 2560 + tile * 16 + row
                )

        if tp == 1:
            if tile < 64:
                store(accumulate(0, 24))
            else:
                first = accumulate(0, 12)
                store(first + accumulate(3072, 12))
        else:
            store(accumulate(0, width // 256))

    @launcher(name)
    def launch(normed: Int64, weight: Int64, out: Int64, stream: fx.Stream = _STREAM):
        _ = SOURCES
        kernel(normed, weight, out).launch(grid=(160,), block=(512,), stream=stream)

    return launch


@fx.struct
class RecurrenceNormSmem:
    core: fx.Array[fx.BFloat16, 128, 16]


def build_recurrence_norm(tokens, tp=8):
    """One complete-head state update followed by the original BF16 norm."""
    dims = Dims(tp)
    heads = dims.gdn_value_heads
    emit = create_qwen_gdr_decode(dims, complete_head=True)
    name = kernel_symbol("qwen_batched_recurrence_norm", b=tokens, tp=tp)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        projected: Int64,
        dt_bias: Int64,
        a_log: Int64,
        read: Int64,
        write: Int64,
        state: Int64,
        norm_weight: Int64,
        out: Int64,
    ):
        task = fx.block_idx.x
        token = task // heads
        head = task % heads
        tid = fx.thread_idx.x
        lane = tid % 64
        sm = fx.SharedAllocator().allocate(RecurrenceNormSmem).peek()
        values = recurrent_views(
            projected + fx.Int64(token) * fx.Int64(dims.gdn_proj_dim * 2),
            dt_bias,
            a_log,
            read + fx.Int64(token) * 4,
            write + fx.Int64(token) * 4,
            state,
            out + fx.Int64(token) * fx.Int64(dims.gdn_value_dim * 2),
            dims,
        )
        emit(*values, fx.Int32(1), tid, head, sm.core.ptr)
        gpu.barrier()
        if tid < 64:
            x0 = fx.Float32(fx.ptr_load(sm.core.ptr + lane))
            x1 = fx.Float32(fx.ptr_load(sm.core.ptr + lane + 64))
            affine0, affine1 = norm_head_values(x0, x1, norm_weight, lane)
            i = head * 128 + lane
            gate = (
                projected + fx.Int64(token * dims.gdn_proj_dim + dims.gdn_conv_dim) * 2
            )
            base = out + fx.Int64(token * dims.gdn_value_dim) * 2
            _st_bf(base, i, affine0 * _sigmoid(_ld_bf(gate, i)), CM_DEV)
            _st_bf(base, i + 64, affine1 * _sigmoid(_ld_bf(gate, i + 64)), CM_DEV)

    @launcher(name)
    def launch(
        projected: Int64,
        dt_bias: Int64,
        a_log: Int64,
        read: Int64,
        write: Int64,
        state: Int64,
        norm_weight: Int64,
        out: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = SOURCES
        kernel(projected, dt_bias, a_log, read, write, state, norm_weight, out).launch(
            grid=(tokens * heads,), block=(512,), stream=stream
        )

    return launch
