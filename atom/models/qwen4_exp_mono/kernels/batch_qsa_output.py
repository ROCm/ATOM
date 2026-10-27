# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA batch BF16 output gate staged directly into projection LDS."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.gdn_common import _bf16_round_strict
from atom.models.qwen4_exp_mono.stages.hc import _sigmoid
from atom.mono.device.ops import kernel_symbol, launcher, mfma_bf16, row_sum, rsrc

_STREAM = fx.Stream(None)


def build(rows):
    width = 768
    assert rows in (2, 4)

    @fx.struct
    class Smem:
        activation: fx.Array[fx.BFloat16, rows * width, 16]
        reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]

    name = kernel_symbol("qwen_batch_qsa_gated_output", b=rows, k=width)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(x: Int64, gate: Int64, w: Int64, out: Int64, gate_stride: Int64):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        tile = fx.block_idx.x
        sm = fx.SharedAllocator().allocate(Smem).peek()
        for offset in range_constexpr(0, width, 512):
            col = offset + tid
            if col < width:
                for row in range_constexpr(rows):
                    value = fx.Float32(
                        bo.buffer_load(
                            rsrc(x), row * width + col, vec_width=1, dtype=T.bf16
                        )
                    )
                    raw = fx.Float32(
                        bo.buffer_load(
                            rsrc(gate),
                            fx.Int64(row) * gate_stride + col,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                    result = _bf16_round_strict(
                        value * _bf16_round_strict(_sigmoid(raw))
                    )
                    fx.ptr_store(
                        result.to(fx.BFloat16),
                        sm.activation.ptr + row * width + col,
                    )
        gpu.barrier()
        token = fx.min(lane % 16, rows - 1)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for k in range_constexpr(width // 256):
            at = (wave + 8 * k) * 32 + 8 * (lane // 16)
            weight = fx.Vector(
                bo.buffer_load(
                    rsrc(w),
                    (tile * 16 + lane % 16) * (width // 2) + at // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            ).bitcast(fx.BFloat16)
            value = fx.Vector(
                fx.ptr_load(
                    sm.activation.ptr + token * width + at,
                    result_type=T.vec(8, T.bf16),
                )
            )
            acc = mfma_bf16(weight, value, acc)
        fx.ptr_store(acc, sm.reduction.ptr + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < 16 * rows:
            token = tid // 16
            col = tile * 16 + tid % 16
            value = row_sum(sm.reduction.ptr, tid % 16, token)
            bo.buffer_store(value.to(fx.BFloat16), rsrc(out), token * 2560 + col)

    @launcher(name)
    def launch(
        x: Int64,
        gate: Int64,
        w: Int64,
        out: Int64,
        gate_stride: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(x, gate, w, out, gate_stride).launch(
            grid=(160,), block=(512,), stream=stream
        )

    return launch
