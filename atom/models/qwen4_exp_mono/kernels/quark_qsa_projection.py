# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 Quark QKV and replicated index QK share one BF16 projection grid."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_projection import G1Smem, stage_activation
from atom.mono.device.ops import kernel_symbol, launcher, mfma_bf16, row_sum, rsrc
from atom.mono.runtime.abi import KernelAbi

ABI = KernelAbi(("x", "qkv_weight", "index_weight", "qkv", "index"))
_STREAM = fx.Stream(None)


@cache
def build(rows):
    assert rows in (1, 2, 4)
    name = kernel_symbol("qwen_quark_qsa_projection", b=rows)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        x: Int64, qkv_weight: Int64, index_weight: Int64, qkv: Int64, index: Int64
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        task = fx.block_idx.x
        sm = fx.SharedAllocator().allocate(G1Smem).peek()
        is_index = task >= 128
        tile = is_index.select(task - 128, task)
        weight = is_index.select(index_weight, qkv_weight)
        out = is_index.select(index, qkv)
        stride = is_index.select(fx.Int64(640), fx.Int64(2048))
        stage_activation(x, sm.activation.ptr, tid, rows)
        n = tile * 16 + lane % 16
        token = fx.min(lane % 16, rows - 1)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for k in range_constexpr(10):
            at = (wave + 8 * k) * 32 + 8 * (lane // 16)
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(weight), n * 1280 + at // 2, vec_width=4, dtype=T.i32
                )
            ).bitcast(fx.BFloat16)
            a = fx.Vector(
                fx.ptr_load(
                    sm.activation.ptr + token * 1280 + at // 2,
                    result_type=T.vec(4, T.i32),
                )
            ).bitcast(fx.BFloat16)
            acc = mfma_bf16(w, a, acc)
        fx.ptr_store(acc, sm.reduction.ptr + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < 16 * rows:
            t = tid // 16
            col = tile * 16 + tid % 16
            bo.buffer_store(
                row_sum(sm.reduction.ptr, tid % 16, t).to(fx.BFloat16),
                rsrc(out),
                fx.Int64(t) * stride + col,
            )

    @launcher(name)
    def launch(
        x: Int64,
        qkv_weight: Int64,
        index_weight: Int64,
        qkv: Int64,
        index: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(x, qkv_weight, index_weight, qkv, index).launch(
            grid=(168,), block=(512,), stream=stream
        )

    return launch
