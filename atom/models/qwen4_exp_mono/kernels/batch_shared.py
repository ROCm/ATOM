# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 shared expert batch seam preserving the accepted G1/G2 arithmetic."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.layout import (
    finish_scratch_bytes,
    finish_scratch_layout,
)
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.g1 import _quant_routed_mid
from atom.models.qwen4_exp_mono.stages.hc import _ld_f32
from atom.mono.device.ops import (
    CM_DEV,
    bf16_round,
    kernel_symbol,
    launcher,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)

_STREAM = fx.Stream(None)


@fx.struct
class G2Smem:
    activation: fx.Array[fx.Int32, 4 * 320, 16]
    reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]


@traced
def _shared_output(mid, weight, gate, out, tokens, intermediate):
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    tile = fx.block_idx.x
    sm = fx.SharedAllocator().allocate(G2Smem).peek()
    if tid < (intermediate + 7) // 8:
        for t in range_constexpr(tokens):
            value = fx.Vector(
                bo.buffer_load(
                    rsrc(mid, nbytes=tokens * intermediate * 2),
                    t * (intermediate // 2) + tid * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            fx.ptr_store(
                value,
                sm.activation.ptr + t * ((intermediate + 31) // 32 * 16) + tid * 4,
            )
    if (tid >= (intermediate + 7) // 8) & (tid < (intermediate + 31) // 32 * 4):
        for t in range_constexpr(tokens):
            fx.ptr_store(
                fx.Vector.filled(4, 0, fx.Int32),
                sm.activation.ptr + t * ((intermediate + 31) // 32 * 16) + tid * 4,
            )
    gpu.barrier()
    token = fx.min(lane % 16, tokens - 1)
    row = tile * 16 + lane % 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for turn in range_constexpr(((intermediate + 31) // 32 + 7) // 8):
        step = wave + 8 * turn
        at = fx.min(step, (intermediate + 31) // 32 - 1) * 32 + 8 * (lane // 16)
        w = fx.Vector(
            bo.buffer_load(
                rsrc(weight, nbytes=2560 * intermediate * 2),
                row * (intermediate // 2) + at // 2,
                vec_width=4,
                dtype=T.i32,
                mask=at < intermediate,
            )
        ).bitcast(fx.BFloat16)
        x = fx.Vector(
            fx.ptr_load(
                sm.activation.ptr + token * ((intermediate + 31) // 32 * 16) + at // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        block = mfma_bf16(w, x, fx.Vector.filled(4, 0.0, fx.Float32))
        acc = acc + block * (step < (intermediate + 31) // 32).select(
            fx.Float32(1.0), fx.Float32(0.0)
        )
    fx.ptr_store(acc, sm.reduction.ptr + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < 16 * tokens:
        t = tid // 16
        value = bf16_round(row_sum(sm.reduction.ptr, tid % 16, t))
        shared = bf16_round(
            fx.Float32(bo.buffer_load(rsrc(gate), t, vec_width=1, dtype=T.f32))
        )
        bo.buffer_store(
            (value * shared).to(fx.BFloat16),
            rsrc(out),
            t * 2560 + tile * 16 + tid % 16,
        )


def build_g2_mid_quant(tokens, tp=8):
    """Shared output and routed MidQ retain independent task owners."""
    intermediate = Dims(tp).shared_intermediate
    layout, stride = finish_scratch_layout(), finish_scratch_bytes()
    name = kernel_symbol("qwen_batched_shared_g2_mid_quant", b=tokens, i=intermediate)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(mid: Int64, weight: Int64, gate: Int64, out: Int64, scratch: Int64):
        if fx.block_idx.x < 160:
            _shared_output(mid, weight, gate, out, tokens, intermediate)
        else:
            task = fx.block_idx.x - 160
            token = fx.Int64(task // 50)
            slot, group = (task % 50) // 5, task % 5
            c = {
                "tid": fx.thread_idx.x,
                "poll": lambda specs: None,
                "put": lambda *args: None,
            }
            for field, (offset, _) in layout.items():
                c[field] = scratch + token * stride + fx.Int64(offset)
            if _ld_f32(c["route_weights"], slot, CM_DEV) > fx.Float32(0):
                _quant_routed_mid(c, slot, group)

    @launcher(name)
    def launch(
        mid: Int64,
        weight: Int64,
        gate: Int64,
        out: Int64,
        scratch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(mid, weight, gate, out, scratch).launch(
            grid=(160 + tokens * 50,), block=(512,), stream=stream
        )

    return launch
