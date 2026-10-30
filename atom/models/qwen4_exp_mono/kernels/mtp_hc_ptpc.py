# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HC up/mix -> BF16 and per-token FP8, in one resident launch.

The 160 N16 producers publish tagged BF16 pairs. One consumer CTA per row
reads all 2560 elements before computing the row scale; group scales are not
compatible with the Native PTPC GEMMs. All producers run on every invocation
on the same stream, including graph replay and padded rows.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.kernels.batch_hc_front import (
    load_payload,
    store_payload,
)
from atom.models.qwen4_exp_mono.shapes import MTP_TARGET_ROWS
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_hc import emit_up, smem_type
from atom.models.qwen4_exp_mono.stages.hc import _st_bf
from atom.mono.device.mx import FP8_MAX, clamp_fp8
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    block_max,
    fp8_pack4,
    kernel_symbol,
    launcher,
    rsrc,
)
from atom.mono.runtime.abi import KernelAbi

ABI = KernelAbi(
    ("part", "normed", "weight", "raw", "stride", "mixed", "q", "scale", "payload")
)
_STREAM = fx.Stream(None)


@cache
def build(tokens):
    assert tokens in MTP_TARGET_ROWS
    hc_type = smem_type(tokens)

    @fx.struct
    class QuantSmem:
        reduction: fx.Array[fx.Float32, 8, 16]

    @fx.union
    class Smem:
        hc: hc_type
        quant: QuantSmem

    name = kernel_symbol("qwen_mtp_hc_ptpc", b=tokens)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        mixed: Int64,
        q: Int64,
        scale: Int64,
        payload: Int64,
    ):
        sm = fx.SharedAllocator().allocate(Smem)
        task, tid = fx.block_idx.x, fx.thread_idx.x
        epoch = (
            fx.Int32(
                bo.buffer_load(
                    rsrc(payload),
                    task * 16 + 1,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            + 1
        )

        def store(col, value):
            _st_bf(mixed, col, value)
            store_payload(payload, col, value, epoch)

        emit_up(
            part,
            normed,
            weight,
            raw,
            stride,
            sm.hc.peek(),
            tokens,
            True,
            store,
            down_parts=1,
        )
        gpu.barrier()
        quant_sm = sm.quant.peek()
        if task < tokens:
            words = load_payload(payload, task * 1280 + tid * 2, epoch, 2)
            tail = fx.Vector.filled(2, 0, fx.Int32)
            if tid < 128:
                tail = load_payload(payload, task * 1280 + 1024 + tid * 2, epoch, 2)
            values = [
                bf_lo(words[0]),
                bf_hi(words[0]),
                bf_lo(words[1]),
                bf_hi(words[1]),
            ]
            extra = [bf_lo(tail[0]), bf_hi(tail[0]), bf_lo(tail[1]), bf_hi(tail[1])]
            maximum = fx.Float32(0.0)
            for value in values + extra:
                maximum = fx.max(maximum, fx.max(value, -value))
            maximum = block_max(maximum, tid % 64, tid // 64, quant_sm.reduction.ptr)
            dequant = maximum / fx.Float32(FP8_MAX)
            inv = (dequant > 0).select(fx.Float32(1.0) / dequant, fx.Float32(0.0))
            packed = fp8_pack4(*[clamp_fp8(v * inv) for v in values])
            bo.buffer_store(packed, rsrc(q), task * 640 + tid)
            if tid < 128:
                packed = fp8_pack4(*[clamp_fp8(v * inv) for v in extra])
                bo.buffer_store(packed, rsrc(q), task * 640 + 512 + tid)
            if tid == 0:
                bo.buffer_store(dequant, rsrc(scale), task)

    @launcher(name)
    def launch(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        mixed: Int64,
        q: Int64,
        scale: Int64,
        payload: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(part, normed, weight, raw, stride, mixed, q, scale, payload).launch(
            grid=(160,), block=(512,), stream=stream
        )

    ABI.check(kernel, launch)
    return launch
