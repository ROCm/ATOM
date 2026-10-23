# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""S=1 dynamic FP8 activation quantization inside the persistent Qwen K2.

The routed experts consume AITER's ``per_1x128`` contract: twenty contiguous
groups cover H=2560, every group stores 128 E4M3 values and one continuous
FP32 dequantization scale.  A quant CTA waits only for the eight HC N16 tiles
that produce its group, so groups can become ready while later HC output tiles
are still running.  The packed values and scale are plain device-scope scratch;
the tagged ``xq_ready`` pair publishes them to G1 consumers.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.layout import (
    HC_OUT_COLS,
    XQ_GROUP,
    XQ_GROUPS,
)
from atom.models.qwen4_exp_mono.stages.hc import _st_f32
from atom.mono.device.mx import FP8_MAX, clamp_fp8
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import first_task

XQ0 = 0
XQ_THREADS = XQ_GROUP // 4
XQ_HC_TILES = XQ_GROUP // HC_OUT_COLS


def quantize_values(values):
    amax = fx.Float32(1e-10)
    for value in values:
        amax = fx.max(amax, abs(value))
    amax = butterfly(amax, (16, 8, 4, 2, 1), fx.max)
    scale = amax * fx.Float32(1.0 / FP8_MAX)
    inv = fx.Float32(1.0) / scale

    return fp8_pack4(*[clamp_fp8(value * inv) for value in values]), scale


@traced
def _stage_xq(c, group):
    """Quantize one contiguous 128-value group with one half wave."""

    tid = c["tid"]
    c["poll"](
        [(c["up_ready"], group * XQ_HC_TILES + tile, 1) for tile in range(XQ_HC_TILES)]
    )

    words = fx.Vector.filled(2, 0, fx.Int32)
    if tid < XQ_THREADS:
        words = fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["pending_out"]),
                group * (XQ_GROUP // 2) + tid * 2,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
    values = [bf_lo(words[0]), bf_hi(words[0]), bf_lo(words[1]), bf_hi(words[1])]
    packed, scale = quantize_values(values)

    if tid < XQ_THREADS:
        bo.buffer_store(
            packed,
            rsrc(c["xq"]),
            group * XQ_THREADS + tid,
            cache_modifier=CM_DEV,
        )
    if tid == 0:
        _st_f32(c["xq_scale"], group, scale, CM_DEV)
    publish(c["put"], c["xq_ready"], group, 1, tid == 0)


@traced
def run_input_quant(c, bid):
    """Emit the twenty independent per-1x128 quant tasks."""

    task = first_task(bid, XQ0)
    if task < XQ_GROUPS:
        _stage_xq(c, task)
