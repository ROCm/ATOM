# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent router/shared projections and input quant in one batch launch."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int64

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_projection import (
    G1Smem,
    _router_projection,
    _shared_projection,
)
from atom.models.qwen4_exp_mono.stages.quant import _stage_xq
from atom.mono.device.ops import kernel_symbol, launcher

_STREAM = fx.Stream(None)


def build(tokens, tp, row_stride):
    intermediate = Dims(tp).shared_intermediate
    shared_tiles = intermediate // 16
    quant0 = 33 + shared_tiles
    name = kernel_symbol("qwen_batched_moe_front", b=tokens, i=intermediate)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        x: Int64,
        router_weight: Int64,
        logits: Int64,
        shared_weight: Int64,
        shared_mid: Int64,
        q: Int64,
        scale: Int64,
    ):
        sm = fx.SharedAllocator().allocate(G1Smem).peek()
        task = fx.block_idx.x
        if task < 33:
            _router_projection(x, router_weight, logits, task, tokens, sm)
        elif task < quant0:
            _shared_projection(
                x, shared_weight, shared_mid, task - 33, tokens, intermediate, sm
            )
        else:
            token = fx.Int64((task - quant0) // 20)
            group = (task - quant0) % 20
            c = {
                "tid": fx.thread_idx.x,
                "args": {"pending_out": x + token * 5120},
                "xq": q + token * row_stride,
                "xq_scale": scale + token * row_stride,
                "up_ready": None,
                "xq_ready": None,
                "poll": lambda specs: None,
                "put": lambda *args: None,
            }
            _stage_xq(c, group)

    @launcher(name)
    def launch(
        x: Int64,
        router_weight: Int64,
        logits: Int64,
        shared_weight: Int64,
        shared_mid: Int64,
        q: Int64,
        scale: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(x, router_weight, logits, shared_weight, shared_mid, q, scale).launch(
            grid=(quant0 + tokens * 20,), block=(512,), stream=stream
        )

    return launch
