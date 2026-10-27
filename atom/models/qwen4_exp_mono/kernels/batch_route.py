# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ordinary requests keep independent per-1x128 FP8 scales and values."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int64

from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.finish import FinishSmem
from atom.models.qwen4_exp_mono.stages.router import _stage_route
from atom.mono.device.ops import (
    kernel_symbol,
    launcher,
)

_STREAM = fx.Stream(None)


def build_route(tokens, expert_owned=False, row_stride=40):
    name = kernel_symbol(
        "qwen_batched_rank_exact_route_seam",
        b=tokens,
        ep=int(expert_owned),
        stride=row_stride,
    )

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        logits: Int64, ids: Int64, weights: Int64, gate: Int64, expert_map: Int64
    ):
        token = fx.Int64(fx.block_idx.x)
        sm = fx.SharedAllocator().allocate(FinishSmem).peek()
        c = {
            "tid": fx.thread_idx.x,
            "lane": fx.thread_idx.x % 64,
            "wave": fx.thread_idx.x // 64,
            "red": sm.red.ptr,
            "up": sm.up.ptr,
            "router_logits": logits + token * 1026,
            "route_ids": ids + token * row_stride,
            "route_weights": weights + token * row_stride,
            "shared_gate": gate + token * 4,
            "router_tile_ready": None,
            "route_ready": None,
            "poll": lambda specs: None,
            "put": lambda *args: None,
            "expert_owned": expert_owned,
            "args": {"expert_map": expert_map},
        }
        _stage_route(c)

    @launcher(name)
    def launch(
        logits: Int64,
        ids: Int64,
        weights: Int64,
        gate: Int64,
        expert_map: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(logits, ids, weights, gate, expert_map).launch(
            grid=(tokens,), block=(512,), stream=stream
        )

    return launch
