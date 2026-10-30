# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL emitters and launchers for Qwen's deferred TP8 hand-off."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from atom.models.qwen4_exp_mono.peer_layout import PARTS, peer_offsets
from atom.models.qwen4_exp_mono.stages.collective import pull_part_to_bf16
from atom.mono.device.ops import kernel_symbol, launcher
from atom.mono.device.sync import Mailbox
from atom.mono.plan.execution import BLOCKS, THREADS
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

PULL_ABI = KernelAbi(("peers", "output", "layer"))


@functools.cache
def build_pull(tokens: int, tp: int):
    """Build the TP8/S1 deferred MoE peer consumer."""

    if tokens != 1 or tp != 8:
        raise ValueError("Qwen deferred pull supports only S=1 TP8")
    region_off = peer_offsets(tokens, tp)["moe"]
    name = kernel_symbol("qwen_tp_pull", s=tokens, tp=tp)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(peers: Int64, output: Int64, layer: Int32):
        task = fx.block_idx.x
        token = task // PARTS
        part = task % PARTS
        if task < PARTS:
            pull_part_to_bf16(
                Mailbox(layer),
                peers,
                output,
                region_off,
                "moe",
                layer,
                1,
                token,
                part,
                tp,
            )

    @launcher(name)
    def launch(
        peers: Int64,
        output: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        kernel(peers, output, layer).launch(
            grid=(max(BLOCKS, PARTS * tokens),), block=(THREADS,), stream=stream
        )

    PULL_ABI.check(kernel, launch)
    return launch
