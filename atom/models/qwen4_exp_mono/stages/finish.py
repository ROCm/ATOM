# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared HC/MoE stage composition, scratch layout and mailbox bindings."""

import flydsl.expr as fx

from atom.models.qwen4_exp_mono.shapes import (
    HIDDEN,
)
from atom.models.qwen4_exp_mono.stages.g1 import run_g1, run_g1_tp8
from atom.models.qwen4_exp_mono.stages.g2 import run_g2, run_wave_expert_g2
from atom.models.qwen4_exp_mono.stages.hc import (
    run_hc_finish,
)
from atom.models.qwen4_exp_mono.stages.quant import run_input_quant
from atom.models.qwen4_exp_mono.stages.router import run_router
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import (
    WAVES,
)


@fx.struct
class FinishSmem:
    red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]
    up: fx.Array[fx.Float32, 4 * 16, 16]
    x: fx.Array[fx.Int32, HIDDEN // 2, 16]
    queue: fx.Array[fx.Int32, 1, 16]


def finish_context(lds, mailbox, layout, scratch, args):
    """Stage context shared by the standalone and composed kernels."""

    tid = fx.thread_idx.x
    c = {
        "tid": tid,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "up": lds.up.ptr,
        "x": lds.x.ptr,
        "queue": lds.queue.ptr,
        "mailbox": mailbox,
        "put": mailbox.put,
        "poll": mailbox.poll,
        "tag": mailbox.tag,
        "scratch": scratch,
        "layout": layout,
        "args": args,
    }
    for name, (off, _) in layout.items():
        key = name.removeprefix("hc_")
        c[key] = (
            sreg(scratch, off, name)
            if name.endswith("_ready")
            else scratch + fx.Int64(off)
        )
    return c


def run_finish(c, bid, g1=run_g1):
    """Common stage order; the build chooses the TP-specific G1 schedule."""

    run_hc_finish(c, bid)
    run_input_quant(c, bid)
    run_router(c, bid)
    g1(c, bid)
    if c.get("wave_experts", False):
        run_wave_expert_g2(c, bid)
    else:
        run_g2(c, bid)


def run_finish_tp8(c, bid):
    """Run the same finish with rank-local expert ownership."""
    run_finish(c, bid, run_g1_tp8)


def distributed_finish_context(c, a, dims, region_off, layer):
    """Attach the TP8 ownership/publication contract at its composition seam."""
    c.update(
        expert_owned=True,
        hc_compact_join=True,
        shared_intermediate=dims.shared_intermediate,
        publish_partial=True,
        peer_sym=a.peer_sym,
        peer_region_off=region_off,
        layer=layer,
    )
    return c
