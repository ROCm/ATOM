# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen GDN attention and HC+MoE finish composed into one layer kernel."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64

from atom.models.qwen4_exp_mono.abi import (
    StructuredAbi,
    finish_bindings,
)
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels.arguments import (
    HcArgs,
    MoeArgs,
    OutputArgs,
)
from atom.models.qwen4_exp_mono.kernels.gdn_tp1 import GdnSmem, gdn_bindings, run_gdn
from atom.models.qwen4_exp_mono.layout import (
    GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
    GDR_OUT_EARLY_TASKS,
    ROWS,
    finish_scratch_layout,
)
from atom.models.qwen4_exp_mono.layout import (
    gdn_scratch_layout as scratch_layout,
)
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.finish import (
    FinishSmem,
    finish_context,
    run_finish,
)
from atom.models.qwen4_exp_mono.stages.gdr_decode import create_qwen_gdr_decode
from atom.models.qwen4_exp_mono.stages.hc import (
    NORM_COLS_PER_PASS,
    NORM_EARLY_PASSES,
    combine_gdn_output_pair,
)
from atom.models.qwen4_exp_mono.stages.memory import workspace as _workspace
from atom.mono.device.ops import kernel_symbol
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import THREADS

assert GDR_OUT_EARLY_TASKS * ROWS == NORM_EARLY_PASSES * NORM_COLS_PER_PASS


@fx.struct
class GdnLayerArgs:
    __annotations__ = {
        "streams": Int64,
        "gdn_input": Int64,
        "attn_raw": Int64,
        "gdn_qkvzba": Int64,
        "gdn_conv": Int64,
        "gdn_a_log": Int64,
        "gdn_dt_bias": Int64,
        "gdn_norm": Int64,
        "gdn_out": Int64,
        "gdn_conv_state": Int64,
        "gdn_recurrent_state": Int64,
        "gdn_state_read": Int64,
        "gdn_state_write": Int64,
        **HcArgs.__annotations__,
        **MoeArgs.__annotations__,
        **OutputArgs.__annotations__,
    }


ABI = StructuredAbi(GdnLayerArgs)


@fx.union
class LayerSmem:
    gdn: GdnSmem
    finish: FinishSmem


def build_gdn_layer(key: MonoBuild):
    """Build TP1 GDN -> HC -> MoE for the single-token decode."""

    if key.tokens != 1:
        raise ValueError("Qwen GDN layer currently has S=1 specialization")
    keyed = key_tuple(key, SOURCES)
    kind = tuple(type(key.kind)).index(key.kind)
    finish_layout = finish_scratch_layout()
    layer_layout = scratch_layout()
    name = kernel_symbol("qwen_gdn_layer", s=key.tokens, k=kind)
    smem = LayerSmem

    emit_gdr_decode_task = create_qwen_gdr_decode(Dims(1))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(a: GdnLayerArgs, scratch: Int64, layer: Int32):
        bid = fx.block_idx.x
        halves = fx.SharedAllocator().allocate(smem)
        mailbox = Mailbox(layer)

        ready = _workspace(scratch, layer_layout, "gdn_ready")
        output = _workspace(scratch, layer_layout, "gdn_out")
        gargs = gdn_bindings(a, scratch, layer_layout)
        args = finish_bindings(a, attn_out=output)
        c = finish_context(
            halves.finish.peek(),
            mailbox,
            finish_layout,
            scratch,
            args,
        )
        c["attn_group_ready"] = sreg(
            ready,
            GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
            "gdn_full_output_group_ready",
        )
        c["attn_precombined"] = True
        c["norm_cache"] = True
        c["gdn_up_prefetch"] = True
        c["gdn_down_prefetch"] = True
        c["parallel_ready"] = True
        c["wave_experts"] = True
        c["router_prefetch"] = True
        run_gdn(
            emit_gdr_decode_task,
            halves.gdn.peek(),
            mailbox,
            ready,
            gargs,
            bid,
            lambda tile, row0, value0, value1: combine_gdn_output_pair(
                c, tile, row0, value0, value1
            ),
        )
        gpu.barrier()
        run_finish(c, bid)
        _ = keyed

    launch = ABI.launcher(kernel, name, keyed)
    return launch
