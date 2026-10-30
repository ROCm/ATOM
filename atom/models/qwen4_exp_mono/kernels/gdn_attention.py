# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP1 S1 GDN composition ending at the native MXFP4 MoE seam."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from atom.models.qwen4_exp_mono.abi import StructuredAbi
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels.gdn_tp1 import GdnSmem, gdn_bindings, run_gdn
from atom.models.qwen4_exp_mono.layout import gdn_scratch_layout
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.gdr_decode import create_qwen_gdr_decode
from atom.models.qwen4_exp_mono.stages.memory import workspace
from atom.mono.device.ops import kernel_symbol
from atom.mono.device.sync import Mailbox
from atom.mono.plan.execution import THREADS


@fx.struct
class GdnAttentionArgs:
    __annotations__ = {
        name: Int64
        for name in (
            "gdn_input",
            "gdn_qkvzba",
            "gdn_conv",
            "gdn_a_log",
            "gdn_dt_bias",
            "gdn_norm",
            "gdn_out",
            "gdn_conv_state",
            "gdn_recurrent_state",
            "gdn_state_read",
            "gdn_state_write",
        )
    }


ABI = StructuredAbi(GdnAttentionArgs)


def build():
    layout = gdn_scratch_layout()
    name = kernel_symbol("qwen_gdn_native_mxfp4_seam", s=1)
    emit = create_qwen_gdr_decode(Dims(1))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(a: GdnAttentionArgs, scratch: Int64, layer: Int32):
        sm = fx.SharedAllocator().allocate(GdnSmem).peek()
        run_gdn(
            emit,
            sm,
            Mailbox(layer),
            workspace(scratch, layout, "gdn_ready"),
            gdn_bindings(a, scratch, layout),
            fx.block_idx.x,
            lambda tile, row, value0, value1: None,
        )

    return ABI.launcher(kernel, name, SOURCES)
