# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen common HC+MoE finish, standalone or composed after attention."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from atom.models.qwen4_exp_mono.abi import StructuredAbi, finish_bindings
from atom.models.qwen4_exp_mono.kernels.arguments import (
    HcArgs,
    MoeArgs,
    OutputArgs,
)
from atom.models.qwen4_exp_mono.layout import finish_scratch_layout as scratch_layout
from atom.models.qwen4_exp_mono.peer_layout import peer_offsets
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.finish import (
    FinishSmem,
    distributed_finish_context,
    finish_context,
    run_finish_tp8,
)
from atom.mono.device.ops import kernel_symbol
from atom.mono.device.sync import Mailbox
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import (
    THREADS,
)


@fx.struct
class FinishArgs:
    __annotations__ = {
        "streams": Int64,
        "attn_out": Int64,
        "attn_raw": Int64,
        **HcArgs.__annotations__,
        **MoeArgs.__annotations__,
        "expert_map": Int64,
        "peer_sym": Int64,
        **OutputArgs.__annotations__,
    }


TP8_ABI = StructuredAbi(FinishArgs)


def build_finish_kernel(key: MonoBuild, abi, name, before_finish=None):
    if key.tokens != 1:
        raise ValueError("Qwen TP8 finish supports only S=1")
    keyed = key_tuple(key, SOURCES)
    layout = scratch_layout()
    region_off = peer_offsets(key.tokens, 8)["moe"]
    argument_type = abi.argument_type

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(a: argument_type, scratch: Int64, layer: Int32):
        lds = fx.SharedAllocator().allocate(FinishSmem).peek()
        mailbox = Mailbox(layer)
        args = finish_bindings(a)

        c = finish_context(lds, mailbox, layout, scratch, args)
        c["norm_cache"] = True
        distributed_finish_context(c, a, key.dims, region_off, layer)

        if before_finish is not None:
            before_finish(c, a, layer, fx.block_idx.x)
        run_finish_tp8(c, fx.block_idx.x)
        _ = keyed

    launch = abi.launcher(kernel, name, keyed)
    return launch


def build_finish(key: MonoBuild):
    """Build TP8 HC/MoE finish; TP1 composes these stages into its layer."""
    if key.tp != 8:
        raise ValueError("standalone Qwen finish requires TP8")
    kind = tuple(type(key.kind)).index(key.kind)
    name = kernel_symbol("qwen_finish_tp8", s=key.tokens, k=kind)
    return build_finish_kernel(key, TP8_ABI, name)
