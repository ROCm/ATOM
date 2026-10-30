# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen QSA K1: attention HC front, projections, norms, RoPE and cache writes.

The launch ends at the same durable seam as DeepSeek-V4.1 ``attn_pre``:
rotated query/gate/index-query plus the settled HC streams.  Main K/V, raw
index K and completed compressed index groups are written directly to their
paged caches, so K2 never repeats K1 work or reads a private projection ABI.

The first specialization is the production text-only TP1/S=1 build.  QKV and
index projections consume the checkpoint's contiguous BF16 weights directly.
Multimodal rows continue to fall back before this kernel is selected.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen4_exp_mono.abi import (
    StructuredAbi,
    argument_values,
)
from atom.models.qwen4_exp_mono.kernels.arguments import (
    HcArgs,
    QsaProjectionArgs,
)
from atom.models.qwen4_exp_mono.layout import (
    qsa_pre_scratch_layout as scratch_layout,
)
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.shapes import (
    HIDDEN,
    QSA_HEADS,
    QSA_INDEX_QK_SIZE,
    QSA_KV_HEADS,
    QSA_KV_SIZE,
    QSA_MROPE_INTERLEAVED,
    QSA_MROPE_SECTION,
    QSA_Q_SIZE,
    QSA_QKV_SIZE,
    QSA_ROTARY_DIM,
)
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.finish import (
    FinishSmem,
    finish_context,
)
from atom.models.qwen4_exp_mono.stages.hc import run_hc_finish
from atom.models.qwen4_exp_mono.stages.memory import workspace as _workspace
from atom.models.qwen4_exp_mono.stages.qsa_projection import (
    project_qkv,
    project_tile,
    run_epilogue,
)
from atom.mono.device.ops import (
    CM_DEV,
    kernel_symbol,
    rsrc,
    traced,
)
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import (
    BLOCKS,
    THREADS,
    first_task,
)

ROWS = 16
K_STEP = 32
QKV_GROUPS = QSA_QKV_SIZE // ROWS
INDEX_GROUPS = QSA_INDEX_QK_SIZE // ROWS
GATE_GROUPS = QSA_Q_SIZE // ROWS
Q_GROUP0 = GATE_GROUPS
K_GROUP0 = Q_GROUP0 + QSA_Q_SIZE // ROWS
V_GROUP0 = K_GROUP0 + QSA_KV_SIZE // ROWS
TOTAL_PROJECTION_GROUPS = QKV_GROUPS + INDEX_GROUPS

assert QSA_QKV_SIZE % ROWS == 0
assert QSA_INDEX_QK_SIZE % ROWS == 0
assert QSA_ROTARY_DIM % 2 == 0
assert sum(QSA_MROPE_SECTION) == QSA_ROTARY_DIM // 2
assert QSA_MROPE_INTERLEAVED


@fx.struct
class QsaPreArgs:
    __annotations__ = {
        "streams": Int64,
        "pending_out": Int64,
        "pending_raw": Int64,
        **HcArgs.__annotations__,
        **QsaProjectionArgs.__annotations__,
        "streams_out": Int64,
        "attn_raw_out": Int64,
        "query_out": Int64,
        "gate_out": Int64,
        "index_query_out": Int64,
    }


ABI = StructuredAbi(QsaPreArgs)


@traced
def _stage_projection_input(c):
    """Wait for HC front once per CTA and stage its BF16 row in LDS."""

    c["poll"]([(c["up_group_ready"], group, 1) for group in range(2)])
    if c["tid"] < HIDDEN // 8:
        at = c["tid"] * 8
        value = fx.Vector(
            bo.buffer_load(
                rsrc(c["qsa_mixed"]),
                at // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(value, c["x"] + at // 2)
    gpu.barrier()


@traced
def _run_projection_tasks(c, bid):
    _stage_projection_input(c)
    for turn in range_constexpr(4):
        task = bid + turn * BLOCKS
        if task < QKV_GROUPS:
            project_qkv(c, task, GATE_GROUPS)
        elif task < TOTAL_PROJECTION_GROUPS:
            group = task - QKV_GROUPS
            project_tile(
                c,
                group,
                c["args"]["index_weight"],
                c["qsa_index"],
                c["index_ready"],
            )


def _compose_tp1(a, scratch, layer, bid, lds, mailbox, layout):
    args = {
        **argument_values(a),
        "attn_out": a.pending_out,
        "attn_raw": a.pending_raw,
        "pending_out": _workspace(scratch, layout, "qsa_mixed"),
        "pending_raw": a.attn_raw_out,
    }
    c = finish_context(lds.peek(), mailbox, layout, scratch, args)
    c["qsa_mixed"] = _workspace(scratch, layout, "qsa_mixed")
    c["qsa_qkv"] = _workspace(scratch, layout, "qsa_qkv")
    c["qsa_index"] = _workspace(scratch, layout, "qsa_index")
    c["qkv_ready"] = sreg(scratch, layout["qsa_qkv_ready"][0], "qsa_qkv_ready")
    c["index_ready"] = sreg(scratch, layout["qsa_index_ready"][0], "qsa_index_ready")

    run_hc_finish(c, bid)
    _run_projection_tasks(c, bid)
    run_epilogue(c, first_task(bid, 0), QSA_HEADS, QSA_KV_HEADS)


def build_projection_kernel(key, abi, stem, smem, layout, compose):
    """One QSA K1 launch contract with explicit TP input and placement policies."""
    if key.tokens != 1:
        raise ValueError("Qwen QSA K1 currently has S=1 specialization")
    keyed = key_tuple(key, SOURCES)
    name = kernel_symbol(stem, s=key.tokens)
    argument_type = abi.argument_type

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(a: argument_type, scratch: Int64, layer: Int32):
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(smem)
        mailbox = Mailbox(layer)
        compose(a, scratch, layer, bid, lds, mailbox, layout)
        _ = keyed

    return abi.launcher(kernel, name, keyed)


def build_qsa_pre(key: MonoBuild):
    return build_projection_kernel(
        key, ABI, "qwen_qsa_pre", FinishSmem, scratch_layout(), _compose_tp1
    )
