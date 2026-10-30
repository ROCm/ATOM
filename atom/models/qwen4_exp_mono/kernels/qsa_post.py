# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen QSA K2: selection, sparse attention, output projection and finish.

The TP1/S=1 specialization is one persistent 256-CTA launch.  Index scoring
and stable top-k publish a 2051-entry logical-token list.  Attention keeps one
16-token tile per low-register task, prunes tiles past the current context and
reduces only the live partials per query head.  The output projection
precombines the four HC streams while its BF16 output is resident and hands two
ready groups directly to ``run_finish``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
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
    QsaDecodeArgs,
)
from atom.models.qwen4_exp_mono.layout import (
    ATTN_HEAD_GROUPS,
    ATTN_SPLITS,
    DIRECT_ATTN_SPLITS,
    HEADS_PER_ATTN_GROUP,
    OPROJ_K_STEPS_PER_WAVE,
    OPROJ_ROWS,
    OPROJ_TASKS,
    finish_scratch_layout,
)
from atom.models.qwen4_exp_mono.layout import qsa_post_scratch_layout as scratch_layout
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.qsa_plan import QsaPlan
from atom.models.qwen4_exp_mono.shapes import (
    QSA_DIRECT_CONTEXT,
    QSA_HEADS,
    QSA_KV_HEADS,
    QSA_Q_SIZE,
)
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.finish import (
    FinishSmem,
    finish_context,
    run_finish,
)
from atom.models.qwen4_exp_mono.stages.hc import (
    _st_bf,
    combine_gdn_output_pair_raw,
)
from atom.models.qwen4_exp_mono.stages.memory import workspace as _workspace
from atom.models.qwen4_exp_mono.stages.projection import publish_output_groups
from atom.models.qwen4_exp_mono.stages.qsa_attention import (
    AttentionSmem,
    SelectionSmem,
    _run_selection,
    _run_selection_tree,
    _stage_attention_reduce,
    _stage_attention_split,
    _stage_index_score,
    attention_context,
    publish_attention_ready,
)
from atom.models.qwen4_exp_mono.stages.qsa_projection import oproj_accumulate
from atom.mono.device.ops import (
    CM_DEV,
    bf16_round,
    kernel_symbol,
    row_sum,
    traced,
)
from atom.mono.device.sync import Mailbox, publish, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import (
    BLOCKS,
    THREADS,
    first_task,
)


@fx.struct
class QsaPostArgs:
    __annotations__ = {
        "streams": Int64,
        "attn_raw": Int64,
        **QsaDecodeArgs.__annotations__,
        **HcArgs.__annotations__,
        **MoeArgs.__annotations__,
        **OutputArgs.__annotations__,
    }


ABI = StructuredAbi(QsaPostArgs)


@fx.union
class QsaPostSmem:
    selection: SelectionSmem
    attention: AttentionSmem
    finish: FinishSmem


@traced
def _publish_attention_groups(c, bid, owner=0):
    tid, poll, put = c["tid"], c["poll"], c["put"]
    head_ready = c["head_ready"]
    attention_ready, selection_ready = c["attention_ready"], c["selection_ready"]
    if bid == owner:
        if tid == 0:
            poll([(head_ready, head, 1) for head in range(HEADS_PER_ATTN_GROUP)])
        gpu.barrier()
        if tid == 0:
            put(attention_ready, 0, 1)
    elif bid == owner + 1:
        if tid == 0:
            poll(
                [
                    (head_ready, HEADS_PER_ATTN_GROUP + head, 1)
                    for head in range(HEADS_PER_ATTN_GROUP)
                ]
            )
        gpu.barrier()
        if tid == 0:
            put(selection_ready, 0, 1)


@traced
def _stage_oproj(c, task, direct):
    """N16 o_proj; direct attention retains its two ready-group waits."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    poll = c["poll"]
    attention_ready = c["attention_ready"]
    red, attn_out = c["red"], c["attn_out"]
    streams, attn_raw, streams_out = (
        c["args"]["streams"],
        c["args"]["attn_raw"],
        c["args"]["streams_out"],
    )
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    if const_expr(direct):
        for group in range_constexpr(ATTN_HEAD_GROUPS):
            ready = c["attention_ready"] if group == 0 else c["selection_ready"]
            if tid == 0:
                poll([(ready, 0, 1)])
            gpu.barrier()
            acc = oproj_accumulate(
                c,
                task,
                c["args"]["o_proj"],
                QSA_Q_SIZE,
                acc,
                group * HEADS_PER_ATTN_GROUP,
                (group + 1) * HEADS_PER_ATTN_GROUP,
            )
    else:
        if tid == 0:
            poll([(attention_ready, 0, 1)])
        gpu.barrier()
        acc = oproj_accumulate(
            c, task, c["args"]["o_proj"], QSA_Q_SIZE, acc, 0, OPROJ_K_STEPS_PER_WAVE
        )
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < OPROJ_ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(red, row0, 0))
        value1 = bf16_round(row_sum(red, row0 + 1, 0))
        _st_bf(attn_out, task * OPROJ_ROWS + row0, value0, CM_DEV)
        _st_bf(attn_out, task * OPROJ_ROWS + row0 + 1, value1, CM_DEV)
        combine_gdn_output_pair_raw(
            streams, attn_raw, streams_out, task, row0, value0, value1
        )
    publish(c["put"], c["oproj_ready"], task, 1, tid == 0)


def build_post_kernel(abi, name, keyed, smem, compose):
    """One QSA K2 allocation/ABI/launch lifecycle around explicit output seams."""
    argument_type = abi.argument_type

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(a: argument_type, scratch: Int64, layer: Int32):
        bid = fx.block_idx.x
        halves = fx.SharedAllocator().allocate(smem)
        mailbox = Mailbox(layer)
        compose(a, scratch, layer, bid, halves, mailbox)
        _ = keyed

    return abi.launcher(kernel, name, keyed)


def build_qsa_post(key: MonoBuild, direct: bool = False):
    """Build TP1 QSA K2 for the single-token decode."""

    if key.tokens != 1:
        raise ValueError("Qwen QSA K2 currently has S=1 specialization")
    task_splits = DIRECT_ATTN_SPLITS if direct else ATTN_SPLITS
    attention_tasks = QSA_KV_HEADS * task_splits
    keyed = (
        *key_tuple(key, SOURCES),
        ("direct", direct),
        ("direct_splits", task_splits),
    )
    plan = QsaPlan(key.max_model_len)
    layout = scratch_layout(max_model_len=key.max_model_len)
    finish_layout = finish_scratch_layout()
    publish_attention = (
        _publish_attention_groups
        if direct
        else (lambda c, bid: publish_attention_ready(c, bid, QSA_HEADS))
    )
    stem = "qwen_qsa_post"
    if direct:
        stem += f"_direct{QSA_DIRECT_CONTEXT}"
    name = kernel_symbol(stem, s=key.tokens)
    smem = QsaPostSmem

    @traced
    def compose(a, scratch, layer, bid, halves, mailbox):
        args = finish_bindings(a, attn_out=_workspace(scratch, layout, "qsa_attn_out"))
        c = finish_context(
            halves.finish.peek(),
            mailbox,
            finish_layout,
            scratch,
            args,
        )
        c.update(
            attention_context(
                halves.attention.peek(),
                mailbox,
                scratch,
                layout,
                c["args"],
                Dims(1),
                halves.selection.peek(),
            )
        )
        c["index_columns"] = plan.columns
        c["index_score_tasks"] = plan.score_tasks
        if len(plan.lists) > 1:
            c["selection_lists"] = plan.lists
            c["select_ready"] = tuple(
                sreg(
                    scratch,
                    layout[f"qsa_select_ready_{level}"][0],
                    f"qsa_select_ready_{level}",
                )
                for level in range(len(plan.lists) - 1)
            )
            c["candidates"] = tuple(
                _workspace(scratch, layout, f"qsa_candidates_{level}")
                for level in range(len(plan.lists) - 1)
            )

        if const_expr(not direct):
            for task in range(first_task(bid, 0), plan.score_tasks, BLOCKS):
                _stage_index_score(c, task)
            if const_expr(len(plan.lists) == 1):
                _run_selection(c, bid)
            else:
                _run_selection_tree(c, bid)
        gpu.barrier()
        for task in range(first_task(bid, 0), attention_tasks, BLOCKS):
            _stage_attention_split(c, task, direct, task_splits)
        head = first_task(bid, 0)
        if head < QSA_HEADS:
            _stage_attention_reduce(c, head, task_splits)
        publish_attention(c, bid)
        c["red"] = halves.finish.peek().red.ptr
        for task in range(first_task(bid, 0), OPROJ_TASKS, BLOCKS):
            _stage_oproj(c, task, direct)
        publish_output_groups(
            c["poll"],
            c["put"],
            c["oproj_ready"],
            c["output_group_ready"],
            bid,
            c["tid"],
        )
        c["attn_group_ready"] = c["output_group_ready"]
        c["attn_precombined"] = True
        c["norm_cache"] = True
        c["wave_experts"] = True
        c["router_prefetch"] = True
        gpu.barrier()
        run_finish(c, bid)

    return build_post_kernel(ABI, name, keyed, smem, compose)
