# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 QSA K1 after the native attention hyper-connection front.

Qwen has 24 query heads but only two KV heads.  TP8 therefore owns three
query heads per rank while four ranks replicate each physical KV head.  The
indexer remains replicated.  This launch consumes the rank-local mixed row,
computes the local QKVG and replicated index projections, normalizes and
rotates only the heads that depend on each completed projection tile, and
writes K/V/index cache rows directly.

The native HC front remains the input boundary. For short contexts, QSA K2
composes the shared attention stages and TP8 HC/MoE finish; longer contexts
use native sparse attention between Mono projection and finish.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.abi import (
    StructuredAbi,
    argument_values,
    finish_bindings,
)
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels.arguments import (
    HcArgs,
    MoeArgs,
    OutputArgs,
    QsaDecodeArgs,
    QsaProjectionArgs,
)
from atom.models.qwen4_exp_mono.kernels.qsa_post import build_post_kernel
from atom.models.qwen4_exp_mono.kernels.qsa_pre import build_projection_kernel
from atom.models.qwen4_exp_mono.layout import (
    DIRECT_ATTN_SPLITS,
    OPROJ_ROWS,
    OPROJ_TASKS,
    finish_scratch_layout,
    qsa_post_scratch_layout,
)
from atom.models.qwen4_exp_mono.layout import align as _align
from atom.models.qwen4_exp_mono.peer_layout import PART_COLS, PARTS, peer_offsets
from atom.models.qwen4_exp_mono.plan import LayerKind, MonoBuild
from atom.models.qwen4_exp_mono.shapes import (
    HIDDEN,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    QSA_INDEX_QK_SIZE,
)
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages import finish, qsa_attention
from atom.models.qwen4_exp_mono.stages.collective import (
    publish_pair,
    pull_sum_pair,
)
from atom.models.qwen4_exp_mono.stages.hc import (
    combine_gdn_output_pair_raw,
    poll_ready_tiles,
)
from atom.models.qwen4_exp_mono.stages.memory import store_bf16 as _store_bf
from atom.models.qwen4_exp_mono.stages.memory import workspace as _workspace
from atom.models.qwen4_exp_mono.stages.projection import (
    output_accumulator,
    output_weights,
)
from atom.models.qwen4_exp_mono.stages.qsa_projection import (
    _projection_accumulator,
    project_qkv,
    run_epilogue,
)
from atom.mono.device.ops import (
    CM_DEV,
    bf16_round,
    kernel_symbol,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
)
from atom.mono.plan.layout import PAIR_BYTES

ROWS = 16
K_STEP = 32
K_STEPS_PER_WAVE = HIDDEN // (K_STEP * WAVES)

_DIMS = Dims(8)
Q_HEADS = _DIMS.qsa_heads
KV_HEADS = _DIMS.qsa_kv_heads
Q_SIZE = _DIMS.qsa_q_size
KV_SIZE = _DIMS.qsa_kv_size
QKV_ROWS = _DIMS.qsa_qkv_size
INDEX_Q_SIZE = QSA_INDEX_HEADS * QSA_INDEX_HEAD_DIM
INDEX_ROWS = QSA_INDEX_QK_SIZE

GATE_GROUPS = Q_SIZE // ROWS
Q_GROUP0 = GATE_GROUPS
K_GROUP0 = Q_GROUP0 + Q_SIZE // ROWS
V_GROUP0 = K_GROUP0 + KV_SIZE // ROWS
QKV_GROUPS = QKV_ROWS // ROWS
INDEX_GROUPS = INDEX_ROWS // ROWS
TOTAL_PROJECTION_TASKS = QKV_GROUPS + INDEX_GROUPS
EPILOGUE_TASKS = Q_HEADS + 2 * KV_HEADS + QSA_INDEX_HEADS + 1

assert (Q_HEADS, KV_HEADS, _DIMS.qsa_kv_replicas) == (3, 1, 4)
assert (Q_SIZE, KV_SIZE, QKV_ROWS) == (768, 256, 2048)
assert (INDEX_Q_SIZE, INDEX_ROWS) == (512, 640)
assert TOTAL_PROJECTION_TASKS == 168
assert BLOCKS == 256 and THREADS == 512 and WAVES == 8
assert K_STEPS_PER_WAVE * K_STEP * WAVES == HIDDEN


@fx.struct
class QsaTp8Args:
    __annotations__ = {
        "mixed": Int64,
        **QsaProjectionArgs.__annotations__,
        "query_out": Int64,
        "gate_out": Int64,
        "index_query_out": Int64,
    }


ABI = StructuredAbi(QsaTp8Args)


@fx.struct
class QsaTp8K2Args:
    __annotations__ = {
        "peers": Int64,
        "streams": Int64,
        "attn_raw": Int64,
        **QsaDecodeArgs.__annotations__,
        **HcArgs.__annotations__,
        **MoeArgs.__annotations__,
        "expert_map": Int64,
        "peer_sym": Int64,
        "attn_out": Int64,
        **OutputArgs.__annotations__,
    }


K2_ABI = StructuredAbi(QsaTp8K2Args)


def scratch_layout() -> dict[str, tuple[int, int]]:
    offset = 0
    out = {}
    for name, pairs in (
        ("qsa_tp8_qkv_ready", QKV_GROUPS),
        ("qsa_tp8_index_ready", INDEX_GROUPS),
    ):
        offset = _align(offset)
        size = pairs * PAIR_BYTES
        out[name] = (offset, size)
        offset += size
    for name, size in (
        ("qsa_tp8_qkv", (Q_SIZE + 2 * KV_SIZE) * 2),
        ("qsa_tp8_index", INDEX_ROWS * 2),
    ):
        offset = _align(offset)
        out[name] = (offset, size)
        offset += size
    return out


@fx.struct
class QsaTp8Smem:
    activation: fx.Array[fx.Int32, HIDDEN // 2, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


@fx.union
class QsaTp8K2Smem:
    selection: qsa_attention.SelectionSmem
    attention: qsa_attention.AttentionSmem
    finish: finish.FinishSmem


@traced
def _wait_k2_attention_ready(poll, ready, tid):
    if tid == 0:
        poll([(ready, 0, 1)])


@traced
def _stage_k2_oproj_peer(c, mailbox, peer_sym, region_off, layer, task):
    """One N16 local o_proj tile published as this rank's TP partial."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    acc = output_accumulator(
        c["gated"], 0, lane, wave, c["weights"], cache_modifier=CM_DEV
    )
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < OPROJ_ROWS // 2:
        row0 = 2 * tid
        col = task * OPROJ_ROWS + row0
        publish_pair(
            mailbox,
            peer_sym,
            region_off,
            "attention",
            layer,
            1,
            0,
            col,
            bf16_round(row_sum(c["red"], row0, 0)),
            bf16_round(row_sum(c["red"], row0 + 1, 0)),
        )


@traced
def _stage_k2_peer_consume(c, mailbox, peers, attention_region_off, layer, part):
    """Reduce one N32 attention part and precombine it into all HC streams."""

    tid = c["tid"]
    if tid < PART_COLS // 2:
        col = part * PART_COLS + 2 * tid
        acc0, acc1 = pull_sum_pair(
            mailbox,
            peers,
            attention_region_off,
            "attention",
            layer,
            1,
            0,
            col,
            8,
        )
        value0 = acc0.to(fx.BFloat16)
        value1 = acc1.to(fx.BFloat16)
        _store_bf(c["attn_out"], col, value0, CM_DEV)
        _store_bf(c["attn_out"], col + 1, value1, CM_DEV)
        combine_gdn_output_pair_raw(
            c["streams"],
            c["attn_raw"],
            c["streams_out"],
            col // OPROJ_ROWS,
            col % OPROJ_ROWS,
            fx.Float32(value0),
            fx.Float32(value1),
        )


@traced
def _publish_k2_part_ready(put, ready, part, tid):
    gpu.barrier()
    if tid == 0:
        put(ready, 2 * part, 1)
        put(ready, 2 * part + 1, 1)


@traced
def _stage_input(c):
    tid = c["tid"]
    if tid < HIDDEN // 8:
        at = tid * 8
        value = fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["mixed"], nbytes=HIDDEN * 2),
                at // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(value, c["x"] + at // 2)
    gpu.barrier()


@traced
def _stage_index_projection(c, group):
    _stage_input(c)
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    acc = _projection_accumulator(c, c["args"]["index_weight"], group)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        _store_bf(
            c["qsa_index"],
            group * ROWS + tid,
            bf16_round(row_sum(c["red"], tid, 0)),
        )
    publish(c["mailbox"].put, c["index_ready"], group, 1, tid == 0)


@traced
def _compose_tp8(a, scratch, layer, bid, storage, mailbox, layout):
    tid = fx.thread_idx.x
    lds = storage.peek()
    c = {
        "put": mailbox.put,
        "tid": tid,
        "lane": tid % 64,
        "wave": tid // 64,
        "x": lds.activation.ptr,
        "red": lds.reduction.ptr,
        "mailbox": mailbox,
        "poll": mailbox.poll,
        "tp": 8,
        "cache_read": CM_DEV,
        "qkv_ready": sreg(
            scratch,
            layout["qsa_tp8_qkv_ready"][0],
            "qsa_tp8_qkv_ready",
        ),
        "index_ready": sreg(
            scratch,
            layout["qsa_tp8_index_ready"][0],
            "qsa_tp8_index_ready",
        ),
        "qsa_qkv": _workspace(scratch, layout, "qsa_tp8_qkv"),
        "qsa_index": _workspace(scratch, layout, "qsa_tp8_index"),
        "args": argument_values(a),
    }
    if bid < QKV_GROUPS:
        _stage_input(c)
        project_qkv(c, bid, GATE_GROUPS, _store_bf)
    elif bid < TOTAL_PROJECTION_TASKS:
        _stage_index_projection(c, bid - QKV_GROUPS)
    elif bid < TOTAL_PROJECTION_TASKS + EPILOGUE_TASKS:
        run_epilogue(c, bid - TOTAL_PROJECTION_TASKS, Q_HEADS, KV_HEADS)


def build_qsa_k1(key: MonoBuild):
    if key.tp != 8 or key.tokens != 1 or key.kind is not LayerKind.QSA:
        raise ValueError("Qwen TP8 QSA K1 requires the TP8/S1/QSA build")
    return build_projection_kernel(
        key, ABI, "qwen_tp8_qsa_k1", QsaTp8Smem, scratch_layout(), _compose_tp8
    )


def k2_phase_scratch_layout() -> dict[str, tuple[int, int]]:
    """TP8-local QSA K2 scratch used before peer reduction and common finish."""

    return qsa_post_scratch_layout(
        qsa_heads=Q_HEADS,
        qsa_kv_heads=KV_HEADS,
        qsa_q_size=Q_SIZE,
    )


def build_qsa_k2(
    key: MonoBuild,
):
    """Build TP8 QSA attention, peer hand-off and common finish as one K2."""

    if key.tokens != 1 or key.tp != 8 or key.kind is not LayerKind.QSA:
        raise ValueError("Qwen TP8 QSA K2 supports only S=1 TP8 QSA")
    task_splits = DIRECT_ATTN_SPLITS
    attention_tasks = KV_HEADS * task_splits
    layout = k2_phase_scratch_layout()
    finish_layout = finish_scratch_layout()
    attention_region_off = peer_offsets(1, 8)["attention"]
    moe_region_off = peer_offsets(1, 8)["moe"]
    keyed = (
        *key_tuple(key, SOURCES),
        ("phase", "qsa_k2_finish"),
        ("direct", True),
        ("direct_splits", task_splits),
    )
    stem = "qwen_tp8_qsa_k2_direct"
    name = kernel_symbol(stem, s=1)

    @traced
    def compose(a, scratch, layer, bid, halves, mailbox):
        args = finish_bindings(a)
        finish_c = finish.finish_context(
            halves.finish.peek(),
            mailbox,
            finish_layout,
            scratch,
            args,
        )
        qsa_c = qsa_attention.attention_context(
            halves.attention.peek(), mailbox, scratch, layout, args, _DIMS
        )

        gpu.barrier()
        for task in range(first_task(bid, 0), attention_tasks, BLOCKS):
            qsa_attention._stage_attention_split(qsa_c, task, True, task_splits)
        head = first_task(bid, 0)
        if head < Q_HEADS:
            qsa_attention._stage_attention_reduce(qsa_c, head, task_splits)
        qsa_attention.publish_attention_ready(qsa_c, bid, Q_HEADS)
        weights = output_weights(
            a.o_proj,
            fx.min(bid, OPROJ_TASKS - 1),
            0,
            Q_SIZE // (32 * WAVES),
            qsa_c["lane"],
            qsa_c["wave"],
            width=Q_SIZE,
        )
        _wait_k2_attention_ready(mailbox.poll, qsa_c["attention_ready"], qsa_c["tid"])
        gpu.barrier()
        oproj_c = {
            "tid": qsa_c["tid"],
            "lane": qsa_c["lane"],
            "wave": qsa_c["wave"],
            "red": qsa_c["red"],
            "gated": qsa_c["gated"],
            "weights": weights,
        }
        for task in range(first_task(bid, 0), OPROJ_TASKS, BLOCKS):
            _stage_k2_oproj_peer(
                oproj_c,
                mailbox,
                a.peer_sym,
                attention_region_off,
                layer,
                task,
            )
        peer_c = {
            **argument_values(a),
            "tid": qsa_c["tid"],
        }
        for part in range(first_task(bid, 0), PARTS, BLOCKS):
            _stage_k2_peer_consume(
                peer_c, mailbox, a.peers, attention_region_off, layer, part
            )
            _publish_k2_part_ready(
                mailbox.put, qsa_c["oproj_ready"], part, qsa_c["tid"]
            )
        for owner, start, count, group in (
            (PARTS, 0, 96, 0),
            (PARTS + 1, 96, HIDDEN // 16 - 96, 1),
        ):
            if bid == owner:
                poll_ready_tiles(
                    {**qsa_c, "parallel_ready": True},
                    qsa_c["oproj_ready"],
                    start,
                    count,
                )
                publish(
                    mailbox.put,
                    qsa_c["output_group_ready"],
                    group,
                    1,
                    qsa_c["tid"] == 0,
                )

        finish_c["attn_group_ready"] = qsa_c["output_group_ready"]
        finish_c["attn_precombined"] = True
        finish_c["norm_cache"] = True
        finish.distributed_finish_context(finish_c, a, key.dims, moe_region_off, layer)
        finish.run_finish_tp8(finish_c, bid)

    return build_post_kernel(K2_ABI, name, keyed, QsaTp8K2Smem, compose)
