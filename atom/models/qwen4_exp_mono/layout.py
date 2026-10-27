# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cold Qwen task geometry and scratch contracts, independent of device kernels."""

import math

from atom.models.qwen4_exp_mono.qsa_plan import QsaPlan
from atom.models.qwen4_exp_mono.shapes import (
    ALIGN,
    GDN_CONV_DIM,
    GDN_PROJ_DIM,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HC,
    HC_RANK,
    HIDDEN,
    INTERMEDIATE,
    QSA_COMPRESS_RATIO,
    QSA_DIRECT_CONTEXT,
    QSA_HEAD_DIM,
    QSA_HEADS,
    QSA_INDEX_BUDGET,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_QK_SIZE,
    QSA_KV_HEADS,
    QSA_KV_SIZE,
    QSA_Q_SIZE,
    QSA_QKV_SIZE,
    ROUTER_OUTPUTS,
    TOPK,
)
from atom.mono.plan.execution import BLOCKS, WAVES
from atom.mono.plan.layout import PAIR_BYTES

HC_NOUT = HC_RANK + HC
HC_DOWN_COLS = 16
HC_DOWN_TASKS_PER_STREAM = (HC_NOUT + HC_DOWN_COLS - 1) // HC_DOWN_COLS
HC_DOWN_TASKS = HC * HC_DOWN_TASKS_PER_STREAM
HC_OUT_COLS = 16
HC_OUT_TASKS = HIDDEN // HC_OUT_COLS
ROUTER_COLS = 16
ROUTER_TASKS = (ROUTER_OUTPUTS + ROUTER_COLS - 1) // ROUTER_COLS
XQ_GROUP = 128
XQ_GROUPS = HIDDEN // XQ_GROUP
XQ_WORDS = HIDDEN // 4
G1_COLS = 16
G1_PARTS = INTERMEDIATE // G1_COLS
G1_SHARED_TASKS = G1_PARTS
G1_ROUTED_TASKS = TOPK * G1_PARTS
G1_TASKS = G1_SHARED_TASKS + G1_ROUTED_TASKS
G1_QUEUE_WORDS = 128
MIDQ_GROUP = 128
MIDQ_GROUPS = INTERMEDIATE // MIDQ_GROUP
MIDQ_WORDS = TOPK * INTERMEDIATE // 4
G2_COLS = 16
G2_TASKS = HIDDEN // G2_COLS


def align(offset: int, alignment: int = ALIGN) -> int:
    return (offset + alignment - 1) // alignment * alignment


def scratch_extent(layout: dict[str, tuple[int, int]]) -> int:
    return align(max(offset + size for offset, size in layout.values()))


QKV_GROUPS = QSA_QKV_SIZE // 16
INDEX_GROUPS = QSA_INDEX_QK_SIZE // 16
ROWS = 16
K_STEP = 32
K_STEPS_PER_WAVE = HIDDEN // (K_STEP * WAVES)
PROJECTION_TASKS = GDN_PROJ_DIM // ROWS
CONV_TASKS = GDN_CONV_DIM // ROWS
GDR_V_SPLITS = 2
GDR_TASKS = GDN_VALUE_HEADS * GDR_V_SPLITS
GDR_B_ROW = GDN_CONV_DIM + GDN_VALUE_DIM
GDR_A_ROW = GDR_B_ROW + GDN_VALUE_HEADS
GDR_GATE_TASKS = 2 * GDN_VALUE_HEADS // ROWS
GDR_FRONT_ROUNDS = (CONV_TASKS + BLOCKS - 1) // BLOCKS
GDR_Z_TASKS = GDN_VALUE_DIM // ROWS
GDR_Z_WORKERS = BLOCKS - GDR_TASKS
GDR_Z_ROUNDS = (GDR_Z_TASKS + GDR_Z_WORKERS - 1) // GDR_Z_WORKERS
GDR_NORM_HEADS_PER_TASK = WAVES
GDR_NORM_TASKS = GDN_VALUE_HEADS // GDR_NORM_HEADS_PER_TASK
GDR_OUT_TASKS = HIDDEN // ROWS
GDR_OUT_SPLITS = 2
GDR_OUT_SPLIT_K = GDN_VALUE_DIM // GDR_OUT_SPLITS
GDR_OUT_DIRECT_TASKS = 2 * GDR_OUT_TASKS - BLOCKS
GDR_OUT_SPLIT_TILES = GDR_OUT_TASKS - GDR_OUT_DIRECT_TASKS
GDR_OUT_PARTIAL_TASKS = GDR_OUT_SPLIT_TILES * GDR_OUT_SPLITS
GDR_OUT_K_STEPS_PER_WAVE = GDR_OUT_SPLIT_K // (K_STEP * WAVES)
GDR_OUT_FULL_K_STEPS_PER_WAVE = GDN_VALUE_DIM // (K_STEP * WAVES)
GDR_OUT_EARLY_TASKS = 96
GDR_OUT_READY_GROUPS = 2
GDR_OUT_EARLY_OWNER = GDR_OUT_TASKS + GDR_OUT_EARLY_TASKS - GDR_OUT_DIRECT_TASKS - 1
GDR_OUT_LATE_OWNER = BLOCKS - 1
MAILBOX_PAIR_BYTES = 8
GDR_FULL_PROJECTION_READY_OFFSET = 0
GDR_FULL_RECURRENCE_READY_OFFSET = (
    GDR_FULL_PROJECTION_READY_OFFSET + MAILBOX_PAIR_BYTES * PROJECTION_TASKS
)
GDR_FULL_HEAD_READY_OFFSET = (
    GDR_FULL_RECURRENCE_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_TASKS
)
GDR_FULL_NORM_READY_OFFSET = (
    GDR_FULL_HEAD_READY_OFFSET + MAILBOX_PAIR_BYTES * GDN_VALUE_HEADS
)
GDR_FULL_PARTIAL_READY_OFFSET = (
    GDR_FULL_NORM_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_NORM_TASKS
)
GDR_FULL_OUTPUT_READY_OFFSET = (
    GDR_FULL_PARTIAL_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_OUT_PARTIAL_TASKS
)
GDR_FULL_OUTPUT_GROUP_READY_OFFSET = (
    GDR_FULL_OUTPUT_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_OUT_TASKS
)
GDR_FULL_READY_BYTES = (
    GDR_FULL_OUTPUT_GROUP_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_OUT_READY_GROUPS
)

assert GDN_PROJ_DIM % ROWS == 0
assert GDN_CONV_DIM % ROWS == 0
assert K_STEPS_PER_WAVE * K_STEP * WAVES == HIDDEN
assert GDN_VALUE_HEADS % GDR_NORM_HEADS_PER_TASK == 0
assert GDN_VALUE_DIM % GDR_OUT_SPLITS == 0
assert GDR_OUT_K_STEPS_PER_WAVE * K_STEP * WAVES == GDR_OUT_SPLIT_K
assert GDR_OUT_FULL_K_STEPS_PER_WAVE * K_STEP * WAVES == GDN_VALUE_DIM
assert GDR_OUT_DIRECT_TASKS + GDR_OUT_PARTIAL_TASKS == BLOCKS
assert GDR_OUT_DIRECT_TASKS < GDR_OUT_EARLY_TASKS < GDR_OUT_TASKS
assert GDR_OUT_TASKS <= GDR_OUT_EARLY_OWNER < GDR_OUT_LATE_OWNER < BLOCKS


def hc_scratch_layout() -> dict[str, tuple[int, int]]:
    """HC mailbox and numerical hand-offs, in bytes."""

    out: dict[str, tuple[int, int]] = {}
    off = 0
    for name, pairs in (
        ("hc_norm_ready", HC),
        ("hc_down_ready", HC_DOWN_TASKS),
        ("hc_gate_ready", 1),
        ("hc_up_ready", HC_OUT_TASKS),
    ):
        off = align(off, PAIR_BYTES)
        size = pairs * PAIR_BYTES
        out[name] = (off, size)
        off += size
    for name, size in (
        ("hc_normed", HC * HIDDEN * 2),
        ("hc_part", HC * HC_NOUT * 4),
        ("hc_gate", HC_RANK * 2),
    ):
        off = align(off)
        out[name] = (off, size)
        off += size
    return out


def hc_scratch_bytes() -> int:
    layout = hc_scratch_layout()
    return align(max(off + size for off, size in layout.values()))


def finish_scratch_layout() -> dict[str, tuple[int, int]]:
    """All global hand-offs owned by the HC+MoE body."""

    out = hc_scratch_layout()
    off = hc_scratch_bytes()
    for name, pairs in (
        ("up_group_ready", 2),
        ("xq_ready", XQ_GROUPS),
        ("router_tile_ready", ROUTER_TASKS),
        ("route_ready", 1),
        ("g1_ready", G1_TASKS),
        ("midq_ready", TOPK * MIDQ_GROUPS),
        ("shared_mid_ready", 1),
    ):
        off = align(off, PAIR_BYTES)
        size = pairs * PAIR_BYTES
        out[name] = (off, size)
        off += size
    for name, size in (
        ("xq", XQ_WORDS * 4),
        ("xq_scale", XQ_GROUPS * 4),
        ("router_logits", ROUTER_OUTPUTS * 2),
        ("route_ids", TOPK * 4),
        ("route_weights", TOPK * 4),
        ("shared_gate", 4),
        ("g1_queue", G1_QUEUE_WORDS * 4),
        ("routed_mid", TOPK * INTERMEDIATE * 2),
        ("shared_mid", INTERMEDIATE * 2),
        ("midq", MIDQ_WORDS * 4),
        ("midq_scale", TOPK * MIDQ_GROUPS * 4),
    ):
        off = align(off)
        out[name] = (off, size)
        off += size
    return out


def finish_scratch_bytes() -> int:
    return scratch_extent(finish_scratch_layout())


def gdn_scratch_layout() -> dict[str, tuple[int, int]]:
    """Disjoint global hand-offs shared by the two halves of the layer."""

    out: dict[str, tuple[int, int]] = {}
    offset = align(finish_scratch_bytes())
    for name, size in (
        ("gdn_ready", GDR_FULL_READY_BYTES),
        ("gdn_projected", GDN_PROJ_DIM * 2),
        ("gdn_core", GDN_VALUE_DIM * 2),
        ("gdn_normed", GDN_VALUE_DIM * 2),
        ("gdn_out", HIDDEN * 2),
        ("gdn_partials", GDR_OUT_PARTIAL_TASKS * 16 * 4),
    ):
        offset = align(offset)
        out[name] = (offset, size)
        offset += size
    return out


def gdn_scratch_bytes() -> int:
    return scratch_extent(gdn_scratch_layout())


def qsa_pre_scratch_layout() -> dict[str, tuple[int, int]]:
    """QSA K1 hand-offs.  The common HC/quant regions retain their names."""

    out = hc_scratch_layout()
    offset = align(max(off + size for off, size in out.values()))
    for name, pairs in (
        # run_hc_finish publishes the two up-projection groups even though
        # QSA projection consumers also retain the per-tile hc_up_ready ABI.
        ("up_group_ready", 2),
        ("qsa_qkv_ready", QKV_GROUPS),
        ("qsa_index_ready", INDEX_GROUPS),
    ):
        offset = align(offset, PAIR_BYTES)
        size = pairs * PAIR_BYTES
        out[name] = (offset, size)
        offset += size
    for name, size in (
        ("qsa_mixed", HIDDEN * 2),
        # Gate is written straight to the public seam.  This workspace holds
        # only Q, K and V in that order.
        ("qsa_qkv", (QSA_Q_SIZE + 2 * QSA_KV_SIZE) * 2),
        ("qsa_index", QSA_INDEX_QK_SIZE * 2),
    ):
        offset = align(offset)
        out[name] = (offset, size)
        offset += size
    return out


def qsa_pre_scratch_bytes() -> int:
    return scratch_extent(qsa_pre_scratch_layout())


SELECTED_GROUPS = QSA_INDEX_BUDGET // QSA_COMPRESS_RATIO
SELECTED_TOKENS = QSA_INDEX_BUDGET + QSA_COMPRESS_RATIO - 1
ATTN_KEYS_PER_SPLIT = 16
ATTN_SPLITS = (SELECTED_TOKENS + ATTN_KEYS_PER_SPLIT - 1) // ATTN_KEYS_PER_SPLIT
DIRECT_ATTN_SPLITS = QSA_DIRECT_CONTEXT // ATTN_KEYS_PER_SPLIT
HEADS_PER_KV = QSA_HEADS // QSA_KV_HEADS
ATTN_HEAD_GROUPS = 2
HEADS_PER_ATTN_GROUP = QSA_HEADS // ATTN_HEAD_GROUPS
OPROJ_ROWS = 16
OPROJ_TASKS = HIDDEN // OPROJ_ROWS
OPROJ_K_STEPS_PER_WAVE = QSA_Q_SIZE // (32 * WAVES)
SCORE_SCALE = 1.0 / math.sqrt(QSA_INDEX_HEAD_DIM)
ATTN_LOG2_SCALE = 1.0 / math.sqrt(QSA_HEAD_DIM) / math.log(2.0)
NEG = -3.4028234663852886e38
assert QSA_HEAD_DIM == WAVES * 32
assert QSA_Q_SIZE % (32 * WAVES) == 0
assert QSA_HEADS % QSA_KV_HEADS == 0
assert QSA_HEADS % ATTN_HEAD_GROUPS == 0
assert HEADS_PER_KV <= 16
assert ATTN_SPLITS == 129
assert DIRECT_ATTN_SPLITS == 16


def qsa_post_scratch_layout(
    *,
    qsa_heads: int = QSA_HEADS,
    qsa_kv_heads: int = QSA_KV_HEADS,
    qsa_q_size: int = QSA_Q_SIZE,
    max_model_len: int = 8192,
) -> dict[str, tuple[int, int]]:
    """Disjoint common-finish and QSA K2 hand-offs, in bytes."""

    plan = QsaPlan(max_model_len)
    out = finish_scratch_layout()
    offset = align(finish_scratch_bytes())
    for name, pairs in (
        ("qsa_score_ready", plan.score_tasks),
        ("qsa_selection_ready", 1),
        ("qsa_split_ready", qsa_kv_heads * ATTN_SPLITS),
        ("qsa_head_ready", qsa_heads),
        ("qsa_attention_ready", 1),
        ("qsa_oproj_ready", OPROJ_TASKS),
        ("qsa_output_group_ready", 2),
    ):
        offset = align(offset, PAIR_BYTES)
        size = pairs * PAIR_BYTES
        out[name] = (offset, size)
        offset += size
    for name, size in (
        ("qsa_scores", plan.columns * 4),
        ("qsa_selected_groups", SELECTED_GROUPS * 4),
        ("qsa_selected_tokens", SELECTED_TOKENS * 4),
        ("qsa_attn_max", ATTN_SPLITS * qsa_heads * 4),
        ("qsa_attn_sum", ATTN_SPLITS * qsa_heads * 4),
        (
            "qsa_attn_partials",
            ATTN_SPLITS * qsa_heads * QSA_HEAD_DIM * 4,
        ),
        ("qsa_gated", qsa_q_size * 2),
        ("qsa_attn_out", HIDDEN * 2),
    ):
        offset = align(offset)
        out[name] = (offset, size)
        offset += size
    # Append tree storage so the accepted one-list layout is unchanged.
    for level, lists in enumerate(plan.lists[:-1]):
        for name, size in (
            (f"qsa_select_ready_{level}", lists * PAIR_BYTES),
            (f"qsa_candidates_{level}", lists * SELECTED_GROUPS * 8),
        ):
            offset = align(offset, PAIR_BYTES)
            out[name] = (offset, size)
            offset += size
    return out


def qsa_post_scratch_bytes(max_model_len: int = 8192) -> int:
    return scratch_extent(qsa_post_scratch_layout(max_model_len=max_model_len))
