# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paged index selection and sparse attention shared by TP1 and TP8 K2.

The context carries rank-local head geometry. Scheduling and output
projection/collective publication belong to the launch compositions.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.layout import (
    ATTN_KEYS_PER_SPLIT,
    ATTN_LOG2_SCALE,
    ATTN_SPLITS,
    HEADS_PER_KV,
    NEG,
    SCORE_SCALE,
    SELECTED_GROUPS,
    SELECTED_TOKENS,
)
from atom.models.qwen4_exp_mono.qsa_plan import (
    INDEX_GROUPS_PER_TASK,
    INDEX_SEGMENT,
    SELECT_FAN,
)
from atom.models.qwen4_exp_mono.shapes import (
    QSA_COMPRESS_RATIO,
    QSA_HEAD_DIM,
    QSA_HEADS,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    QSA_KV_HEADS,
)
from atom.models.qwen4_exp_mono.stages.hc import (
    _ld_bf8,
    _sigmoid,
    _st_bf,
)
from atom.models.qwen4_exp_mono.stages.memory import (
    load_i32 as _load_i32,
)
from atom.models.qwen4_exp_mono.stages.memory import (
    load_i64 as _load_i64,
)
from atom.models.qwen4_exp_mono.stages.memory import workspace
from atom.mono.device.ops import (
    CM_DEV,
    bf16_round,
    butterfly,
    hw_exp2,
    lane_gather,
    mfma_bf16,
    rsrc,
    traced,
    xshfl,
)
from atom.mono.device.sync import publish, sreg
from atom.mono.device.topk import key_bits, lds_topk, lds_words
from atom.mono.plan.execution import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
)

_SEL = lds_words(4)
SEL_KEYS = _SEL["keys"]
SEL_HIST = _SEL["hist"]
SEL_BUF = _SEL["buf"]


@fx.struct
class SelectionSmem:
    keys: fx.Array[fx.Int32, SEL_KEYS, 16]
    hist: fx.Array[fx.Int32, SEL_HIST, 16]
    buf: fx.Array[fx.Int32, SEL_BUF, 16]


@fx.struct
class AttentionSmem:
    red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
    kv: fx.Array[fx.Int32, WAVES * ATTN_KEYS_PER_SPLIT * 16, 16]


def _load_f32(ptr, index, cache_modifier=CM_DEV):
    return fx.Float32(
        bo.buffer_load(
            rsrc(ptr), index, vec_width=1, dtype=T.f32, cache_modifier=cache_modifier
        )
    )


def _store_i32(ptr, index, value, cache_modifier=CM_DEV):
    bo.buffer_store(fx.Int32(value), rsrc(ptr), index, cache_modifier=cache_modifier)


def _store_f32(ptr, index, value, cache_modifier=CM_DEV):
    bo.buffer_store(fx.Float32(value), rsrc(ptr), index, cache_modifier=cache_modifier)


def _bf8_at(address):
    """Eight BF16 values at a 64-bit global address."""

    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16), fx.Int64(address)
    )
    return fx.Vector(
        fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32))
    ).bitcast(fx.BFloat16)


def _cache_address(base, physical, offset, head, page_size, heads, dim, col):
    row = (fx.Int64(physical) * fx.Int64(page_size) + fx.Int64(offset)) * fx.Int64(
        heads
    ) + fx.Int64(head)
    return base + (row * fx.Int64(dim) + fx.Int64(col)) * fx.Int64(2)


def _visible_groups(c):
    request = _load_i32(c["args"]["token_to_req"], 0)
    safe_request = fx.max(request, 0)
    position = _load_i64(c["args"]["logical_positions"], 0)
    context = _load_i32(c["args"]["seq_lens"], safe_request)
    visible = fx.min(
        (position + fx.Int64(1)) // QSA_COMPRESS_RATIO, context // QSA_COMPRESS_RATIO
    )
    return (
        request,
        position,
        context,
        fx.Int32(fx.max(fx.min(visible, c.get("index_columns", INDEX_SEGMENT)), 0)),
    )


def _attention_width(c):
    """Number of valid entries emitted by selection for this decode row."""

    request, position, context, _ = _visible_groups(c)
    length = fx.Int32(
        ((request >= 0) & (position >= 0)).select(
            fx.min(position + fx.Int64(1), fx.Int64(context)), fx.Int64(0)
        )
    )
    full_groups = length // QSA_COMPRESS_RATIO
    history = fx.min(full_groups, SELECTED_GROUPS) * QSA_COMPRESS_RATIO
    tail = length - full_groups * QSA_COMPRESS_RATIO
    width = fx.Int32(fx.max(fx.min(history + tail, SELECTED_TOKENS), 0))
    return width


def _physical_row(c, logical, compressed=False):
    a = c["args"]
    page = a["page_size"] // QSA_COMPRESS_RATIO if compressed else a["page_size"]
    safe = fx.max(logical, 0)
    logical_page = safe // page
    page_offset = safe % page
    request = _load_i32(a["token_to_req"], 0)
    safe_request = fx.max(request, 0)
    in_table = (request >= 0) & (logical_page < a["table_width"])
    physical = _load_i32(
        a["block_tables"],
        safe_request * a["table_stride"] + fx.min(logical_page, a["table_width"] - 1),
    )
    live = (logical >= 0) & in_table & (physical >= 0) & (physical < a["cache_pages"])
    return fx.max(physical, 0), page_offset, live


def _masked_bf8(values, live):
    return fx.Vector.from_elements(
        [live.select(values[e], fx.BFloat16(0.0)) for e in range(8)],
        fx.BFloat16,
    )


@traced
def _stage_index_score(c, task):
    """Sixteen compressed keys, four heads and eight K partitions per dot."""

    tid, lane = c["tid"], c["lane"]
    _, _, _, visible = _visible_groups(c)
    if task * INDEX_GROUPS_PER_TASK < visible:
        key = task * INDEX_GROUPS_PER_TASK + tid // (QSA_INDEX_HEADS * 8)
        head = tid // 8 % QSA_INDEX_HEADS
        part = tid % 8
        physical, offset, live = _physical_row(c, key, compressed=True)
        live = live & (key < visible)
        dot = fx.Float32(0.0)
        for half in range_constexpr(2):
            chunk = part * 16 + half * 8
            address = _cache_address(
                c["args"]["compressed_key_cache"],
                physical,
                offset,
                0,
                c["args"]["page_size"] // QSA_COMPRESS_RATIO,
                1,
                QSA_INDEX_HEAD_DIM,
                chunk,
            )
            kvals = _masked_bf8(_bf8_at(address), live)
            qvals = _ld_bf8(
                c["args"]["index_query"],
                head * QSA_INDEX_HEAD_DIM + chunk,
                CM_DEV,
            )
            for e in range_constexpr(8):
                dot = fx.Float32(
                    fmath.fma(fx.Float32(kvals[e]), fx.Float32(qvals[e]), dot)
                )
        dot = butterfly(dot, (4, 2, 1))
        score = fx.max(dot, fx.Float32(0.0))
        if part == 0:
            key_lane = lane // 32 * 32
            total = fx.Float32(0.0)
            for h in range_constexpr(QSA_INDEX_HEADS):
                value = lane_gather(score.bitcast(fx.Int32), key_lane + h * 8).bitcast(
                    fx.Float32
                )
                total = total + value
            if head == 0:
                _store_f32(c["scores"], key, total * fx.Float32(SCORE_SCALE))
    publish(c["put"], c["score_ready"], task, 1, tid == 0)


@traced
def _run_selection(c, bid, owner=0):
    if bid != owner:
        return
    if c["tid"] == 0:
        for start in range_constexpr(0, c["index_score_tasks"], 8):
            c["poll"](
                [
                    (c["score_ready"], start + j, 1)
                    for j in range(min(8, c["index_score_tasks"] - start))
                ]
            )
    gpu.barrier()
    _, position, context, visible = _visible_groups(c)

    def load(_q, i):
        values = fx.Vector(
            bo.buffer_load(
                rsrc(c["scores"]),
                i,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        ).bitcast(fx.Int32)
        return [values[e] for e in range(4)]

    def emit(pos, d, i, _extra):
        _store_i32(c["selected_groups"], pos, -1 if d is None else i)

    lds_topk(
        visible,
        SELECTED_GROUPS,
        load,
        emit,
        c["selection_lds"],
        c["tid"],
        per_thread=4,
    )
    full_groups = fx.Int32(
        fx.max(
            fx.min((position + 1) // QSA_COMPRESS_RATIO, context // QSA_COMPRESS_RATIO),
            0,
        )
    )
    length = fx.Int32(
        (position >= 0).select(fx.min(position + 1, fx.Int64(context)), fx.Int64(0))
    )
    history_end = fx.min(full_groups, SELECTED_GROUPS) * QSA_COMPRESS_RATIO
    tail_start = full_groups * QSA_COMPRESS_RATIO
    for r in range_constexpr((SELECTED_TOKENS + THREADS - 1) // THREADS):
        col = c["tid"] + r * THREADS
        if col < SELECTED_TOKENS:
            history = col < history_end
            group = _load_i32(
                c["selected_groups"],
                fx.min(col // QSA_COMPRESS_RATIO, SELECTED_GROUPS - 1),
            )
            history_value = ((group >= 0) & (group < full_groups)).select(
                group * QSA_COMPRESS_RATIO + col % QSA_COMPRESS_RATIO,
                fx.Int32(-1),
            )
            tail_offset = col - history_end
            tail_value = (
                (tail_offset >= 0) & (tail_offset < length - tail_start)
            ).select(tail_start + tail_offset, fx.Int32(-1))
            _store_i32(
                c["selected_tokens"], col, history.select(history_value, tail_value)
            )
    publish(c["put"], c["selection_ready"], 0, 1, c["tid"] == 0)


@traced
def _select_tree_list(c, visible, task, level):
    """Select one fixed-size leaf or merge, retaining original key/index pairs."""
    tid = c["tid"]
    final = level == len(c["selection_lists"]) - 1
    if const_expr(level == 0):
        n = fx.max(fx.min(visible - task * INDEX_SEGMENT, INDEX_SEGMENT), 0)
        first = task * (INDEX_SEGMENT // INDEX_GROUPS_PER_TASK)
        if tid == 0:
            for j in range_constexpr(INDEX_SEGMENT // INDEX_GROUPS_PER_TASK):
                index = first + j
                if index < c["index_score_tasks"]:
                    c["poll"]([(c["score_ready"], index, 1)])
        gpu.barrier()
    else:
        span = INDEX_SEGMENT * SELECT_FAN ** (level - 1)
        below = (visible + span - 1) // span
        children = fx.max(fx.min(below - task * SELECT_FAN, SELECT_FAN), 0)
        n = children * SELECTED_GROUPS
        if tid == 0:
            for j in range_constexpr(SELECT_FAN):
                index = task * SELECT_FAN + j
                if index < c["selection_lists"][level - 1]:
                    c["poll"]([(c["select_ready"][level - 1], index, 1)])
        gpu.barrier()

    def load(_q, i):
        if const_expr(level == 0):
            return [
                _load_f32(c["scores"], task * INDEX_SEGMENT + i + e).bitcast(fx.Int32)
                for e in range(4)
            ]
        return [
            _load_i32(
                c["candidates"][level - 1],
                (task * SELECT_FAN * SELECTED_GROUPS + i + e) * 2,
            )
            for e in range(4)
        ]

    def emit(pos, d, i, _extra):
        if const_expr(d is None):
            ident, bits = fx.Int32(-1), fx.Float32(NEG).bitcast(fx.Int32)
        else:
            ident = (
                task * INDEX_SEGMENT + i
                if level == 0
                else _load_i32(
                    c["candidates"][level - 1],
                    (task * SELECT_FAN * SELECTED_GROUPS + i) * 2 + 1,
                )
            )
            bits = key_bits(fx.ptr_load(c["selection_lds"]["keys"] + i))
        if const_expr(final):
            _store_i32(c["selected_groups"], pos, ident)
        else:
            offset = (task * SELECTED_GROUPS + pos) * 2
            _store_i32(c["candidates"][level], offset, bits)
            _store_i32(c["candidates"][level], offset + 1, ident)

    lds_topk(n, SELECTED_GROUPS, load, emit, c["selection_lds"], tid, per_thread=4)
    if const_expr(not final):
        publish(c["put"], c["select_ready"][level], task, 1, tid == 0)


@traced
def _run_selection_tree(c, bid):
    """All score CTAs finish before waiting on fixed-LDS selection lists."""
    _, position, context, visible = _visible_groups(c)
    for level in range_constexpr(len(c["selection_lists"])):
        for task in range(first_task(bid, 0), c["selection_lists"][level], BLOCKS):
            _select_tree_list(c, visible, task, level)
    if bid == 0:
        length = fx.Int32(
            (position >= 0).select(fx.min(position + 1, fx.Int64(context)), fx.Int64(0))
        )
        full_groups = length // QSA_COMPRESS_RATIO
        history_end = fx.min(full_groups, SELECTED_GROUPS) * QSA_COMPRESS_RATIO
        for r in range_constexpr((SELECTED_TOKENS + THREADS - 1) // THREADS):
            col = c["tid"] + r * THREADS
            if col < SELECTED_TOKENS:
                group = _load_i32(
                    c["selected_groups"],
                    fx.min(col // QSA_COMPRESS_RATIO, SELECTED_GROUPS - 1),
                )
                history = ((group >= 0) & (group < full_groups)).select(
                    group * QSA_COMPRESS_RATIO + col % QSA_COMPRESS_RATIO, fx.Int32(-1)
                )
                tail = col - history_end
                tail_value = (
                    (tail >= 0) & (tail < length - full_groups * QSA_COMPRESS_RATIO)
                ).select(full_groups * QSA_COMPRESS_RATIO + tail, fx.Int32(-1))
                _store_i32(
                    c["selected_tokens"],
                    col,
                    (col < history_end).select(history, tail_value),
                )
        publish(c["put"], c["selection_ready"], 0, 1, c["tid"] == 0)


@traced
def _stage_attention_split(c, task, direct, task_splits):
    """One KV head and one 16-token tile -> 12-head FP32 partials."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    qsa_heads = c.get("qsa_heads", QSA_HEADS)
    qsa_kv_heads = c.get("qsa_kv_heads", QSA_KV_HEADS)
    heads_per_kv = c.get("heads_per_kv", HEADS_PER_KV)
    kv_head = task // task_splits
    split = task % task_splits
    active_tokens = _attention_width(c)
    if split * ATTN_KEYS_PER_SPLIT < active_tokens:
        if const_expr(not direct):  # noqa: SIM102 - compile-time specialization
            if tid == 0:
                c["poll"]([(c["selection_ready"], 0, 1)])
        gpu.barrier()
        g = lane // 16
        head_local = lane % 16
        query_head = kv_head * heads_per_kv + fx.min(head_local, heads_per_kv - 1)
        global_head = kv_head * heads_per_kv + head_local
        chunk = wave * 32 + 8 * g
        qraw = _ld_bf8(c["args"]["query"], query_head * QSA_HEAD_DIM + chunk, CM_DEV)
        qvals = fx.Vector.from_elements(
            [
                (fx.Float32(qraw[e]) * fx.Float32(ATTN_LOG2_SCALE)).to(fx.BFloat16)
                for e in range(8)
            ],
            fx.BFloat16,
        )
        key_index = split * ATTN_KEYS_PER_SPLIT + lane % ATTN_KEYS_PER_SPLIT
        logical = fx.Int32(key_index)
        if const_expr(not direct):
            logical = _load_i32(
                c["selected_tokens"], fx.min(key_index, SELECTED_TOKENS - 1)
            )
        physical, offset, live = _physical_row(c, logical)
        live = live & (key_index < active_tokens)
        kaddr = _cache_address(
            c["args"]["k_cache"],
            physical,
            offset,
            kv_head,
            c["args"]["page_size"],
            qsa_kv_heads,
            QSA_HEAD_DIM,
            chunk,
        )
        kvals = _masked_bf8(_bf8_at(kaddr), live)
        acc = mfma_bf16(kvals, qvals, fx.Vector.filled(4, 0.0, fx.Float32))
        fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
        gpu.barrier()

        scores = fx.Vector.filled(4, 0.0, fx.Float32)
        for w in range_constexpr(WAVES):
            part = fx.Vector(
                fx.ptr_load(
                    c["red"] + (w * 64 + lane) * 4,
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
            )
            scores = scores + part
        if const_expr(direct):
            selected = [
                fx.Int32(split * ATTN_KEYS_PER_SPLIT + 4 * g + e) for e in range(4)
            ]
        else:
            selected = [
                _load_i32(
                    c["selected_tokens"],
                    fx.min(
                        split * ATTN_KEYS_PER_SPLIT + 4 * g + e,
                        SELECTED_TOKENS - 1,
                    ),
                )
                for e in range(4)
            ]
        selected_live = [
            (split * ATTN_KEYS_PER_SPLIT + 4 * g + e < active_tokens)
            & (selected[e] >= 0)
            for e in range(4)
        ]
        sv = [
            fx.Float32(selected_live[e].select(scores[e], fx.Float32(NEG)))
            for e in range(4)
        ]
        maximum = butterfly(
            fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])),
            (32, 16),
            fx.max,
        )
        probs = [
            selected_live[e].select(hw_exp2(sv[e] - maximum), fx.Float32(0.0))
            for e in range(4)
        ]
        total = (probs[0] + probs[2]) + (probs[1] + probs[3])
        total = total + xshfl(total, 32)
        total = total + xshfl(total, 16)
        if (wave == 0) & (g == 0) & (head_local < heads_per_kv):
            stat = split * qsa_heads + global_head
            _store_f32(c["attn_max"], stat, maximum)
            _store_f32(c["attn_sum"], stat, total)

        value_index = split * ATTN_KEYS_PER_SPLIT + lane // 4
        vlogical = fx.Int32(value_index)
        if const_expr(not direct):
            vlogical = _load_i32(
                c["selected_tokens"], fx.min(value_index, SELECTED_TOKENS - 1)
            )
        vphysical, voffset, vlive = _physical_row(c, vlogical)
        vlive = vlive & (value_index < active_tokens)
        col0 = wave * 32 + 8 * (lane % 4)
        vaddr = _cache_address(
            c["args"]["v_cache"],
            vphysical,
            voffset,
            kv_head,
            c["args"]["page_size"],
            qsa_kv_heads,
            QSA_HEAD_DIM,
            col0,
        )
        values = _masked_bf8(_bf8_at(vaddr), vlive)
        tile = c["kv"] + wave * (ATTN_KEYS_PER_SPLIT * 16)
        fx.ptr_store(values.bitcast(fx.Int32), tile + (lane // 4 * 16 + lane % 4 * 4))
        gpu.barrier()

        pwords = (
            fx.Vector.from_elements(probs, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        )
        pb = fx.Vector.from_elements([pwords[0], pwords[1]], fx.Int32).bitcast(fx.Int16)
        outputs = []
        for half in range_constexpr(2):
            col = half * 16 + head_local
            pieces = [
                fx.Int32(fx.ptr_load(tile + ((4 * g + row) * 16 + col // 2)))
                >> (col % 2 * 16)
                & 0xFFFF
                for row in range(4)
            ]
            av = fx.Vector.from_elements(
                [pieces[0] | (pieces[1] << 16), pieces[2] | (pieces[3] << 16)],
                fx.Int32,
            )
            out = fx.Vector(
                rocdl.mfma_f32_16x16x16bf16_1k(
                    T.vec(4, T.f32),
                    [
                        av.bitcast(fx.Int16),
                        pb,
                        fx.Vector.filled(4, 0.0, fx.Float32),
                    ],
                )
            )
            for e in range_constexpr(4):
                outputs.append(out[e])
        if head_local < heads_per_kv:
            for e in range_constexpr(8):
                col = wave * 32 + (e // 4) * 16 + g * 4 + e % 4
                index = (split * qsa_heads + global_head) * QSA_HEAD_DIM + col
                _store_f32(c["attn_partials"], index, outputs[e])
    ready = kv_head * c.get("attn_split_stride", ATTN_SPLITS) + split
    publish(c["put"], c["split_ready"], ready, 1, tid == 0)


@traced
def _reduce_attention_bucket(c, head, splits, live_splits):
    """Reduce one compile-time split bucket without loop-carried bounds."""

    tid = c["tid"]
    qsa_heads = c.get("qsa_heads", QSA_HEADS)
    heads_per_kv = c.get("heads_per_kv", HEADS_PER_KV)
    split_stride = c.get("attn_split_stride", ATTN_SPLITS)
    kv_head = head // heads_per_kv
    if tid == 0:
        for start in range_constexpr(0, splits, 8):
            c["poll"](
                [
                    (c["split_ready"], kv_head * split_stride + start + j, 1)
                    for j in range(min(8, splits - start))
                ]
            )
    gpu.barrier()
    if tid < QSA_HEAD_DIM:
        maximum = fx.Float32(NEG)
        denominator = fx.Float32(0.0)
        value = fx.Float32(0.0)
        for split in range_constexpr(splits):
            stat = split * qsa_heads + head
            live = fx.Int32(split) < live_splits
            split_max = live.select(_load_f32(c["attn_max"], stat), fx.Float32(NEG))
            split_sum = live.select(_load_f32(c["attn_sum"], stat), fx.Float32(0.0))
            partial = live.select(
                _load_f32(c["attn_partials"], stat * QSA_HEAD_DIM + tid),
                fx.Float32(0.0),
            )
            next_max = fx.max(maximum, split_max)
            alpha = hw_exp2(maximum - next_max)
            beta = hw_exp2(split_max - next_max)
            value = value * alpha + partial * beta
            denominator = denominator * alpha + split_sum * beta
            maximum = next_max
        attn = bf16_round(
            (denominator > fx.Float32(0.0)).select(value / denominator, fx.Float32(0.0))
        )
        gate = bf16_round(
            _sigmoid(
                _ld_bf8(c["args"]["gate"], head * QSA_HEAD_DIM + (tid & ~7), CM_DEV)[
                    tid % 8
                ]
            )
        )
        gated = bf16_round(attn * gate)
        _st_bf(c["gated"], head * QSA_HEAD_DIM + tid, gated, CM_DEV)


@traced
def _stage_attention_reduce(c, head, max_splits):
    """Select a compile-time reduction bucket from the runtime context."""

    tid = c["tid"]
    live_splits = (_attention_width(c) + ATTN_KEYS_PER_SPLIT - 1) // ATTN_KEYS_PER_SPLIT
    if max_splits == 16:
        if live_splits <= 4:
            _reduce_attention_bucket(c, head, 4, live_splits)
        elif live_splits <= 8:
            _reduce_attention_bucket(c, head, 8, live_splits)
        else:
            _reduce_attention_bucket(c, head, 16, live_splits)
    else:
        if live_splits <= 4:
            _reduce_attention_bucket(c, head, 4, live_splits)
        elif live_splits <= 16:
            _reduce_attention_bucket(c, head, 16, live_splits)
        elif live_splits <= 64:
            _reduce_attention_bucket(c, head, 64, live_splits)
        else:
            _reduce_attention_bucket(c, head, ATTN_SPLITS, live_splits)
    publish(c["put"], c["head_ready"], head, 1, tid == 0)


def attention_context(lds, mailbox, scratch, layout, args, dims, selection=None):
    """Bind the same attention stages to each TP layout without changing offsets."""
    tid = fx.thread_idx.x
    c = {
        "tid": tid,
        "lane": tid % 64,
        "wave": tid // 64,
        "put": mailbox.put,
        "poll": mailbox.poll,
        "args": args,
        "red": lds.red.ptr,
        "kv": lds.kv.ptr,
    }
    if dims.tp == 8:
        c.update(
            qsa_heads=dims.qsa_heads,
            qsa_kv_heads=dims.qsa_kv_heads,
            heads_per_kv=dims.qsa_heads // dims.qsa_kv_heads,
            attn_split_stride=ATTN_SPLITS,
        )
    ready = (
        "split_ready",
        "head_ready",
        "attention_ready",
        "oproj_ready",
        "output_group_ready",
    )
    buffers = ("attn_max", "attn_sum", "attn_partials", "gated")
    if selection is not None:
        ready = ("score_ready", "selection_ready", *ready)
        buffers = ("scores", "selected_groups", "selected_tokens", *buffers, "attn_out")
        c["selection_lds"] = {
            name: getattr(selection, name).ptr for name in ("keys", "hist", "buf")
        }
    for region in ready:
        name = "qsa_" + region
        c[region] = sreg(scratch, layout[name][0], name)
    for region in buffers:
        c[region] = workspace(scratch, layout, "qsa_" + region)
    return c


@traced
def publish_attention_ready(c, bid, heads, owner=0):
    tid, poll, put = c["tid"], c["poll"], c["put"]
    attention_ready, head_ready = c["attention_ready"], c["head_ready"]
    if bid == owner:
        if tid == 0:
            poll([(head_ready, head, 1) for head in range(heads)])
        gpu.barrier()
        if tid == 0:
            put(attention_ready, 0, 1)
