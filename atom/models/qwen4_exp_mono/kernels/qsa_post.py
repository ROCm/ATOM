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

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen4_exp_mono.config import (
    ALIGN,
    HC,
    HIDDEN,
    QSA_COMPRESS_RATIO,
    QSA_DIRECT_CONTEXT,
    QSA_HEAD_DIM,
    QSA_HEADS,
    QSA_INDEX_BUDGET,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    QSA_INDEX_Q_SIZE,
    QSA_KV_HEADS,
    QSA_MAX_MODEL_LEN,
    QSA_Q_SIZE,
)
from atom.models.qwen4_exp_mono.kernels.finish import (
    FinishS2Smem,
    FinishSmem,
    finish_args,
    finish_context,
    run_finish,
    run_finish_s2,
    run_finish_s4,
)
from atom.models.qwen4_exp_mono.kernels.finish import (
    scratch_bytes as finish_scratch_bytes,
)
from atom.models.qwen4_exp_mono.kernels.finish import (
    scratch_layout as finish_scratch_layout,
)
from atom.models.qwen4_exp_mono.kernels.hc import (
    _ld_bf8,
    _sigmoid,
    _st_bf,
    combine_gdn_output_pair_raw,
)
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    butterfly,
    hw_exp2,
    kernel_symbol,
    lane_gather,
    launcher,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    xshfl,
)
from atom.mono.device.sync import Mailbox, publish, sreg
from atom.mono.device.topk import lds_topk, lds_words
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import (
    BLOCKS,
    PERSISTENT_VALUE_ATTRS,
    THREADS,
    WAVES,
    first_task,
)
from atom.mono.plan.layout import PAIR_BYTES
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

INDEX_COLUMNS = QSA_MAX_MODEL_LEN // QSA_COMPRESS_RATIO
INDEX_GROUPS_PER_TASK = 16
INDEX_SCORE_TASKS = INDEX_COLUMNS // INDEX_GROUPS_PER_TASK
SELECTED_GROUPS = QSA_INDEX_BUDGET // QSA_COMPRESS_RATIO
SELECTED_TOKENS = QSA_INDEX_BUDGET + QSA_COMPRESS_RATIO - 1
ATTN_KEYS_PER_SPLIT = 16
ATTN_SPLITS = (SELECTED_TOKENS + ATTN_KEYS_PER_SPLIT - 1) // ATTN_KEYS_PER_SPLIT
DIRECT_ATTN_SPLITS = QSA_DIRECT_CONTEXT // ATTN_KEYS_PER_SPLIT
HEADS_PER_KV = QSA_HEADS // QSA_KV_HEADS
ATTN_TASKS = QSA_KV_HEADS * ATTN_SPLITS
ATTN_HEAD_GROUPS = 2
HEADS_PER_ATTN_GROUP = QSA_HEADS // ATTN_HEAD_GROUPS
OPROJ_ROWS = 16
OPROJ_TASKS = HIDDEN // OPROJ_ROWS
OPROJ_K_STEPS_PER_WAVE = QSA_Q_SIZE // (32 * WAVES)
SCORE_SCALE = 1.0 / math.sqrt(QSA_INDEX_HEAD_DIM)
ATTN_LOG2_SCALE = 1.0 / math.sqrt(QSA_HEAD_DIM) / math.log(2.0)
NEG = -3.4028234663852886e38

assert INDEX_COLUMNS == 2048
assert INDEX_COLUMNS % INDEX_GROUPS_PER_TASK == 0
assert QSA_HEAD_DIM == WAVES * 32
assert QSA_Q_SIZE % (32 * WAVES) == 0
assert QSA_HEADS % QSA_KV_HEADS == 0
assert QSA_HEADS % ATTN_HEAD_GROUPS == 0
assert HEADS_PER_KV <= 16
assert ATTN_SPLITS == 129
assert DIRECT_ATTN_SPLITS == 16


ABI = KernelAbi(
    (
        "streams",
        "attn_raw",
        "query",
        "gate",
        "index_query",
        "block_tables",
        "table_stride",
        "table_width",
        "seq_lens",
        "token_to_req",
        "logical_positions",
        "page_size",
        "cache_pages",
        "k_cache",
        "v_cache",
        "compressed_key_cache",
        "o_proj",
        "hc_norm",
        "hc_down_inject",
        "hc_up",
        "router",
        "shared_w13",
        "shared_w2",
        "routed_w13",
        "routed_w13_scale",
        "routed_w2",
        "routed_w2_scale",
        "streams_out",
        "pending_out",
        "pending_raw",
        "moe_out",
        "scratch",
        "layer",
    )
)


def _align(offset: int, alignment: int = ALIGN) -> int:
    return (offset + alignment - 1) // alignment * alignment


def scratch_layout(
    *,
    qsa_heads: int = QSA_HEADS,
    qsa_kv_heads: int = QSA_KV_HEADS,
    qsa_q_size: int = QSA_Q_SIZE,
) -> dict[str, tuple[int, int]]:
    """Disjoint common-finish and QSA K2 hand-offs, in bytes."""

    out = finish_scratch_layout()
    offset = _align(finish_scratch_bytes())
    for name, pairs in (
        ("qsa_score_ready", INDEX_SCORE_TASKS),
        ("qsa_selection_ready", 1),
        ("qsa_split_ready", qsa_kv_heads * ATTN_SPLITS),
        ("qsa_head_ready", qsa_heads),
        ("qsa_attention_ready", 1),
        ("qsa_oproj_ready", OPROJ_TASKS),
        ("qsa_output_group_ready", 2),
    ):
        offset = _align(offset, PAIR_BYTES)
        size = pairs * PAIR_BYTES
        out[name] = (offset, size)
        offset += size
    for name, size in (
        ("qsa_scores", INDEX_COLUMNS * 4),
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
        offset = _align(offset)
        out[name] = (offset, size)
        offset += size
    return out


def scratch_bytes() -> int:
    layout = scratch_layout()
    return _align(max(off + size for off, size in layout.values()))


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


@fx.union
class QsaPostSmem:
    selection: SelectionSmem
    attention: AttentionSmem
    finish: FinishSmem


@fx.union
class QsaPostS2Smem:
    selection: SelectionSmem
    attention: AttentionSmem
    finish: FinishS2Smem


@fx.union
class QsaPostS4Smem:
    selection: SelectionSmem
    attention: AttentionSmem
    finish: FinishS2Smem


def _workspace(scratch, layout, name):
    return scratch + fx.Int64(layout[name][0])


def _load_i32(ptr, index, cache_modifier=CM_DEV):
    return fx.Int32(
        bo.buffer_load(
            rsrc(ptr), index, vec_width=1, dtype=T.i32, cache_modifier=cache_modifier
        )
    )


def _load_i64(ptr, index, cache_modifier=CM_DEV):
    return fx.Int64(
        bo.buffer_load(
            rsrc(ptr), index, vec_width=1, dtype=T.i64, cache_modifier=cache_modifier
        )
    )


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
        fx.Int32(fx.max(fx.min(visible, INDEX_COLUMNS), 0)),
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
        for start in range_constexpr(0, INDEX_SCORE_TASKS, 8):
            c["poll"](
                [
                    (c["score_ready"], start + j, 1)
                    for j in range(min(8, INDEX_SCORE_TASKS - start))
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


@traced
def _publish_attention_ready(c, bid, owner=0):
    # Keep the host-side context (including Mailbox) outside dynamic scf.if
    # regions. The AST rewriter treats calls through c[...] as context writes
    # and would otherwise try to carry the entire dict as IR loop/if state.
    tid = c["tid"]
    poll = c["poll"]
    put = c["put"]
    attention_ready = c["attention_ready"]
    head_ready = c["head_ready"]
    heads = c.get("qsa_heads", QSA_HEADS)
    if bid == owner:
        if tid == 0:
            poll([(head_ready, head, 1) for head in range(heads)])
        gpu.barrier()
        if tid == 0:
            put(attention_ready, 0, 1)


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
def _stage_oproj(c, task):
    """One N16 BF16 o_proj tile, immediately precombined into HC streams."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    poll = c["poll"]
    attention_ready = c["attention_ready"]
    red, attn_out = c["red"], c["attn_out"]
    partial = c.get("oproj_partial", False)
    streams, attn_raw, streams_out = (
        c["args"]["streams"],
        c["args"]["attn_raw"],
        c["args"]["streams_out"],
    )
    if tid == 0:
        poll([(attention_ready, 0, 1)])
    gpu.barrier()
    row = task * OPROJ_ROWS + lane % OPROJ_ROWS
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    qsa_q_size = c.get("qsa_q_size", QSA_Q_SIZE)
    oproj_k_steps = c.get("oproj_k_steps", OPROJ_K_STEPS_PER_WAVE)
    for k_pass in range_constexpr(oproj_k_steps):
        k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // OPROJ_ROWS)
        weight = _ld_bf8(c["args"]["o_proj"], row * qsa_q_size + k0, CM_NT)
        value = _ld_bf8(c["gated"], k0, CM_DEV)
        acc = mfma_bf16(weight, value, acc)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < OPROJ_ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(red, row0, 0))
        value1 = bf16_round(row_sum(red, row0 + 1, 0))
        _st_bf(attn_out, task * OPROJ_ROWS + row0, value0, CM_DEV)
        _st_bf(attn_out, task * OPROJ_ROWS + row0 + 1, value1, CM_DEV)
        if const_expr(partial):
            pass
        else:
            combine_gdn_output_pair_raw(
                streams, attn_raw, streams_out, task, row0, value0, value1
            )
    publish(c["put"], c["oproj_ready"], task, 1, tid == 0)


@traced
def _stage_oproj_grouped(c, task):
    """Direct attention's two producer-consumer o_proj groups."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    poll = c["poll"]
    red, attn_out = c["red"], c["attn_out"]
    streams, attn_raw, streams_out = (
        c["args"]["streams"],
        c["args"]["attn_raw"],
        c["args"]["streams_out"],
    )
    row = task * OPROJ_ROWS + lane % OPROJ_ROWS
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for group in range_constexpr(ATTN_HEAD_GROUPS):
        ready = c["attention_ready"] if group == 0 else c["selection_ready"]
        if tid == 0:
            poll([(ready, 0, 1)])
        gpu.barrier()
        for k_pass in range_constexpr(
            group * HEADS_PER_ATTN_GROUP,
            (group + 1) * HEADS_PER_ATTN_GROUP,
        ):
            k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // OPROJ_ROWS)
            weight = _ld_bf8(c["args"]["o_proj"], row * QSA_Q_SIZE + k0, CM_NT)
            value = _ld_bf8(c["gated"], k0, CM_DEV)
            acc = mfma_bf16(weight, value, acc)
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


@traced
def _stage_oproj_columns(contexts, task, grouped):
    c0 = contexts[0]
    tid, lane, wave = c0["tid"], c0["lane"], c0["wave"]
    token = fx.min(lane % 16, len(contexts) - 1)
    gated = contexts[-1]["gated"]
    for t, c in enumerate(contexts[:-1]):
        gated = (token == t).select(c["gated"], gated)
    row = task * OPROJ_ROWS + lane % OPROJ_ROWS
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for group in range_constexpr(ATTN_HEAD_GROUPS if grouped else 1):
        for c in contexts:
            ready = c["attention_ready"] if group == 0 else c["selection_ready"]
            if tid == 0:
                c["poll"]([(ready, 0, 1)])
        gpu.barrier()
        for k_pass in range_constexpr(
            group * HEADS_PER_ATTN_GROUP if grouped else 0,
            (group + 1) * HEADS_PER_ATTN_GROUP if grouped else OPROJ_K_STEPS_PER_WAVE,
        ):
            k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // OPROJ_ROWS)
            weight = _ld_bf8(c0["args"]["o_proj"], row * QSA_Q_SIZE + k0, CM_NT)
            value = _ld_bf8(gated, k0, CM_DEV)
            acc = mfma_bf16(weight, value, acc)
    fx.ptr_store(acc, c0["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    for t, c in enumerate(contexts):
        if tid < OPROJ_ROWS // 2:
            row0 = 2 * tid
            v0 = bf16_round(row_sum(c0["red"], row0, t))
            v1 = bf16_round(row_sum(c0["red"], row0 + 1, t))
            _st_bf(c["attn_out"], task * OPROJ_ROWS + row0, v0, CM_DEV)
            _st_bf(c["attn_out"], task * OPROJ_ROWS + row0 + 1, v1, CM_DEV)
            combine_gdn_output_pair_raw(
                c["args"]["streams"],
                c["args"]["attn_raw"],
                c["args"]["streams_out"],
                task,
                row0,
                v0,
                v1,
            )
        publish(c["put"], c["oproj_ready"], task, 1, tid == 0)


@traced
def _publish_output_groups(c, bid, owner=0):
    tid, poll, put = c["tid"], c["poll"], c["put"]
    oproj_ready, output_ready = c["oproj_ready"], c["output_group_ready"]
    if bid == owner:
        if tid == 0:
            for start in range_constexpr(0, 96, 8):
                poll([(oproj_ready, start + j, 1) for j in range(min(8, 96 - start))])
        gpu.barrier()
        if tid == 0:
            put(output_ready, 0, 1)
    elif bid == owner + 1:
        if tid == 0:
            for start in range_constexpr(0, OPROJ_TASKS - 96, 8):
                poll(
                    [
                        (oproj_ready, 96 + start + j, 1)
                        for j in range(min(8, OPROJ_TASKS - 96 - start))
                    ]
                )
        gpu.barrier()
        if tid == 0:
            put(output_ready, 1, 1)


def build_qsa_post(
    key: MonoBuild, direct: bool = False, direct_splits: int = ATTN_SPLITS
):
    """Build TP1 QSA K2 for the accepted narrow token widths."""

    if key.tokens not in (1, 2, 4):
        raise ValueError("Qwen QSA K2 currently has S=1/S=2/S=4 specializations")
    if direct_splits not in (DIRECT_ATTN_SPLITS, ATTN_SPLITS):
        raise ValueError("QSA K2 direct_splits must be the short or full bucket")
    if not direct and direct_splits != ATTN_SPLITS:
        raise ValueError("only direct QSA K2 supports a short-context split bucket")
    task_splits = direct_splits if direct else ATTN_SPLITS
    attention_tasks = QSA_KV_HEADS * task_splits
    keyed = (
        *key_tuple(key, SOURCES),
        ("direct", direct),
        ("direct_splits", direct_splits),
    )
    layout = scratch_layout()
    finish_layout = finish_scratch_layout()
    publish_attention = (
        _publish_attention_groups if direct else _publish_attention_ready
    )
    stage_oproj = _stage_oproj_grouped if direct else _stage_oproj
    stem = "qwen_qsa_post"
    if direct:
        stem += (
            f"_direct{QSA_DIRECT_CONTEXT}"
            if direct_splits == DIRECT_ATTN_SPLITS
            else "_direct"
        )
    name = kernel_symbol(stem, s=key.tokens)
    smem = (
        QsaPostSmem
        if key.tokens == 1
        else QsaPostS2Smem if key.tokens == 2 else QsaPostS4Smem
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        streams: Int64,
        attn_raw: Int64,
        query: Int64,
        gate: Int64,
        index_query: Int64,
        block_tables: Int64,
        table_stride: Int32,
        table_width: Int32,
        seq_lens: Int64,
        token_to_req: Int64,
        logical_positions: Int64,
        page_size: Int32,
        cache_pages: Int32,
        k_cache: Int64,
        v_cache: Int64,
        compressed_key_cache: Int64,
        o_proj: Int64,
        hc_norm: Int64,
        hc_down_inject: Int64,
        hc_up: Int64,
        router: Int64,
        shared_w13: Int64,
        shared_w2: Int64,
        routed_w13: Int64,
        routed_w13_scale: Int64,
        routed_w2: Int64,
        routed_w2_scale: Int64,
        streams_out: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        moe_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        bid = fx.block_idx.x
        halves = fx.SharedAllocator().allocate(smem)
        mailbox = Mailbox(layer)

        def token_context(token):
            token_scratch = scratch + fx.Int64(token * scratch_bytes())
            args = finish_args(
                streams + fx.Int64(token * HC * HIDDEN * 2),
                _workspace(token_scratch, layout, "qsa_attn_out"),
                attn_raw + fx.Int64(token * HC * 2),
                hc_norm,
                hc_down_inject,
                hc_up,
                router,
                shared_w13,
                shared_w2,
                routed_w13,
                routed_w13_scale,
                routed_w2,
                routed_w2_scale,
                streams_out + fx.Int64(token * HC * HIDDEN * 2),
                pending_out + fx.Int64(token * HIDDEN * 2),
                pending_raw + fx.Int64(token * HC * 2),
                moe_out + fx.Int64(token * HIDDEN * 2),
            )
            c = finish_context(
                halves.finish.peek(),
                mailbox,
                finish_layout,
                token_scratch,
                args,
            )
            c["args"].update(
                {
                    "query": query + fx.Int64(token * QSA_Q_SIZE * 2),
                    "gate": gate + fx.Int64(token * QSA_Q_SIZE * 2),
                    "index_query": index_query + fx.Int64(token * QSA_INDEX_Q_SIZE * 2),
                    "block_tables": block_tables,
                    "table_stride": table_stride,
                    "table_width": table_width,
                    "seq_lens": seq_lens,
                    "token_to_req": token_to_req + fx.Int64(token * 4),
                    "logical_positions": logical_positions + fx.Int64(token * 8),
                    "page_size": page_size,
                    "cache_pages": cache_pages,
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                    "compressed_key_cache": compressed_key_cache,
                    "o_proj": o_proj,
                }
            )
            c["red"] = halves.attention.peek().red.ptr
            c["kv"] = halves.attention.peek().kv.ptr
            c["selection_lds"] = {
                "keys": halves.selection.peek().keys.ptr,
                "hist": halves.selection.peek().hist.ptr,
                "buf": halves.selection.peek().buf.ptr,
            }
            for region in (
                "score_ready",
                "selection_ready",
                "split_ready",
                "head_ready",
                "attention_ready",
                "oproj_ready",
                "output_group_ready",
            ):
                name_ = f"qsa_{region}"
                c[region] = sreg(token_scratch, layout[name_][0], name_)
            for region in (
                "scores",
                "selected_groups",
                "selected_tokens",
                "attn_max",
                "attn_sum",
                "attn_partials",
                "gated",
                "attn_out",
            ):
                c[region] = _workspace(token_scratch, layout, f"qsa_{region}")
            return c

        c0 = token_context(0)
        if key.tokens == 1:
            if const_expr(not direct):
                for task in range(first_task(bid, 0), INDEX_SCORE_TASKS, BLOCKS):
                    _stage_index_score(c0, task)
                _run_selection(c0, bid)
            gpu.barrier()
            for task in range(first_task(bid, 0), attention_tasks, BLOCKS):
                _stage_attention_split(c0, task, direct, task_splits)
            head = first_task(bid, 0)
            if head < QSA_HEADS:
                _stage_attention_reduce(c0, head, task_splits)
            publish_attention(c0, bid)
            c0["red"] = halves.finish.peek().red.ptr
            for task in range(first_task(bid, 0), OPROJ_TASKS, BLOCKS):
                stage_oproj(c0, task)
            _publish_output_groups(c0, bid)
            c0["attn_group_ready"] = c0["output_group_ready"]
            c0["attn_precombined"] = True
            gpu.barrier()
            run_finish(c0, bid)
        elif key.tokens == 2:
            c1 = token_context(1)
            if const_expr(not direct):
                for unit in range(first_task(bid, 0), 2 * INDEX_SCORE_TASKS, BLOCKS):
                    if unit < INDEX_SCORE_TASKS:
                        _stage_index_score(c0, unit)
                    if unit >= INDEX_SCORE_TASKS:
                        _stage_index_score(c1, unit - INDEX_SCORE_TASKS)
                _run_selection(c0, bid, 0)
                _run_selection(c1, bid, 1)
            gpu.barrier()
            for unit in range(first_task(bid, 0), 2 * attention_tasks, BLOCKS):
                if unit < attention_tasks:
                    _stage_attention_split(c0, unit, direct, task_splits)
                if unit >= attention_tasks:
                    _stage_attention_split(
                        c1, unit - attention_tasks, direct, task_splits
                    )
            head = first_task(bid, 0)
            if head < QSA_HEADS:
                _stage_attention_reduce(c0, head, task_splits)
            if (head >= QSA_HEADS) & (head < 2 * QSA_HEADS):
                _stage_attention_reduce(c1, head - QSA_HEADS, task_splits)
            publish_attention(c0, bid, 0)
            publish_attention(c1, bid, 2)
            c0["red"] = halves.finish.peek().red.ptr
            c1["red"] = halves.finish.peek().red.ptr
            for unit in range(first_task(bid, 0), 2 * OPROJ_TASKS, BLOCKS):
                if unit < OPROJ_TASKS:
                    stage_oproj(c0, unit)
                if unit >= OPROJ_TASKS:
                    stage_oproj(c1, unit - OPROJ_TASKS)
            _publish_output_groups(c0, bid, 0)
            _publish_output_groups(c1, bid, 2)
            c0["attn_group_ready"] = c0["output_group_ready"]
            c1["attn_group_ready"] = c1["output_group_ready"]
            c0["attn_precombined"] = True
            c1["attn_precombined"] = True
            gpu.barrier()
            run_finish_s2(c0, c1, bid)
        else:
            c1 = token_context(1)
            c2 = token_context(2)
            c3 = token_context(3)
            if const_expr(not direct):
                for unit in range(first_task(bid, 0), 4 * INDEX_SCORE_TASKS, BLOCKS):
                    if unit < INDEX_SCORE_TASKS:
                        _stage_index_score(c0, unit)
                    if (unit >= INDEX_SCORE_TASKS) & (unit < 2 * INDEX_SCORE_TASKS):
                        _stage_index_score(c1, unit - INDEX_SCORE_TASKS)
                    if (unit >= 2 * INDEX_SCORE_TASKS) & (unit < 3 * INDEX_SCORE_TASKS):
                        _stage_index_score(c2, unit - 2 * INDEX_SCORE_TASKS)
                    if unit >= 3 * INDEX_SCORE_TASKS:
                        _stage_index_score(c3, unit - 3 * INDEX_SCORE_TASKS)
                _run_selection(c0, bid, 0)
                _run_selection(c1, bid, 1)
                _run_selection(c2, bid, 2)
                _run_selection(c3, bid, 3)
            gpu.barrier()
            for unit in range(first_task(bid, 0), 4 * attention_tasks, BLOCKS):
                if unit < attention_tasks:
                    _stage_attention_split(c0, unit, direct, task_splits)
                if (unit >= attention_tasks) & (unit < 2 * attention_tasks):
                    _stage_attention_split(
                        c1, unit - attention_tasks, direct, task_splits
                    )
                if (unit >= 2 * attention_tasks) & (unit < 3 * attention_tasks):
                    _stage_attention_split(
                        c2, unit - 2 * attention_tasks, direct, task_splits
                    )
                if unit >= 3 * attention_tasks:
                    _stage_attention_split(
                        c3, unit - 3 * attention_tasks, direct, task_splits
                    )
            head = first_task(bid, 0)
            if head < QSA_HEADS:
                _stage_attention_reduce(c0, head, task_splits)
            if (head >= QSA_HEADS) & (head < 2 * QSA_HEADS):
                _stage_attention_reduce(c1, head - QSA_HEADS, task_splits)
            if (head >= 2 * QSA_HEADS) & (head < 3 * QSA_HEADS):
                _stage_attention_reduce(c2, head - 2 * QSA_HEADS, task_splits)
            if (head >= 3 * QSA_HEADS) & (head < 4 * QSA_HEADS):
                _stage_attention_reduce(c3, head - 3 * QSA_HEADS, task_splits)
            publish_attention(c0, bid, 0)
            publish_attention(c1, bid, 2)
            publish_attention(c2, bid, 4)
            publish_attention(c3, bid, 6)
            c0["red"] = halves.finish.peek().red.ptr
            c1["red"] = halves.finish.peek().red.ptr
            c2["red"] = halves.finish.peek().red.ptr
            c3["red"] = halves.finish.peek().red.ptr
            for task in range(first_task(bid, 0), OPROJ_TASKS, BLOCKS):
                _stage_oproj_columns((c0, c1, c2, c3), task, direct)
            _publish_output_groups(c0, bid, 0)
            _publish_output_groups(c1, bid, 2)
            _publish_output_groups(c2, bid, 4)
            _publish_output_groups(c3, bid, 6)
            for c in (c0, c1, c2, c3):
                c["attn_group_ready"] = c["output_group_ready"]
                c["attn_precombined"] = True
            gpu.barrier()
            run_finish_s4(c0, c1, c2, c3, bid)
        _ = keyed

    @launcher(name)
    def launch(
        streams: Int64,
        attn_raw: Int64,
        query: Int64,
        gate: Int64,
        index_query: Int64,
        block_tables: Int64,
        table_stride: Int32,
        table_width: Int32,
        seq_lens: Int64,
        token_to_req: Int64,
        logical_positions: Int64,
        page_size: Int32,
        cache_pages: Int32,
        k_cache: Int64,
        v_cache: Int64,
        compressed_key_cache: Int64,
        o_proj: Int64,
        hc_norm: Int64,
        hc_down_inject: Int64,
        hc_up: Int64,
        router: Int64,
        shared_w13: Int64,
        shared_w2: Int64,
        routed_w13: Int64,
        routed_w13_scale: Int64,
        routed_w2: Int64,
        routed_w2_scale: Int64,
        streams_out: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        moe_out: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = keyed
        kernel(
            streams,
            attn_raw,
            query,
            gate,
            index_query,
            block_tables,
            table_stride,
            table_width,
            seq_lens,
            token_to_req,
            logical_positions,
            page_size,
            cache_pages,
            k_cache,
            v_cache,
            compressed_key_cache,
            o_proj,
            hc_norm,
            hc_down_inject,
            hc_up,
            router,
            shared_w13,
            shared_w2,
            routed_w13,
            routed_w13_scale,
            routed_w2,
            routed_w2_scale,
            streams_out,
            pending_out,
            pending_raw,
            moe_out,
            scratch,
            layer,
            value_attrs=PERSISTENT_VALUE_ATTRS,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kernel, launch)
    return launch
