# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 QSA K1 after the native attention hyper-connection front.

Qwen has 24 query heads but only two KV heads.  TP8 therefore owns three
query heads per rank while four ranks replicate each physical KV head.  The
indexer remains replicated.  This launch consumes the rank-local mixed row,
computes the local QKVG and replicated index projections, normalizes and
rotates only the heads that depend on each completed projection tile, and
writes K/V/index cache rows directly.

The native HC front stays outside this kernel until a dedicated HC probe beats
its measured boundary.  QSA K2 may consume the public query/gate/index-query
seam through the native implementation while its persistent replacement is
built independently.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen4_exp_mono.config import (
    ALIGN,
    HIDDEN,
    QSA_COMPRESS_RATIO,
    QSA_HEAD_DIM,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    QSA_INDEX_QK_SIZE,
    QSA_MROPE_SECTION,
    QSA_ROTARY_DIM,
)
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels import finish, qsa_post
from atom.models.qwen4_exp_mono.kernels.collective import (
    publish_pair,
    pull_sum_pair,
)
from atom.models.qwen4_exp_mono.kernels.hc import combine_gdn_output_pair_raw
from atom.models.qwen4_exp_mono.peer_layout import PART_COLS, PARTS, peer_offsets
from atom.models.qwen4_exp_mono.plan import LayerKind, MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    butterfly,
    hw_rsq,
    kernel_symbol,
    launcher,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import Mailbox, publish, sreg
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
EPS = 1e-6
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
INDEX_K_GROUP0 = INDEX_Q_SIZE // ROWS
INDEX_GROUPS = INDEX_ROWS // ROWS
TOTAL_PROJECTION_TASKS = QKV_GROUPS + INDEX_GROUPS
EPILOGUE_TASKS = Q_HEADS + 2 * KV_HEADS + QSA_INDEX_HEADS + 1

assert (Q_HEADS, KV_HEADS, _DIMS.qsa_kv_replicas) == (3, 1, 4)
assert (Q_SIZE, KV_SIZE, QKV_ROWS) == (768, 256, 2048)
assert (INDEX_Q_SIZE, INDEX_ROWS) == (512, 640)
assert TOTAL_PROJECTION_TASKS == 168
assert BLOCKS == 256 and THREADS == 512 and WAVES == 8
assert K_STEPS_PER_WAVE * K_STEP * WAVES == HIDDEN


ABI = KernelAbi(
    (
        "mixed",
        "qkv_weight",
        "q_norm",
        "k_norm",
        "index_weight",
        "index_q_norm",
        "index_k_norm",
        "cos",
        "sin",
        "index_cos",
        "index_sin",
        "positions",
        "position_rows",
        "block_tables",
        "table_stride",
        "slot_mapping",
        "compressed_slot_mapping",
        "token_to_req",
        "logical_positions",
        "page_size",
        "k_cache",
        "v_cache",
        "raw_key_cache",
        "compressed_key_cache",
        "rope_position_cache",
        "query_out",
        "gate_out",
        "index_query_out",
        "scratch",
        "layer",
    )
)

K2_PHASE_ABI = KernelAbi(
    (
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
        "partial_out",
        "scratch",
        "layer",
    )
)

K2_ABI = KernelAbi(
    (
        "peers",
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
        "expert_map",
        "peer_sym",
        "attn_out",
        "streams_out",
        "pending_out",
        "pending_raw",
        "moe_out",
        "scratch",
        "layer",
    )
)


def _align(offset: int) -> int:
    return (offset + ALIGN - 1) // ALIGN * ALIGN


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


def scratch_bytes() -> int:
    return _align(max(offset + size for offset, size in scratch_layout().values()))


def _workspace(scratch, layout, name):
    return scratch + fx.Int64(layout[name][0])


@fx.struct
class QsaTp8Smem:
    activation: fx.Array[fx.Int32, HIDDEN // 2, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


@fx.union
class QsaTp8K2Smem:
    selection: qsa_post.SelectionSmem
    attention: qsa_post.AttentionSmem
    finish: finish.FinishSmem


@traced
def _publish_k2_attention_ready(c, bid):
    """Publish the local three-head barrier without capturing the full context."""

    tid = c["tid"]
    poll = c["poll"]
    put = c["put"]
    head_ready = c["head_ready"]
    attention_ready = c["attention_ready"]
    if bid == 0:
        if tid == 0:
            poll([(head_ready, head, 1) for head in range(Q_HEADS)])
        gpu.barrier()
        if tid == 0:
            put(attention_ready, 0, 1)


@traced
def _wait_k2_attention_ready(poll, ready, tid):
    if tid == 0:
        poll([(ready, 0, 1)])


@traced
def _stage_k2_oproj_local(c, task):
    """One N16 local o_proj tile for the isolated numerical phase."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    row = task * qsa_post.OPROJ_ROWS + lane % qsa_post.OPROJ_ROWS
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(Q_SIZE // (32 * WAVES)):
        k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // qsa_post.OPROJ_ROWS)
        weight = _load_bf8(c["o_proj"], row * Q_SIZE + k0, CM_NT)
        value = _load_bf8(c["gated"], k0, CM_DEV)
        acc = mfma_bf16(weight, value, acc)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < qsa_post.OPROJ_ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(c["red"], row0, 0))
        value1 = bf16_round(row_sum(c["red"], row0 + 1, 0))
        col = task * qsa_post.OPROJ_ROWS + row0
        _store_bf(c["attn_out"], col, value0, CM_DEV)
        _store_bf(c["attn_out"], col + 1, value1, CM_DEV)


@traced
def _stage_k2_oproj_peer(c, mailbox, peer_sym, region_off, layer, task):
    """One N16 local o_proj tile published as this rank's TP partial."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    row = task * qsa_post.OPROJ_ROWS + lane % qsa_post.OPROJ_ROWS
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(Q_SIZE // (32 * WAVES)):
        k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // qsa_post.OPROJ_ROWS)
        weight = _load_bf8(c["o_proj"], row * Q_SIZE + k0, CM_NT)
        value = _load_bf8(c["gated"], k0, CM_DEV)
        acc = mfma_bf16(weight, value, acc)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < qsa_post.OPROJ_ROWS // 2:
        row0 = 2 * tid
        col = task * qsa_post.OPROJ_ROWS + row0
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
def _stage_k2_peer_consume(
    c, mailbox, peers, attention_region_off, layer, part
):
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
            col // qsa_post.OPROJ_ROWS,
            col % qsa_post.OPROJ_ROWS,
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
def _publish_k2_output_groups(poll, put, ready, group_ready, bid, tid):
    """Publish the two HC precombined groups from N32 peer consumers."""

    if bid == PARTS:
        if tid == 0:
            for start in range_constexpr(0, 96, 8):
                poll(
                    [
                        (ready, start + j, 1)
                        for j in range(min(8, 96 - start))
                    ]
                )
        gpu.barrier()
        if tid == 0:
            put(group_ready, 0, 1)
    elif bid == PARTS + 1:
        if tid == 0:
            for start in range_constexpr(0, qsa_post.OPROJ_TASKS - 96, 8):
                poll(
                    [
                        (ready, 96 + start + j, 1)
                        for j in range(
                            min(8, qsa_post.OPROJ_TASKS - 96 - start)
                        )
                    ]
                )
        gpu.barrier()
        if tid == 0:
            put(group_ready, 1, 1)


def _load_bf(ptr, index, cache_modifier=CM_DEV):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(
                rsrc(ptr),
                index,
                vec_width=1,
                dtype=T.bf16,
                cache_modifier=cache_modifier,
            )
        )
    )


def _store_bf(ptr, index, value, cache_modifier=CM_DEV):
    bo.buffer_store(
        fx.BFloat16(value),
        rsrc(ptr),
        index,
        cache_modifier=cache_modifier,
    )


def _load_bf8(ptr, index, cache_modifier=CM_DEV):
    return fx.Vector(
        bo.buffer_load(
            rsrc(ptr),
            index // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    ).bitcast(fx.BFloat16)


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


def _store_i64(ptr, index, value, cache_modifier=CM_DEV):
    bo.buffer_store(fx.Int64(value), rsrc(ptr), index, cache_modifier=cache_modifier)


def _position_index(c, axis):
    return (c["args"]["position_rows"] == 3).select(axis, fx.Int32(0))


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
        fx.ptr_store(value, c["activation"] + at // 2)
    gpu.barrier()


def _projection_accumulator(c, weight, group):
    row = group * ROWS + c["lane"] % ROWS
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(K_STEPS_PER_WAVE):
        k0 = (c["wave"] + WAVES * k_pass) * K_STEP + 8 * (c["lane"] // ROWS)
        w = fx.Vector(
            bo.buffer_load(
                rsrc(weight),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        ).bitcast(fx.BFloat16)
        x = fx.Vector(
            fx.ptr_load(
                c["activation"] + k0 // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, x, acc)
    return acc


@traced
def _stage_qkv_projection(c, group):
    _stage_input(c)
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    acc = _projection_accumulator(c, c["args"]["qkv_weight"], group)
    fx.ptr_store(acc, c["reduction"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        value = bf16_round(row_sum(c["reduction"], tid, 0))
        if group < GATE_GROUPS:
            _store_bf(c["args"]["gate_out"], group * ROWS + tid, value)
        else:
            _store_bf(c["qkv"], (group - GATE_GROUPS) * ROWS + tid, value)
    publish(c["mailbox"].put, c["qkv_ready"], group, 1, tid == 0)


@traced
def _stage_index_projection(c, group):
    _stage_input(c)
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    acc = _projection_accumulator(c, c["args"]["index_weight"], group)
    fx.ptr_store(acc, c["reduction"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        _store_bf(
            c["index"],
            group * ROWS + tid,
            bf16_round(row_sum(c["reduction"], tid, 0)),
        )
    publish(c["mailbox"].put, c["index_ready"], group, 1, tid == 0)


def _poll_groups(c, ready, first, groups):
    c["mailbox"].poll([(ready, first + group, 1) for group in range(groups)])


@traced
def _rope(c, normalized, dim, cos, sin, position0, position1, position2):
    half = QSA_ROTARY_DIM // 2
    rotary = dim < QSA_ROTARY_DIM
    safe_dim = fx.min(dim, QSA_ROTARY_DIM - 1)
    pair = (safe_dim < half).select(safe_dim + half, safe_dim - half)
    other = fx.ptr_load(c["reduction"] + pair)
    freq = safe_dim % half
    axis = fx.Int32(0)
    axis = (
        ((freq % 3) == 1) & (freq < 3 * QSA_MROPE_SECTION[1])
    ).select(fx.Int32(1), axis)
    axis = (
        ((freq % 3) == 2) & (freq < 3 * QSA_MROPE_SECTION[2])
    ).select(fx.Int32(2), axis)
    position = (axis == 0).select(position0, (axis == 1).select(position1, position2))
    table = position * half + freq
    co = _load_bf(cos, table)
    si = _load_bf(sin, table)
    signed = (safe_dim < half).select(-other, other)
    xc = bf16_round(normalized * co)
    ys = bf16_round(signed * si)
    return rotary.select(bf16_round(xc + ys), normalized)


@traced
def _stage_norm_rope(
    c,
    src,
    src_offset,
    weight,
    dim,
    dst,
    dst_offset,
    store_valid,
    ready,
    ready_first,
    ready_groups,
    cos,
    sin,
):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    _poll_groups(c, ready, ready_first, ready_groups)
    x = (tid < dim).select(_load_bf(src, src_offset + fx.min(tid, dim - 1)), 0.0)
    ss = butterfly(x * x, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ss, c["reduction"] + wave)
    gpu.barrier()
    total = fx.ptr_load(c["reduction"] + lane % WAVES)
    total = butterfly(total, (4, 2, 1))
    rstd = hw_rsq(total * fx.Float32(1.0 / dim) + fx.Float32(EPS))
    normalized = bf16_round(
        x
        * rstd
        * (fx.Float32(1.0) + _load_bf(weight, fx.min(tid, dim - 1), CM_NT))
    )
    if tid < dim:
        fx.ptr_store(normalized, c["reduction"] + tid)
    gpu.barrier()
    if (tid < dim) & store_valid:
        position0 = _load_i64(c["args"]["positions"], _position_index(c, 0))
        position1 = position0
        position2 = position0
        if c["args"]["position_rows"] == 3:
            position1 = _load_i64(c["args"]["positions"], _position_index(c, 1))
            position2 = _load_i64(c["args"]["positions"], _position_index(c, 2))
        _store_bf(
            dst,
            dst_offset + tid,
            _rope(
                c,
                normalized,
                tid,
                cos,
                sin,
                position0,
                position1,
                position2,
            ),
        )


@traced
def _stage_value_cache(c):
    _poll_groups(c, c["qkv_ready"], V_GROUP0, QSA_HEAD_DIM // ROWS)
    tid = c["tid"]
    slot = _load_i64(c["args"]["slot_mapping"], 0)
    if (tid < QSA_HEAD_DIM) & (slot >= 0):
        _store_bf(
            c["args"]["v_cache"],
            slot * (KV_HEADS * QSA_HEAD_DIM) + tid,
            _load_bf(c["qkv"], Q_SIZE + KV_SIZE + tid),
        )


@traced
def _stage_index_key(c):
    _poll_groups(
        c,
        c["index_ready"],
        INDEX_K_GROUP0,
        QSA_INDEX_HEAD_DIM // ROWS,
    )
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    slot = _load_i64(c["args"]["slot_mapping"], 0)
    if (tid < QSA_INDEX_HEAD_DIM) & (slot >= 0):
        _store_bf(
            c["args"]["raw_key_cache"],
            slot * QSA_INDEX_HEAD_DIM + tid,
            _load_bf(c["index"], INDEX_Q_SIZE + tid),
        )
    if (c["args"]["rope_position_cache"] != 0) & (tid < 3) & (slot >= 0):
        axis = (tid < c["args"]["position_rows"]).select(tid, fx.Int32(0))
        pos = _load_i64(c["args"]["positions"], _position_index(c, axis))
        _store_i64(c["args"]["rope_position_cache"], slot * 3 + tid, pos)
    gpu.barrier()

    compressed_slot = _load_i64(c["args"]["compressed_slot_mapping"], 0)
    if compressed_slot >= 0:
        logical = _load_i64(c["args"]["logical_positions"], 0)
        request = _load_i32(c["args"]["token_to_req"], 0)
        first_position = logical - (QSA_COMPRESS_RATIO - 1)
        pooled = fx.Float32(0.0)
        if tid < QSA_INDEX_HEAD_DIM:
            for offset in range_constexpr(QSA_COMPRESS_RATIO):
                position = first_position + offset
                table_at = request * c["args"]["table_stride"] + fx.Int32(
                    position // c["args"]["page_size"]
                )
                page = _load_i32(c["args"]["block_tables"], table_at)
                row = (
                    fx.Int64(page) * c["args"]["page_size"]
                    + position % c["args"]["page_size"]
                )
                pooled = pooled + _load_bf(
                    c["args"]["raw_key_cache"], row * QSA_INDEX_HEAD_DIM + tid
                )
            pooled = bf16_round(pooled * fx.Float32(1.0 / QSA_COMPRESS_RATIO))
        ss = butterfly(pooled * pooled, (32, 16, 8, 4, 2, 1))
        if lane == 0:
            fx.ptr_store(ss, c["reduction"] + wave)
        gpu.barrier()
        total = fx.ptr_load(c["reduction"] + lane % WAVES)
        total = butterfly(total, (4, 2, 1))
        rstd = hw_rsq(total * fx.Float32(1.0 / QSA_INDEX_HEAD_DIM) + fx.Float32(EPS))
        normalized = bf16_round(
            pooled
            * rstd
            * (
                fx.Float32(1.0)
                + _load_bf(
                    c["args"]["index_k_norm"],
                    fx.min(tid, QSA_INDEX_HEAD_DIM - 1),
                    CM_NT,
                )
            )
        )
        if tid < QSA_INDEX_HEAD_DIM:
            fx.ptr_store(normalized, c["reduction"] + tid)
        gpu.barrier()
        if tid < QSA_INDEX_HEAD_DIM:
            position0 = first_position
            position1 = first_position
            position2 = first_position
            if c["args"]["rope_position_cache"] != 0:
                first_table_at = request * c["args"]["table_stride"] + fx.Int32(
                    first_position // c["args"]["page_size"]
                )
                first_page = _load_i32(c["args"]["block_tables"], first_table_at)
                first_row = (
                    fx.Int64(first_page) * c["args"]["page_size"]
                    + first_position % c["args"]["page_size"]
                )
                position0 = _load_i64(c["args"]["rope_position_cache"], first_row * 3)
                position1 = _load_i64(
                    c["args"]["rope_position_cache"], first_row * 3 + 1
                )
                position2 = _load_i64(
                    c["args"]["rope_position_cache"], first_row * 3 + 2
                )
            _store_bf(
                c["args"]["compressed_key_cache"],
                compressed_slot * QSA_INDEX_HEAD_DIM + tid,
                _rope(
                    c,
                    normalized,
                    tid,
                    c["args"]["index_cos"],
                    c["args"]["index_sin"],
                    position0,
                    position1,
                    position2,
                ),
            )


@traced
def _run_epilogue(c, task):
    if task < Q_HEADS:
        head = task
        _stage_norm_rope(
            c,
            c["qkv"],
            head * QSA_HEAD_DIM,
            c["args"]["q_norm"],
            QSA_HEAD_DIM,
            c["args"]["query_out"],
            head * QSA_HEAD_DIM,
            fx.Boolean(True),
            c["qkv_ready"],
            Q_GROUP0 + head * (QSA_HEAD_DIM // ROWS),
            QSA_HEAD_DIM // ROWS,
            c["args"]["cos"],
            c["args"]["sin"],
        )
    elif task == Q_HEADS:
        slot = _load_i64(c["args"]["slot_mapping"], 0)
        _stage_norm_rope(
            c,
            c["qkv"],
            Q_SIZE,
            c["args"]["k_norm"],
            QSA_HEAD_DIM,
            c["args"]["k_cache"],
            slot * KV_HEADS * QSA_HEAD_DIM,
            slot >= 0,
            c["qkv_ready"],
            K_GROUP0,
            QSA_HEAD_DIM // ROWS,
            c["args"]["cos"],
            c["args"]["sin"],
        )
    elif task == Q_HEADS + KV_HEADS:
        _stage_value_cache(c)
    elif task < Q_HEADS + 2 * KV_HEADS + QSA_INDEX_HEADS:
        head = task - Q_HEADS - 2 * KV_HEADS
        _stage_norm_rope(
            c,
            c["index"],
            head * QSA_INDEX_HEAD_DIM,
            c["args"]["index_q_norm"],
            QSA_INDEX_HEAD_DIM,
            c["args"]["index_query_out"],
            head * QSA_INDEX_HEAD_DIM,
            fx.Boolean(True),
            c["index_ready"],
            head * (QSA_INDEX_HEAD_DIM // ROWS),
            QSA_INDEX_HEAD_DIM // ROWS,
            c["args"]["index_cos"],
            c["args"]["index_sin"],
        )
    else:
        _stage_index_key(c)


def build_qsa_k1(key: MonoBuild):
    if key.tp != 8 or key.tokens != 1 or key.kind is not LayerKind.QSA:
        raise ValueError("Qwen TP8 QSA K1 requires the TP8/S1/QSA build")
    keyed = key_tuple(key, SOURCES)
    layout = scratch_layout()
    name = kernel_symbol("qwen_tp8_qsa_k1", s=key.tokens)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        mixed: Int64,
        qkv_weight: Int64,
        q_norm: Int64,
        k_norm: Int64,
        index_weight: Int64,
        index_q_norm: Int64,
        index_k_norm: Int64,
        cos: Int64,
        sin: Int64,
        index_cos: Int64,
        index_sin: Int64,
        positions: Int64,
        position_rows: Int32,
        block_tables: Int64,
        table_stride: Int32,
        slot_mapping: Int64,
        compressed_slot_mapping: Int64,
        token_to_req: Int64,
        logical_positions: Int64,
        page_size: Int32,
        k_cache: Int64,
        v_cache: Int64,
        raw_key_cache: Int64,
        compressed_key_cache: Int64,
        rope_position_cache: Int64,
        query_out: Int64,
        gate_out: Int64,
        index_query_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(QsaTp8Smem).peek()
        mailbox = Mailbox(layer)
        c = {
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "activation": lds.activation.ptr,
            "reduction": lds.reduction.ptr,
            "mailbox": mailbox,
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
            "qkv": _workspace(scratch, layout, "qsa_tp8_qkv"),
            "index": _workspace(scratch, layout, "qsa_tp8_index"),
            "args": {
                "mixed": mixed,
                "qkv_weight": qkv_weight,
                "q_norm": q_norm,
                "k_norm": k_norm,
                "index_weight": index_weight,
                "index_q_norm": index_q_norm,
                "index_k_norm": index_k_norm,
                "cos": cos,
                "sin": sin,
                "index_cos": index_cos,
                "index_sin": index_sin,
                "positions": positions,
                "position_rows": position_rows,
                "block_tables": block_tables,
                "table_stride": table_stride,
                "slot_mapping": slot_mapping,
                "compressed_slot_mapping": compressed_slot_mapping,
                "token_to_req": token_to_req,
                "logical_positions": logical_positions,
                "page_size": page_size,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "raw_key_cache": raw_key_cache,
                "compressed_key_cache": compressed_key_cache,
                "rope_position_cache": rope_position_cache,
                "query_out": query_out,
                "gate_out": gate_out,
                "index_query_out": index_query_out,
            },
        }
        if bid < QKV_GROUPS:
            _stage_qkv_projection(c, bid)
        elif bid < TOTAL_PROJECTION_TASKS:
            _stage_index_projection(c, bid - QKV_GROUPS)
        elif bid < TOTAL_PROJECTION_TASKS + EPILOGUE_TASKS:
            _run_epilogue(c, bid - TOTAL_PROJECTION_TASKS)
        _ = keyed

    @launcher(name)
    def launch(
        mixed: Int64,
        qkv_weight: Int64,
        q_norm: Int64,
        k_norm: Int64,
        index_weight: Int64,
        index_q_norm: Int64,
        index_k_norm: Int64,
        cos: Int64,
        sin: Int64,
        index_cos: Int64,
        index_sin: Int64,
        positions: Int64,
        position_rows: Int32,
        block_tables: Int64,
        table_stride: Int32,
        slot_mapping: Int64,
        compressed_slot_mapping: Int64,
        token_to_req: Int64,
        logical_positions: Int64,
        page_size: Int32,
        k_cache: Int64,
        v_cache: Int64,
        raw_key_cache: Int64,
        compressed_key_cache: Int64,
        rope_position_cache: Int64,
        query_out: Int64,
        gate_out: Int64,
        index_query_out: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = keyed
        kernel(
            mixed,
            qkv_weight,
            q_norm,
            k_norm,
            index_weight,
            index_q_norm,
            index_k_norm,
            cos,
            sin,
            index_cos,
            index_sin,
            positions,
            position_rows,
            block_tables,
            table_stride,
            slot_mapping,
            compressed_slot_mapping,
            token_to_req,
            logical_positions,
            page_size,
            k_cache,
            v_cache,
            raw_key_cache,
            compressed_key_cache,
            rope_position_cache,
            query_out,
            gate_out,
            index_query_out,
            scratch,
            layer,
        ).launch(
            grid=(BLOCKS,),
            block=(THREADS,),
            stream=stream,
            value_attrs=PERSISTENT_VALUE_ATTRS,
        )

    ABI.check(kernel, launch)
    return launch


def k2_phase_scratch_layout() -> dict[str, tuple[int, int]]:
    """TP8-local QSA K2 scratch used before peer reduction and common finish."""

    return qsa_post.scratch_layout(
        qsa_heads=Q_HEADS,
        qsa_kv_heads=KV_HEADS,
        qsa_q_size=Q_SIZE,
    )


def k2_phase_scratch_bytes() -> int:
    layout = k2_phase_scratch_layout()
    return _align(max(offset + size for offset, size in layout.values()))


def build_qsa_k2_phase(
    key: MonoBuild,
    *,
    direct: bool,
    direct_splits: int = qsa_post.DIRECT_ATTN_SPLITS,
):
    """Build the production-disabled TP8 attention + local-o_proj phase.

    The phase deliberately stops before the peer hand-off.  It proves the
    rank-local QSA shape and establishes an event/resource lower bound before
    the attention reduce and common finish are composed into one K2 launch.
    """

    if key.tokens != 1 or key.tp != 8 or key.kind is not LayerKind.QSA:
        raise ValueError("Qwen TP8 QSA K2 phase supports only S=1 TP8 QSA")
    if direct_splits not in (
        qsa_post.DIRECT_ATTN_SPLITS,
        qsa_post.ATTN_SPLITS,
    ):
        raise ValueError("Qwen TP8 QSA K2 phase has no requested split bucket")
    if not direct and direct_splits != qsa_post.ATTN_SPLITS:
        raise ValueError("selected QSA K2 requires the full split bucket")

    task_splits = direct_splits if direct else qsa_post.ATTN_SPLITS
    attention_tasks = KV_HEADS * task_splits
    layout = k2_phase_scratch_layout()
    keyed = (
        *key_tuple(key, SOURCES),
        ("phase", "qsa_k2_local"),
        ("direct", direct),
        ("direct_splits", direct_splits),
    )
    stem = "qwen_tp8_qsa_k2_phase_direct" if direct else "qwen_tp8_qsa_k2_phase"
    name = kernel_symbol(stem, s=1)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
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
        partial_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        bid = fx.block_idx.x
        halves = fx.SharedAllocator().allocate(QsaTp8K2Smem)
        mailbox = Mailbox(layer)
        tid = fx.thread_idx.x
        c = {
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "put": mailbox.put,
            "poll": mailbox.poll,
            "args": {
                "query": query,
                "gate": gate,
                "index_query": index_query,
                "block_tables": block_tables,
                "table_stride": table_stride,
                "table_width": table_width,
                "seq_lens": seq_lens,
                "token_to_req": token_to_req,
                "logical_positions": logical_positions,
                "page_size": page_size,
                "cache_pages": cache_pages,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "compressed_key_cache": compressed_key_cache,
                "o_proj": o_proj,
            },
            "qsa_heads": Q_HEADS,
            "qsa_kv_heads": KV_HEADS,
            "heads_per_kv": Q_HEADS // KV_HEADS,
            "qsa_q_size": Q_SIZE,
            "oproj_k_steps": Q_SIZE // (32 * WAVES),
            "attn_split_stride": qsa_post.ATTN_SPLITS,
            "oproj_partial": True,
        }
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
            c[region] = sreg(scratch, layout[name_][0], name_)
        for region in (
            "scores",
            "selected_groups",
            "selected_tokens",
            "attn_max",
            "attn_sum",
            "attn_partials",
            "gated",
        ):
            c[region] = qsa_post._workspace(scratch, layout, f"qsa_{region}")
        c["attn_out"] = partial_out

        if const_expr(not direct):
            for task in range(first_task(bid, 0), qsa_post.INDEX_SCORE_TASKS, BLOCKS):
                qsa_post._stage_index_score(c, task)
            qsa_post._run_selection(c, bid)
        gpu.barrier()
        for task in range(first_task(bid, 0), attention_tasks, BLOCKS):
            qsa_post._stage_attention_split(c, task, direct, task_splits)
        head = first_task(bid, 0)
        if head < Q_HEADS:
            qsa_post._stage_attention_reduce(c, head, task_splits)
        _publish_k2_attention_ready(c, bid)
        _wait_k2_attention_ready(mailbox.poll, c["attention_ready"], c["tid"])
        gpu.barrier()
        oproj_c = {
            "tid": c["tid"],
            "lane": c["lane"],
            "wave": c["wave"],
            "red": c["red"],
            "gated": c["gated"],
            "o_proj": o_proj,
            "attn_out": partial_out,
        }
        for task in range(first_task(bid, 0), qsa_post.OPROJ_TASKS, BLOCKS):
            _stage_k2_oproj_local(oproj_c, task)
        _ = keyed

    @launcher(name)
    def launch(
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
        partial_out: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        kernel(
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
            partial_out,
            scratch,
            layer,
            value_attrs=PERSISTENT_VALUE_ATTRS,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    K2_PHASE_ABI.check(kernel, launch)
    return launch


def build_qsa_k2(
    key: MonoBuild,
    *,
    direct: bool,
    direct_splits: int = qsa_post.DIRECT_ATTN_SPLITS,
):
    """Build TP8 QSA attention, peer hand-off and common finish as one K2."""

    if key.tokens != 1 or key.tp != 8 or key.kind is not LayerKind.QSA:
        raise ValueError("Qwen TP8 QSA K2 supports only S=1 TP8 QSA")
    if direct_splits not in (
        qsa_post.DIRECT_ATTN_SPLITS,
        qsa_post.ATTN_SPLITS,
    ):
        raise ValueError("Qwen TP8 QSA K2 has no requested split bucket")
    if not direct and direct_splits != qsa_post.ATTN_SPLITS:
        raise ValueError("selected QSA K2 requires the full split bucket")

    task_splits = direct_splits if direct else qsa_post.ATTN_SPLITS
    attention_tasks = KV_HEADS * task_splits
    layout = k2_phase_scratch_layout()
    finish_layout = finish.scratch_layout()
    attention_region_off = peer_offsets(1, 8)["attention"]
    moe_region_off = peer_offsets(1, 8)["moe"]
    keyed = (
        *key_tuple(key, SOURCES),
        ("phase", "qsa_k2_finish"),
        ("direct", direct),
        ("direct_splits", direct_splits),
    )
    stem = "qwen_tp8_qsa_k2_direct" if direct else "qwen_tp8_qsa_k2"
    name = kernel_symbol(stem, s=1)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        peers: Int64,
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
        expert_map: Int64,
        peer_sym: Int64,
        attn_out: Int64,
        streams_out: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        moe_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        bid = fx.block_idx.x
        halves = fx.SharedAllocator().allocate(QsaTp8K2Smem)
        mailbox = Mailbox(layer)
        args = {
            "streams": streams,
            "attn_out": attn_out,
            "attn_raw": attn_raw,
            "hc_norm": hc_norm,
            "hc_down_inject": hc_down_inject,
            "hc_up": hc_up,
            "router": router,
            "shared_w13": shared_w13,
            "shared_w2": shared_w2,
            "routed_w13": routed_w13,
            "routed_w13_scale": routed_w13_scale,
            "routed_w2": routed_w2,
            "routed_w2_scale": routed_w2_scale,
            "expert_map": expert_map,
            "streams_out": streams_out,
            "pending_out": pending_out,
            "pending_raw": pending_raw,
            "moe_out": moe_out,
            "query": query,
            "gate": gate,
            "index_query": index_query,
            "block_tables": block_tables,
            "table_stride": table_stride,
            "table_width": table_width,
            "seq_lens": seq_lens,
            "token_to_req": token_to_req,
            "logical_positions": logical_positions,
            "page_size": page_size,
            "cache_pages": cache_pages,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "compressed_key_cache": compressed_key_cache,
            "o_proj": o_proj,
        }
        finish_c = finish.finish_context(
            halves.finish.peek(),
            mailbox,
            finish_layout,
            scratch,
            args,
        )
        tid = fx.thread_idx.x
        qsa_c = {
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "put": mailbox.put,
            "poll": mailbox.poll,
            "args": args,
            "qsa_heads": Q_HEADS,
            "qsa_kv_heads": KV_HEADS,
            "heads_per_kv": Q_HEADS // KV_HEADS,
            "qsa_q_size": Q_SIZE,
            "oproj_k_steps": Q_SIZE // (32 * WAVES),
            "attn_split_stride": qsa_post.ATTN_SPLITS,
            "red": halves.attention.peek().red.ptr,
            "kv": halves.attention.peek().kv.ptr,
        }
        qsa_c["selection_lds"] = {
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
            qsa_c[region] = sreg(scratch, layout[name_][0], name_)
        for region in (
            "scores",
            "selected_groups",
            "selected_tokens",
            "attn_max",
            "attn_sum",
            "attn_partials",
            "gated",
        ):
            qsa_c[region] = qsa_post._workspace(scratch, layout, f"qsa_{region}")

        if const_expr(not direct):
            for task in range(first_task(bid, 0), qsa_post.INDEX_SCORE_TASKS, BLOCKS):
                qsa_post._stage_index_score(qsa_c, task)
            qsa_post._run_selection(qsa_c, bid)
        gpu.barrier()
        for task in range(first_task(bid, 0), attention_tasks, BLOCKS):
            qsa_post._stage_attention_split(qsa_c, task, direct, task_splits)
        head = first_task(bid, 0)
        if head < Q_HEADS:
            qsa_post._stage_attention_reduce(qsa_c, head, task_splits)
        _publish_k2_attention_ready(qsa_c, bid)
        _wait_k2_attention_ready(
            mailbox.poll, qsa_c["attention_ready"], qsa_c["tid"]
        )
        gpu.barrier()
        oproj_c = {
            "tid": qsa_c["tid"],
            "lane": qsa_c["lane"],
            "wave": qsa_c["wave"],
            "red": qsa_c["red"],
            "gated": qsa_c["gated"],
            "o_proj": o_proj,
        }
        for task in range(first_task(bid, 0), qsa_post.OPROJ_TASKS, BLOCKS):
            _stage_k2_oproj_peer(
                oproj_c,
                mailbox,
                peer_sym,
                attention_region_off,
                layer,
                task,
            )
        peer_c = {
            "tid": qsa_c["tid"],
            "attn_out": attn_out,
            "streams": streams,
            "attn_raw": attn_raw,
            "streams_out": streams_out,
        }
        for part in range(first_task(bid, 0), PARTS, BLOCKS):
            _stage_k2_peer_consume(
                peer_c, mailbox, peers, attention_region_off, layer, part
            )
            _publish_k2_part_ready(
                mailbox.put, qsa_c["oproj_ready"], part, qsa_c["tid"]
            )
        _publish_k2_output_groups(
            mailbox.poll,
            mailbox.put,
            qsa_c["oproj_ready"],
            qsa_c["output_group_ready"],
            bid,
            qsa_c["tid"],
        )

        finish_c.update(
            {
                "attn_group_ready": qsa_c["output_group_ready"],
                "attn_precombined": True,
                "expert_owned": True,
                "shared_intermediate": key.dims.shared_intermediate,
                "publish_partial": True,
                "peer_sym": peer_sym,
                "peers": peers,
                "peer_region_off": moe_region_off,
                "layer": layer,
            }
        )
        finish.run_finish_tp8(finish_c, bid)
        _ = keyed

    @launcher(name)
    def launch(
        peers: Int64,
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
        expert_map: Int64,
        peer_sym: Int64,
        attn_out: Int64,
        streams_out: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        moe_out: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        kernel(
            peers,
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
            expert_map,
            peer_sym,
            attn_out,
            streams_out,
            pending_out,
            pending_raw,
            moe_out,
            scratch,
            layer,
            value_attrs=PERSISTENT_VALUE_ATTRS,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    K2_ABI.check(kernel, launch)
    return launch
