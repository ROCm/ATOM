# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared QSA projection and cache stages; launch ownership stays in K1.

Both TP specializations use the same BF16 projection, Gemma RMSNorm,
interleaved mRoPE and compressed-index cache math. Contexts expose the same
activation/reduction/workspace names; dimensions and mailbox ownership are
chosen by the launch composition. Cache offsets are BF16 element offsets.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from atom.models.qwen4_exp_mono.shapes import (
    QSA_COMPRESS_RATIO,
    QSA_HEAD_DIM,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    QSA_INDEX_Q_SIZE,
    QSA_MROPE_SECTION,
    QSA_ROTARY_DIM,
)
from atom.models.qwen4_exp_mono.stages.hc import _ld_bf, _ld_bf8, _st_bf
from atom.models.qwen4_exp_mono.stages.memory import (
    load_i32 as _load_i32,
)
from atom.models.qwen4_exp_mono.stages.memory import (
    load_i64 as _load_i64,
)
from atom.models.qwen4_exp_mono.stages.memory import (
    store_i64 as _store_i64,
)
from atom.models.qwen4_exp_mono.stages.projection import projection_accumulator
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    butterfly,
    hw_rsq,
    mfma_bf16,
    row_sum,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import WAVES

EPS = 1e-6
ROWS = 16
K_STEP = 32
INDEX_K_GROUP0 = QSA_INDEX_Q_SIZE // ROWS


def _poll_groups(c, ready, first, groups):
    c["poll"]([(ready, first + group, 1) for group in range(groups)])


def _position_index(c, axis):
    if c.get("tp", 1) == 8:
        return (c["args"]["position_rows"] == 3).select(axis, fx.Int32(0))
    return axis


def _projection_accumulator(c, weight, group):
    return projection_accumulator(
        weight, group, c["lane"], c["wave"], c["x"], cache_modifier=CM_NT
    )


@traced
def _rope(c, normalized, dim, position0, position1, position2, cos, sin):
    """Apply the checkpoint's interleaved mRoPE with native BF16 roundings."""

    half = QSA_ROTARY_DIM // 2
    # Keep this branchless.  A traced helper called inside another traced
    # helper does not reliably rewrite a Python ``if`` over ``dim``; the first
    # version therefore returned the unrotated value for every rotary lane.
    rotary = dim < QSA_ROTARY_DIM
    safe_dim = fx.min(dim, QSA_ROTARY_DIM - 1)
    pair = (safe_dim < half).select(safe_dim + half, safe_dim - half)
    other = fx.ptr_load(c["red"] + pair)
    freq = safe_dim % half
    axis = fx.Int32(0)
    axis = (((freq % 3) == 1) & (freq < 3 * QSA_MROPE_SECTION[1])).select(
        fx.Int32(1), axis
    )
    axis = (((freq % 3) == 2) & (freq < 3 * QSA_MROPE_SECTION[2])).select(
        fx.Int32(2), axis
    )
    position = (axis == 0).select(position0, (axis == 1).select(position1, position2))
    table = position * half + freq
    co = _ld_bf(cos, table, CM_DEV)
    si = _ld_bf(sin, table, CM_DEV)
    signed = (safe_dim < half).select(-other, other)
    # qsa_apply_mrope casts each product to BF16 before the final add.
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
    """One Gemma RMSNorm row followed by text mRoPE and a BF16 store."""

    cache_read = c.get("cache_read", 0)
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    _poll_groups(c, ready, ready_first, ready_groups)
    x = (tid < dim).select(
        _ld_bf(src, src_offset + fx.min(tid, dim - 1), cache_read), 0.0
    )
    ss = butterfly(x * x, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ss, c["red"] + wave)
    gpu.barrier()
    total = fx.ptr_load(c["red"] + lane % WAVES)
    total = butterfly(total, (4, 2, 1))
    rstd = hw_rsq(total * fx.Float32(1.0 / dim) + fx.Float32(EPS))
    normalized = bf16_round(
        x * rstd * (fx.Float32(1.0) + _ld_bf(weight, fx.min(tid, dim - 1), CM_NT))
    )
    if tid < dim:
        fx.ptr_store(normalized, c["red"] + tid)
    gpu.barrier()
    if (tid < dim) & store_valid:
        position0 = _load_i64(c["args"]["positions"], _position_index(c, 0))
        position1 = position0
        position2 = position0
        if c["args"]["position_rows"] == 3:
            position1 = _load_i64(c["args"]["positions"], _position_index(c, 1))
            position2 = _load_i64(c["args"]["positions"], _position_index(c, 2))
        value = _rope(
            c,
            normalized,
            tid,
            position0,
            position1,
            position2,
            cos,
            sin,
        )
        _st_bf(dst, dst_offset + tid, value, CM_DEV)


@traced
def _stage_index_key(c):
    _poll_groups(
        c,
        c["index_ready"],
        INDEX_K_GROUP0,
        QSA_INDEX_HEAD_DIM // ROWS,
    )
    cache_read = c.get("cache_read", 0)
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    slot = _load_i64(c["args"]["slot_mapping"], 0)
    raw = fx.Float32(0.0)
    if (tid < QSA_INDEX_HEAD_DIM) & (slot >= 0):
        raw = _ld_bf(c["qsa_index"], QSA_INDEX_Q_SIZE + tid, cache_read)
        _st_bf(
            c["args"]["raw_key_cache"],
            slot * QSA_INDEX_HEAD_DIM + tid,
            raw,
            CM_DEV,
        )
    if (c["args"]["rope_position_cache"] != 0) & (tid < 3) & (slot >= 0):
        pos = _load_i64(
            c["args"]["positions"],
            _position_index(c, (tid < c["args"]["position_rows"]).select(tid, 0)),
        )
        _store_i64(c["args"]["rope_position_cache"], slot * 3 + tid, pos, CM_DEV)
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
                pooled = pooled + _ld_bf(
                    c["args"]["raw_key_cache"],
                    row * QSA_INDEX_HEAD_DIM + tid,
                    cache_read,
                )
            # qsa_compress_groups materializes its mean into a BF16 tensor
            # before GemmaRMSNorm consumes it.
            pooled = bf16_round(pooled * fx.Float32(1.0 / QSA_COMPRESS_RATIO))
        ss = butterfly(pooled * pooled, (32, 16, 8, 4, 2, 1))
        if lane == 0:
            fx.ptr_store(ss, c["red"] + wave)
        gpu.barrier()
        total = fx.ptr_load(c["red"] + lane % WAVES)
        total = butterfly(total, (4, 2, 1))
        rstd = hw_rsq(total * fx.Float32(1.0 / QSA_INDEX_HEAD_DIM) + fx.Float32(EPS))
        normalized = bf16_round(
            pooled
            * rstd
            * (
                fx.Float32(1.0)
                + _ld_bf(
                    c["args"]["index_k_norm"],
                    fx.min(tid, QSA_INDEX_HEAD_DIM - 1),
                    CM_NT,
                )
            )
        )
        if tid < QSA_INDEX_HEAD_DIM:
            fx.ptr_store(normalized, c["red"] + tid)
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
            value = _rope(
                c,
                normalized,
                tid,
                position0,
                position1,
                position2,
                c["args"]["index_cos"],
                c["args"]["index_sin"],
            )
            _st_bf(
                c["args"]["compressed_key_cache"],
                compressed_slot * QSA_INDEX_HEAD_DIM + tid,
                value,
                CM_DEV,
            )


@traced
def value_cache(c, head, q_size, kv_heads):
    first = (2 * q_size + kv_heads * QSA_HEAD_DIM) // ROWS + head * (
        QSA_HEAD_DIM // ROWS
    )
    _poll_groups(c, c["qkv_ready"], first, QSA_HEAD_DIM // ROWS)
    tid = c["tid"]
    slot = _load_i64(c["args"]["slot_mapping"], 0)
    cache_read = c.get("cache_read", 0)
    if (tid < QSA_HEAD_DIM) & (slot >= 0):
        value = _ld_bf(
            c["qsa_qkv"],
            q_size + kv_heads * QSA_HEAD_DIM + head * QSA_HEAD_DIM + tid,
            cache_read,
        )
        _st_bf(
            c["args"]["v_cache"],
            slot * (kv_heads * QSA_HEAD_DIM) + head * QSA_HEAD_DIM + tid,
            value,
            CM_DEV,
        )


@traced
def run_epilogue(c, task, heads, kv_heads):
    """Projection consumers; every dependency is a named tagged mailbox."""

    if task < heads:
        head = task
        _stage_norm_rope(
            c,
            c["qsa_qkv"],
            (head) * QSA_HEAD_DIM,
            c["args"]["q_norm"],
            QSA_HEAD_DIM,
            c["args"]["query_out"],
            (head) * QSA_HEAD_DIM,
            fx.Boolean(True),
            c["qkv_ready"],
            heads * QSA_HEAD_DIM // ROWS + head * (QSA_HEAD_DIM // ROWS),
            QSA_HEAD_DIM // ROWS,
            c["args"]["cos"],
            c["args"]["sin"],
        )
    elif (task == heads) if kv_heads == 1 else (task < heads + kv_heads):
        head = 0 if kv_heads == 1 else task - heads
        slot = _load_i64(c["args"]["slot_mapping"], 0)
        _stage_norm_rope(
            c,
            c["qsa_qkv"],
            (heads + head) * QSA_HEAD_DIM,
            c["args"]["k_norm"],
            QSA_HEAD_DIM,
            c["args"]["k_cache"],
            (slot * kv_heads + head) * QSA_HEAD_DIM,
            slot >= 0,
            c["qkv_ready"],
            2 * heads * QSA_HEAD_DIM // ROWS + head * (QSA_HEAD_DIM // ROWS),
            QSA_HEAD_DIM // ROWS,
            c["args"]["cos"],
            c["args"]["sin"],
        )
    elif (task == heads + kv_heads) if kv_heads == 1 else (task < heads + 2 * kv_heads):
        head = 0 if kv_heads == 1 else task - heads - kv_heads
        value_cache(c, head, heads * QSA_HEAD_DIM, kv_heads)
    elif task < heads + 2 * kv_heads + QSA_INDEX_HEADS:
        head = task - heads - 2 * kv_heads
        _stage_norm_rope(
            c,
            c["qsa_index"],
            (head) * QSA_INDEX_HEAD_DIM,
            c["args"]["index_q_norm"],
            QSA_INDEX_HEAD_DIM,
            c["args"]["index_query_out"],
            (head) * QSA_INDEX_HEAD_DIM,
            fx.Boolean(True),
            c["index_ready"],
            head * (QSA_INDEX_HEAD_DIM // ROWS),
            QSA_INDEX_HEAD_DIM // ROWS,
            c["args"]["index_cos"],
            c["args"]["index_sin"],
        )
    elif True if kv_heads == 1 else (task == heads + 2 * kv_heads + QSA_INDEX_HEADS):
        _stage_index_key(c)


@traced
def project_tile(c, group, weight, out, ready):
    tid, wave, lane = c["tid"], c["wave"], c["lane"]
    acc = _projection_accumulator(c, weight, group)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        value = bf16_round(row_sum(c["red"], tid, 0))
        _st_bf(out, group * ROWS + tid, value, CM_DEV)
    publish(c["put"], ready, group, 1, tid == 0)


@traced
def project_qkv(c, task, gate_groups, store=_st_bf):
    tid, wave, lane = c["tid"], c["wave"], c["lane"]
    acc = _projection_accumulator(c, c["args"]["qkv_weight"], task)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        value = bf16_round(row_sum(c["red"], tid, 0))
        if task < gate_groups:
            store(c["args"]["gate_out"], task * ROWS + tid, value, CM_DEV)
        else:
            store(c["qsa_qkv"], (task - gate_groups) * ROWS + tid, value, CM_DEV)
    publish(c["put"], c["qkv_ready"], task, 1, tid == 0)


@traced
def oproj_accumulate(c, task, weight, width, acc, start, end, load=_ld_bf8):
    """Accumulate a contiguous K range; callers own waits and output publication."""
    lane, wave = c["lane"], c["wave"]
    row = task * ROWS + lane % ROWS
    for k_pass in range_constexpr(start, end):
        k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // ROWS)
        w = load(weight, row * width + k0, CM_NT)
        value = load(c["gated"], k0, CM_DEV)
        acc = mfma_bf16(w, value, acc)
    return acc
