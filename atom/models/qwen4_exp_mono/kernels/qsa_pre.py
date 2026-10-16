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

from atom.models.qwen4_exp_mono.config import (
    ALIGN,
    HC,
    HIDDEN,
    QSA_COMPRESS_RATIO,
    QSA_HEAD_DIM,
    QSA_HEADS,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    QSA_INDEX_Q_SIZE,
    QSA_INDEX_QK_SIZE,
    QSA_KV_HEADS,
    QSA_KV_SIZE,
    QSA_MROPE_INTERLEAVED,
    QSA_MROPE_SECTION,
    QSA_Q_SIZE,
    QSA_QKV_SIZE,
    QSA_ROTARY_DIM,
)
from atom.models.qwen4_exp_mono.kernels.finish import (
    FinishS2Smem,
    FinishSmem,
    finish_context,
    hc_scratch_layout,
)
from atom.models.qwen4_exp_mono.kernels.hc import (
    _ld_bf,
    _st_bf,
    run_hc_finish,
    run_hc_finish_s2,
    run_hc_finish_s2_partition,
)
from atom.models.qwen4_exp_mono.plan import MonoBuild
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
QKV_GROUPS = QSA_QKV_SIZE // ROWS
INDEX_GROUPS = QSA_INDEX_QK_SIZE // ROWS
GATE_GROUPS = QSA_Q_SIZE // ROWS
Q_GROUP0 = GATE_GROUPS
K_GROUP0 = Q_GROUP0 + QSA_Q_SIZE // ROWS
V_GROUP0 = K_GROUP0 + QSA_KV_SIZE // ROWS
INDEX_K_GROUP0 = QSA_INDEX_Q_SIZE // ROWS
TOTAL_PROJECTION_GROUPS = QKV_GROUPS + INDEX_GROUPS
S4_PAIR_BLOCKS = BLOCKS // 2
S4_PROJECTION_TURNS = (TOTAL_PROJECTION_GROUPS + S4_PAIR_BLOCKS - 1) // S4_PAIR_BLOCKS

assert QSA_QKV_SIZE % ROWS == 0
assert QSA_INDEX_QK_SIZE % ROWS == 0
assert QSA_ROTARY_DIM % 2 == 0
assert sum(QSA_MROPE_SECTION) == QSA_ROTARY_DIM // 2
assert QSA_MROPE_INTERLEAVED


ABI = KernelAbi(
    (
        "streams",
        "pending_out",
        "pending_raw",
        "hc_norm",
        "hc_down_inject",
        "hc_up",
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
        "streams_out",
        "attn_raw_out",
        "query_out",
        "gate_out",
        "index_query_out",
        "scratch",
        "layer",
    )
)


def _align(offset: int, alignment: int = ALIGN) -> int:
    return (offset + alignment - 1) // alignment * alignment


def scratch_layout() -> dict[str, tuple[int, int]]:
    """QSA K1 hand-offs.  The common HC/quant regions retain their names."""

    out = hc_scratch_layout()
    offset = _align(max(off + size for off, size in out.values()))
    for name, pairs in (
        # run_hc_finish publishes the two up-projection groups even though
        # QSA projection consumers also retain the per-tile hc_up_ready ABI.
        ("up_group_ready", 2),
        ("qsa_qkv_ready", QKV_GROUPS),
        ("qsa_index_ready", INDEX_GROUPS),
    ):
        offset = _align(offset, PAIR_BYTES)
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
        offset = _align(offset)
        out[name] = (offset, size)
        offset += size
    return out


def scratch_bytes() -> int:
    layout = scratch_layout()
    return _align(max(off + size for off, size in layout.values()))


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


def _store_i64(ptr, index, value, cache_modifier=CM_DEV):
    bo.buffer_store(fx.Int64(value), rsrc(ptr), index, cache_modifier=cache_modifier)


def _position_index(c, axis):
    """Index the current token in either [S] or [3, S] position storage."""

    if c.get("tokens", 1) == 1:
        return axis
    return (c["args"]["position_rows"] == 3).select(
        axis * c["tokens"] + c["token"], c["token"]
    )


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
def _stage_projection_input_s2(c0, c1):
    """Stage two HC rows once so one projection weight load serves both."""

    c0["poll"]([(c0["up_group_ready"], group, 1) for group in range(2)])
    c1["poll"]([(c1["up_group_ready"], group, 1) for group in range(2)])
    if c0["tid"] < HIDDEN // 8:
        at = c0["tid"] * 8
        value0 = fx.Vector(
            bo.buffer_load(
                rsrc(c0["qsa_mixed"]),
                at // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        value1 = fx.Vector(
            bo.buffer_load(
                rsrc(c1["qsa_mixed"]),
                at // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(value0, c0["x"] + at // 2)
        fx.ptr_store(value1, c0["x"] + (HIDDEN + at) // 2)
    gpu.barrier()


def _projection_accumulator(c, weight, group):
    """One wave's K split for a contiguous BF16 N16 projection tile."""

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
            fx.ptr_load(c["x"] + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, x, acc)
    return acc


def _projection_accumulator_s2(c, weight, group):
    """One weight tile against two token columns of the N16 MFMA."""

    row = group * ROWS + c["lane"] % ROWS
    token = fx.min(c["lane"] % ROWS, 1)
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
                c["x"] + (token * HIDDEN + k0) // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, x, acc)
    return acc


@traced
def _stage_projection(c, group, weight, out, ready):
    """One N16 BF16 projection tile."""

    tid, wave, lane = c["tid"], c["wave"], c["lane"]
    acc = _projection_accumulator(c, weight, group)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        value = bf16_round(row_sum(c["red"], tid, 0))
        _st_bf(out, group * ROWS + tid, value, CM_DEV)
    publish(c["put"], ready, group, 1, tid == 0)


@traced
def _stage_qkv_projection(c, task):
    """QKV projection with public gate rows and compact Q/K/V workspace rows."""

    tid, wave, lane = c["tid"], c["wave"], c["lane"]
    acc = _projection_accumulator(c, c["args"]["qkv_weight"], task)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        value = bf16_round(row_sum(c["red"], tid, 0))
        if task < GATE_GROUPS:
            _st_bf(c["args"]["gate_out"], task * ROWS + tid, value, CM_DEV)
        else:
            _st_bf(
                c["qsa_qkv"],
                (task - GATE_GROUPS) * ROWS + tid,
                value,
                CM_DEV,
            )
    publish(c["put"], c["qkv_ready"], task, 1, tid == 0)


@traced
def _stage_qkv_projection_s2(c0, c1, task):
    """One QKV weight tile produces both token rows."""

    tid, wave, lane = c0["tid"], c0["wave"], c0["lane"]
    acc = _projection_accumulator_s2(c0, c0["args"]["qkv_weight"], task)
    fx.ptr_store(acc, c0["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        value = bf16_round(row_sum(c0["red"], tid, 0))
        if task < GATE_GROUPS:
            _st_bf(c0["args"]["gate_out"], task * ROWS + tid, value, CM_DEV)
        else:
            _st_bf(
                c0["qsa_qkv"],
                (task - GATE_GROUPS) * ROWS + tid,
                value,
                CM_DEV,
            )
    if (tid >= ROWS) & (tid < 2 * ROWS):
        row = tid - ROWS
        value = bf16_round(row_sum(c0["red"], row, 1))
        if task < GATE_GROUPS:
            _st_bf(c1["args"]["gate_out"], task * ROWS + row, value, CM_DEV)
        else:
            _st_bf(
                c1["qsa_qkv"],
                (task - GATE_GROUPS) * ROWS + row,
                value,
                CM_DEV,
            )
    publish(c0["put"], c0["qkv_ready"], task, 1, tid == 0)
    publish(c1["put"], c1["qkv_ready"], task, 1, tid == 0)


@traced
def _stage_projection_s2(c0, c1, group, weight, out0, out1, ready0, ready1):
    """One generic N16 projection weight tile for both token rows."""

    tid, wave, lane = c0["tid"], c0["wave"], c0["lane"]
    acc = _projection_accumulator_s2(c0, weight, group)
    fx.ptr_store(acc, c0["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        value = bf16_round(row_sum(c0["red"], tid, 0))
        _st_bf(out0, group * ROWS + tid, value, CM_DEV)
    if (tid >= ROWS) & (tid < 2 * ROWS):
        row = tid - ROWS
        value = bf16_round(row_sum(c0["red"], row, 1))
        _st_bf(out1, group * ROWS + row, value, CM_DEV)
    publish(c0["put"], ready0, group, 1, tid == 0)
    publish(c1["put"], ready1, group, 1, tid == 0)


@traced
def _run_projection_tasks(c, bid):
    _stage_projection_input(c)
    for turn in range_constexpr(4):
        task = bid + turn * BLOCKS
        if task < QKV_GROUPS:
            _stage_qkv_projection(c, task)
        elif task < TOTAL_PROJECTION_GROUPS:
            group = task - QKV_GROUPS
            _stage_projection(
                c,
                group,
                c["args"]["index_weight"],
                c["qsa_index"],
                c["index_ready"],
            )


@traced
def _run_projection_tasks_s2(c0, c1, bid):
    _stage_projection_input_s2(c0, c1)
    for turn in range_constexpr(4):
        task = bid + turn * BLOCKS
        if task < QKV_GROUPS:
            _stage_qkv_projection_s2(c0, c1, task)
        elif task < TOTAL_PROJECTION_GROUPS:
            group = task - QKV_GROUPS
            _stage_projection_s2(
                c0,
                c1,
                group,
                c0["args"]["index_weight"],
                c0["qsa_index"],
                c1["qsa_index"],
                c0["index_ready"],
                c1["index_ready"],
            )


@traced
def _run_projection_tasks_s2_partition(c0, c1, bid):
    """Project one token pair on 128 CTAs without a cross-pair dependency."""

    _stage_projection_input_s2(c0, c1)
    for turn in range_constexpr(S4_PROJECTION_TURNS):
        task = bid + turn * S4_PAIR_BLOCKS
        if task < QKV_GROUPS:
            _stage_qkv_projection_s2(c0, c1, task)
        elif task < TOTAL_PROJECTION_GROUPS:
            group = task - QKV_GROUPS
            _stage_projection_s2(
                c0,
                c1,
                group,
                c0["args"]["index_weight"],
                c0["qsa_index"],
                c1["qsa_index"],
                c0["index_ready"],
                c1["index_ready"],
            )


def _poll_groups(c, ready, first, groups):
    c["poll"]([(ready, first + group, 1) for group in range(groups)])


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
    src_row,
    weight,
    dim,
    dst,
    dst_row,
    store_valid,
    ready,
    ready_first,
    ready_groups,
    cos,
    sin,
):
    """One Gemma RMSNorm row followed by text mRoPE and a BF16 store."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    _poll_groups(c, ready, ready_first, ready_groups)
    x = (tid < dim).select(_ld_bf(src, src_row * dim + fx.min(tid, dim - 1)), 0.0)
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
        _st_bf(dst, dst_row * dim + tid, value, CM_DEV)


@traced
def _stage_value_cache(c, head):
    first = V_GROUP0 + head * (QSA_HEAD_DIM // ROWS)
    _poll_groups(c, c["qkv_ready"], first, QSA_HEAD_DIM // ROWS)
    tid = c["tid"]
    slot = _load_i64(c["args"]["slot_mapping"], 0)
    if (tid < QSA_HEAD_DIM) & (slot >= 0):
        value = _ld_bf(
            c["qsa_qkv"], QSA_Q_SIZE + QSA_KV_SIZE + head * QSA_HEAD_DIM + tid
        )
        _st_bf(
            c["args"]["v_cache"],
            slot * (QSA_KV_HEADS * QSA_HEAD_DIM) + head * QSA_HEAD_DIM + tid,
            value,
            CM_DEV,
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
    raw = fx.Float32(0.0)
    if (tid < QSA_INDEX_HEAD_DIM) & (slot >= 0):
        raw = _ld_bf(c["qsa_index"], QSA_INDEX_Q_SIZE + tid)
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
                    c["args"]["raw_key_cache"], row * QSA_INDEX_HEAD_DIM + tid
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
def _run_epilogues(c, bid):
    """Projection consumers; every dependency is a named tagged mailbox."""

    task = first_task(bid, 0)
    if task < QSA_HEADS:
        head = task
        _stage_norm_rope(
            c,
            c["qsa_qkv"],
            head,
            c["args"]["q_norm"],
            QSA_HEAD_DIM,
            c["args"]["query_out"],
            head,
            fx.Boolean(True),
            c["qkv_ready"],
            Q_GROUP0 + head * (QSA_HEAD_DIM // ROWS),
            QSA_HEAD_DIM // ROWS,
            c["args"]["cos"],
            c["args"]["sin"],
        )
    elif task < QSA_HEADS + QSA_KV_HEADS:
        head = task - QSA_HEADS
        slot = _load_i64(c["args"]["slot_mapping"], 0)
        _stage_norm_rope(
            c,
            c["qsa_qkv"],
            QSA_HEADS + head,
            c["args"]["k_norm"],
            QSA_HEAD_DIM,
            c["args"]["k_cache"],
            slot * QSA_KV_HEADS + head,
            slot >= 0,
            c["qkv_ready"],
            K_GROUP0 + head * (QSA_HEAD_DIM // ROWS),
            QSA_HEAD_DIM // ROWS,
            c["args"]["cos"],
            c["args"]["sin"],
        )
    elif task < QSA_HEADS + 2 * QSA_KV_HEADS:
        _stage_value_cache(c, task - QSA_HEADS - QSA_KV_HEADS)
    elif task < QSA_HEADS + 2 * QSA_KV_HEADS + QSA_INDEX_HEADS:
        head = task - QSA_HEADS - 2 * QSA_KV_HEADS
        _stage_norm_rope(
            c,
            c["qsa_index"],
            head,
            c["args"]["index_q_norm"],
            QSA_INDEX_HEAD_DIM,
            c["args"]["index_query_out"],
            head,
            fx.Boolean(True),
            c["index_ready"],
            head * (QSA_INDEX_HEAD_DIM // ROWS),
            QSA_INDEX_HEAD_DIM // ROWS,
            c["args"]["index_cos"],
            c["args"]["index_sin"],
        )
    elif task == QSA_HEADS + 2 * QSA_KV_HEADS + QSA_INDEX_HEADS:
        _stage_index_key(c)


def build_qsa_pre(key: MonoBuild):
    """Build QSA K1; narrow widths share projection loads in token pairs."""

    if key.tokens not in (1, 2, 4):
        raise ValueError("Qwen QSA K1 currently has S=1/S=2/S=4 specializations")
    keyed = key_tuple(key, SOURCES)
    layout = scratch_layout()
    name = kernel_symbol("qwen_qsa_pre", s=key.tokens)
    smem = FinishSmem if key.tokens == 1 else FinishS2Smem

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        streams: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        hc_norm: Int64,
        hc_down_inject: Int64,
        hc_up: Int64,
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
        streams_out: Int64,
        attn_raw_out: Int64,
        query_out: Int64,
        gate_out: Int64,
        index_query_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(smem)
        mailbox = Mailbox(layer)

        def token_context(token):
            token_scratch = scratch + fx.Int64(token * scratch_bytes())
            args = {
                "streams": streams + fx.Int64(token * HC * HIDDEN * 2),
                "attn_out": pending_out + fx.Int64(token * HIDDEN * 2),
                "attn_raw": pending_raw + fx.Int64(token * HC * 2),
                "hc_norm": hc_norm,
                "hc_down_inject": hc_down_inject,
                "hc_up": hc_up,
                "streams_out": streams_out + fx.Int64(token * HC * HIDDEN * 2),
                "pending_out": _workspace(token_scratch, layout, "qsa_mixed"),
                "pending_raw": attn_raw_out + fx.Int64(token * HC * 2),
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
                "slot_mapping": slot_mapping + fx.Int64(token * 8),
                "compressed_slot_mapping": compressed_slot_mapping
                + fx.Int64(token * 8),
                "token_to_req": token_to_req + fx.Int64(token * 4),
                "logical_positions": logical_positions + fx.Int64(token * 8),
                "page_size": page_size,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "raw_key_cache": raw_key_cache,
                "compressed_key_cache": compressed_key_cache,
                "rope_position_cache": rope_position_cache,
                "query_out": query_out + fx.Int64(token * QSA_Q_SIZE * 2),
                "gate_out": gate_out + fx.Int64(token * QSA_Q_SIZE * 2),
                "index_query_out": index_query_out
                + fx.Int64(token * QSA_INDEX_Q_SIZE * 2),
            }
            c = finish_context(lds.peek(), mailbox, layout, token_scratch, args)
            c["token"] = token
            c["tokens"] = key.tokens
            c["qsa_mixed"] = _workspace(token_scratch, layout, "qsa_mixed")
            c["qsa_qkv"] = _workspace(token_scratch, layout, "qsa_qkv")
            c["qsa_index"] = _workspace(token_scratch, layout, "qsa_index")
            c["qkv_ready"] = sreg(
                token_scratch, layout["qsa_qkv_ready"][0], "qsa_qkv_ready"
            )
            c["index_ready"] = sreg(
                token_scratch, layout["qsa_index_ready"][0], "qsa_index_ready"
            )
            return c

        if key.tokens == 1:
            c = token_context(0)
            run_hc_finish(c, bid)
            _run_projection_tasks(c, bid)
            _run_epilogues(c, bid)
        else:

            def hc_context(token):
                token_scratch = scratch + fx.Int64(token * scratch_bytes())
                args = {
                    "streams": streams + fx.Int64(token * HC * HIDDEN * 2),
                    "attn_out": pending_out + fx.Int64(token * HIDDEN * 2),
                    "attn_raw": pending_raw + fx.Int64(token * HC * 2),
                    "hc_norm": hc_norm,
                    "hc_down_inject": hc_down_inject,
                    "hc_up": hc_up,
                    "streams_out": streams_out + fx.Int64(token * HC * HIDDEN * 2),
                    "pending_out": _workspace(token_scratch, layout, "qsa_mixed"),
                    "pending_raw": attn_raw_out + fx.Int64(token * HC * 2),
                }
                return finish_context(lds.peek(), mailbox, layout, token_scratch, args)

            def projection_context(token):
                token_scratch = scratch + fx.Int64(token * scratch_bytes())
                tid = fx.thread_idx.x
                return {
                    "tid": tid,
                    "lane": tid % 64,
                    "wave": tid // 64,
                    "red": lds.peek().red.ptr,
                    "x": lds.peek().x.ptr,
                    "put": mailbox.put,
                    "poll": mailbox.poll,
                    "up_ready": sreg(
                        token_scratch,
                        layout["hc_up_ready"][0],
                        "hc_up_ready",
                    ),
                    "up_group_ready": sreg(
                        token_scratch, layout["up_group_ready"][0], "up_group_ready"
                    ),
                    "qkv_ready": sreg(
                        token_scratch,
                        layout["qsa_qkv_ready"][0],
                        "qsa_qkv_ready",
                    ),
                    "index_ready": sreg(
                        token_scratch,
                        layout["qsa_index_ready"][0],
                        "qsa_index_ready",
                    ),
                    "qsa_mixed": _workspace(token_scratch, layout, "qsa_mixed"),
                    "qsa_qkv": _workspace(token_scratch, layout, "qsa_qkv"),
                    "qsa_index": _workspace(token_scratch, layout, "qsa_index"),
                    "args": {
                        "qkv_weight": qkv_weight,
                        "index_weight": index_weight,
                        "gate_out": gate_out + fx.Int64(token * QSA_Q_SIZE * 2),
                    },
                }

            def epilogue_context(token):
                token_scratch = scratch + fx.Int64(token * scratch_bytes())
                tid = fx.thread_idx.x
                return {
                    "tid": tid,
                    "lane": tid % 64,
                    "wave": tid // 64,
                    "red": lds.peek().red.ptr,
                    "poll": mailbox.poll,
                    "token": token,
                    "tokens": key.tokens,
                    "qsa_qkv": _workspace(token_scratch, layout, "qsa_qkv"),
                    "qsa_index": _workspace(token_scratch, layout, "qsa_index"),
                    "qkv_ready": sreg(
                        token_scratch,
                        layout["qsa_qkv_ready"][0],
                        "qsa_qkv_ready",
                    ),
                    "index_ready": sreg(
                        token_scratch,
                        layout["qsa_index_ready"][0],
                        "qsa_index_ready",
                    ),
                    "args": {
                        "q_norm": q_norm,
                        "k_norm": k_norm,
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
                        "slot_mapping": slot_mapping + fx.Int64(token * 8),
                        "compressed_slot_mapping": compressed_slot_mapping
                        + fx.Int64(token * 8),
                        "token_to_req": token_to_req + fx.Int64(token * 4),
                        "logical_positions": logical_positions + fx.Int64(token * 8),
                        "page_size": page_size,
                        "k_cache": k_cache,
                        "v_cache": v_cache,
                        "raw_key_cache": raw_key_cache,
                        "compressed_key_cache": compressed_key_cache,
                        "rope_position_cache": rope_position_cache,
                        "query_out": query_out + fx.Int64(token * QSA_Q_SIZE * 2),
                        "index_query_out": index_query_out
                        + fx.Int64(token * QSA_INDEX_Q_SIZE * 2),
                    },
                }

            if key.tokens == 2:
                hc0 = hc_context(0)
                hc1 = hc_context(1)
                run_hc_finish_s2(hc0, hc1, bid)
                projection0 = projection_context(0)
                projection1 = projection_context(1)
                _run_projection_tasks_s2(projection0, projection1, bid)
                _run_epilogues(epilogue_context(0), bid)
                _run_epilogues(epilogue_context(1), bid)
            elif bid < S4_PAIR_BLOCKS:
                hc0 = hc_context(0)
                hc1 = hc_context(1)
                run_hc_finish_s2_partition(hc0, hc1, bid)
                projection0 = projection_context(0)
                projection1 = projection_context(1)
                _run_projection_tasks_s2_partition(projection0, projection1, bid)
                _run_epilogues(epilogue_context(0), bid)
                _run_epilogues(epilogue_context(1), bid)
            else:
                local = bid - S4_PAIR_BLOCKS
                hc2 = hc_context(2)
                hc3 = hc_context(3)
                run_hc_finish_s2_partition(hc2, hc3, local)
                projection2 = projection_context(2)
                projection3 = projection_context(3)
                _run_projection_tasks_s2_partition(projection2, projection3, local)
                _run_epilogues(epilogue_context(2), local)
                _run_epilogues(epilogue_context(3), local)
        _ = keyed

    @launcher(name)
    def launch(
        streams: Int64,
        pending_out: Int64,
        pending_raw: Int64,
        hc_norm: Int64,
        hc_down_inject: Int64,
        hc_up: Int64,
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
        streams_out: Int64,
        attn_raw_out: Int64,
        query_out: Int64,
        gate_out: Int64,
        index_query_out: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = keyed
        kernel(
            streams,
            pending_out,
            pending_raw,
            hc_norm,
            hc_down_inject,
            hc_up,
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
            streams_out,
            attn_raw_out,
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
