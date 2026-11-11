# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 BF16 GEMV that streams both operands through the Tensor Data Mover.

A CTA owns up to ``tasks`` 16-row weight tasks and every token row. K is walked
in 512-column slabs: the TDM copies each slab's token rows into one of two LDS
buffers and each task's 16 weight rows into one of ``DEPTH + 1`` buffers, so
``DEPTH`` weight slabs and the next token slab are in flight while the waves
multiply the current one. Every wave owns 64 columns of a slab (two K32 WMMA
steps); the v8 f32 accumulators keep ``gemv_gfx1250``'s lane mapping, so its
``store_accumulator``/``row_sum`` finish the cross-wave reduction.

Kernels using it launch in clusters of ``TDM_CLUSTER`` CTAs (``CLUSTER_LAUNCH``):
the token slab loads carry the cluster's multicast mask and a cluster barrier
opens every slab, so the CTAs' identical loads are in flight together and merge.

LDS rows are padded by 8 BF16 after each 512 columns: a TDM pad interval is at
most 256 dwords, and the 1040-byte row pitch spreads the 16 rows one operand
read touches over distinct banks.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl._mlir.dialects import vector
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages.gemv_gfx1250 import (
    ACCUMULATOR_ELEMENTS,
    fp8x8_to_bf16,
)
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX1250_GEMV,
    K_PER_MMA,
    LDS_PAD,
    PACK_ELEMENTS,
    ROWS,
    TDM_CLUSTER,
    TDM_DEPTH,
    TDM_FP8_DEPTH,
    TDM_FP8_SCALE_WORDS,
    TDM_ROW_STRIDE,
    TDM_SLAB,
    tdm_lds_words,
    tdm_task_words,
)
from atom.mono.device.ops import mma_bf16, rsrc
from atom.mono.plan.architecture import GFX1250

SLAB = TDM_SLAB
ROW_STRIDE = TDM_ROW_STRIDE
DEPTH = TDM_DEPTH
FP8_DEPTH = TDM_FP8_DEPTH
WAVES = 8
WAVE_COLUMNS = SLAB // WAVES
WAVE_STEPS = WAVE_COLUMNS // K_PER_MMA
assert WAVES * GFX1250.wave_size == TDM_FP8_SCALE_WORDS
_DESCRIPTOR_TYPE = 2 << 30
# gfx12 CPol TH_NT: each weight slab is read once per launch.
_WEIGHT_CACHE_POLICY = 1
_CLUSTER_MASK = (1 << TDM_CLUSTER) - 1 if TDM_CLUSTER > 1 else 0
CLUSTER_LAUNCH = {
    "value_attrs": {"rocdl.cluster_dims": f"{TDM_CLUSTER},1,1"},
    "cluster": (TDM_CLUSTER, 1, 1),
}


def _i32(value: int):
    return fx.Int32(value - (1 << 32) if value >= 1 << 31 else value)


def load_rows(
    address,
    row_stride: int,
    lds,
    rows,
    rows_per_wave: int,
    mask: int = 0,
    padded: bool = True,
    cache_policy: int = 0,
):
    """Issue this wave's TDM copy of ``rows_per_wave`` of a [rows, SLAB] BF16
    tile starting at ``address`` into ``lds``; rows past ``rows`` are skipped.
    ``mask`` is the cluster multicast mask of identical loads to merge.
    ``padded``: LDS rows ``ROW_STRIDE`` apart, else packed ``SLAB`` apart."""

    wave = fx.Int32(rocdl.wave_id())
    row0 = wave * rows_per_wave
    source = address + fx.Int64(row0) * fx.Int64(row_stride * 2)
    pitch = ROW_STRIDE if padded else SLAB
    destination = fx.Int32(fx.ptrtoint(lds)) + row0 * (pitch * 2)
    live = fx.max(fx.min(fx.Int32(rows) - row0, rows_per_wave), 0)
    group0 = vector.from_elements(
        T.vec(4, T.i32),
        [
            as_ir_value(value)
            for value in (
                (live > 0).select(fx.Int32(1), fx.Int32(0)),
                destination,
                fx.Int32(source & fx.Int64(0xFFFFFFFF)),
                fx.Int32(source >> fx.Int64(32)) | _i32(_DESCRIPTOR_TYPE),
            )
        ],
    )
    pad_interval = int(math.log2(SLAB * 2 // 4)) - 1
    pad_amount = LDS_PAD * 2 // 4 - 1
    config = (1 << 16) | mask
    if padded:
        config |= (1 << 20) | (pad_interval << 22) | (pad_amount << 25)
    group1 = vector.from_elements(
        T.vec(8, T.i32),
        [
            as_ir_value(value)
            for value in (
                _i32(config),
                _i32(SLAB << 16),
                (live & 0xFFFF) << 16,
                (live >> 16) | _i32(SLAB << 16),
                live,
                _i32(row_stride),
                _i32(0),
                _i32(0),
            )
        ],
    )
    zero4 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(_i32(0))] * 4)
    zero8 = vector.from_elements(T.vec(8, T.i32), [as_ir_value(_i32(0))] * 8)
    rocdl_dialect.tensor_load_to_lds(
        group0,
        group1,
        zero4,
        zero4,
        zero8,
        cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_policy),
    )


def _operand(lds, row, wave, lane):
    """One lane's v16 BF16 operands of row ``row`` for this wave's columns."""

    lane_group = lane // ROWS
    column = wave * WAVE_COLUMNS
    steps = []
    for step in range_constexpr(WAVE_STEPS):
        words = []
        for pack in range_constexpr(GFX1250_GEMV.operand_packs_per_lane):
            element = (
                column
                + step * K_PER_MMA
                + lane_group * PACK_ELEMENTS
                + pack * GFX1250_GEMV.operand_pack_stride
            )
            raw = fx.Vector(
                fx.ptr_load(
                    lds + (row * ROW_STRIDE + element) // 2,
                    result_type=fx.Vector.make_type(PACK_ELEMENTS // 2, fx.Int32),
                )
            )
            words.extend(raw[word] for word in range_constexpr(PACK_ELEMENTS // 2))
        steps.append(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16))
    return steps


def _fp8_operand(lds, wave, lane, exponent):
    """``_operand`` of a 16-row FP8 slab in AITER's gfx1250 preshuffle
    (``[K/32][2][16 rows][16 B]``), each value times the E8M0 ``exponent``'s
    power of two, so the BF16 operands are exact."""

    row = lane % ROWS
    lane_group = lane // ROWS
    steps = []
    for step in range_constexpr(WAVE_STEPS):
        chunk = wave * WAVE_STEPS + step
        elements = []
        for half in range_constexpr(2):
            byte = ((chunk * 2 + half) * ROWS + row) * 16 + lane_group * PACK_ELEMENTS
            raw = fx.Vector(
                fx.ptr_load(
                    lds + byte // 4,
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
            )
            converted = fp8x8_to_bf16(raw, exponent)
            elements.extend(converted[i] for i in range_constexpr(PACK_ELEMENTS))
        steps.append(fx.Vector.from_elements(elements, fx.BFloat16))
    return steps


def project(
    lds,
    x,
    x_stride: int,
    rows: int,
    w,
    w_rows: int,
    k: int,
    task_rows,
    lane,
    wave,
    w_scale=None,
    depth: int = DEPTH,
):
    """``x[rows, k] @ w[task_rows[i]:+16, k]^T`` for each task, as v8 f32 WMMA
    accumulators ``[task][16-token tile]``. ``w`` has ``w_rows`` rows of ``k``
    BF16; rows past it are not read and their outputs are meaningless.

    With ``w_scale``, ``w`` is FP8 in AITER's gfx1250 preshuffle with its rows
    padded to 128 and ``w_scale`` its ``[rows / 128, k / 128]`` E8M0 block
    scales: the per_1x128 layout ATOM's online quant leaves."""

    if k % SLAB:
        raise ValueError(f"TDM GEMV needs K divisible by {SLAB}, got {k}")
    slabs = k // SLAB
    fp8 = w_scale is not None
    if fp8 and slabs > GFX1250.wave_size:
        raise ValueError(f"FP8 TDM GEMV holds one slab's scales a lane, K={k}")
    tasks = len(task_rows)
    tiles = (rows + ROWS - 1) // ROWS
    x_words, _ = tdm_lds_words(rows, tasks, fp8)
    token_words = rows * ROW_STRIDE // 2
    task_words = tdm_task_words(fp8)
    stages = depth + 1
    rows_per_wave = (rows + WAVES - 1) // WAVES
    issued = []
    # FP8: wave t copies task t's scale words, one a slab (its four K blocks'
    # E8M0 bytes), to LDS past the weight slabs; the waves past the tasks and
    # the lanes past the slabs fill unread words. Read back from LDS a slab at
    # a time, the exponents stay out of SGPRs, which K1 has none to spare.
    scales = lds + x_words + stages * tasks * task_words
    if fp8:
        row = task_rows[0]
        for task in range(1, tasks):
            row = (wave == task).select(task_rows[task], row)
        word = bo.buffer_load(
            rsrc(w_scale),
            row // 128 * (k // 512) + fx.min(lane, slabs - 1),
            vec_width=1,
            dtype=T.i32,
        )
        fx.ptr_store(fx.Int32(word), scales + wave * GFX1250.wave_size + lane)

    def slab_exponent(task, slab):
        word = fx.ptr_load(scales + task * GFX1250.wave_size + slab)
        return (word >> (wave // (128 // WAVE_COLUMNS) * 8)) & 0xFF

    def issue_tokens(slab):
        load_rows(
            x + fx.Int64(slab * SLAB * 2),
            x_stride,
            lds + (slab % 2) * token_words,
            rows,
            rows_per_wave,
            _CLUSTER_MASK,
        )
        issued.append(slab)

    def issue_weights(slab):
        for task in range(tasks):
            destination = lds + x_words + ((slab % stages) * tasks + task) * task_words
            if fp8:
                # The task's slab is one contiguous 16 x 512 B block: copied as
                # WAVES packed 512-element rows.
                load_rows(
                    w
                    + fx.Int64(task_rows[task]) * fx.Int64(k)
                    + fx.Int64(slab * SLAB * ROWS),
                    SLAB,
                    destination,
                    WAVES,
                    1,
                    padded=False,
                    cache_policy=_WEIGHT_CACHE_POLICY,
                )
            else:
                load_rows(
                    w
                    + fx.Int64(task_rows[task]) * fx.Int64(k * 2)
                    + fx.Int64(slab * SLAB * 2),
                    k,
                    destination,
                    fx.Int32(w_rows) - task_rows[task],
                    ROWS // WAVES,
                    cache_policy=_WEIGHT_CACHE_POLICY,
                )
            issued.append(slab)

    accumulators = [
        [fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32) for _ in range(tiles)]
        for _ in range(tasks)
    ]
    issue_tokens(0)
    for slab in range(min(depth, slabs)):
        issue_weights(slab)
    for slab in range(slabs):
        if slab:
            # The buffers slab - 1 read are the ones refilled below.
            if TDM_CLUSTER > 1:
                cluster.cluster_barrier()
            else:
                gpu.barrier()
            if slab + 1 < slabs:
                issue_tokens(slab + 1)
        elif slabs > 1:
            issue_tokens(1)
        if slab + depth < slabs:
            issue_weights(slab + depth)
        last = max(index for index, tag in enumerate(issued) if tag == slab)
        rocdl.s_wait_tensorcnt(len(issued) - 1 - last)
        gpu.barrier()
        weights = [
            (
                _fp8_operand(
                    lds + x_words + ((slab % stages) * tasks + task) * task_words,
                    wave,
                    lane,
                    slab_exponent(task, slab),
                )
                if fp8
                else _operand(
                    lds + x_words + ((slab % stages) * tasks + task) * task_words,
                    lane % ROWS,
                    wave,
                    lane,
                )
            )
            for task in range(tasks)
        ]
        tokens = lds + (slab % 2) * token_words
        for tile in range(tiles):
            activations = _operand(
                tokens,
                tile * ROWS + fx.min(lane % ROWS, min(ROWS, rows - tile * ROWS) - 1),
                wave,
                lane,
            )
            for task in range(tasks):
                for step in range(WAVE_STEPS):
                    accumulators[task][tile] = mma_bf16(
                        weights[task][step],
                        activations[step],
                        accumulators[task][tile],
                        GFX1250,
                    )
    return accumulators
