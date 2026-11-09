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

LDS rows are padded by 8 BF16 after each 512 columns: a TDM pad interval is at
most 256 dwords, and the 1040-byte row pitch spreads the 16 rows one operand
read touches over distinct banks.
"""

import math

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl._mlir.dialects import vector
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages.gemv_gfx1250 import ACCUMULATOR_ELEMENTS
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX1250_GEMV,
    K_PER_MMA,
    LDS_PAD,
    PACK_ELEMENTS,
    ROWS,
    TDM_DEPTH,
    TDM_ROW_STRIDE,
    TDM_SLAB,
    tdm_lds_words,
)
from atom.mono.device.ops import mma_bf16
from atom.mono.plan.architecture import GFX1250

SLAB = TDM_SLAB
ROW_STRIDE = TDM_ROW_STRIDE
DEPTH = TDM_DEPTH
WAVES = 8
WAVE_COLUMNS = SLAB // WAVES
WAVE_STEPS = WAVE_COLUMNS // K_PER_MMA
_DESCRIPTOR_TYPE = 2 << 30


def _i32(value: int):
    return fx.Int32(value - (1 << 32) if value >= 1 << 31 else value)


def load_rows(address, row_stride: int, lds, rows, rows_per_wave: int):
    """Issue this wave's TDM copy of ``rows_per_wave`` of a [rows, SLAB] BF16
    tile starting at ``address`` into ``lds``; rows past ``rows`` are skipped."""

    wave = fx.Int32(rocdl.wave_id())
    row0 = wave * rows_per_wave
    source = address + fx.Int64(row0) * fx.Int64(row_stride * 2)
    destination = fx.Int32(fx.ptrtoint(lds)) + row0 * (ROW_STRIDE * 2)
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
    config = (1 << 16) | (1 << 20) | (pad_interval << 22) | (pad_amount << 25)
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
        cache_policy=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
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


def project(lds, x, x_stride: int, rows: int, w, w_rows: int, k: int, task_rows,
            lane, wave):
    """``x[rows, k] @ w[task_rows[i]:+16, k]^T`` for each task, as v8 f32 WMMA
    accumulators ``[task][16-token tile]``. ``w`` has ``w_rows`` rows of ``k``
    BF16; rows past it are not read and their outputs are meaningless."""

    if k % SLAB:
        raise ValueError(f"TDM GEMV needs K divisible by {SLAB}, got {k}")
    slabs = k // SLAB
    tasks = len(task_rows)
    tiles = (rows + ROWS - 1) // ROWS
    x_words, _ = tdm_lds_words(rows, tasks)
    token_words = rows * ROW_STRIDE // 2
    task_words = ROWS * ROW_STRIDE // 2
    stages = DEPTH + 1
    rows_per_wave = (rows + WAVES - 1) // WAVES
    issued = []

    def issue_tokens(slab):
        load_rows(
            x + fx.Int64(slab * SLAB * 2),
            x_stride,
            lds + (slab % 2) * token_words,
            rows,
            rows_per_wave,
        )
        issued.append(slab)

    def issue_weights(slab):
        for task in range(tasks):
            load_rows(
                w + fx.Int64(task_rows[task]) * fx.Int64(k * 2) + fx.Int64(slab * SLAB * 2),
                k,
                lds + x_words + ((slab % stages) * tasks + task) * task_words,
                fx.Int32(w_rows) - task_rows[task],
                ROWS // WAVES,
            )
            issued.append(slab)

    accumulators = [
        [fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32) for _ in range(tiles)]
        for _ in range(tasks)
    ]
    issue_tokens(0)
    for slab in range(min(DEPTH, slabs)):
        issue_weights(slab)
    for slab in range(slabs):
        if slab:
            # The buffers slab - 1 read are the ones refilled below.
            gpu.barrier()
            if slab + 1 < slabs:
                issue_tokens(slab + 1)
        elif slabs > 1:
            issue_tokens(1)
        if slab + DEPTH < slabs:
            issue_weights(slab + DEPTH)
        last = max(index for index, tag in enumerate(issued) if tag == slab)
        rocdl.s_wait_tensorcnt(len(issued) - 1 - last)
        gpu.barrier()
        weights = [
            _operand(
                lds + x_words + ((slab % stages) * tasks + task) * task_words,
                lane % ROWS,
                wave,
                lane,
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
