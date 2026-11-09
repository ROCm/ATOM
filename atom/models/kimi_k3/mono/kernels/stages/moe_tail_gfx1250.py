# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared-expert down projection used by the packed gfx1250 K2 finish."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.config import (
    GEMV_ROWS,
    HIDDEN,
    SHARED_INTER,
    SUPPORTED_TP,
)
from atom.models.kimi_k3.mono.kernels.stages import (
    gemv_gfx1250 as gemv_device,
)
from atom.models.kimi_k3.mono.kernels.stages import gemv_layout, moe_expert_gfx1250
from atom.mono.device.ops import bf16_round, bf_hi, bf_lo, rsrc, traced

SHARED_INTERMEDIATE = SHARED_INTER // SUPPORTED_TP[0]
SHARED_ROW_STRIDE = SHARED_INTERMEDIATE + gemv_layout.LDS_PAD


def _pair(pointer, pair_index):
    return fx.Int32(
        bo.buffer_load(
            rsrc(pointer),
            pair_index,
            vec_width=1,
            dtype=T.i32,
        )
    )


def _store_pair(pointer, pair_index, low, high):
    word = (
        fx.Vector.from_elements([low, high], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    bo.buffer_store(word, rsrc(pointer), pair_index)


@traced
def _rows_to_lds(c, source, columns, row_stride):
    tokens, tid, threads = c["S"], c["tid"], c["threads"]
    pairs = tokens * columns // 2
    for index in range_constexpr((pairs + threads - 1) // threads):
        pair = tid + threads * index
        if pair < pairs:
            token, column_pair = pair // (columns // 2), pair % (columns // 2)
            fx.ptr_store(
                _pair(source, pair),
                c["tail"]["gemv_rows"] + (token * row_stride + 2 * column_pair) // 2,
            )
    gpu.barrier()


@traced
def stage_activated_rows(c, gate_up):
    """Stage SiTU(gate, up) of every token's shared gate/up row in
    ``gemv_rows`` once, for ``sdown_staged`` to reuse across tasks."""
    row_stride = SHARED_ROW_STRIDE
    tokens, tid, threads = c["S"], c["tid"], c["threads"]
    pairs = tokens * SHARED_INTERMEDIATE // 2
    for index in range_constexpr((pairs + threads - 1) // threads):
        pair = tid + threads * index
        if pair < pairs:
            token = pair // (SHARED_INTERMEDIATE // 2)
            column_pair = pair % (SHARED_INTERMEDIATE // 2)
            gate = _pair(gate_up, token * SHARED_INTERMEDIATE + column_pair)
            up = _pair(
                gate_up,
                token * SHARED_INTERMEDIATE + SHARED_INTERMEDIATE // 2 + column_pair,
            )
            low = moe_expert_gfx1250._situ(bf_lo(gate), bf_lo(up))
            high = moe_expert_gfx1250._situ(bf_hi(gate), bf_hi(up))
            fx.ptr_store(
                fx.Vector.from_elements([low, high], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0],
                c["tail"]["gemv_rows"] + (token * row_stride + 2 * column_pair) // 2,
            )
    gpu.barrier()


@traced
def stage_hidden_rows(c, hidden):
    """Stage every token's shared hidden row in ``gemv_rows`` once, for
    ``sdown_staged`` to reuse across tasks."""
    _rows_to_lds(c, hidden, SHARED_INTERMEDIATE, SHARED_ROW_STRIDE)


@traced
def stage_sdown(c, task, operands, waves):
    """Shared-down GEMV into the rank-local final contribution."""

    _rows_to_lds(c, c["tail"]["h"], SHARED_INTERMEDIATE, SHARED_ROW_STRIDE)
    sdown_staged(c, task, operands, waves, c["tail"]["gemv_rows"])


@traced
def sdown_staged(c, task, operands, waves, rows):
    """``stage_sdown``'s GEMV over ``c["S"]`` rows already staged at ``rows``."""

    accumulator = gemv_device.mfmas(
        c["lane"],
        c["wave"],
        waves,
        rows,
        SHARED_ROW_STRIDE,
        SHARED_INTERMEDIATE,
        c["S"],
        operands,
    )
    gemv_device.store_accumulator(
        accumulator,
        c["tail"]["gemv_accumulator"],
        c["wave"],
        c["lane"],
    )
    gpu.barrier()
    total = c["S"] * (GEMV_ROWS // 2)
    for index in range_constexpr((total + c["threads"] - 1) // c["threads"]):
        work = c["tid"] + c["threads"] * index
        if work < total:
            token, pair = work // (GEMV_ROWS // 2), work % (GEMV_ROWS // 2)
            row = task * GEMV_ROWS + 2 * pair
            low = bf16_round(
                gemv_device.row_sum(
                    c["tail"]["gemv_accumulator"],
                    2 * pair,
                    token,
                    waves,
                )
            )
            high = bf16_round(
                gemv_device.row_sum(
                    c["tail"]["gemv_accumulator"],
                    2 * pair + 1,
                    token,
                    waves,
                )
            )
            _store_pair(
                c["shared_partial"],
                (token * HIDDEN + row) // 2,
                low,
                high,
            )
    gpu.barrier()
