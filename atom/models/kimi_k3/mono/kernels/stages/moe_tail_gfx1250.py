# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 local-compute callables for Kimi K2's post-expert tail.

``latr_partials`` and ``rank_final`` are rank-major staging interfaces.  This
module computes against them but does not publish, poll, fence, or claim XGMI
visibility.  Persistent orchestration must establish readiness before calling
the consumers.
"""

from dataclasses import dataclass, field

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.kernels.stages import (
    gemv_gfx1250 as gemv_device,
)
from atom.models.kimi_k3.mono.kernels.stages import gemv_layout
from atom.models.kimi_k3.mono.kernels.stages.moe_tail_layout import (
    HIDDEN,
    LATENT,
    LNORM_TASKS,
    ROWS,
    SHARED_INTERMEDIATE,
    TP,
    UP_ROWS,
    TailOwnershipContract,
)
from atom.mono.device.ops import (
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    rsrc,
    traced,
    wave_sum,
)
from atom.mono.plan.architecture import GFX1250


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


def _rank_pair(pointer, rank, tokens, width, token, column):
    return _pair(
        pointer,
        ((rank * tokens + token) * width + column) // 2,
    )


def _rank_order_pair(pointer, tokens, width, token, column):
    word = _rank_pair(pointer, 0, tokens, width, token, column)
    low, high = bf_lo(word), bf_hi(word)
    for rank in range_constexpr(1, TP):
        word = _rank_pair(pointer, rank, tokens, width, token, column)
        low = low + bf_lo(word)
        high = high + bf_hi(word)
    return low, high


@traced
def stage_latr_reduce(c, task):
    """Rank-order latent sum into local BF16 staging, 128 columns per task."""

    tokens, tid, threads = c["S"], c["tid"], c["threads"]
    total = tokens * 64
    for index in range_constexpr((total + threads - 1) // threads):
        work = tid + threads * index
        if work < total:
            token, pair = work // 64, work % 64
            column = task * 128 + 2 * pair
            low, high = _rank_order_pair(
                c["tail"]["latr_partials"],
                tokens,
                LATENT,
                token,
                column,
            )
            _store_pair(
                c["tail"]["latr_sum"],
                (token * LATENT + column) // 2,
                bf16_round(low),
                bf16_round(high),
            )
    gpu.barrier()


@traced
def stage_lnorm_stats(c, task):
    """Exact gfx950 XOR tree: LDS offset-32 pair, then wave32 reduction."""

    tokens = c["S"]
    tid, lane, wave, threads = (
        c["tid"],
        c["lane"],
        c["wave"],
        c["threads"],
    )
    total = tokens * 64
    for index in range_constexpr((total + threads - 1) // threads):
        work = tid + threads * index
        if work < total:
            token, pair = work // 64, work % 64
            column = task * 128 + 2 * pair
            word = _pair(
                c["tail"]["latr_sum"],
                (token * LATENT + column) // 2,
            )
            low, high = bf_lo(word), bf_hi(word)
            fx.ptr_store(
                low * low + high * high,
                c["tail"]["norm_tree"] + token * 64 + pair,
            )
    gpu.barrier()

    for index in range_constexpr((tokens + c["waves"] - 1) // c["waves"]):
        token = wave + c["waves"] * index
        if token < tokens:
            value = fx.ptr_load(
                c["tail"]["norm_tree"] + token * 64 + lane
            ) + fx.ptr_load(c["tail"]["norm_tree"] + token * 64 + lane + 32)
            total_square = wave_sum(value, GFX1250)
            if lane == 0:
                fx.ptr_store(
                    total_square,
                    c["tail"]["norm_partial"] + task * tokens + token,
                )
    gpu.barrier()


@traced
def stage_lnorm_rstd(c, token):
    """Serial task-order fold of the 28 norm partials for one token."""

    if c["tid"] == 0:
        total = fx.ptr_load(c["tail"]["norm_partial"] + token)
        for task in range_constexpr(1, LNORM_TASKS):
            total = total + fx.ptr_load(
                c["tail"]["norm_partial"] + task * c["S"] + token
            )
        inverse = hw_rsq(total * fx.Float32(1.0 / LATENT) + fx.Float32(c["ln_eps"]))
        fx.ptr_store(inverse, c["tail"]["rstd"] + token)
    gpu.barrier()


@traced
def stage_lnorm_apply(c, task):
    """Apply staged rstd and BF16 norm weights to one 128-column slab."""

    tokens, tid, threads = c["S"], c["tid"], c["threads"]
    total = tokens * 64
    for index in range_constexpr((total + threads - 1) // threads):
        work = tid + threads * index
        if work < total:
            token, pair = work // 64, work % 64
            column = task * 128 + 2 * pair
            word = _pair(
                c["tail"]["latr_sum"],
                (token * LATENT + column) // 2,
            )
            weight_low = fx.Float32(
                fx.BFloat16(
                    bo.buffer_load(
                        rsrc(c["args"]["ln_w"]),
                        column,
                        vec_width=1,
                        dtype=T.bf16,
                    )
                )
            )
            weight_high = fx.Float32(
                fx.BFloat16(
                    bo.buffer_load(
                        rsrc(c["args"]["ln_w"]),
                        column + 1,
                        vec_width=1,
                        dtype=T.bf16,
                    )
                )
            )
            inverse = fx.ptr_load(c["tail"]["rstd"] + token)
            _store_pair(
                c["tail"]["ln"],
                (token * LATENT + column) // 2,
                bf_lo(word) * inverse * weight_low,
                bf_hi(word) * inverse * weight_high,
            )
    gpu.barrier()


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
def stage_sdown(c, task, operands, waves):
    """Shared-down GEMV into this rank's final-contribution plane."""

    row_stride = SHARED_INTERMEDIATE + gemv_layout.LDS_PAD
    _rows_to_lds(c, c["tail"]["h"], SHARED_INTERMEDIATE, row_stride)
    accumulator = gemv_device.mfmas(
        c["lane"],
        c["wave"],
        waves,
        c["tail"]["gemv_rows"],
        row_stride,
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
    total = c["S"] * (ROWS // 2)
    for index in range_constexpr((total + c["threads"] - 1) // c["threads"]):
        work = c["tid"] + c["threads"] * index
        if work < total:
            token, pair = work // (ROWS // 2), work % (ROWS // 2)
            row = task * ROWS + 2 * pair
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
            if const_expr(c.get("shared_partial") is not None):
                _store_pair(
                    c["shared_partial"],
                    (token * HIDDEN + row) // 2,
                    low,
                    high,
                )
            else:
                _store_pair(
                    c["tail"]["rank_final"],
                    ((c["rank"] * c["S"] + token) * HIDDEN + row) // 2,
                    low,
                    high,
                )
    gpu.barrier()


@traced
def stage_up(c, task, operands, waves):
    """Rank-local latent up GEMV plus this rank's shared-down contribution."""

    row_stride = LATENT + gemv_layout.LDS_PAD
    _rows_to_lds(c, c["tail"]["ln"], LATENT, row_stride)
    accumulator = gemv_device.mfmas(
        c["lane"],
        c["wave"],
        waves,
        c["tail"]["gemv_rows"],
        row_stride,
        LATENT,
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
    total = c["S"] * (ROWS // 2)
    for index in range_constexpr((total + c["threads"] - 1) // c["threads"]):
        work = c["tid"] + c["threads"] * index
        if work < total:
            token, pair = work // (ROWS // 2), work % (ROWS // 2)
            row = c["rank"] * UP_ROWS + task * ROWS + 2 * pair
            location = ((c["rank"] * c["S"] + token) * HIDDEN + row) // 2
            shared = _pair(c["tail"]["rank_final"], location)
            low = bf16_round(
                bf_lo(shared)
                + gemv_device.row_sum(
                    c["tail"]["gemv_accumulator"],
                    2 * pair,
                    token,
                    waves,
                )
            )
            high = bf16_round(
                bf_hi(shared)
                + gemv_device.row_sum(
                    c["tail"]["gemv_accumulator"],
                    2 * pair + 1,
                    token,
                    waves,
                )
            )
            _store_pair(c["tail"]["rank_final"], location, low, high)
    gpu.barrier()


@traced
def stage_final_reduce(c, task):
    """Rank-order final reduction from staged contributions to BF16 output."""

    tokens, tid, threads = c["S"], c["tid"], c["threads"]
    total = tokens * 64
    for index in range_constexpr((total + threads - 1) // threads):
        work = tid + threads * index
        if work < total:
            token, pair = work // 64, work % 64
            column = task * 128 + 2 * pair
            low, high = _rank_order_pair(
                c["tail"]["rank_final"],
                tokens,
                HIDDEN,
                token,
                column,
            )
            word = (
                fx.Vector.from_elements([low, high], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
            bo.buffer_store(
                word,
                rsrc(c["args"]["out"]),
                (token * HIDDEN + column) // 2,
            )
    gpu.barrier()


@dataclass(frozen=True)
class TailBackend:
    ownership: TailOwnershipContract
    _matrix: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.ownership.architecture is not GFX1250:
            raise ValueError("gfx1250 tail backend needs gfx1250 ownership")
        object.__setattr__(
            self,
            "_matrix",
            gemv_layout.stage_backend_for(GFX1250, waves=self.ownership.waves),
        )

    def sdown_loads(self, c, task):
        return self._matrix.loads(
            c["lane"],
            c["wave"],
            c["args"]["w_sd"],
            task * ROWS,
            SHARED_INTERMEDIATE,
        )

    @property
    def token_tiles(self):
        return tuple(self._matrix.token_tiles(self.ownership.tokens))

    @staticmethod
    def _require_matrix_tile_context(c):
        if c["S"] > 16:
            raise ValueError(
                "gfx1250 tail WMMA callables require an M16 tile context " "for S > 16"
            )

    def stage_sdown(self, c, task, operands):
        self._require_matrix_tile_context(c)
        stage_sdown(c, task, operands, self.ownership.waves)

    @staticmethod
    def stage_latr_reduce(c, task):
        stage_latr_reduce(c, task)

    @staticmethod
    def stage_lnorm_stats(c, task):
        stage_lnorm_stats(c, task)

    @staticmethod
    def stage_lnorm_rstd(c, token):
        stage_lnorm_rstd(c, token)

    @staticmethod
    def stage_lnorm_apply(c, task):
        stage_lnorm_apply(c, task)

    def up_loads(self, c, task):
        return self._matrix.loads(
            c["lane"],
            c["wave"],
            c["args"]["w_up"],
            task * ROWS,
            LATENT,
        )

    def stage_up(self, c, task, operands):
        self._require_matrix_tile_context(c)
        stage_up(c, task, operands, self.ownership.waves)

    @staticmethod
    def stage_final_reduce(c, task):
        stage_final_reduce(c, task)
