# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL entry points for independently launchable gfx1250 K2 local stages.

These kernels contain no collective, transport pointer, or cross-CTA wait.
The host runner owns all three reductions and stream ordering.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float32, Int32, Int64, T

from atom.models.kimi_k3.mono.config import (
    HIDDEN,
    LATENT,
    SUPPORTED_TP,
    Dims,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import LDS_PAD
from atom.models.kimi_k3.mono.kernels.stages.moe_tail_layout import (
    LNORM_TASKS,
    ROWS,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import WMMA_TOKEN_ROWS
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

WAVES = 8
THREADS = WAVES * GFX1250.wave_size
OUTPUT_TASKS = HIDDEN // ROWS
TP = SUPPORTED_TP[0]
UP_ROWS = HIDDEN // TP
UP_TASKS = UP_ROWS // ROWS


@dataclass(frozen=True)
class LocalKernelEntryPoint:
    """One emitted kernel and its JIT launcher."""

    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    rows: int


def _require_plan(plan: Gfx1250KimiBuildPlan) -> None:
    if plan.architecture is not GFX1250:
        raise ValueError("local K2 kernels require gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError(f"local K2 kernels require {THREADS} threads/{WAVES} waves")


def _load_pair(pointer, pair):
    return fx.Int32(
        bo.buffer_load(
            rsrc(pointer),
            pair,
            vec_width=1,
            dtype=T.i32,
        )
    )


def _store_pair(pointer, pair, low, high):
    word = (
        fx.Vector.from_elements([low, high], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    bo.buffer_store(word, rsrc(pointer), pair)


def build_oproj_local(
    plan: Gfx1250KimiBuildPlan,
) -> LocalKernelEntryPoint:
    """Emit rank-local ``[S,1536] x [7168,1536]`` BF16 projection."""

    _require_plan(plan)
    rows = plan.tokens
    local_k = Dims(TP).proj
    tile_count = plan.token_tiling.tile_count
    row_stride = local_k + LDS_PAD
    name = f"k3_k2_gfx1250_oproj_local_s{rows}"

    @fx.struct
    class Smem:
        activation: fx.Array[
            fx.Int32,
            WMMA_TOKEN_ROWS * row_stride // 2,
            16,
        ]
        accumulator: fx.Array[
            fx.Float32,
            WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS,
            16,
        ]

    @traced
    def compute_tile(
        activation,
        weight,
        output,
        shared,
        tid,
        lane,
        wave,
        task,
        tile,
    ):
        pairs = WMMA_TOKEN_ROWS * local_k // 2
        for index in range_constexpr((pairs + THREADS - 1) // THREADS):
            pair = tid + THREADS * index
            if pair < pairs:
                local_token = pair // (local_k // 2)
                token = tile * WMMA_TOKEN_ROWS + local_token
                source_token = fx.min(token, rows - 1)
                column_pair = pair % (local_k // 2)
                word = (token < rows).select(
                    _load_pair(
                        activation,
                        source_token * (local_k // 2) + column_pair,
                    ),
                    fx.Int32(0),
                )
                fx.ptr_store(
                    word,
                    shared.activation.ptr
                    + (local_token * row_stride + 2 * column_pair) // 2,
                )
        gpu.barrier()

        operands = gemv_gfx1250.loads(
            lane,
            wave,
            WAVES,
            weight,
            task * ROWS,
            local_k,
        )
        accumulator = gemv_gfx1250.mfmas(
            lane,
            wave,
            WAVES,
            shared.activation.ptr,
            row_stride,
            local_k,
            WMMA_TOKEN_ROWS,
            operands,
        )
        gemv_gfx1250.store_accumulator(
            accumulator,
            shared.accumulator.ptr,
            wave,
            lane,
        )
        gpu.barrier()

        work_items = WMMA_TOKEN_ROWS * (ROWS // 2)
        for index in range_constexpr((work_items + THREADS - 1) // THREADS):
            work = tid + THREADS * index
            if work < work_items:
                local_token = work // (ROWS // 2)
                pair = work % (ROWS // 2)
                token = tile * WMMA_TOKEN_ROWS + local_token
                if token < rows:
                    low = gemv_gfx1250.row_sum(
                        shared.accumulator.ptr,
                        2 * pair,
                        local_token,
                        WAVES,
                    )
                    high = gemv_gfx1250.row_sum(
                        shared.accumulator.ptr,
                        2 * pair + 1,
                        local_token,
                        WAVES,
                    )
                    _store_pair(
                        output,
                        (token * HIDDEN + task * ROWS + 2 * pair) // 2,
                        low,
                        high,
                    )
        rocdl.s_wait_storecnt(0)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(activation: Int64, weight: Int64, output: Int64):
        tid = fx.thread_idx.x
        block = fx.block_idx.x
        task = block % OUTPUT_TASKS
        tile = block // OUTPUT_TASKS
        shared = fx.SharedAllocator().allocate(Smem).peek()
        compute_tile(
            activation,
            weight,
            output,
            shared,
            tid,
            tid % GFX1250.wave_size,
            tid // GFX1250.wave_size,
            task,
            tile,
        )

    @flyc.jit
    def launch(
        activation: Int64,
        weight: Int64,
        output: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(activation, weight, output).launch(
            grid=(OUTPUT_TASKS * tile_count,),
            block=(THREADS,),
            stream=stream,
        )

    return LocalKernelEntryPoint(
        name,
        launch,
        (OUTPUT_TASKS * tile_count,),
        (THREADS,),
        rows,
    )


def build_finalize_local(
    plan: Gfx1250KimiBuildPlan,
) -> LocalKernelEntryPoint:
    """Emit RMSNorm, sharded routed-up, and shared-partial merge."""

    _require_plan(plan)
    rows = plan.tokens
    tile_count = plan.token_tiling.tile_count
    row_stride = LATENT + LDS_PAD
    name = f"k3_k2_gfx1250_finalize_local_s{rows}"

    @fx.struct
    class Smem:
        normalized: fx.Array[
            fx.Int32,
            WMMA_TOKEN_ROWS * row_stride // 2,
            16,
        ]
        accumulator: fx.Array[
            fx.Float32,
            WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS,
            16,
        ]
        rstd: fx.Array[fx.Float32, WMMA_TOKEN_ROWS, 16]

    @traced
    def copy_shared(shared_partial, output, tid, task, tile):
        work_items = WMMA_TOKEN_ROWS * (ROWS // 2)
        for index in range_constexpr((work_items + THREADS - 1) // THREADS):
            work = tid + THREADS * index
            if work < work_items:
                local_token = work // (ROWS // 2)
                pair = work % (ROWS // 2)
                token = tile * WMMA_TOKEN_ROWS + local_token
                if token < rows:
                    location = (token * HIDDEN + task * ROWS + 2 * pair) // 2
                    bo.buffer_store(
                        _load_pair(shared_partial, location),
                        rsrc(output),
                        location,
                    )

    @traced
    def normalize_tile(
        routed_reduced,
        norm_weight,
        shared,
        tid,
        lane,
        wave,
        tile,
        epsilon,
    ):
        if wave == 0:
            for local_token in range_constexpr(WMMA_TOKEN_ROWS):
                token = fx.min(
                    tile * WMMA_TOKEN_ROWS + local_token,
                    rows - 1,
                )
                total = fx.Float32(0.0)
                for task in range_constexpr(LNORM_TASKS):
                    pair0 = token * (LATENT // 2) + task * 64 + lane
                    pair1 = pair0 + 32
                    word0 = _load_pair(routed_reduced, pair0)
                    word1 = _load_pair(routed_reduced, pair1)
                    low0, high0 = bf_lo(word0), bf_hi(word0)
                    low1, high1 = bf_lo(word1), bf_hi(word1)
                    pair0_sum = low0 * low0 + high0 * high0
                    pair1_sum = low1 * low1 + high1 * high1
                    partial = pair0_sum + pair1_sum
                    task_sum = wave_sum(partial, GFX1250)
                    total = total + task_sum
                if lane == 0:
                    fx.ptr_store(
                        hw_rsq(total * fx.Float32(1.0 / LATENT) + epsilon),
                        shared.rstd.ptr + local_token,
                    )
        gpu.barrier()

        pairs = WMMA_TOKEN_ROWS * LATENT // 2
        for index in range_constexpr((pairs + THREADS - 1) // THREADS):
            pair = tid + THREADS * index
            if pair < pairs:
                local_token = pair // (LATENT // 2)
                column_pair = pair % (LATENT // 2)
                token = fx.min(
                    tile * WMMA_TOKEN_ROWS + local_token,
                    rows - 1,
                )
                word = _load_pair(
                    routed_reduced,
                    token * (LATENT // 2) + column_pair,
                )
                weight = _load_pair(norm_weight, column_pair)
                inverse = fx.ptr_load(shared.rstd.ptr + local_token)
                normalized = (
                    fx.Vector.from_elements(
                        [
                            bf_lo(word) * inverse * bf_lo(weight),
                            bf_hi(word) * inverse * bf_hi(weight),
                        ],
                        fx.Float32,
                    )
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                fx.ptr_store(
                    normalized,
                    shared.normalized.ptr
                    + (local_token * row_stride + 2 * column_pair) // 2,
                )
        gpu.barrier()

    @traced
    def add_routed_band(
        weight,
        output,
        shared,
        tid,
        lane,
        wave,
        task,
        tile,
        rank,
    ):
        local_task = task - rank * UP_TASKS
        operands = gemv_gfx1250.loads(
            lane,
            wave,
            WAVES,
            weight,
            local_task * ROWS,
            LATENT,
        )
        accumulator = gemv_gfx1250.mfmas(
            lane,
            wave,
            WAVES,
            shared.normalized.ptr,
            row_stride,
            LATENT,
            WMMA_TOKEN_ROWS,
            operands,
        )
        gemv_gfx1250.store_accumulator(
            accumulator,
            shared.accumulator.ptr,
            wave,
            lane,
        )
        gpu.barrier()

        work_items = WMMA_TOKEN_ROWS * (ROWS // 2)
        for index in range_constexpr((work_items + THREADS - 1) // THREADS):
            work = tid + THREADS * index
            if work < work_items:
                local_token = work // (ROWS // 2)
                pair = work % (ROWS // 2)
                token = tile * WMMA_TOKEN_ROWS + local_token
                if token < rows:
                    location = (token * HIDDEN + task * ROWS + 2 * pair) // 2
                    base = _load_pair(output, location)
                    low = bf16_round(
                        bf_lo(base)
                        + gemv_gfx1250.row_sum(
                            shared.accumulator.ptr,
                            2 * pair,
                            local_token,
                            WAVES,
                        )
                    )
                    high = bf16_round(
                        bf_hi(base)
                        + gemv_gfx1250.row_sum(
                            shared.accumulator.ptr,
                            2 * pair + 1,
                            local_token,
                            WAVES,
                        )
                    )
                    _store_pair(output, location, low, high)
        rocdl.s_wait_storecnt(0)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        routed_reduced: Int64,
        shared_partial: Int64,
        norm_weight: Int64,
        weight: Int64,
        output: Int64,
        rank: Int32,
        epsilon: Float32,
    ):
        tid = fx.thread_idx.x
        block = fx.block_idx.x
        task = block % OUTPUT_TASKS
        tile = block // OUTPUT_TASKS
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        shared = fx.SharedAllocator().allocate(Smem).peek()
        copy_shared(shared_partial, output, tid, task, tile)
        rocdl.s_wait_storecnt(0)
        gpu.barrier()
        routed_task = (task >= rank * UP_TASKS) & (task < (rank + 1) * UP_TASKS)
        if routed_task:
            normalize_tile(
                routed_reduced,
                norm_weight,
                shared,
                tid,
                lane,
                wave,
                tile,
                epsilon,
            )
            add_routed_band(
                weight,
                output,
                shared,
                tid,
                lane,
                wave,
                task,
                tile,
                rank,
            )

    @flyc.jit
    def launch(
        routed_reduced: Int64,
        shared_partial: Int64,
        norm_weight: Int64,
        weight: Int64,
        output: Int64,
        rank: Int32,
        epsilon: Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            routed_reduced,
            shared_partial,
            norm_weight,
            weight,
            output,
            rank,
            epsilon,
        ).launch(
            grid=(OUTPUT_TASKS * tile_count,),
            block=(THREADS,),
            stream=stream,
        )

    return LocalKernelEntryPoint(
        name,
        launch,
        (OUTPUT_TASKS * tile_count,),
        (THREADS,),
        rows,
    )
