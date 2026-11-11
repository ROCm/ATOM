# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replicated gfx1250 K2 finalize: the whole routed up-projection on every rank.

The routed and shared partials are all-reduced together, so a rank finishes the
layer by itself: ``out = bf16(shared + up(normed routed))`` over all HIDDEN rows,
through ``gemv_tdm_gfx1250``, with no second all-reduce. CTA ``p`` owns the
16-row tasks ``2p`` and ``2p + 1`` of the up weight.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_words
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import bf16_round, bf_hi, bf_lo, rsrc
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = 256
WAVES = 8
ROWS = 16
PAIRS = HIDDEN // (2 * ROWS)
ACCUMULATOR_WORDS = WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS
# The epilogue reduces both tasks of a tile in one round.
assert THREADS == 2 * ROWS * 8

REPLICATED_FINALIZE_ABI = KernelAbi(("routed_normed", "shared", "up_weight", "output"))


@dataclass(frozen=True)
class K2ReplicatedFinalizeKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = REPLICATED_FINALIZE_ABI
    launch_count: int = 1


def build_replicated_finalize(plan: Gfx1250KimiBuildPlan) -> K2ReplicatedFinalizeKernel:
    if plan.architecture is not GFX1250:
        raise ValueError("replicated K2 finalize requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("replicated K2 finalize requires 256 threads/eight waves")
    if PAIRS > plan.blocks:
        raise ValueError(f"replicated K2 finalize needs {PAIRS} CTAs")
    rows = plan.tokens
    tiles = (rows + ROWS - 1) // ROWS

    @fx.struct
    class Lds:
        gemv: fx.Array[fx.Int32, sum(tdm_lds_words(rows, 2)), 16]
        # Both tasks' accumulators of every tile.
        red: fx.Array[fx.Float32, 2 * tiles * ACCUMULATOR_WORDS, 16]

    name = f"k3_k2_gfx1250_replicated_finalize_s{rows}"

    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(routed_normed: Int64, shared: Int64, up_weight: Int64, output: Int64):
        _ = sources
        tid = fx.thread_idx.x
        row0 = fx.block_idx.x * (2 * ROWS)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        lds = fx.SharedAllocator().allocate(Lds).peek()
        # Threads [0, 128) finish a tile's first task, the rest its second; the
        # shared rows they add are loaded before the projection.
        task = tid // (ROWS * 8)
        local_token = tid % (ROWS * 8) // 8
        row_pair = tid % 8
        locations = [
            (fx.min(tile * ROWS + local_token, rows - 1) * HIDDEN + row0 + task * ROWS)
            // 2
            + row_pair
            for tile in range_constexpr(tiles)
        ]
        bases = [
            fx.Int32(bo.buffer_load(rsrc(shared), location, vec_width=1, dtype=T.i32))
            for location in locations
        ]
        accumulators = gemv_tdm.project(
            lds.gemv.ptr,
            routed_normed,
            LATENT,
            rows,
            up_weight,
            HIDDEN,
            LATENT,
            [row0, row0 + ROWS],
            lane,
            wave,
        )
        for tile in range_constexpr(tiles):
            for index in range_constexpr(2):
                gemv_gfx1250.store_accumulator(
                    accumulators[index][tile],
                    lds.red.ptr + (2 * tile + index) * ACCUMULATOR_WORDS,
                    wave,
                    lane,
                )
        gpu.barrier()
        for tile in range_constexpr(tiles):
            slot = lds.red.ptr + (2 * tile + task) * ACCUMULATOR_WORDS
            if local_token < min(ROWS, rows - tile * ROWS):
                low = bf16_round(
                    bf_lo(bases[tile])
                    + gemv_gfx1250.row_sum(slot, 2 * row_pair, local_token, WAVES)
                )
                high = bf16_round(
                    bf_hi(bases[tile])
                    + gemv_gfx1250.row_sum(slot, 2 * row_pair + 1, local_token, WAVES)
                )
                bo.buffer_store(
                    fx.Vector.from_elements([low, high], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0],
                    rsrc(output),
                    locations[tile],
                )

    @flyc.jit
    def launch(
        routed_normed: Int64,
        shared: Int64,
        up_weight: Int64,
        output: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            routed_normed,
            shared,
            up_weight,
            output,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(PAIRS,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    REPLICATED_FINALIZE_ABI.check(kernel, launch)
    return K2ReplicatedFinalizeKernel(name, launch, (PAIRS,), (THREADS,))
