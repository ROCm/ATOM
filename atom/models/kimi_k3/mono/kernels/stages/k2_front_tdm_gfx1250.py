# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replicated gfx1250 K2 front: no all-gather.

Every rank projects the whole router and latent, like the production MoE, plus
its shared gate/up shard, through ``gemv_tdm_gfx1250``. CTA ``p`` owns two
16-row tasks of one weight: in the latent they are one 32-column E8M0 group,
and in the shared gate/up they are the gate and the up rows of the same 16
columns, so a CTA quantizes its latent columns or activates its shared columns
without waiting on any other CTA.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT, SUPPORTED_TP
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import (
    gemv_gfx1250,
    moe_expert_gfx1250,
    moe_gfx1250,
)
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_words
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import bf_hi, bf_lo, butterfly, rsrc
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = 256
WAVES = 8
ROWS = 16
SHARED_ROWS = 2 * 2 * 3072 // SUPPORTED_TP[0]
SHARED_COLUMNS = SHARED_ROWS // 2
ROUTER_TASKS = EXPERTS // ROWS
LATENT_TASKS = LATENT // ROWS
SHARED_TASKS = SHARED_ROWS // ROWS
LATENT_PAIR0 = ROUTER_TASKS // 2
SHARED_PAIR0 = (ROUTER_TASKS + LATENT_TASKS) // 2
PAIRS = (ROUTER_TASKS + LATENT_TASKS + SHARED_TASKS) // 2
ACCUMULATOR_WORDS = WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS
# The projection epilogue reduces both tasks of a tile in one round.
assert THREADS == 2 * ROWS * 8

REPLICATED_FRONT_ABI = KernelAbi(
    (
        "x",
        "router_weight",
        "latent_weight",
        "shared_weight",
        "shared_scale",
        "router_logits",
        "latent_fp4",
        "latent_scales",
        "shared_hidden",
    )
)


@dataclass(frozen=True)
class K2ReplicatedFrontKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = REPLICATED_FRONT_ABI
    launch_count: int = 1


def _stage_accumulators(accumulators, red, tiles, wave, lane):
    for tile in range_constexpr(tiles):
        for task in range_constexpr(2):
            gemv_gfx1250.store_accumulator(
                accumulators[task][tile],
                red + (2 * tile + task) * ACCUMULATOR_WORDS,
                wave,
                lane,
            )


def build_replicated_front(
    plan: Gfx1250KimiBuildPlan, fp8_shared: bool = False
) -> K2ReplicatedFrontKernel:
    """``fp8_shared``: the shared gate/up is ATOM's online per_1x128 FP8 with
    its E8M0 block scales in ``shared_scale``."""
    if plan.architecture is not GFX1250:
        raise ValueError("replicated K2 front requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("replicated K2 front requires 256 threads/eight waves")
    if PAIRS > plan.blocks:
        raise ValueError(f"replicated K2 front needs {PAIRS} CTAs")
    rows = plan.tokens
    tiles = (rows + ROWS - 1) // ROWS
    staged = sum(tdm_lds_words(rows, 2))
    # The FP8 shared slabs reuse the LDS planned for the BF16 ones, and the
    # shared CTAs start a cluster, so no cluster mixes the two pipelines.
    assert sum(tdm_lds_words(rows, 2, True, gemv_tdm.FP8_DEPTH)) <= staged
    assert SHARED_PAIR0 % gemv_tdm.CLUSTER_LAUNCH["cluster"][0] == 0

    @fx.struct
    class Lds:
        gemv: fx.Array[fx.Int32, staged, 16]
        # Both tasks' accumulators of every tile.
        red: fx.Array[fx.Float32, 2 * tiles * ACCUMULATOR_WORDS, 16]
        # Per token, 32 projected latent columns or 16 shared gate then 16 up
        # columns, as 16 BF16 pairs.
        group: fx.Array[fx.Int32, rows * 16, 16]

    name = f"k3_k2_gfx1250_replicated_front_s{rows}{'_f8' if fp8_shared else ''}"

    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
    ):
        _ = sources
        tid = fx.thread_idx.x
        pair = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        lds = fx.SharedAllocator().allocate(Lds).peek()
        if pair < PAIRS:
            router = pair < LATENT_PAIR0
            latent = (pair >= LATENT_PAIR0) & (pair < SHARED_PAIR0)
            weight = router.select(
                router_weight, latent.select(latent_weight, shared_weight)
            )
            weight_rows = router.select(
                fx.Int32(EXPERTS),
                latent.select(fx.Int32(LATENT), fx.Int32(SHARED_ROWS)),
            )
            first_pair = router.select(
                fx.Int32(0),
                latent.select(fx.Int32(LATENT_PAIR0), fx.Int32(SHARED_PAIR0)),
            )
            row0 = (pair - first_pair) * (2 * ROWS)
            shared_column0 = (pair - first_pair) * ROWS
            second = (router | latent).select(
                row0 + ROWS, shared_column0 + SHARED_COLUMNS
            )
            first = (router | latent).select(row0, shared_column0)
            if const_expr(fp8_shared):
                if pair >= SHARED_PAIR0:
                    _stage_accumulators(
                        gemv_tdm.project(
                            lds.gemv.ptr,
                            x,
                            HIDDEN,
                            rows,
                            shared_weight,
                            SHARED_ROWS,
                            HIDDEN,
                            [first, second],
                            lane,
                            wave,
                            w_scale=shared_scale,
                            depth=gemv_tdm.FP8_DEPTH,
                        ),
                        lds.red.ptr,
                        tiles,
                        wave,
                        lane,
                    )
                else:
                    _stage_accumulators(
                        gemv_tdm.project(
                            lds.gemv.ptr,
                            x,
                            HIDDEN,
                            rows,
                            weight,
                            weight_rows,
                            HIDDEN,
                            [first, second],
                            lane,
                            wave,
                        ),
                        lds.red.ptr,
                        tiles,
                        wave,
                        lane,
                    )
            else:
                _stage_accumulators(
                    gemv_tdm.project(
                        lds.gemv.ptr,
                        x,
                        HIDDEN,
                        rows,
                        weight,
                        weight_rows,
                        HIDDEN,
                        [first, second],
                        lane,
                        wave,
                    ),
                    lds.red.ptr,
                    tiles,
                    wave,
                    lane,
                )
            gpu.barrier()
            # A round reduces one tile: threads [0, 128) the first task, the
            # rest the second.
            task = tid // (ROWS * 8)
            for tile in range_constexpr(tiles):
                tile_rows = min(ROWS, rows - tile * ROWS)
                slot = lds.red.ptr + (2 * tile + task) * ACCUMULATOR_WORDS
                local_token = tid % (ROWS * 8) // 8
                row_pair = tid % 8
                token = tile * ROWS + local_token
                low = gemv_gfx1250.row_sum(slot, 2 * row_pair, local_token, WAVES)
                high = gemv_gfx1250.row_sum(slot, 2 * row_pair + 1, local_token, WAVES)
                row = row0 + task * ROWS + 2 * row_pair
                packed = (
                    fx.Vector.from_elements([low, high], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                if local_token < tile_rows:
                    if router:
                        bo.buffer_store(low, rsrc(router_logits), token * EXPERTS + row)
                        bo.buffer_store(
                            high, rsrc(router_logits), token * EXPERTS + row + 1
                        )
                    else:
                        fx.ptr_store(
                            packed, lds.group.ptr + token * 16 + task * 8 + row_pair
                        )
            gpu.barrier()
            if pair >= SHARED_PAIR0:
                for round_index in range_constexpr((rows * 8 + THREADS - 1) // THREADS):
                    work = tid + THREADS * round_index
                    if work < rows * 8:
                        token = work // 8
                        column_pair = work % 8
                        gate = fx.ptr_load(lds.group.ptr + token * 16 + column_pair)
                        up = fx.ptr_load(lds.group.ptr + token * 16 + 8 + column_pair)
                        low = moe_expert_gfx1250._situ(bf_lo(gate), bf_lo(up))
                        high = moe_expert_gfx1250._situ(bf_hi(gate), bf_hi(up))
                        bo.buffer_store(
                            fx.Vector.from_elements([low, high], fx.Float32)
                            .to(fx.BFloat16)
                            .bitcast(fx.Int32)[0],
                            rsrc(shared_hidden),
                            (token * SHARED_COLUMNS + shared_column0) // 2 + column_pair,
                        )
            if latent & (tid < rows * 4):
                token = tid // 4
                packed_word = tid % 4
                words = [
                    fx.ptr_load(lds.group.ptr + token * 16 + packed_word * 4 + index)
                    for index in range_constexpr(4)
                ]
                maximum_bits = fx.Int32(0)
                for word in words:
                    maximum_bits = fx.max(maximum_bits, word & 0x7FFF)
                    maximum_bits = fx.max(maximum_bits, (word >> 16) & 0x7FFF)
                maximum = (maximum_bits << 16).bitcast(fx.Float32)
                maximum = butterfly(maximum, (2, 1), fx.max, architecture=GFX1250)
                code = moe_gfx1250._e8m0_roundup(maximum)
                reciprocal_scale = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
                group = row0 // 32
                bo.buffer_store(
                    moe_gfx1250._pack_bf16_words(words, reciprocal_scale),
                    rsrc(latent_fp4),
                    token * (LATENT // 8) + group * 4 + packed_word,
                )
                if packed_word == 0:
                    bo.buffer_store(
                        fx.Int8(code),
                        rsrc(latent_scales),
                        token * (LATENT // 32) + group,
                    )

    @flyc.jit
    def launch(
        x: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            x,
            router_weight,
            latent_weight,
            shared_weight,
            shared_scale,
            router_logits,
            latent_fp4,
            latent_scales,
            shared_hidden,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(PAIRS,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    REPLICATED_FRONT_ABI.check(kernel, launch)
    return K2ReplicatedFrontKernel(name, launch, (PAIRS,), (THREADS,))
