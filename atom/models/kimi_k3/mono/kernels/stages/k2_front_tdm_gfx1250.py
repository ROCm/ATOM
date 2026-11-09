# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replicated gfx1250 K2 front: no all-gather.

Every rank projects the whole router and latent, like the production MoE, plus
its shared gate/up shard, through ``gemv_tdm_gfx1250``. CTA ``p`` owns the
16-row tasks ``2p`` and ``2p + 1``; both always fall in one weight, and in the
latent they are one 32-column E8M0 group, so a CTA quantizes the latent columns
it projected without waiting on any other CTA.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT, SUPPORTED_TP
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250, moe_gfx1250
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_words
from atom.mono.device.ops import butterfly, rsrc
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = 256
WAVES = 8
ROWS = 16
SHARED_ROWS = 2 * 2 * 3072 // SUPPORTED_TP[0]
ROUTER_TASKS = EXPERTS // ROWS
LATENT_TASKS = LATENT // ROWS
SHARED_TASKS = SHARED_ROWS // ROWS
LATENT_PAIR0 = ROUTER_TASKS // 2
SHARED_PAIR0 = (ROUTER_TASKS + LATENT_TASKS) // 2
PAIRS = (ROUTER_TASKS + LATENT_TASKS + SHARED_TASKS) // 2
ACCUMULATOR_WORDS = WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS

REPLICATED_FRONT_ABI = KernelAbi(
    (
        "x",
        "router_weight",
        "latent_weight",
        "shared_weight",
        "router_logits",
        "latent_fp4",
        "latent_scales",
        "shared_gate_up",
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


def build_replicated_front(plan: Gfx1250KimiBuildPlan) -> K2ReplicatedFrontKernel:
    if plan.architecture is not GFX1250:
        raise ValueError("replicated K2 front requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("replicated K2 front requires 256 threads/eight waves")
    if PAIRS > plan.blocks:
        raise ValueError(f"replicated K2 front needs {PAIRS} CTAs")
    rows = plan.tokens
    tiles = (rows + ROWS - 1) // ROWS
    staged = sum(tdm_lds_words(rows, 2))

    @fx.struct
    class Lds:
        gemv: fx.Array[fx.Int32, staged, 16]
        red: fx.Array[fx.Float32, ACCUMULATOR_WORDS, 16]
        # One token's 32 projected latent columns, as 16 BF16 pairs.
        group: fx.Array[fx.Int32, rows * 16, 16]

    name = f"k3_k2_gfx1250_replicated_front_s{rows}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_gate_up: Int64,
    ):
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
            accumulators = gemv_tdm.project(
                lds.gemv.ptr,
                x,
                HIDDEN,
                rows,
                weight,
                weight_rows,
                HIDDEN,
                [row0, row0 + ROWS],
                lane,
                wave,
            )
            for tile in range_constexpr(tiles):
                tile_rows = min(ROWS, rows - tile * ROWS)
                for task in range_constexpr(2):
                    gpu.barrier()
                    gemv_gfx1250.store_accumulator(
                        accumulators[task][tile], lds.red.ptr, wave, lane
                    )
                    gpu.barrier()
                    if tid < ROWS * 8:
                        local_token = tid // 8
                        row_pair = tid % 8
                        token = tile * ROWS + local_token
                        low = gemv_gfx1250.row_sum(
                            lds.red.ptr, 2 * row_pair, local_token, WAVES
                        )
                        high = gemv_gfx1250.row_sum(
                            lds.red.ptr, 2 * row_pair + 1, local_token, WAVES
                        )
                        row = row0 + task * ROWS + 2 * row_pair
                        packed = (
                            fx.Vector.from_elements([low, high], fx.Float32)
                            .to(fx.BFloat16)
                            .bitcast(fx.Int32)[0]
                        )
                        if local_token < tile_rows:
                            if router:
                                bo.buffer_store(
                                    low, rsrc(router_logits), token * EXPERTS + row
                                )
                                bo.buffer_store(
                                    high,
                                    rsrc(router_logits),
                                    token * EXPERTS + row + 1,
                                )
                            elif latent:
                                fx.ptr_store(
                                    packed,
                                    lds.group.ptr
                                    + token * 16
                                    + task * 8
                                    + row_pair,
                                )
                            else:
                                bo.buffer_store(
                                    packed,
                                    rsrc(shared_gate_up),
                                    (token * SHARED_ROWS + row) // 2,
                                )
            gpu.barrier()
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
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_gate_up: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            x,
            router_weight,
            latent_weight,
            shared_weight,
            router_logits,
            latent_fp4,
            latent_scales,
            shared_gate_up,
        ).launch(grid=(PAIRS,), block=(THREADS,), stream=stream)

    REPLICATED_FRONT_ABI.check(kernel, launch)
    return K2ReplicatedFrontKernel(name, launch, (PAIRS,), (THREADS,))
