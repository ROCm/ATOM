# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-launch resident-grid sharded K2 finish for gfx1250.

The kernel consumes the packed TP8 front all-gather and fuses router,
quantization, routed experts, and the shared expert into one launch.  The
routed latent partial is reduced before finalization by the host-side RCCL
runner.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT, SUPPORTED_TP
from atom.models.kimi_k3.mono.external_collectives import FrontPayloadLayout
from atom.models.kimi_k3.mono.kernels.attention import (
    kda_gfx1250 as protocol_device,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages import (
    gemv_gfx1250,
    moe_expert_gfx1250,
    moe_gfx1250,
    moe_tail_gfx1250,
)
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import LDS_PAD
from atom.models.kimi_k3.mono.kernels.token_tiling import WMMA_TOKEN_ROWS
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import bf_hi, bf_lo, rsrc, traced
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = 256
WAVES = 8
TP = SUPPORTED_TP[0]
LOCAL_LATENT = LATENT // TP
FINISH_REQUIRED_LDS_BYTES = 113_664
SHARED_INTERMEDIATE = 768
ROUTED_INTERMEDIATE = 384
ROUTED_DOWN_TASKS = LATENT // 32
SHARED_DOWN_TASKS = HIDDEN // 16
GATE_UP_GROUPS = 12
LOCAL_UNION_MAX_TILES = 2

FINISH_PACKED_FRONT_ABI = KernelAbi(
    (
        "front_gathered",
        "router_bias",
        "shared_gate_up",
        "shared_down_weight",
        "shared_down_scale",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "shared_hidden",
        "route",
        "tile_route",
        "flags",
        "experts",
        "kof",
        "union_sizes",
        "latent_fp4",
        "latent_scales",
        "routed_intermediate_fp4",
        "routed_intermediate_scales",
        "routed_partial",
        "shared_partial",
        "scratch",
    )
)


@dataclass(frozen=True)
class K2FinishKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    required_lds_bytes: int
    scratch_bytes: int
    abi: KernelAbi = FINISH_PACKED_FRONT_ABI
    launch_count: int = 1


def _direct_poll(requests):
    values = []
    for pointer, index, _tag in requests:
        word = fx.Int32(bo.buffer_load(rsrc(pointer), index, vec_width=1, dtype=T.i32))
        values.append((word, fx.Int32(1)))
    return values


def _direct_put(pointer, index, value):
    bo.buffer_store(value, rsrc(pointer), index)


@traced
def _grid_barrier(scratch, ticket, blocks):
    """All-resident monotonic grid barrier using the validated K1 primitive."""

    tid = fx.thread_idx.x
    # Every wave has drained its stores before thread 0 arrives for the CTA.
    gpu.barrier()
    if tid == 0:
        generation = protocol_device.epoch_arrive_system_atomic(scratch, fx.Int32(1))
        fx.ptr_store(generation, ticket)
    gpu.barrier()
    generation = fx.ptr_load(ticket)
    target = (generation // fx.Int32(blocks) + fx.Int32(1)) * fx.Int32(blocks)
    if tid == 0:
        arrived = protocol_device.epoch_load_system_acquire(scratch)
        while arrived < target:
            arrived = protocol_device.epoch_load_system_acquire(scratch)
    protocol_device.epoch_fence_system_acquire()
    gpu.barrier()


@traced
def _build_tile_union(
    tid, lane, wave, tile, tile_rows, tile_route_in, lds, publish, outputs
):
    """Build one tile's ascending expert union in ``lds``.

    ``publish`` CTAs also store the union for the routed-down phase.
    """

    context = {
        "S": tile_rows,
        "tid": tid,
        "lane": lane,
        "wave": wave,
        "poll": _direct_poll,
        "scan": lds["scan"],
        "route": tile_route_in,
        "rt": {
            "route": lds["route"],
            "flag": lds["flag"],
            "expert": lds["expert"],
            "kof": lds["kof"],
        },
    }
    union_size = moe_gfx1250.load_route(context, THREADS, WAVES)
    if publish:
        if tid == 0:
            bo.buffer_store(union_size, rsrc(outputs["union_sizes"]), tile)
        for iteration in range_constexpr((16 * 32 + THREADS - 1) // THREADS):
            index = tid + THREADS * iteration
            if index < tile_rows * 32:
                bo.buffer_store(
                    fx.ptr_load(lds["route"] + index),
                    rsrc(outputs["tile_route"]),
                    tile * 16 * 32 + index,
                )
        for iteration in range_constexpr((EXPERTS + THREADS - 1) // THREADS):
            index = tid + THREADS * iteration
            if index < EXPERTS:
                bo.buffer_store(
                    fx.ptr_load(lds["flag"] + index),
                    rsrc(outputs["flags"]),
                    tile * EXPERTS + index,
                )
        for iteration in range_constexpr((256 + THREADS - 1) // THREADS):
            index = tid + THREADS * iteration
            if index < union_size:
                bo.buffer_store(
                    fx.ptr_load(lds["expert"] + index),
                    rsrc(outputs["experts"]),
                    tile * 256 + index,
                )
        for iteration in range_constexpr((256 * 16 + THREADS - 1) // THREADS):
            index = tid + THREADS * iteration
            if index < union_size * tile_rows:
                bo.buffer_store(
                    fx.ptr_load(lds["kof"] + index),
                    rsrc(outputs["kof"]),
                    tile * 256 * 16 + index,
                )
    return union_size


def build_packed_finish(
    plan: Gfx1250KimiBuildPlan,
    *,
    gu_interleave: bool = False,
    include_shared: bool = True,
    include_routed: bool = True,
    replicated_front: bool = False,
    herd: bool = False,
    local_union: bool | None = None,
    fp8_shared_down: bool = False,
) -> K2FinishKernel:
    """Without ``include_routed`` the routed experts are left to AITER's grouped
    MoE: ``route`` and ``tile_route`` then receive contiguous [rows, 16] expert
    ids and weights, and ``latent_scales`` one E8M0 byte per 32-column group.
    After a ``replicated_front`` there is no gathered payload: ``front_gathered``
    is the rank's [rows, EXPERTS] f32 router logits, the latent planes are
    already in place and ``shared_hidden`` holds the activated shared rows.
    With ``herd`` the routes are [rows, 17] Top-(K+1) candidates and their raw
    sigmoid scores, for AITER's HERD drop. With ``fp8_shared_down`` the
    grouped-MoE finish reads ATOM's online per_1x128 FP8 shared down, its E8M0
    block scales in ``shared_down_scale``."""
    if fp8_shared_down and (include_routed or not include_shared):
        raise ValueError("only the grouped-MoE finish reads an FP8 shared down")
    if replicated_front and include_routed:
        raise ValueError("a replicated front feeds only the grouped-MoE finish")
    if herd and include_routed:
        raise ValueError("HERD candidates feed only the grouped-MoE finish")
    if plan.architecture is not GFX1250:
        raise ValueError("persistent K2 finish requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("persistent K2 finish requires 256 threads/eight waves")
    if include_shared and not include_routed and SHARED_DOWN_TASKS > 2 * plan.blocks:
        raise ValueError("grouped-MoE finish holds two shared-down tasks a CTA")
    rows = plan.tokens
    tiles = plan.token_tiling.tile_count
    payload = FrontPayloadLayout(rows)
    if local_union is None:
        # Rebuilding a union costs each CTA about as much as one grid
        # barrier, so three-tile widths keep the published union.
        local_union = tiles <= LOCAL_UNION_MAX_TILES

    @fx.struct
    class RouteRole:
        red: fx.Array[fx.Float32, moe_gfx1250.ROUTE_RED_WORDS, 16]

    @fx.struct
    class PrepRole:
        scan: fx.Array[fx.Int32, WAVES, 16]
        route: fx.Array[fx.Int32, 16 * 32, 16]
        flag: fx.Array[fx.Int32, EXPERTS, 16]
        expert: fx.Array[fx.Int32, 256, 16]
        kof: fx.Array[fx.Int32, 256 * 16, 16]
        latent_fp4: fx.Array[fx.Int32, 16 * (LATENT // 8), 16]
        latent_scales: fx.Array[fx.Int32, 16 * (LATENT // 32), 16]

    # Without routed experts every token's activated row stays staged.
    staged_rows = WMMA_TOKEN_ROWS * (1 if include_routed else tiles)
    shared_accumulator_words = (
        WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS
        if include_routed
        else moe_tail_gfx1250.sdown_accumulator_words(2, rows, WAVES)
    )

    @fx.struct
    class SharedDownRole:
        rows: fx.Array[
            fx.Int32,
            staged_rows * (SHARED_INTERMEDIATE + LDS_PAD) // 2,
            16,
        ]
        accumulator: fx.Array[fx.Float32, shared_accumulator_words, 16]

    @fx.struct
    class GateUpRole:
        latent_fp4: fx.Array[fx.Int32, 16 * (LATENT // 8), 16]
        latent_scales: fx.Array[fx.Int32, 16 * (LATENT // 32), 16]
        experts: fx.Array[fx.Int32, 256, 16]
        kof: fx.Array[fx.Int32, 256 * 16, 16]
        accumulator: fx.Array[fx.Float32, 8 * 32 * 16, 16]
        route: fx.Array[fx.Int32, 16 * 32, 16]
        flag: fx.Array[fx.Int32, EXPERTS, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class DownRole:
        route: fx.Array[fx.Int32, 16 * 32, 16]
        experts: fx.Array[fx.Int32, 256, 16]
        kof: fx.Array[fx.Int32, 256 * 16, 16]
        intermediate_fp4: fx.Array[fx.Int32, 256 * (ROUTED_INTERMEDIATE // 8), 16]
        intermediate_scales: fx.Array[fx.Int32, 256 * (ROUTED_INTERMEDIATE // 32), 16]
        contributions: fx.Array[fx.Float32, 256 * 32, 16]

    @fx.union
    class RoleLds:
        route: RouteRole
        prep: PrepRole
        shared_down: SharedDownRole
        gate_up: GateUpRole
        down: DownRole

    name = (
        f"k3_k2_gfx1250_persistent_finish_s{rows}_b{plan.blocks}"
        f"_gu{int(gu_interleave)}_sh{int(include_shared)}"
        f"_pq_u{int(local_union)}_r{int(include_routed)}"
        f"{'_rf' if replicated_front else ''}{'_herd' if herd else ''}"
        f"{'_f8' if fp8_shared_down else ''}"
    )

    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        front_gathered: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        shared_down_scale: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        shared_hidden: Int64,
        route: Int64,
        tile_route: Int64,
        flags: Int64,
        experts: Int64,
        kof: Int64,
        union_sizes: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        routed_intermediate_fp4: Int64,
        routed_intermediate_scales: Int64,
        routed_partial: Int64,
        shared_partial: Int64,
        scratch: Int64,
    ):
        _ = sources
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        role = fx.SharedAllocator().allocate(RoleLds)
        route_role = role.route.peek()
        prep = role.prep.peek()
        shared_down = role.shared_down.peek()
        gate_up = role.gate_up.peek()
        down = role.down.peek()
        ticket = down.route.ptr

        stride = plan.blocks * THREADS
        # Rank-major compressed payload -> logical token-major FP4/scales.
        total_words = 0 if replicated_front else rows * (LATENT // 8)
        work = bid * THREADS + tid
        while work < total_words:
            token = work // (LATENT // 8)
            column_word = work % (LATENT // 8)
            rank = column_word // (LOCAL_LATENT // 8)
            local_word = column_word % (LOCAL_LATENT // 8)
            source = (
                rank * payload.local_bytes
                + payload.latent_offset
                + (token * (LOCAL_LATENT // 8) + local_word) * 4
            )
            value = fx.Int32(
                bo.buffer_load(
                    rsrc(front_gathered + fx.Int64(source)),
                    0,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            bo.buffer_store(value, rsrc(latent_fp4), work)
            work = work + stride
        total_scales = 0 if replicated_front else rows * (LATENT // 32)
        work = bid * THREADS + tid
        while work < total_scales:
            token = work // (LATENT // 32)
            group = work % (LATENT // 32)
            rank = group // (LOCAL_LATENT // 32)
            local_group = group % (LOCAL_LATENT // 32)
            source = (
                rank * payload.local_bytes
                + payload.latent_scale_offset
                + token * (LOCAL_LATENT // 32)
                + local_group
            )
            code = fx.Int32(
                bo.buffer_load(
                    rsrc(front_gathered + fx.Int64(source)),
                    0,
                    vec_width=1,
                    dtype=T.i8,
                )
            ) & fx.Int32(0xFF)
            if const_expr(include_routed):
                bo.buffer_store(code, rsrc(latent_scales), work)
            else:
                bo.buffer_store(fx.Int8(code), rsrc(latent_scales), work)
            work = work + stride
        # The unpacked latent is first read by gate/up, so the unpack shares
        # the routing phase instead of paying for its own grid barrier.

        # Routing and shared SiTU are independent after their gathered/local
        # inputs become visible.
        route_outputs = (
            {"route": route}
            if include_routed
            else {"route": route, "route_weights": tile_route}
        )
        # Without routed experts nothing below waits on a grid barrier, so
        # routing takes the CTAs with the fewest shared-down tasks.
        route_token = bid if include_routed else bid - (plan.blocks - rows)
        if (route_token >= 0) & (route_token < rows):
            context = {
                "S": rows,
                "tid": tid,
                "lane": lane,
                "wave": wave,
                "red": route_role.red.ptr,
                "poll": _direct_poll,
                "put": _direct_put,
                "peer": {"rlog_own": front_gathered},
                "args": {"bias": router_bias},
                "contiguous_logits": replicated_front,
                "herd": herd,
                **route_outputs,
            }
            # One CTA owns one token.  The old candidate serialized all rows
            # through CTA 0 even though routing has no cross-token reduction.
            moe_gfx1250.stage_route(
                context,
                route_token,
                payload.local_bytes // 4,
            )

        if include_shared and include_routed:
            pairs = rows * SHARED_INTERMEDIATE // 2
            pair = bid * THREADS + tid
            while pair < pairs:
                token = pair // (SHARED_INTERMEDIATE // 2)
                column_pair = pair % (SHARED_INTERMEDIATE // 2)
                gate_at = (token * (2 * SHARED_INTERMEDIATE) + 2 * column_pair) // 2
                up_at = (
                    token * (2 * SHARED_INTERMEDIATE)
                    + SHARED_INTERMEDIATE
                    + 2 * column_pair
                ) // 2
                gate = fx.Int32(
                    bo.buffer_load(
                        rsrc(shared_gate_up),
                        gate_at,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                up = fx.Int32(
                    bo.buffer_load(
                        rsrc(shared_gate_up),
                        up_at,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                low = moe_expert_gfx1250._situ(bf_lo(gate), bf_lo(up))
                high = moe_expert_gfx1250._situ(bf_hi(gate), bf_hi(up))
                packed = (
                    fx.Vector.from_elements([low, high], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                bo.buffer_store(
                    packed,
                    rsrc(shared_hidden),
                    token * (SHARED_INTERMEDIATE // 2) + column_pair,
                )
                pair = pair + stride
        if const_expr(include_routed):
            rocdl.s_wait_storecnt(0)
            _grid_barrier(scratch, ticket, plan.blocks)

        union_outputs = {
            "union_sizes": union_sizes,
            "tile_route": tile_route,
            "flags": flags,
            "experts": experts,
            "kof": kof,
        }
        if const_expr(include_routed and not local_union):
            # One control CTA per token tile builds and publishes the union.
            prep_lds = {
                "scan": prep.scan.ptr,
                "route": prep.route.ptr,
                "flag": prep.flag.ptr,
                "expert": prep.expert.ptr,
                "kof": prep.kof.ptr,
            }
            for tile in range_constexpr(tiles):
                if bid == tile:
                    tile_start = tile * WMMA_TOKEN_ROWS
                    _build_tile_union(
                        tid,
                        lane,
                        wave,
                        tile,
                        min(WMMA_TOKEN_ROWS, rows - tile_start),
                        route + tile_start * 32 * 4,
                        prep_lds,
                        bid == tile,
                        union_outputs,
                    )
            rocdl.s_wait_storecnt(0)
            _grid_barrier(scratch, ticket, plan.blocks)

        if const_expr(include_shared and not include_routed):
            # Both tasks' weights are in flight before the rows are staged.
            tasks = [bid, bid + plan.blocks]
            operands = [
                (
                    gemv_gfx1250.fp8_loads(
                        lane,
                        wave,
                        WAVES,
                        shared_down_weight,
                        shared_down_scale,
                        fx.min(task, SHARED_DOWN_TASKS - 1) * 16,
                        SHARED_INTERMEDIATE,
                    )
                    if fp8_shared_down
                    else gemv_gfx1250.loads(
                        lane,
                        wave,
                        WAVES,
                        shared_down_weight,
                        fx.min(task, SHARED_DOWN_TASKS - 1) * 16,
                        SHARED_INTERMEDIATE,
                    )
                )
                for task in tasks
            ]
            staging = {
                "S": rows,
                "tid": tid,
                "threads": THREADS,
                "tail": {"gemv_rows": shared_down.rows.ptr},
            }
            if const_expr(replicated_front):
                # The replicated front already activated the shared columns.
                moe_tail_gfx1250.stage_hidden_rows(staging, shared_hidden)
            else:
                moe_tail_gfx1250.stage_activated_rows(staging, shared_gate_up)
            moe_tail_gfx1250.sdown_tasks(
                {
                    "S": rows,
                    "tid": tid,
                    "threads": THREADS,
                    "lane": lane,
                    "wave": wave,
                    "tail": {"gemv_accumulator": shared_down.accumulator.ptr},
                    "shared_partial": shared_partial,
                },
                tasks,
                operands,
                WAVES,
                shared_down.rows.ptr,
            )

        # Shared-down and routed gate/up are independent and share a static
        # resident work queue.  Runtime union masks never change barriers.
        for tile in range_constexpr(tiles if include_routed else 0):
            tile_start = tile * WMMA_TOKEN_ROWS
            tile_rows = min(WMMA_TOKEN_ROWS, rows - tile_start)
            if include_shared:
                task = bid
                while task < SHARED_DOWN_TASKS:
                    operands = gemv_gfx1250.loads(
                        lane,
                        wave,
                        WAVES,
                        shared_down_weight,
                        task * 16,
                        SHARED_INTERMEDIATE,
                    )
                    context = {
                        "S": tile_rows,
                        "tid": tid,
                        "threads": THREADS,
                        "lane": lane,
                        "wave": wave,
                        "tail": {
                            "h": (shared_hidden + tile_start * SHARED_INTERMEDIATE * 2),
                            "gemv_rows": shared_down.rows.ptr,
                            "gemv_accumulator": shared_down.accumulator.ptr,
                        },
                        "shared_partial": (shared_partial + tile_start * HIDDEN * 2),
                    }
                    moe_tail_gfx1250.stage_sdown(context, task, operands, WAVES)
                    task = task + plan.blocks

            if const_expr(include_routed):
                if const_expr(local_union):
                    # Every CTA rebuilds the tile union in its own LDS instead of
                    # waiting on a grid barrier for the published copy.
                    union_size = _build_tile_union(
                        tid,
                        lane,
                        wave,
                        tile,
                        tile_rows,
                        route + tile_start * 32 * 4,
                        {
                            "scan": gate_up.scan.ptr,
                            "route": gate_up.route.ptr,
                            "flag": gate_up.flag.ptr,
                            "expert": gate_up.experts.ptr,
                            "kof": gate_up.kof.ptr,
                        },
                        bid == tile,
                        union_outputs,
                    )
                else:
                    union_size = fx.Int32(
                        bo.buffer_load(
                            rsrc(union_sizes),
                            tile,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                maximum_union = tile_rows * 16
                task = bid
                active = task < union_size * GATE_UP_GROUPS
                if active:
                    for iteration in range_constexpr(
                        (16 * (LATENT // 8) + THREADS - 1) // THREADS
                    ):
                        index = tid + THREADS * iteration
                        if index < tile_rows * (LATENT // 8):
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(latent_fp4),
                                    tile_start * (LATENT // 8) + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, gate_up.latent_fp4.ptr + index)
                    for iteration in range_constexpr(
                        (16 * (LATENT // 32) + THREADS - 1) // THREADS
                    ):
                        index = tid + THREADS * iteration
                        if index < tile_rows * (LATENT // 32):
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(latent_scales),
                                    tile_start * (LATENT // 32) + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, gate_up.latent_scales.ptr + index)
                    for iteration in range_constexpr(
                        0 if local_union else (256 + THREADS - 1) // THREADS
                    ):
                        index = tid + THREADS * iteration
                        if index < maximum_union:
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(experts),
                                    tile * 256 + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, gate_up.experts.ptr + index)
                    for iteration in range_constexpr(
                        0 if local_union else (256 * 16 + THREADS - 1) // THREADS
                    ):
                        index = tid + THREADS * iteration
                        if index < maximum_union * tile_rows:
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(kof),
                                    tile * 256 * 16 + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, gate_up.kof.ptr + index)
                    gpu.barrier()
                # Even waves own ``task`` and odd waves the task one grid stride
                # later, so all eight waves run WMMAs.
                while task < maximum_union * GATE_UP_GROUPS:
                    if task < union_size * GATE_UP_GROUPS:
                        own_task = task + (wave % 2) * plan.blocks
                        own_valid = own_task < union_size * GATE_UP_GROUPS
                        own_task = fx.min(own_task, union_size * GATE_UP_GROUPS - 1)
                        context = {
                            "S": tile_rows,
                            "tid": tid,
                            "lane": lane,
                            "wave": wave,
                            "rt": {
                                "expert": gate_up.experts.ptr,
                                "kof": gate_up.kof.ptr,
                            },
                            "latq": gate_up.latent_fp4.ptr,
                            "latsc": gate_up.latent_scales.ptr,
                            "expert_red": gate_up.accumulator.ptr,
                            "args": {"w13": w13, "w13s": w13s},
                            "interq_out": (
                                routed_intermediate_fp4
                                + tile * 256 * (ROUTED_INTERMEDIATE // 8) * 4
                            ),
                            "intsc_out": (
                                routed_intermediate_scales
                                + tile * 256 * (ROUTED_INTERMEDIATE // 32) * 4
                            ),
                        }
                        operands = moe_expert_gfx1250.ug_loads(
                            context,
                            own_task,
                            gu_interleave,
                        )
                        moe_expert_gfx1250.stage_ug(
                            context,
                            own_task,
                            operands,
                            gu_interleave,
                            valid=own_valid,
                        )
                    task = task + 2 * plan.blocks
        rocdl.s_wait_storecnt(0)
        if const_expr(include_routed):
            _grid_barrier(scratch, ticket, plan.blocks)

            for tile in range_constexpr(tiles):
                tile_start = tile * WMMA_TOKEN_ROWS
                tile_rows = min(WMMA_TOKEN_ROWS, rows - tile_start)
                # Tile t's row groups start at CTA t * ROUTED_DOWN_TASKS (mod grid)
                # so later tiles run on otherwise idle CTAs.
                down_task = (bid + (-tile * ROUTED_DOWN_TASKS) % plan.blocks) % plan.blocks
                if down_task < ROUTED_DOWN_TASKS:
                    for iteration in range_constexpr((16 * 32 + THREADS - 1) // THREADS):
                        index = tid + THREADS * iteration
                        if index < tile_rows * 32:
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(tile_route),
                                    tile * 16 * 32 + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, down.route.ptr + index)
                    for iteration in range_constexpr((256 + THREADS - 1) // THREADS):
                        index = tid + THREADS * iteration
                        if index < tile_rows * 16:
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(experts),
                                    tile * 256 + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, down.experts.ptr + index)
                    for iteration in range_constexpr((256 * 16 + THREADS - 1) // THREADS):
                        index = tid + THREADS * iteration
                        if index < tile_rows * 16 * tile_rows:
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(kof),
                                    tile * 256 * 16 + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, down.kof.ptr + index)
                    for iteration in range_constexpr(
                        (256 * (ROUTED_INTERMEDIATE // 8) + THREADS - 1) // THREADS
                    ):
                        index = tid + THREADS * iteration
                        if index < 256 * (ROUTED_INTERMEDIATE // 8):
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(routed_intermediate_fp4),
                                    tile * 256 * (ROUTED_INTERMEDIATE // 8) + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, down.intermediate_fp4.ptr + index)
                    for iteration in range_constexpr(
                        (256 * (ROUTED_INTERMEDIATE // 32) + THREADS - 1) // THREADS
                    ):
                        index = tid + THREADS * iteration
                        if index < 256 * (ROUTED_INTERMEDIATE // 32):
                            value = fx.Int32(
                                bo.buffer_load(
                                    rsrc(routed_intermediate_scales),
                                    tile * 256 * (ROUTED_INTERMEDIATE // 32) + index,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                            fx.ptr_store(value, down.intermediate_scales.ptr + index)
                    for iteration in range_constexpr((256 * 32 + THREADS - 1) // THREADS):
                        index = tid + THREADS * iteration
                        if index < 256 * 32:
                            fx.ptr_store(
                                fx.Float32(0.0),
                                down.contributions.ptr + index,
                            )
                    gpu.barrier()
                    union_size = fx.Int32(
                        bo.buffer_load(
                            rsrc(union_sizes),
                            tile,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    context = {
                        "S": tile_rows,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "rt": {
                            "route": down.route.ptr,
                            "expert": down.experts.ptr,
                            "kof": down.kof.ptr,
                        },
                        "interq": down.intermediate_fp4.ptr,
                        "intsc": down.intermediate_scales.ptr,
                        "contrib": down.contributions.ptr,
                        "args": {"w2": w2, "w2s": w2s},
                        "routed_partial": (routed_partial + tile_start * LATENT * 2),
                    }
                    moe_expert_gfx1250.stage_down(context, down_task, union_size, WAVES)
        rocdl.s_wait_storecnt(0)

    @flyc.jit
    def launch(
        front_gathered: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        shared_down_scale: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        shared_hidden: Int64,
        route: Int64,
        tile_route: Int64,
        flags: Int64,
        experts: Int64,
        kof: Int64,
        union_sizes: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        routed_intermediate_fp4: Int64,
        routed_intermediate_scales: Int64,
        routed_partial: Int64,
        shared_partial: Int64,
        scratch: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            front_gathered,
            router_bias,
            shared_gate_up,
            shared_down_weight,
            shared_down_scale,
            w13,
            w13s,
            w2,
            w2s,
            shared_hidden,
            route,
            tile_route,
            flags,
            experts,
            kof,
            union_sizes,
            latent_fp4,
            latent_scales,
            routed_intermediate_fp4,
            routed_intermediate_scales,
            routed_partial,
            shared_partial,
            scratch,
        ).launch(grid=(plan.blocks,), block=(THREADS,), stream=stream)

    FINISH_PACKED_FRONT_ABI.check(kernel, launch)
    return K2FinishKernel(
        name=name,
        launcher=launch,
        grid=(plan.blocks,),
        block=(THREADS,),
        required_lds_bytes=FINISH_REQUIRED_LDS_BYTES,
        scratch_bytes=8,
    )
