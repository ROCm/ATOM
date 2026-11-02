# SPDX-License-Identifier: MIT
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
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
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
from atom.models.kimi_k3.mono.kernels.stages.k2_fused_intervals_gfx1250 import (
    FINISH_REQUIRED_LDS_BYTES,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_resident_protocol import (
    FinishIntervalSchedule,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import WMMA_TOKEN_ROWS
from atom.mono.device.ops import bf_hi, bf_lo, rsrc, traced
from atom.mono.plan.architecture import GFX1250

THREADS = 256
WAVES = 8
LATENT = 3584
SHARED_INTERMEDIATE = 768
ROUTED_INTERMEDIATE = 384
ROUTED_DOWN_TASKS = LATENT // 32
SHARED_DOWN_TASKS = HIDDEN // 16
GATE_UP_GROUPS = 12

FINISH_PACKED_FRONT_ABI = (
    "front_gathered",
    "router_bias",
    "shared_gate_up",
    "shared_down_weight",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "latent_logical",
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


@dataclass(frozen=True)
class K2FinishKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    required_lds_bytes: int
    scratch_bytes: int
    schedule: FinishIntervalSchedule
    abi: tuple[str, ...] = FINISH_PACKED_FRONT_ABI
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
def _quantize_latent(latent, latent_fp4, latent_scales, rows, global_tid, threads):
    blocks = rows * (LATENT // 32)
    for round_index in range_constexpr((blocks + threads - 1) // threads):
        item = global_tid + threads * round_index
        block = fx.min(item, blocks - 1)
        token = block // (LATENT // 32)
        group = block % (LATENT // 32)
        base = (token * LATENT + group * 32) // 2
        words = [
            fx.Int32(
                bo.buffer_load(
                    rsrc(latent),
                    base + index,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            for index in range(16)
        ]
        maximum_bits = fx.Int32(0)
        for word in words:
            maximum_bits = fx.max(maximum_bits, word & 0x7FFF)
            maximum_bits = fx.max(maximum_bits, (word >> 16) & 0x7FFF)
        maximum = (maximum_bits << 16).bitcast(fx.Float32)
        code = moe_gfx1250._e8m0_roundup(maximum)
        reciprocal_scale = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        if item < blocks:
            for packed_word in range_constexpr(4):
                value = moe_gfx1250._pack_bf16_words(
                    words[4 * packed_word : 4 * packed_word + 4],
                    reciprocal_scale,
                )
                bo.buffer_store(
                    value,
                    rsrc(latent_fp4),
                    token * (LATENT // 8) + group * 4 + packed_word,
                )
            bo.buffer_store(
                code,
                rsrc(latent_scales),
                token * (LATENT // 32) + group,
            )
    gpu.barrier()


@traced
def _grid_barrier(scratch, ticket, blocks):
    """All-resident monotonic grid barrier using the validated K1 primitive."""

    tid = fx.thread_idx.x
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


def build_packed_finish(
    plan: Gfx1250KimiBuildPlan,
) -> K2FinishKernel:
    if plan.architecture is not GFX1250:
        raise ValueError("persistent K2 finish requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("persistent K2 finish requires 256 threads/eight waves")
    schedule = FinishIntervalSchedule(plan.tokens, plan.blocks)
    schedule.prove_coverage()
    rows = plan.tokens
    tiles = plan.token_tiling.tile_count

    @fx.struct
    class RouteRole:
        red: fx.Array[fx.Float32, 128, 16]

    @fx.struct
    class PrepRole:
        scan: fx.Array[fx.Int32, WAVES, 16]
        route: fx.Array[fx.Int32, 16 * 32, 16]
        flag: fx.Array[fx.Int32, 896, 16]
        expert: fx.Array[fx.Int32, 256, 16]
        kof: fx.Array[fx.Int32, 256 * 16, 16]
        latent_fp4: fx.Array[fx.Int32, 16 * (LATENT // 8), 16]
        latent_scales: fx.Array[fx.Int32, 16 * (LATENT // 32), 16]

    @fx.struct
    class SharedDownRole:
        rows: fx.Array[
            fx.Int32,
            WMMA_TOKEN_ROWS * (SHARED_INTERMEDIATE + LDS_PAD) // 2,
            16,
        ]
        accumulator: fx.Array[
            fx.Float32,
            WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS,
            16,
        ]

    @fx.struct
    class GateUpRole:
        latent_fp4: fx.Array[fx.Int32, 16 * (LATENT // 8), 16]
        latent_scales: fx.Array[fx.Int32, 16 * (LATENT // 32), 16]
        experts: fx.Array[fx.Int32, 256, 16]
        kof: fx.Array[fx.Int32, 256 * 16, 16]
        accumulator: fx.Array[fx.Float32, 4 * 32 * 16, 16]

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

    name = f"k3_k2_gfx1250_persistent_finish_s{rows}_b{plan.blocks}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        front_gathered: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        latent_logical: Int64,
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

        # Rank-major [8,S,448] -> logical [S,3584].
        total_pairs = rows * (LATENT // 2)
        work = bid * THREADS + tid
        stride = plan.blocks * THREADS
        while work < total_pairs:
            token = work // (LATENT // 2)
            column_pair = work % (LATENT // 2)
            rank = column_pair // (448 // 2)
            local_pair = column_pair % (448 // 2)
            source = (rank * rows + token) * (448 // 2) + local_pair
            payload_words = rows * (896 // 8) + rows * (448 // 2)
            source = (
                rank * payload_words
                + rows * (896 // 8)
                + token * (448 // 2)
                + local_pair
            )
            value = fx.Int32(
                bo.buffer_load(
                    rsrc(front_gathered),
                    source,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            bo.buffer_store(value, rsrc(latent_logical), work)
            work = work + stride
        rocdl.s_wait_storecnt(0)
        _grid_barrier(scratch, ticket, plan.blocks)

        # Routing and shared SiTU are independent after their gathered/local
        # inputs become visible.
        if bid < rows:
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
                "route": route,
            }
            # One CTA owns one token.  The old candidate serialized all rows
            # through CTA 0 even though routing has no cross-token reduction.
            moe_gfx1250.stage_route(
                context,
                bid,
                rows * (896 // 8) + rows * (448 // 2),
            )

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
        rocdl.s_wait_storecnt(0)
        _grid_barrier(scratch, ticket, plan.blocks)

        # One control CTA per token tile builds the union and quantized latent.
        for tile in range_constexpr(tiles):
            if bid == tile:
                tile_start = tile * WMMA_TOKEN_ROWS
                tile_rows = min(WMMA_TOKEN_ROWS, rows - tile_start)
                context = {
                    "S": tile_rows,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "poll": _direct_poll,
                    "scan": prep.scan.ptr,
                    "route": route + tile_start * 32 * 4,
                    "peer": {
                        "lat_own": latent_logical + tile_start * LATENT * 2,
                    },
                    "rt": {
                        "route": prep.route.ptr,
                        "flag": prep.flag.ptr,
                        "expert": prep.expert.ptr,
                        "kof": prep.kof.ptr,
                    },
                    "latq": prep.latent_fp4.ptr,
                    "latsc": prep.latent_scales.ptr,
                }
                union_size = moe_gfx1250.load_route(context, THREADS, WAVES)
                if tid == 0:
                    bo.buffer_store(union_size, rsrc(union_sizes), tile)
                for iteration in range_constexpr((16 * 32 + THREADS - 1) // THREADS):
                    index = tid + THREADS * iteration
                    if index < tile_rows * 32:
                        bo.buffer_store(
                            fx.ptr_load(prep.route.ptr + index),
                            rsrc(tile_route),
                            tile * 16 * 32 + index,
                        )
                for iteration in range_constexpr((896 + THREADS - 1) // THREADS):
                    index = tid + THREADS * iteration
                    if index < 896:
                        bo.buffer_store(
                            fx.ptr_load(prep.flag.ptr + index),
                            rsrc(flags),
                            tile * 896 + index,
                        )
                for iteration in range_constexpr((256 + THREADS - 1) // THREADS):
                    index = tid + THREADS * iteration
                    if index < union_size:
                        bo.buffer_store(
                            fx.ptr_load(prep.expert.ptr + index),
                            rsrc(experts),
                            tile * 256 + index,
                        )
                for iteration in range_constexpr((256 * 16 + THREADS - 1) // THREADS):
                    index = tid + THREADS * iteration
                    if index < union_size * tile_rows:
                        bo.buffer_store(
                            fx.ptr_load(prep.kof.ptr + index),
                            rsrc(kof),
                            tile * 256 * 16 + index,
                        )
        # Quantization groups are independent.  Flatten them over the whole
        # resident grid instead of serializing quantization in tile-control
        # CTAs; this also makes every CTA participate uniformly.
        _quantize_latent(
            latent_logical,
            latent_fp4,
            latent_scales,
            rows,
            bid * THREADS + tid,
            plan.blocks * THREADS,
        )
        rocdl.s_wait_storecnt(0)
        _grid_barrier(scratch, ticket, plan.blocks)

        # Shared-down and routed gate/up are independent and share a static
        # resident work queue.  Runtime union masks never change barriers.
        for tile in range_constexpr(tiles):
            tile_start = tile * WMMA_TOKEN_ROWS
            tile_rows = min(WMMA_TOKEN_ROWS, rows - tile_start)
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
                        "h": shared_hidden + tile_start * SHARED_INTERMEDIATE * 2,
                        "gemv_rows": shared_down.rows.ptr,
                        "gemv_accumulator": shared_down.accumulator.ptr,
                    },
                    "shared_partial": (shared_partial + tile_start * HIDDEN * 2),
                }
                moe_tail_gfx1250.stage_sdown(context, task, operands, WAVES)
                task = task + plan.blocks

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
            while task < maximum_union * GATE_UP_GROUPS:
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
                    for iteration in range_constexpr((256 + THREADS - 1) // THREADS):
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
                        (256 * 16 + THREADS - 1) // THREADS
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
                    operands = moe_expert_gfx1250.ug_loads(context, task)
                    moe_expert_gfx1250.stage_ug(context, task, operands)
                task = task + plan.blocks
        rocdl.s_wait_storecnt(0)
        _grid_barrier(scratch, ticket, plan.blocks)

        for tile in range_constexpr(tiles):
            tile_start = tile * WMMA_TOKEN_ROWS
            tile_rows = min(WMMA_TOKEN_ROWS, rows - tile_start)
            if bid < ROUTED_DOWN_TASKS:
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
                moe_expert_gfx1250.stage_down(context, bid, union_size, WAVES)
        rocdl.s_wait_storecnt(0)

    @flyc.jit
    def launch(
        front_gathered: Int64,
        router_bias: Int64,
        shared_gate_up: Int64,
        shared_down_weight: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        latent_logical: Int64,
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
        kernel(
            front_gathered,
            router_bias,
            shared_gate_up,
            shared_down_weight,
            w13,
            w13s,
            w2,
            w2s,
            latent_logical,
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

    return K2FinishKernel(
        name=name,
        launcher=launch,
        grid=(plan.blocks,),
        block=(THREADS,),
        required_lds_bytes=FINISH_REQUIRED_LDS_BYTES,
        scratch_bytes=8,
        schedule=schedule,
    )
