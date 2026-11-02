# SPDX-License-Identifier: MIT
"""Role-LDS resident K2 front for gfx1250.

The kernel never materializes full ``attnres_values`` or
``attnres_mixed`` tensors.  Each of the 56 128-column slice CTAs retains
values, then mixed values, in its role LDS until normalization has stored
``x_local``.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Float32, Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.attention import kda as attention_base
from atom.models.kimi_k3.mono.kernels.attention import (
    kda_gfx1250 as protocol_device,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import LDS_PAD
from atom.models.kimi_k3.mono.kernels.stages.k2_fused_intervals_gfx1250 import (
    FRONT_REQUIRED_LDS_BYTES,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_resident_protocol import (
    PROJECTION_TASKS,
    AttnResRoleSchedule,
    AttnResRoleTraffic,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import WMMA_TOKEN_ROWS
from atom.mono.device.ops import (
    CM_DEV,
    hw_exp2,
    hw_rsq,
    rsrc,
    traced,
    wave_max,
    wave_sum,
)
from atom.mono.plan.architecture import GFX1250

ATTENTION_CTAS = HIDDEN // 128
COMMON_RED_ELEMENTS = 512
LOG2E = 1.0 / math.log(2.0)
THREADS = 256
WAVES = 8
ROUTER_ROWS = 896 // 8
LATENT_ROWS = 3584 // 8
SHARED_ROWS = 2 * 768
ROUTER_TASKS = ROUTER_ROWS // 16
LATENT_TASKS = LATENT_ROWS // 16
ACTIVATION_SLAB_BUDGET = 120 * 1024
ACCUMULATOR_BYTES = WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS * 4
ROLE_LDS_FRONT_ABI = (
    "oproj_reduced",
    "prefix",
    "blocks",
    "blk_sm",
    "blk_sr",
    "ares_norm_weight",
    "ares_query",
    "input_norm_weight",
    "router_weight",
    "latent_weight",
    "shared_weight",
    "x_local",
    "router_local",
    "latent_local",
    "shared_gate_up",
    "scratch",
    "layer",
    "attention_epsilon",
    "output_epsilon",
)


def _align(value: int, alignment: int = 16) -> int:
    return (value + alignment - 1) // alignment * alignment


@traced
def _stage_activation_slab(
    activation,
    local,
    tid,
    tile_start,
    rows,
    slab_start,
    slab_width,
    row_stride,
):
    words = WMMA_TOKEN_ROWS * slab_width // 8
    for iteration in range_constexpr((words + THREADS - 1) // THREADS):
        work = tid + THREADS * iteration
        if work < words:
            local_token = work // (slab_width // 8)
            column = (work % (slab_width // 8)) * 8
            token = tile_start + local_token
            source_token = fx.min(token, rows - 1)
            value = fx.Vector.from_elements(
                [
                    (token < rows).select(word, fx.Int32(0))
                    for word in fx.Vector(
                        bo.buffer_load(
                            rsrc(activation),
                            (source_token * HIDDEN + slab_start + column) // 2,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    )
                ],
                fx.Int32,
            )
            fx.ptr_store(
                value,
                local + (local_token * row_stride + column) // 2,
            )
    gpu.barrier()


@traced
def _projection_task(
    activation,
    weight,
    output,
    activation_lds,
    accumulator_lds,
    tid,
    lane,
    wave,
    task,
    tile_start,
    tile_rows,
    rows,
    output_rows,
    output_f32,
    slabs,
    row_stride,
):
    accumulator = fx.Vector.filled(gemv_gfx1250.ACCUMULATOR_ELEMENTS, 0.0, fx.Float32)
    for slab_index in range_constexpr(len(slabs)):
        slab_start, slab_width = slabs[slab_index]
        operands = gemv_gfx1250.loads(
            lane,
            wave,
            WAVES,
            weight,
            task * 16,
            slab_width,
            ldk=HIDDEN,
            k0=slab_start,
        )
        if slab_index:
            gpu.barrier()
        _stage_activation_slab(
            activation,
            activation_lds,
            tid,
            tile_start,
            rows,
            slab_start,
            slab_width,
            row_stride,
        )
        accumulator = gemv_gfx1250.mfmas(
            lane,
            wave,
            WAVES,
            activation_lds,
            row_stride,
            slab_width,
            tile_rows,
            operands,
            accumulator,
        )
    gemv_gfx1250.store_accumulator(accumulator, accumulator_lds, wave, lane)
    gpu.barrier()
    total = WMMA_TOKEN_ROWS * 8
    for iteration in range_constexpr((total + THREADS - 1) // THREADS):
        work = tid + THREADS * iteration
        if work < total:
            local_token = work // 8
            pair = work % 8
            token = tile_start + local_token
            low = gemv_gfx1250.row_sum(accumulator_lds, 2 * pair, local_token, WAVES)
            high = gemv_gfx1250.row_sum(
                accumulator_lds, 2 * pair + 1, local_token, WAVES
            )
            row = task * 16 + 2 * pair
            at = token * output_rows + row
            if local_token < tile_rows:
                if output_f32:
                    bo.buffer_store(low, rsrc(output), at)
                    bo.buffer_store(high, rsrc(output), at + 1)
                else:
                    packed = (
                        fx.Vector.from_elements([low, high], fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32)[0]
                    )
                    bo.buffer_store(packed, rsrc(output), at // 2)
    protocol_device.wait_vector_stores()


@traced
def _projection_direct_task(
    x_local,
    router_weight,
    latent_weight,
    shared_weight,
    router_local,
    latent_local,
    shared_gate_up,
    activation_lds,
    accumulator_lds,
    tid,
    lane,
    wave,
    task_id,
    tile_start,
    tile_rows,
    rows,
    slabs,
    row_stride,
):
    if task_id < ROUTER_TASKS:
        _projection_task(
            x_local,
            router_weight,
            router_local,
            activation_lds,
            accumulator_lds,
            tid,
            lane,
            wave,
            task_id,
            tile_start,
            tile_rows,
            rows,
            ROUTER_ROWS,
            True,
            slabs,
            row_stride,
        )
    elif task_id < ROUTER_TASKS + LATENT_TASKS:
        _projection_task(
            x_local,
            latent_weight,
            latent_local,
            activation_lds,
            accumulator_lds,
            tid,
            lane,
            wave,
            task_id - ROUTER_TASKS,
            tile_start,
            tile_rows,
            rows,
            LATENT_ROWS,
            False,
            slabs,
            row_stride,
        )
    else:
        _projection_task(
            x_local,
            shared_weight,
            shared_gate_up,
            activation_lds,
            accumulator_lds,
            tid,
            lane,
            wave,
            task_id - ROUTER_TASKS - LATENT_TASKS,
            tile_start,
            tile_rows,
            rows,
            SHARED_ROWS,
            False,
            slabs,
            row_stride,
        )


@dataclass(frozen=True)
class K2FrontKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    required_lds_bytes: int
    scratch_bytes: int
    traffic: AttnResRoleTraffic
    abi: tuple[str, ...] = ROLE_LDS_FRONT_ABI
    launch_count: int = 1


def build_role_lds_front(
    plan: Gfx1250KimiBuildPlan,
    *,
    include_projection: bool = True,
    reset: bool = False,
) -> K2FrontKernel:
    if plan.architecture is not GFX1250:
        raise ValueError("role-LDS K2 front requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("role-LDS K2 front requires 256 threads/eight waves")
    if plan.nblocks is None or plan.nblocks <= 0:
        raise ValueError("role-LDS K2 front requires explicit nblocks")
    attention_tasks = ATTENTION_CTAS * plan.token_tiling.tile_count
    gate_cta = attention_tasks
    norm_cta = attention_tasks + 1
    if plan.blocks <= norm_cta:
        raise ValueError("role-LDS K2 front lacks gate/norm control CTAs")

    rows = plan.tokens
    sources = plan.nblocks + 1
    if sources > GFX1250.wave_size:
        raise ValueError("role-LDS AttnRes sources exceed one wave32")
    slabs = plan.gemv_layout.slab_plan(
        HIDDEN,
        WMMA_TOKEN_ROWS,
        WAVES,
        ACTIVATION_SLAB_BUDGET,
    )
    max_slab = max(width for _, width in slabs)
    row_stride = max_slab + LDS_PAD
    projection_bytes = WMMA_TOKEN_ROWS * row_stride * 2 + ACCUMULATOR_BYTES
    attention_bytes = WMMA_TOKEN_ROWS * sources * 128 * 4
    role_bytes = max(projection_bytes, attention_bytes)
    fixed_bytes = 16 + COMMON_RED_ELEMENTS * 4
    occupancy_padding = FRONT_REQUIRED_LDS_BYTES - fixed_bytes - role_bytes
    if occupancy_padding < 16 or occupancy_padding % 16:
        raise ValueError("role-LDS front cannot reserve exact occupancy LDS")

    partial_pairs = ATTENTION_CTAS * rows * sources * 2
    weight_pairs = rows * (sources + 1)
    mean_square_pairs = ATTENTION_CTAS * rows
    rstd_pairs = rows
    ready_pairs = attention_tasks
    part_offset = 0
    weight_offset = part_offset + partial_pairs * 8
    mean_square_offset = weight_offset + weight_pairs * 8
    rstd_offset = mean_square_offset + mean_square_pairs * 8
    ready_offset = rstd_offset + rstd_pairs * 8
    sync_offset = _align(ready_offset + ready_pairs * 8)
    scratch_bytes = sync_offset + 8
    reset_pairs = (
        (part_offset, partial_pairs),
        (weight_offset, weight_pairs),
        (mean_square_offset, mean_square_pairs),
        (rstd_offset, rstd_pairs),
        (ready_offset, ready_pairs),
    )
    total_pairs = sum(count for _, count in reset_pairs)
    AttnResRoleSchedule(rows, sources).prove_progress()
    traffic = AttnResRoleTraffic.for_shape(rows, sources)

    @fx.struct
    class AttentionRole:
        values: fx.Array[fx.Float32, WMMA_TOKEN_ROWS * sources * 128, 16]

    @fx.struct
    class ProjectionRole:
        activation: fx.Array[fx.Int32, WMMA_TOKEN_ROWS * row_stride // 2, 16]
        accumulator: fx.Array[
            fx.Float32,
            WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS,
            16,
        ]

    @fx.union
    class RoleLds:
        attention: AttentionRole
        projection: ProjectionRole

    @fx.struct
    class Smem:
        ticket: fx.Array[fx.Int32, 1, 16]
        red: fx.Array[fx.Float32, COMMON_RED_ELEMENTS, 16]

    @fx.struct
    class Guard:
        occupancy_guard: fx.Array[fx.Int32, occupancy_padding // 4, 16]

    name = (
        f"k3_k2_gfx1250_role_lds_front_s{rows}"
        f"_nb{plan.nblocks}_b{plan.blocks}"
        f"_p{int(include_projection)}"
        f"_r{int(reset)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        oproj_reduced: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int64,
        blk_sr: Int64,
        ares_norm_weight: Int64,
        ares_query: Int64,
        input_norm_weight: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        x_local: Int64,
        router_local: Int64,
        latent_local: Int64,
        shared_gate_up: Int64,
        scratch: Int64,
        layer: Int32,
        attention_epsilon: Float32,
        output_epsilon: Float32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        allocator = fx.SharedAllocator()
        shared = allocator.allocate(Smem).peek()
        role = allocator.allocate(RoleLds)
        attention = role.attention.peek()
        projection = role.projection.peek()
        guard = allocator.allocate(Guard).peek()
        if tid == 0:
            fx.ptr_store(
                fx.Int32(0),
                guard.occupancy_guard.ptr + occupancy_padding // 4 - 1,
            )
        gpu.barrier()

        total_threads = plan.blocks * THREADS
        pair_base = bid * THREADS + tid
        for iteration in range_constexpr(
            (total_pairs + total_threads - 1) // total_threads
        ):
            flat = pair_base + iteration * total_threads
            if flat < total_pairs:
                protocol_device.mailbox_reset_device(scratch, flat)
        protocol_device.wait_vector_stores()
        gpu.barrier()
        protocol_device.epoch_fence_system_release()
        if tid == 0:
            ticket = protocol_device.epoch_arrive_system_atomic(
                scratch + fx.Int64(sync_offset), fx.Int32(1)
            )
            fx.ptr_store(ticket, shared.ticket.ptr)
        gpu.barrier()
        generation = fx.ptr_load(shared.ticket.ptr)
        target = (generation // fx.Int32(plan.blocks) + fx.Int32(1)) * fx.Int32(
            plan.blocks
        )
        if tid == 0:
            arrived = protocol_device.epoch_load_system_acquire(
                scratch + fx.Int64(sync_offset)
            )
            while arrived < target:
                arrived = protocol_device.epoch_load_system_acquire(
                    scratch + fx.Int64(sync_offset)
                )
        protocol_device.epoch_fence_system_acquire()
        gpu.barrier()

        tag = layer + fx.Int32(1)
        part = scratch + fx.Int64(part_offset)
        weights = scratch + fx.Int64(weight_offset)
        mean_squares = scratch + fx.Int64(mean_square_offset)
        rstd = scratch + fx.Int64(rstd_offset)
        ready = scratch + fx.Int64(ready_offset)

        if bid < attention_tasks:
            token_tile = bid // ATTENTION_CTAS
            slice_id = bid % ATTENTION_CTAS
            tile_start = token_tile * WMMA_TOKEN_ROWS
            slice_work = WMMA_TOKEN_ROWS * sources * (128 // 4)
            for iteration in range_constexpr((slice_work + THREADS - 1) // THREADS):
                work = tid + THREADS * iteration
                clamped = fx.min(work, slice_work - 1)
                source_row = clamped // (128 // 4)
                local_token = source_row // sources
                source = source_row % sources
                global_token = tile_start + local_token
                source_token = fx.min(global_token, rows - 1)
                live = (work < slice_work) & (global_token < rows)
                column = slice_id * 128 + (clamped % (128 // 4)) * 4
                is_prefix = source == plan.nblocks
                block = attention_base.ld_bf4(
                    blocks,
                    (
                        source_token * blk_sm
                        + fx.min(source, plan.nblocks - 1) * blk_sr
                        + column
                    ),
                )
                delta_values = attention_base.ld_bf4(
                    oproj_reduced, source_token * HIDDEN + column
                )
                if reset:
                    # K1 already banked the old prefix into the new block.
                    # Restart the live prefix at the reduced o_proj output,
                    # exactly like gfx950 K2Build(reset=True), rather than
                    # counting the old prefix both as a block and as prefix.
                    updated = delta_values
                else:
                    prefix_values = attention_base.ld_bf4(
                        prefix, source_token * HIDDEN + column
                    )
                    updated = [
                        prefix_values[element] + delta_values[element]
                        for element in range_constexpr(4)
                    ]
                if live & is_prefix:
                    bo.buffer_store(
                        attention_base.bf4_words(updated),
                        rsrc(prefix),
                        (global_token * HIDDEN + column) // 2,
                    )
                values = [
                    live.select(
                        is_prefix.select(updated[element], block[element]),
                        fx.Float32(0.0),
                    )
                    for element in range_constexpr(4)
                ]
                if work < slice_work:
                    for element in range_constexpr(4):
                        fx.ptr_store(
                            values[element],
                            attention.values.ptr
                            + (source_row * 128 + (clamped % (128 // 4)) * 4 + element),
                        )
                norm = attention_base.ld_bf4(ares_norm_weight, column)
                query = attention_base.ld_bf4(ares_query, column)
                square = (
                    values[0] * values[0]
                    + values[1] * values[1]
                    + values[2] * values[2]
                    + values[3] * values[3]
                )
                dot = (
                    values[0] * norm[0] * query[0]
                    + values[1] * norm[1] * query[1]
                    + values[2] * norm[2] * query[2]
                    + values[3] * norm[3] * query[3]
                )
                square = protocol_device.attention_reduce32(square)
                dot = protocol_device.attention_reduce32(dot)
                if live & (lane == 0):
                    at = ((slice_id * rows + global_token) * sources + source) * 2
                    protocol_device.mailbox_store_device(part, at, square, tag)
                    protocol_device.mailbox_store_device(part, at + 1, dot, tag)
            gpu.barrier()

            weight_count = rows * (sources + 1)
            for iteration in range_constexpr((weight_count + THREADS - 1) // THREADS):
                item = tid + THREADS * iteration
                if item < weight_count:
                    bits = protocol_device.mailbox_poll_device(weights, item, tag)
                    fx.ptr_store(bits.bitcast(fx.Float32), shared.red.ptr + item)
            gpu.barrier()

            rounds = WMMA_TOKEN_ROWS // WAVES
            for round_index in range_constexpr(rounds):
                local_token = wave + WAVES * round_index
                token = tile_start + local_token
                live = token < rows
                column = lane * 4
                mixed = []
                for element in range_constexpr(4):
                    value = fx.Float32(0.0)
                    for source in range_constexpr(sources):
                        value = value + fx.ptr_load(
                            shared.red.ptr
                            + fx.min(token, rows - 1) * (sources + 1)
                            + source
                        ) * fx.ptr_load(
                            attention.values.ptr
                            + (local_token * sources + source) * 128
                            + column
                            + element
                        )
                    mixed.append(
                        value
                        / fx.ptr_load(
                            shared.red.ptr
                            + fx.min(token, rows - 1) * (sources + 1)
                            + sources
                        )
                    )
                    fx.ptr_store(
                        mixed[element],
                        attention.values.ptr + local_token * 128 + column + element,
                    )
                square = (
                    mixed[0] * mixed[0]
                    + mixed[1] * mixed[1]
                    + mixed[2] * mixed[2]
                    + mixed[3] * mixed[3]
                )
                square = wave_sum(square, GFX1250)
                if live & (lane == 0):
                    protocol_device.mailbox_store_device(
                        mean_squares,
                        slice_id * rows + token,
                        square,
                        tag,
                    )
            gpu.barrier()

            for round_index in range_constexpr(rounds):
                local_token = wave + WAVES * round_index
                token = tile_start + local_token
                live = token < rows
                bits = protocol_device.mailbox_poll_device(
                    rstd, fx.min(token, rows - 1), tag
                )
                inverse_norm = bits.bitcast(fx.Float32)
                column = lane * 4
                norm = attention_base.ld_bf4(input_norm_weight, slice_id * 128 + column)
                output = attention_base.bf4_words(
                    [
                        fx.ptr_load(
                            attention.values.ptr + local_token * 128 + column + element
                        )
                        * inverse_norm
                        * norm[element]
                        for element in range_constexpr(4)
                    ]
                )
                if live:
                    bo.buffer_store(
                        output,
                        rsrc(x_local),
                        (token * HIDDEN + slice_id * 128 + column) // 2,
                        cache_modifier=CM_DEV,
                    )
            protocol_device.wait_vector_stores()
            gpu.barrier()
            if tid == 0:
                protocol_device.mailbox_store_device(ready, bid, fx.Int32(1), tag)

        elif bid == gate_cta:
            gate_rounds = (rows + WAVES - 1) // WAVES
            for gate_round in range_constexpr(gate_rounds):
                token = fx.min(wave + WAVES * gate_round, rows - 1)
                live = wave + WAVES * gate_round < rows
                source = fx.min(lane, sources - 1)
                square = fx.Float32(0.0)
                dot = fx.Float32(0.0)
                for task in range_constexpr(ATTENTION_CTAS):
                    base = ((task * rows + token) * sources + source) * 2
                    square_bits = protocol_device.mailbox_poll_device(part, base, tag)
                    dot_bits = protocol_device.mailbox_poll_device(part, base + 1, tag)
                    square = square + square_bits.bitcast(fx.Float32)
                    dot = dot + dot_bits.bitcast(fx.Float32)
                logit = dot * hw_rsq(
                    square * fx.Float32(1.0 / HIDDEN) + attention_epsilon
                )
                logit = (lane < sources).select(
                    logit * fx.Float32(LOG2E),
                    fx.Float32(-3.0e38),
                )
                maximum = wave_max(logit, GFX1250)
                exponential = (lane < sources).select(
                    hw_exp2(logit - maximum), fx.Float32(0.0)
                )
                denominator = wave_sum(exponential, GFX1250)
                if live & (lane < sources):
                    protocol_device.mailbox_store_device(
                        weights,
                        token * (sources + 1) + lane,
                        exponential,
                        tag,
                    )
                if live & (lane == 0):
                    protocol_device.mailbox_store_device(
                        weights,
                        token * (sources + 1) + sources,
                        denominator,
                        tag,
                    )

        elif bid == norm_cta:
            if tid < rows:
                total = fx.Float32(0.0)
                for task in range_constexpr(ATTENTION_CTAS):
                    bits = protocol_device.mailbox_poll_device(
                        mean_squares, task * rows + tid, tag
                    )
                    total = total + bits.bitcast(fx.Float32)
                inverse_norm = hw_rsq(total * fx.Float32(1.0 / HIDDEN) + output_epsilon)
                protocol_device.mailbox_store_device(rstd, tid, inverse_norm, tag)

        if include_projection:  # noqa: SIM102 - compile-time outer gate
            if bid < PROJECTION_TASKS:
                if tid < attention_tasks:
                    protocol_device.mailbox_poll_device(ready, tid, tag)
                gpu.barrier()
                protocol_device.epoch_fence_system_acquire()
                for tile in range_constexpr(plan.token_tiling.tile_count):
                    tile_start = tile * WMMA_TOKEN_ROWS
                    tile_rows = min(WMMA_TOKEN_ROWS, rows - tile_start)
                    _projection_direct_task(
                        x_local,
                        router_weight,
                        latent_weight,
                        shared_weight,
                        router_local,
                        latent_local,
                        shared_gate_up,
                        projection.activation.ptr,
                        projection.accumulator.ptr,
                        tid,
                        lane,
                        wave,
                        bid,
                        tile_start,
                        tile_rows,
                        rows,
                        slabs,
                        row_stride,
                    )

    @flyc.jit
    def launch(
        oproj_reduced: Int64,
        prefix: Int64,
        blocks: Int64,
        blk_sm: Int64,
        blk_sr: Int64,
        ares_norm_weight: Int64,
        ares_query: Int64,
        input_norm_weight: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        x_local: Int64,
        router_local: Int64,
        latent_local: Int64,
        shared_gate_up: Int64,
        scratch: Int64,
        layer: Int32,
        attention_epsilon: Float32,
        output_epsilon: Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            oproj_reduced,
            prefix,
            blocks,
            blk_sm,
            blk_sr,
            ares_norm_weight,
            ares_query,
            input_norm_weight,
            router_weight,
            latent_weight,
            shared_weight,
            x_local,
            router_local,
            latent_local,
            shared_gate_up,
            scratch,
            layer,
            attention_epsilon,
            output_epsilon,
        ).launch(grid=(plan.blocks,), block=(THREADS,), stream=stream)

    return K2FrontKernel(
        name=name,
        launcher=launch,
        grid=(plan.blocks,),
        block=(THREADS,),
        required_lds_bytes=FRONT_REQUIRED_LDS_BYTES,
        scratch_bytes=scratch_bytes,
        traffic=traffic,
    )
