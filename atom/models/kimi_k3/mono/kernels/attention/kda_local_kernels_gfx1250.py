# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Launchable rank-local gfx1250 K1 staged pipeline.

Six current-stream launches implement AttnRes slice/gate/mix, projection,
fused convolution/recurrence, and output normalization. The pipeline has no
transport or remote-rank argument.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.kernels.attention import kda as base
from atom.models.kimi_k3.mono.kernels.attention import (
    kda_gfx1250 as gfx_backend,
)
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    HIDDEN,
    K1_ABI_NAMES,
    K1KernelDeclaration,
    K1LocalBuildSpec,
    local_scratch_layout,
)
from atom.models.kimi_k3.mono.kernels.attention.protocol import (
    ATTENTION_CTAS,
    HEAD_DIM,
    HEADS,
    NPROJ,
    PROJECTION_TASKS,
    STATE_CHUNKS,
)
from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    hw_rsq,
    ld_i32,
    rsrc,
    traced,
    uniform,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.plan.build_key import key_tuple
from atom.mono.runtime.abi import KernelAbi

FUSED_ROWS = 6284
ROWS = 16
PROJECTION_ROWS_PER_CTA = 2
SEAM_COLUMNS = 128
SEAM_GROUPS = 8
SEAM_SLICES_PER_GROUP = 7
VALUE_CHUNK = 16
_STREAM = fx.Stream(None)
_BUILDS: dict[tuple[object, ...], object] = {}
_PIPELINES: dict[tuple[object, ...], object] = {}
LOCAL_ABI = KernelAbi(K1_ABI_NAMES)


class LocalMailbox:
    """Device-scope tagged pairs for already stream-ordered local stages."""

    def __init__(self, tag):
        self.tag = tag

    def put(self, pointer, index, value):
        gfx_backend.mailbox_store_device(pointer, index, value, self.tag)

    @traced
    def poll(self, specs, batch=12):
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        return [
            [
                gfx_backend.mailbox_poll_device(pointer, index + item, self.tag)
                for item in range(count)
            ]
            for pointer, index, count in specs
        ]


@dataclass(frozen=True)
class ProjectionBuildKey:
    """Scalar-only cache key for the emitted projection slice."""

    tokens: int
    qlen: int
    nblocks: int
    threads: int
    blocks: int
    lds_bytes: int
    debug: bool = False


@dataclass(frozen=True)
class LocalPipelineBuildKey:
    tokens: int
    qlen: int
    nblocks: int
    threads: int
    blocks: int
    lds_bytes: int
    delta: bool
    state_len: int
    eps: float
    out_eps: float
    onorm_eps: float
    lower_bound: float
    write_idx: int
    debug: bool
    kda0: int


@dataclass(frozen=True)
class K1LocalProjectionEntryPoint:
    """The emitted K1 projection kernel and JIT launcher."""

    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    tokens: int
    token_tiles: int
    abi_names: tuple[str, ...] = LOCAL_ABI.names
    consumes: tuple[str, ...] = ("w_in", "scratch.x")
    produces: tuple[str, ...] = ("scratch.proj",)


def _require_contract(
    plan: Gfx1250KimiBuildPlan,
    declaration: K1KernelDeclaration,
) -> None:
    if plan.architecture is not GFX1250:
        raise ValueError("local K1 projection requires gfx1250")
    if plan.threads != declaration.threads or plan.blocks != declaration.blocks:
        raise ValueError("K1 declaration geometry does not match build plan")
    if plan.tokens != declaration.tokens or plan.nblocks != declaration.nblocks:
        raise ValueError("K1 declaration shape does not match build plan")
    if declaration.unsupported_device_protocol:
        raise ValueError("local K1 declaration still contains peer transport")
    if declaration.waves != 8 or declaration.threads != 256:
        raise ValueError("local K1 projection requires 256 threads/eight waves")


def build_projection_slice(
    plan: Gfx1250KimiBuildPlan,
    declaration: K1KernelDeclaration,
    spec: K1LocalBuildSpec | None = None,
) -> K1LocalProjectionEntryPoint:
    """Emit K1's rank-local in-projection with tagged local publication."""

    _require_contract(plan, declaration)
    cache_key = (
        declaration.tokens,
        declaration.qlen,
        declaration.nblocks,
        declaration.threads,
        declaration.blocks,
        plan.lds_bytes,
        False if spec is None else spec.debug,
    )
    cached = _BUILDS.get(cache_key)
    if cached is not None:
        return cached

    backend = plan.kda_backend()
    compute = declaration.bind_compute_backend(backend)
    device = declaration.bind_device_protocol(backend)
    lds_types = device["emit_role_lds_types"](declaration.lds_type_spec)
    scratch_layout = local_scratch_layout(declaration)
    slabs = declaration.lds.arena.gemv_slabs[0]
    tile_starts = tuple(tile.row_start for tile in declaration.token_tiles)
    tile_rows = tuple(tile.rows for tile in declaration.token_tiles)
    group_columns = plan.kda_matrix_contract().group_columns
    keyed = key_tuple(
        ProjectionBuildKey(
            declaration.tokens,
            declaration.qlen,
            declaration.nblocks,
            declaration.threads,
            declaration.blocks,
            plan.lds_bytes,
            False if spec is None else spec.debug,
        ),
        SOURCES,
    )
    name = (
        f"k3_mono_kda_pre_gfx1250_projection_s{declaration.tokens}"
        f"_l{declaration.qlen}_nb{declaration.nblocks}"
    )

    @traced
    def stage_x_tile(
        x,
        xl,
        tid,
        tile_start,
        rows,
        first_group,
        groups,
    ):
        columns = groups * group_columns
        words = rows * columns // 8
        for iteration in range_constexpr(
            (words + declaration.threads - 1) // declaration.threads
        ):
            work = tid + declaration.threads * iteration
            if work < words:
                local_token = work // (columns // 8)
                column = (work % (columns // 8)) * 8
                global_token = tile_start + local_token
                global_column = first_group * group_columns + column
                value = fx.Vector(
                    bo.buffer_load(
                        rsrc(x),
                        (global_token * HIDDEN + global_column) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                fx.ptr_store(
                    value,
                    xl + (local_token * declaration.xrow + column) // 2,
                )
        gpu.barrier()

    @flyc.kernel(
        name=name,
        known_block_size=[declaration.threads, 1, 1],
    )
    def projection(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        layer: Int32,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        allocator = fx.SharedAllocator()
        smem = allocator.allocate(lds_types.smem).peek()
        role = allocator.allocate(lds_types.role)
        gemv = role.gemv.peek()
        x = scratch + fx.Int64(scratch_layout["x"][0])
        proj = scratch + fx.Int64(scratch_layout["proj"][0])
        context = {
            "S": ROWS,
            "lane": lane,
            "wave": wave,
            "matrix_waves": declaration.waves,
            "gemv_tasks": PROJECTION_TASKS,
            "fused_rows": FUSED_ROWS,
            "xrow": declaration.xrow,
            "xl": gemv.xl.ptr,
            "args": {"w_in": w_in},
        }

        if bid >= ATTENTION_CTAS:
            first_task = (bid - ATTENTION_CTAS) * PROJECTION_ROWS_PER_CTA
            for task_offset in range_constexpr(PROJECTION_ROWS_PER_CTA):
                task = first_task + task_offset
                if task < PROJECTION_TASKS:
                    for tile_index in range_constexpr(len(tile_starts)):
                        context["S"] = tile_rows[tile_index]
                        accumulator = compute["kda_zero"]()
                        for slab in range_constexpr(len(slabs)):
                            first_group = slabs[slab][0]
                            groups = slabs[slab][1]
                            operands = compute["kda_gemv_loads"](
                                context,
                                task,
                                range(first_group, first_group + groups),
                            )
                            if const_expr(slab):
                                gpu.barrier()
                            stage_x_tile(
                                x,
                                gemv.xl.ptr,
                                tid,
                                tile_starts[tile_index],
                                tile_rows[tile_index],
                                first_group,
                                groups,
                            )
                            accumulator = compute["kda_gemv_slab"](
                                context,
                                operands,
                                groups,
                                accumulator,
                            )
                        compute["kda_gemv_store"](
                            accumulator,
                            smem.red.ptr,
                            wave,
                            lane,
                        )
                        gpu.barrier()

                        work_items = tile_rows[tile_index] * ROWS
                        for iteration in range_constexpr(
                            (work_items + declaration.threads - 1)
                            // declaration.threads
                        ):
                            work = tid + declaration.threads * iteration
                            if work < work_items:
                                local_token = work // ROWS
                                row = work % ROWS
                                value = fx.Float32(
                                    compute["kda_gemv_row_sum"](
                                        smem.red.ptr,
                                        row,
                                        local_token,
                                    )
                                ).to(fx.BFloat16)
                                value = fx.Float32(value)
                                global_token = tile_starts[tile_index] + local_token
                                device["mailbox_store_device"](
                                    proj,
                                    global_token * NPROJ + task * ROWS + row,
                                    value,
                                    layer + 1,
                                )
                                if const_expr(spec is not None and spec.debug):
                                    bo.buffer_store(
                                        value.to(fx.BFloat16),
                                        rsrc(proj_dbg),
                                        global_token * NPROJ + task * ROWS + row,
                                    )
                        gpu.barrier()
        device["wait_vector_stores"]()

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        projection(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_in,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            core_out,
            x_dbg,
            proj_dbg,
            scratch,
            layer,
            tl,
        ).launch(
            grid=(declaration.blocks,),
            block=(declaration.threads,),
            stream=stream,
        )

    LOCAL_ABI.check(projection, launch)
    entry = K1LocalProjectionEntryPoint(
        name,
        launch,
        (declaration.blocks,),
        (declaration.threads,),
        declaration.tokens,
        len(declaration.token_tiles),
    )
    _BUILDS[cache_key] = entry
    return entry


@dataclass(frozen=True)
class LocalStageEntryPoint:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]


@dataclass(frozen=True)
class K1LocalPipelineEntryPoint:
    """Complete rank-local K1 callable composed on one current stream."""

    name: str
    launcher: object
    stages: tuple[LocalStageEntryPoint, ...]
    abi_names: tuple[str, ...] = LOCAL_ABI.names
    launch_count: int = 6
    complete: bool = True


def _task_start(block, first, blocks):
    return (block - first + blocks) % blocks


def build_local_pipeline(
    plan: Gfx1250KimiBuildPlan,
    declaration: K1KernelDeclaration,
    spec: K1LocalBuildSpec,
) -> K1LocalPipelineEntryPoint:
    """Emit complete six-launch and seven-launch rank-local K1 paths."""

    _require_contract(plan, declaration)
    spec.validate(plan)
    cache_key = (
        declaration.tokens,
        declaration.qlen,
        declaration.nblocks,
        declaration.threads,
        declaration.blocks,
        plan.lds_bytes,
        spec.delta,
        spec.state_len,
        spec.eps,
        spec.out_eps,
        spec.onorm_eps,
        spec.lower_bound,
        spec.write_idx,
        spec.debug,
        spec.kda0,
    )
    cached = _PIPELINES.get(cache_key)
    if cached is not None:
        return cached

    backend = plan.kda_backend()
    compute = declaration.bind_compute_backend(backend)
    device = declaration.bind_device_protocol(backend)
    lds_types = device["emit_role_lds_types"](declaration.lds_type_spec)
    layout = local_scratch_layout(declaration)
    s = declaration.tokens
    L = declaration.qlen
    ns = declaration.nblocks + 1
    task_count = (s // L) * HEADS * STATE_CHUNKS
    keyed = key_tuple(
        LocalPipelineBuildKey(
            declaration.tokens,
            declaration.qlen,
            declaration.nblocks,
            declaration.threads,
            declaration.blocks,
            plan.lds_bytes,
            spec.delta,
            spec.state_len,
            spec.eps,
            spec.out_eps,
            spec.onorm_eps,
            spec.lower_bound,
            spec.write_idx,
            spec.debug,
            spec.kda0,
        ),
        SOURCES,
    )

    @flyc.kernel(
        name=f"k3_k1_gfx1250_attn_slice_s{s}_nb{declaration.nblocks}",
        known_block_size=[declaration.threads, 1, 1],
    )
    def attn_slice(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        task = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        part = scratch + fx.Int64(layout["part"][0])
        values = scratch + fx.Int64(layout["vals"][0])
        work_items = s * ns * (SEAM_COLUMNS // 4)
        for iteration in range_constexpr(
            (work_items + declaration.threads - 1) // declaration.threads
        ):
            work = tid + declaration.threads * iteration
            live = work < work_items
            item = fx.min(work, work_items - 1)
            row = item // (SEAM_COLUMNS // 4)
            token = row // ns
            source = row % ns
            column = task * SEAM_COLUMNS + (item % (SEAM_COLUMNS // 4)) * 4
            is_prefix = source == declaration.nblocks
            block_values = base.ld_bf4(
                blocks,
                token * blk_sm
                + fx.min(source, declaration.nblocks - 1) * blk_sr
                + column,
            )
            prefix_values = base.ld_bf4(prefix, token * HIDDEN + column)
            if const_expr(spec.delta):
                delta_values = base.ld_bf4(delta, token * HIDDEN + column)
                prefix_values = [
                    prefix_values[element] + delta_values[element]
                    for element in range(4)
                ]
                if live & is_prefix:
                    bo.buffer_store(
                        base.bf4_words(prefix_values),
                        rsrc(prefix),
                        (token * HIDDEN + column) // 2,
                    )
            if const_expr(spec.write_idx >= 0):  # noqa: SIM102
                if live & is_prefix:
                    bo.buffer_store(
                        base.bf4_words(prefix_values),
                        rsrc(blocks),
                        (token * blk_sm + spec.write_idx * blk_sr + column) // 2,
                    )
            selected = [
                live.select(
                    is_prefix.select(
                        prefix_values[element],
                        block_values[element],
                    ),
                    fx.Float32(0.0),
                )
                for element in range(4)
            ]
            if live:
                bo.buffer_store(
                    fx.Vector.from_elements(selected, fx.Float32),
                    rsrc(values),
                    ((token * ns + source) * HIDDEN + column),
                )
            norm_weight = base.ld_bf4(ares_nw, column)
            query = base.ld_bf4(ares_qk, column)
            square_sum = sum(value * value for value in selected)
            dot = sum(
                selected[element] * (norm_weight[element] * query[element])
                for element in range(4)
            )
            square_sum = compute["attention_reduce32"](square_sum)
            dot = compute["attention_reduce32"](dot)
            if live & (lane == 0):
                index = ((task * s + token) * ns + source) * 2
                gfx_backend.mailbox_store_device(part, index, square_sum, layer + 1)
                gfx_backend.mailbox_store_device(part, index + 1, dot, layer + 1)
        device["wait_vector_stores"]()

    @flyc.jit
    def launch_attn_slice(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        attn_slice(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            scratch,
            layer,
        ).launch(
            grid=(ATTENTION_CTAS,),
            block=(declaration.threads,),
            stream=stream,
        )

    @flyc.kernel(
        name=f"k3_k1_gfx1250_attn_gate_s{s}_nb{declaration.nblocks}",
        known_block_size=[declaration.threads, 1, 1],
    )
    def attn_gate(scratch: Int64, layer: Int32):
        _ = keyed
        tid = fx.thread_idx.x
        allocator = fx.SharedAllocator()
        smem = allocator.allocate(lds_types.smem).peek()
        context = {
            "S": s,
            "NS": ns,
            "tid": tid,
            "lane": tid % GFX1250.wave_size,
            "wave": tid // GFX1250.wave_size,
            "red": smem.red.ptr,
            "eps": spec.eps,
            "part": scratch + fx.Int64(layout["part"][0]),
            "wgt": scratch + fx.Int64(layout["wgt"][0]),
            "put": LocalMailbox(layer + 1).put,
            "poll": LocalMailbox(layer + 1).poll,
        }
        compute["attention_gate"](context, fx.block_idx.x)
        device["wait_vector_stores"]()

    @flyc.jit
    def launch_attn_gate(
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        attn_gate(scratch, layer).launch(
            grid=(s,),
            block=(declaration.threads,),
            stream=stream,
        )

    @flyc.kernel(
        name=f"k3_k1_gfx1250_attn_mix_s{s}_nb{declaration.nblocks}",
        known_block_size=[declaration.threads, 1, 1],
    )
    def attn_mix(
        in_nw: Int64,
        x_dbg: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        task = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        allocator = fx.SharedAllocator()
        smem = allocator.allocate(lds_types.smem).peek()
        red = smem.red.ptr
        weights = scratch + fx.Int64(layout["wgt"][0])
        msq = scratch + fx.Int64(layout["msq"][0])
        values = scratch + fx.Int64(layout["vals"][0])
        x = scratch + fx.Int64(layout["x"][0])
        weight_count = s * (ns + 1)
        for iteration in range_constexpr(
            (weight_count + declaration.threads - 1) // declaration.threads
        ):
            item = tid + declaration.threads * iteration
            if item < weight_count:
                word = gfx_backend.mailbox_poll_device(weights, item, layer + 1)
                fx.ptr_store(word.bitcast(fx.Float32), red + item)
        gpu.barrier()

        rounds = (s + declaration.waves - 1) // declaration.waves
        tokens = []
        live_tokens = []
        mixed_values = []
        column = lane * 4
        column0 = task * SEAM_COLUMNS
        for round_index in range_constexpr(rounds):
            token = fx.min(wave + declaration.waves * round_index, s - 1)
            live = wave + declaration.waves * round_index < s
            mixed = []
            for element in range_constexpr(4):
                total = fx.Float32(0.0)
                for source in range_constexpr(ns):
                    source_value = fx.Float32(
                        bo.buffer_load(
                            rsrc(values),
                            (token * ns + source) * HIDDEN + column0 + column + element,
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )
                    total = (
                        total
                        + fx.ptr_load(red + token * (ns + 1) + source) * source_value
                    )
                mixed.append(total / fx.ptr_load(red + token * (ns + 1) + ns))
            square_sum = sum(value * value for value in mixed)
            square_sum = compute["attention_reduce32"](square_sum)
            if live & (lane == 0):
                gfx_backend.mailbox_store_device(
                    msq,
                    task * s + token,
                    square_sum,
                    layer + 1,
                )
            tokens.append(token)
            live_tokens.append(live)
            mixed_values.append(mixed)
        gpu.barrier()

        group_work = SEAM_GROUPS * s
        for iteration in range_constexpr(
            (group_work + declaration.threads - 1) // declaration.threads
        ):
            item = tid + declaration.threads * iteration
            if item < group_work:
                group = item // s
                token = item % s
                total = gfx_backend.mailbox_poll_device(
                    msq,
                    group * SEAM_SLICES_PER_GROUP * s + token,
                    layer + 1,
                ).bitcast(fx.Float32)
                for index in range(1, SEAM_SLICES_PER_GROUP):
                    total = total + gfx_backend.mailbox_poll_device(
                        msq,
                        (group * SEAM_SLICES_PER_GROUP + index) * s + token,
                        layer + 1,
                    ).bitcast(fx.Float32)
                fx.ptr_store(total, red + 128 + group * s + token)
        gpu.barrier()

        for round_index in range_constexpr(rounds):
            token = tokens[round_index]
            live = live_tokens[round_index]
            mixed = mixed_values[round_index]
            total = fx.ptr_load(red + 128 + token)
            for group in range_constexpr(1, SEAM_GROUPS):
                total = total + fx.ptr_load(red + 128 + group * s + token)
            inverse = hw_rsq(
                total * fx.Float32(1.0 / HIDDEN) + fx.Float32(spec.out_eps)
            )
            norm_weight = base.ld_bf4(in_nw, column0 + column)
            output = base.bf4_words(
                [
                    mixed[element] * inverse * norm_weight[element]
                    for element in range(4)
                ]
            )
            if live:
                output_index = (token * HIDDEN + column0 + column) // 2
                bo.buffer_store(
                    output,
                    rsrc(x),
                    output_index,
                    cache_modifier=CM_DEV,
                )
                if const_expr(spec.debug):
                    bo.buffer_store(output, rsrc(x_dbg), output_index)
        device["mailbox_publish_device"](
            scratch + fx.Int64(layout["xrdy"][0]),
            task,
            fx.Int32(1),
            layer + 1,
            tid == 0,
        )

    @flyc.jit
    def launch_attn_mix(
        in_nw: Int64,
        x_dbg: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        attn_mix(in_nw, x_dbg, scratch, layer).launch(
            grid=(ATTENTION_CTAS,),
            block=(declaration.threads,),
            stream=stream,
        )

    projection = build_projection_slice(plan, declaration, spec)

    @traced
    def conv_metadata(context, request, head, chunk):
        tid = context["tid"]
        args = context["args"]
        metadata = {}
        metadata["acc_idx"] = uniform(ld_i32(args["num_acc"], request)) - 1
        metadata["slot_c"] = uniform(
            ld_i32(
                args["st_idx"],
                request * args["st_stride"],
            )
        )
        channel_thread = fx.min(tid, 3 * VALUE_CHUNK - 1)
        part = channel_thread // VALUE_CHUNK
        channel = (
            part * (HEADS * HEAD_DIM)
            + head * HEAD_DIM
            + chunk * VALUE_CHUNK
            + channel_thread % VALUE_CHUNK
        )
        conv_state = (
            args["conv_st"]
            + fx.Int64(fx.max(metadata["slot_c"], 0)) * fx.Int64(args["cs_seq"]) * 2
        )
        channel_offset = channel * args["cs_dim"]
        metadata["cs"] = conv_state
        metadata["co"] = channel_offset
        metadata["chan"] = channel
        metadata["part"] = part
        metadata["cw"] = fx.Vector(
            bo.buffer_load(
                rsrc(args["conv_w"]),
                channel * 4,
                vec_width=4,
                dtype=T.f32,
            )
        )
        metadata["win"] = [
            base.ld_bf(
                conv_state,
                channel_offset + (metadata["acc_idx"] + index) * args["cs_tok"],
            )
            for index in range(3)
        ]
        metadata["old"] = [
            base.ld_bf(
                conv_state,
                channel_offset + (metadata["acc_idx"] + 1 + index) * args["cs_tok"],
            )
            for index in range(spec.state_len - L)
        ]
        return metadata

    @flyc.kernel(
        name=f"k3_k1_gfx1250_conv_s{s}_l{L}",
        known_block_size=[declaration.threads, 1, 1],
    )
    def conv(
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        allocator = fx.SharedAllocator()
        smem = allocator.allocate(lds_types.smem).peek()
        role = allocator.allocate(lds_types.role)
        recurrence = role.kda.peek()
        context = {
            "S": s,
            "L": L,
            "SL": spec.state_len,
            "tid": tid,
            "matrix_threads": declaration.threads,
            "red": smem.red.ptr,
            "tls": smem.tls.ptr,
            "vloc": recurrence.vloc.ptr,
            "proj": scratch + fx.Int64(layout["proj"][0]),
            "qk": scratch + fx.Int64(layout["qk"][0]),
            "put": LocalMailbox(layer + 1).put,
            "poll": LocalMailbox(layer + 1).poll,
            "on": False,
            "xp": 0,
            "args": {
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
            },
        }
        first = _task_start(bid, spec.kda0, declaration.blocks)
        for task in range(first, task_count, declaration.blocks):
            request = task // (HEADS * STATE_CHUNKS)
            head = (task // STATE_CHUNKS) % HEADS
            chunk = task % STATE_CHUNKS
            metadata = conv_metadata(context, request, head, chunk)
            base.kda_conv(context, metadata, request, head, chunk)
            if tid < L * VALUE_CHUNK:
                token = tid // VALUE_CHUNK
                element = tid % VALUE_CHUNK
                value = fx.ptr_load(recurrence.vloc.ptr + token * VALUE_CHUNK + element)
                bo.buffer_store(
                    value,
                    rsrc(scratch + fx.Int64(layout["vloc"][0])),
                    ((request * L + token) * HEADS + head) * HEAD_DIM
                    + chunk * VALUE_CHUNK
                    + element,
                )
            gpu.barrier()
        device["wait_vector_stores"]()

    @traced
    def recurrent_metadata(context, request, head, chunk):
        args = context["args"]
        metadata = {}
        metadata["acc_idx"] = uniform(ld_i32(args["num_acc"], request)) - 1
        metadata["slot0"] = uniform(
            ld_i32(
                args["st_idx"],
                request * args["st_stride"] + metadata["acc_idx"],
            )
        )
        metadata["slots"] = [
            uniform(
                ld_i32(
                    args["st_idx"],
                    request * args["st_stride"] + token,
                )
            )
            for token in range(L)
        ]
        compute["kda_state_loads"](context, metadata, head, chunk)
        metadata["fb"], metadata["dtb"] = compute["kda_gate_loads"](context, head)
        metadata["aa"] = base.hw_exp2(
            base.ld_f32(args["a_log"], head) * fx.Float32(base.LOG2E)
        )
        return metadata

    @flyc.kernel(
        name=f"k3_k1_gfx1250_conv_recur_s{s}_l{L}",
        known_block_size=[declaration.threads, 1, 1],
    )
    def conv_recur(
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        allocator = fx.SharedAllocator()
        smem = allocator.allocate(lds_types.smem).peek()
        role = allocator.allocate(lds_types.role)
        recurrence = role.kda.peek()
        context = {
            "S": s,
            "L": L,
            "SL": spec.state_len,
            "tid": tid,
            "lane": tid % GFX1250.wave_size,
            "wave": tid // GFX1250.wave_size,
            "matrix_waves": declaration.waves,
            "matrix_threads": declaration.threads,
            "red": smem.red.ptr,
            "tls": smem.tls.ptr,
            "fa": recurrence.fa.ptr,
            "qkl": recurrence.qkl.ptr,
            "decay": recurrence.decay.ptr,
            "vloc": recurrence.vloc.ptr,
            "ol": recurrence.ol.ptr,
            "beta": recurrence.beta.ptr,
            "proj": scratch + fx.Int64(layout["proj"][0]),
            "qk": scratch + fx.Int64(layout["qk"][0]),
            "onrm": scratch + fx.Int64(layout["onrm"][0]),
            "put": LocalMailbox(layer + 1).put,
            "poll": LocalMailbox(layer + 1).poll,
            "on": False,
            "xp": 0,
            "lb": spec.lower_bound,
            "kda_gate_compute": compute["kda_gate_compute"],
            "kda_qk_normalize": compute["kda_qk_normalize"],
            "args": {
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
            },
        }
        first = _task_start(bid, spec.kda0, declaration.blocks)
        for task in range(first, task_count, declaration.blocks):
            request = task // (HEADS * STATE_CHUNKS)
            head = (task // STATE_CHUNKS) % HEADS
            chunk = task % STATE_CHUNKS
            conv_values = conv_metadata(context, request, head, chunk)
            base.kda_conv(context, conv_values, request, head, chunk)
            recurrent_values = recurrent_metadata(context, request, head, chunk)
            base.kda_gates(context, recurrent_values, request, head)
            compute["kda_recur"](context, recurrent_values, request, head, chunk)
            if tid < L * VALUE_CHUNK:
                token = tid // VALUE_CHUNK
                element = tid % VALUE_CHUNK
                value = fx.ptr_load(recurrence.ol.ptr + token * VALUE_CHUNK + element)
                bo.buffer_store(
                    value,
                    rsrc(scratch + fx.Int64(layout["ol"][0])),
                    ((request * L + token) * HEADS + head) * HEAD_DIM
                    + chunk * VALUE_CHUNK
                    + element,
                )
            gpu.barrier()
        device["wait_vector_stores"]()

    @flyc.jit
    def launch_conv_recur(
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        conv_recur(
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            scratch,
            layer,
        ).launch(
            grid=(declaration.blocks,),
            block=(declaration.threads,),
            stream=stream,
        )

    @flyc.kernel(
        name=f"k3_k1_gfx1250_norm_s{s}_l{L}",
        known_block_size=[declaration.threads, 1, 1],
    )
    def norm(
        on_w: Int64,
        core_out: Int64,
        scratch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        allocator = fx.SharedAllocator()
        allocator.allocate(lds_types.smem).peek()
        role = allocator.allocate(lds_types.role)
        recurrence = role.kda.peek()
        context = {
            "L": L,
            "tid": tid,
            "ol": recurrence.ol.ptr,
            "onrm": scratch + fx.Int64(layout["onrm"][0]),
            "proj": scratch + fx.Int64(layout["proj"][0]),
            "poll": LocalMailbox(layer + 1).poll,
            "onorm_eps": spec.onorm_eps,
            "args": {
                "on_w": on_w,
                "core_out": core_out,
            },
        }
        first = _task_start(bid, spec.kda0, declaration.blocks)
        for task in range(first, task_count, declaration.blocks):
            request = task // (HEADS * STATE_CHUNKS)
            head = (task // STATE_CHUNKS) % HEADS
            chunk = task % STATE_CHUNKS
            if tid < L * VALUE_CHUNK:
                token = tid // VALUE_CHUNK
                element = tid % VALUE_CHUNK
                value = fx.Float32(
                    bo.buffer_load(
                        rsrc(scratch + fx.Int64(layout["ol"][0])),
                        ((request * L + token) * HEADS + head) * HEAD_DIM
                        + chunk * VALUE_CHUNK
                        + element,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
                fx.ptr_store(
                    value,
                    recurrence.ol.ptr + token * VALUE_CHUNK + element,
                )
            gpu.barrier()
            base.kda_norm(context, request, head, chunk)

    @flyc.jit
    def launch_norm(
        on_w: Int64,
        core_out: Int64,
        scratch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        norm(on_w, core_out, scratch, layer).launch(
            grid=(declaration.blocks,),
            block=(declaration.threads,),
            stream=stream,
        )

    stages = (
        LocalStageEntryPoint(
            "attn_slice",
            launch_attn_slice,
            (ATTENTION_CTAS,),
            (declaration.threads,),
        ),
        LocalStageEntryPoint(
            "attn_gate",
            launch_attn_gate,
            (s,),
            (declaration.threads,),
        ),
        LocalStageEntryPoint(
            "attn_mix",
            launch_attn_mix,
            (ATTENTION_CTAS,),
            (declaration.threads,),
        ),
        LocalStageEntryPoint(
            "projection",
            projection.launcher,
            projection.grid,
            projection.block,
        ),
        LocalStageEntryPoint(
            "conv_recur",
            launch_conv_recur,
            (declaration.blocks,),
            (declaration.threads,),
        ),
        LocalStageEntryPoint(
            "norm",
            launch_norm,
            (declaration.blocks,),
            (declaration.threads,),
        ),
    )

    def launch_sequence(
        prefix,
        delta,
        blocks,
        blk_sm,
        blk_sr,
        ares_nw,
        ares_qk,
        in_nw,
        w_in,
        w_fb,
        conv_w,
        conv_st,
        cs_seq,
        cs_dim,
        cs_tok,
        a_log,
        dt_bias,
        on_w,
        rstate,
        rs_seq,
        st_idx,
        st_stride,
        num_acc,
        core_out,
        x_dbg,
        proj_dbg,
        scratch,
        layer,
        tl,
        stream=_STREAM,
    ):
        _ = tl
        launch_attn_slice(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            scratch,
            layer,
            stream,
        )
        launch_attn_gate(scratch, layer, stream)
        launch_attn_mix(in_nw, x_dbg, scratch, layer, stream)
        projection.launcher(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_in,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            core_out,
            x_dbg,
            proj_dbg,
            scratch,
            layer,
            tl,
            stream,
        )
        launch_conv_recur(
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            scratch,
            layer,
            stream,
        )
        launch_norm(on_w, core_out, scratch, layer, stream)

    entry = K1LocalPipelineEntryPoint(
        f"k3_k1_gfx1250_local_pipeline_s{s}_l{L}",
        launch_sequence,
        stages,
    )
    _PIPELINES[cache_key] = entry
    return entry
