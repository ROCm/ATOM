# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-launch resident gfx1250 Kimi-K3 K1."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import GFX1250_BLOCKS
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
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    fp8_pack4,
    hw_rcp,
    ld_i32,
    rsrc,
    traced,
    uniform,
    wave_max,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.plan.build_key import key_tuple
from atom.mono.runtime.abi import KernelAbi

FUSED_ROWS = 6284
ROWS = 16
PROJECTION_ROWS_PER_CTA = 2
SEAM_COLUMNS = 128
VALUE_CHUNK = 16
PROJ = HEADS * HEAD_DIM
PTPC_FP8_MAX = 448.0
_STREAM = fx.Stream(None)
_BUILDS: dict[tuple[object, ...], object] = {}
LOCAL_ABI = KernelAbi(K1_ABI_NAMES)


@dataclass(frozen=True)
class ResidentBuildKey:
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
    quant_core: bool
    debug: bool
    kda0: int
    gemv_unit: int


@dataclass(frozen=True)
class ResidentK1EntryPoint:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = LOCAL_ABI
    launch_count: int = 1
    complete: bool = True


def _require_contract(
    plan: Gfx1250KimiBuildPlan,
    declaration: K1KernelDeclaration,
    spec: K1LocalBuildSpec,
) -> None:
    spec.validate(plan)
    if plan.architecture is not GFX1250:
        raise ValueError("resident K1 requires gfx1250")
    if (plan.threads, plan.blocks) != (declaration.threads, declaration.blocks):
        raise ValueError("resident K1 declaration geometry drifted from its plan")
    if declaration.unsupported_device_protocol:
        raise ValueError("resident local K1 must not contain peer transport")
    if declaration.threads != 256 or declaration.waves != 8:
        raise ValueError("resident K1 requires 256 threads/eight wave32 waves")
    if declaration.blocks != GFX1250_BLOCKS:
        raise ValueError(f"resident K1 requires {GFX1250_BLOCKS} all-resident CTAs")
    if declaration.gemv_unit != spec.gemv_unit:
        raise ValueError("resident K1 GEMV unit drifted from its declaration")


@traced
def _opening_generation(sync, ticket, blocks):
    """Reusable all-resident rendezvous returning one tag per invocation."""

    tid = fx.thread_idx.x
    generation = fx.Int32(0)
    if tid == 0:
        previous = gfx_backend.epoch_load_system_acquire(sync + fx.Int64(4))
        arrival = gfx_backend.epoch_arrive_system_atomic(sync, fx.Int32(1))
        if arrival == fx.Int32(blocks - 1):
            gfx_backend.epoch_store_system_release(sync, fx.Int32(0))
            generation = previous + fx.Int32(1)
            gfx_backend.epoch_store_system_release(
                sync + fx.Int64(4),
                generation,
            )
        else:
            generation = gfx_backend.epoch_load_system_acquire(sync + fx.Int64(4))
            while generation == previous:
                rocdl.s_sleep(1)
                generation = gfx_backend.epoch_load_system_acquire(sync + fx.Int64(4))
        fx.ptr_store(generation, ticket)
    gpu.barrier()
    generation = fx.ptr_load(ticket)
    gpu.barrier()
    return generation


def _task_start(block, first, blocks):
    return (block - first + blocks) % blocks


def build_resident_k1(
    plan: Gfx1250KimiBuildPlan,
    declaration: K1KernelDeclaration,
    spec: K1LocalBuildSpec,
) -> ResidentK1EntryPoint:
    """Emit the complete rank-local K1 as one all-resident launch."""

    _require_contract(plan, declaration, spec)
    cache_key = (
        declaration.tokens,
        declaration.qlen,
        declaration.nblocks,
        declaration.threads,
        declaration.blocks,
        plan.lds_bytes,
        spec,
    )
    cached = _BUILDS.get(cache_key)
    if cached is not None:
        return cached

    backend = plan.kda_backend()
    compute = declaration.bind_compute_backend(backend)
    device = declaration.bind_device_protocol(backend)
    lds_types = device["emit_role_lds_types"](declaration.lds_type_spec)
    layout = local_scratch_layout(declaration)
    slabs = declaration.lds.arena.gemv_slabs[0]
    tile_starts = tuple(tile.row_start for tile in declaration.token_tiles)
    tile_rows = tuple(tile.rows for tile in declaration.token_tiles)
    group_columns = plan.kda_matrix_contract().group_columns
    s = declaration.tokens
    qlen = declaration.qlen
    sources = declaration.nblocks + 1
    task_count = (s // qlen) * HEADS * STATE_CHUNKS
    keyed = key_tuple(
        ResidentBuildKey(
            s,
            qlen,
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
            spec.quant_core,
            spec.debug,
            spec.kda0,
            spec.gemv_unit,
        ),
        SOURCES,
    )
    name = (
        f"k3_k1_gfx1250_resident_s{s}_l{qlen}_nb{declaration.nblocks}"
        f"_u{spec.gemv_unit}_q{int(spec.quant_core)}"
    )

    @traced
    def stage_x_tile(x, xl, tid, tile_start, rows, first_group, groups):
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

    @traced
    def conv_metadata(context, request, head, chunk):
        tid = context["tid"]
        args = context["args"]
        metadata = {}
        metadata["acc_idx"] = uniform(ld_i32(args["num_acc"], request)) - 1
        metadata["slot_c"] = uniform(
            ld_i32(args["st_idx"], request * args["st_stride"])
        )
        channel_thread = fx.min(tid, 3 * VALUE_CHUNK - 1)
        metadata["part"] = channel_thread // VALUE_CHUNK
        metadata["chan"] = (
            metadata["part"] * (HEADS * HEAD_DIM)
            + head * HEAD_DIM
            + chunk * VALUE_CHUNK
            + channel_thread % VALUE_CHUNK
        )
        metadata["cs"] = (
            args["conv_st"]
            + fx.Int64(fx.max(metadata["slot_c"], 0)) * fx.Int64(args["cs_seq"]) * 2
        )
        metadata["co"] = metadata["chan"] * args["cs_dim"]
        metadata["cw"] = fx.Vector(
            bo.buffer_load(
                rsrc(args["conv_w"]),
                metadata["chan"] * 4,
                vec_width=4,
                dtype=T.f32,
            )
        )
        metadata["win"] = [
            base.ld_bf(
                metadata["cs"],
                metadata["co"] + (metadata["acc_idx"] + index) * args["cs_tok"],
            )
            for index in range(3)
        ]
        metadata["old"] = [
            base.ld_bf(
                metadata["cs"],
                metadata["co"] + (metadata["acc_idx"] + 1 + index) * args["cs_tok"],
            )
            for index in range(spec.state_len - qlen)
        ]
        return metadata

    @traced
    def gate_metadata(context, head):
        args = context["args"]
        metadata = {}
        metadata["fb"], metadata["dtb"] = compute["kda_gate_loads"](
            context,
            head,
        )
        metadata["aa"] = base.hw_exp2(
            base.ld_f32(args["a_log"], head) * fx.Float32(base.LOG2E)
        )
        return metadata

    @traced
    def load_state_metadata(context, metadata, request, head, chunk):
        tid = context["tid"]
        args = context["args"]
        metadata["acc_idx"] = uniform(ld_i32(args["num_acc"], request)) - 1
        metadata["slot0"] = uniform(
            ld_i32(
                args["st_idx"],
                request * args["st_stride"] + metadata["acc_idx"],
            )
        )
        if tid < qlen:
            fx.ptr_store(
                ld_i32(
                    args["st_idx"],
                    request * args["st_stride"] + tid,
                ),
                context["slots"] + tid,
            )
        gpu.barrier()
        compute["kda_state_loads"](context, metadata, head, chunk)
        return metadata

    @traced
    def quantize_core(token, tid, lane, wave, red, core_out, core_q):
        values = []
        maximum = fx.Float32(0.0)
        for iteration in range_constexpr(
            (PROJ + declaration.threads * 4 - 1) // (declaration.threads * 4)
        ):
            element = (iteration * declaration.threads + tid) * 4
            live = element < PROJ
            loaded = base.ld_bf4(
                core_out,
                token * PROJ + fx.min(element, PROJ - 4),
            )
            loaded = [live.select(value, fx.Float32(0.0)) for value in loaded]
            values.extend(loaded)
            for value in loaded:
                maximum = fx.max(maximum, fmath.absf(value))

        maximum = wave_max(maximum, GFX1250)
        gpu.barrier()
        if lane == 0:
            fx.ptr_store(maximum, red + wave)
        gpu.barrier()
        maximum = fx.ptr_load(red)
        for other_wave in range_constexpr(1, declaration.waves):
            maximum = fx.max(maximum, fx.ptr_load(red + other_wave))
        scale = maximum * fx.Float32(1.0 / PTPC_FP8_MAX)
        if tid == 0:
            bo.buffer_store(
                scale,
                rsrc(core_q + fx.Int64(s * PROJ)),
                token,
            )
        inverse = hw_rcp(scale)

        for iteration in range_constexpr(len(values) // 4):
            element = (iteration * declaration.threads + tid) * 4
            if element < PROJ:
                quantized = [
                    fx.max(
                        fx.min(
                            values[iteration * 4 + offset] * inverse,
                            fx.Float32(PTPC_FP8_MAX),
                        ),
                        fx.Float32(-PTPC_FP8_MAX),
                    )
                    for offset in range(4)
                ]
                bo.buffer_store(
                    fp8_pack4(*quantized),
                    rsrc(core_q),
                    (token * PROJ + element) // 4,
                    cache_modifier=CM_DEV,
                )
        device["wait_vector_stores"]()

    @flyc.kernel(name=name, known_block_size=[declaration.threads, 1, 1])
    def resident(
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
        core_q: Int64,
        scratch: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        allocator = fx.SharedAllocator()
        smem = allocator.allocate(lds_types.smem).peek()
        role = allocator.allocate(lds_types.role)
        attention = role.attn.peek()
        projection = role.gemv.peek()
        recurrence = role.kda.peek()
        sync = scratch + fx.Int64(layout["sync"][0])
        generation = _opening_generation(
            sync,
            projection.xl.ptr,
            declaration.blocks,
        )
        part = scratch + fx.Int64(layout["part"][0])
        weights = scratch + fx.Int64(layout["wgt"][0])
        mean_squares = scratch + fx.Int64(layout["msq"][0])
        x_ready = scratch + fx.Int64(layout["xrdy"][0])
        x = scratch + fx.Int64(layout["x"][0])
        projected = scratch + fx.Int64(layout["proj"][0])
        # The seam's PART handoff is dead once every projection is visible.
        # Reuse its much larger tagged-pair region for post-KDA core readiness
        # instead of keeping another base pointer live through the resident
        # kernel.
        core_ready = part

        if bid < ATTENTION_CTAS:
            for tile in range_constexpr(len(tile_starts)):
                token0 = tile_starts[tile]
                rows = tile_rows[tile]
                work_items = rows * sources * (SEAM_COLUMNS // 4)
                for iteration in range_constexpr(
                    (work_items + declaration.threads - 1) // declaration.threads
                ):
                    work = tid + declaration.threads * iteration
                    live = work < work_items
                    item = fx.min(work, work_items - 1)
                    local_token = item // (sources * (SEAM_COLUMNS // 4))
                    source = (item // (SEAM_COLUMNS // 4)) % sources
                    column = bid * SEAM_COLUMNS + (item % (SEAM_COLUMNS // 4)) * 4
                    global_token = token0 + local_token
                    is_prefix = source == declaration.nblocks
                    block_values = base.ld_bf4(
                        blocks,
                        global_token * blk_sm
                        + fx.min(source, declaration.nblocks - 1) * blk_sr
                        + column,
                    )
                    prefix_values = base.ld_bf4(
                        prefix,
                        global_token * HIDDEN + column,
                    )
                    if const_expr(spec.delta):
                        delta_values = base.ld_bf4(
                            delta,
                            global_token * HIDDEN + column,
                        )
                        prefix_values = [
                            prefix_values[element] + delta_values[element]
                            for element in range(4)
                        ]
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
                        fx.ptr_store(
                            fx.Vector.from_elements(selected, fx.Float32),
                            attention.vals.ptr
                            + (
                                (local_token * sources + source) * SEAM_COLUMNS
                                + column % SEAM_COLUMNS
                            ),
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
                        index = ((bid * s + global_token) * sources + source) * 2
                        gfx_backend.mailbox_store_device(
                            part,
                            index,
                            square_sum,
                            generation,
                        )
                        gfx_backend.mailbox_store_device(
                            part,
                            index + 1,
                            dot,
                            generation,
                        )
                gpu.barrier()
                if const_expr(spec.delta or spec.write_idx >= 0):
                    copy_items = rows * (SEAM_COLUMNS // 4)
                    for iteration in range_constexpr(
                        (copy_items + declaration.threads - 1) // declaration.threads
                    ):
                        work = tid + declaration.threads * iteration
                        if work < copy_items:
                            local_token = work // (SEAM_COLUMNS // 4)
                            local_column = (work % (SEAM_COLUMNS // 4)) * 4
                            global_token = token0 + local_token
                            column = bid * SEAM_COLUMNS + local_column
                            at = (
                                local_token * sources + declaration.nblocks
                            ) * SEAM_COLUMNS + local_column
                            value = [
                                fx.ptr_load(attention.vals.ptr + at + element)
                                for element in range(4)
                            ]
                            packed = base.bf4_words(value)
                            if const_expr(spec.delta):
                                bo.buffer_store(
                                    packed,
                                    rsrc(prefix),
                                    (global_token * HIDDEN + column) // 2,
                                )
                            if const_expr(spec.write_idx >= 0):
                                bo.buffer_store(
                                    packed,
                                    rsrc(blocks),
                                    (
                                        global_token * blk_sm
                                        + spec.write_idx * blk_sr
                                        + column
                                    )
                                    // 2,
                                )
                device["wait_vector_stores"]()
                if bid < rows:
                    gate_context = {
                        "S": rows,
                        "total_S": s,
                        "token0": token0,
                        "NS": sources,
                        "tid": tid,
                        "lane": lane,
                        "wave": wave,
                        "red": smem.red.ptr,
                        "eps": spec.eps,
                        "part": part,
                        "wgt": weights,
                        "tag": generation,
                    }
                    compute["attention_gate_tile"](gate_context, bid)
                    device["wait_vector_stores"]()
                mix_context = {
                    "S": rows,
                    "total_S": s,
                    "token0": token0,
                    "tile": tile,
                    "NS": sources,
                    "tid": tid,
                    "lane": lane,
                    "wave": wave,
                    "matrix_waves": declaration.waves,
                    "matrix_threads": declaration.threads,
                    "red": smem.red.ptr,
                    "vals": attention.vals.ptr,
                    "out_eps": spec.out_eps,
                    "debug": False,
                    "mix_lds_handoff": False,
                    "wgt": weights,
                    "msq": mean_squares,
                    "xrdy": x_ready,
                    "xbuf": x,
                    "tag": generation,
                    "args": {"in_nw": in_nw},
                }
                compute["attention_mix_tile"](mix_context, bid)

        if bid >= ATTENTION_CTAS:
            # Observe every tile before issuing weights: a gfx buffer poll
            # retires behind prior vmcnt loads, so prefetching weights first
            # would serialize every spin iteration behind those loads.
            for tile in range_constexpr(len(tile_starts)):
                gfx_backend.mailbox_poll_device(
                    x_ready,
                    tile * ATTENTION_CTAS + tid % ATTENTION_CTAS,
                    generation,
                )
                device["device_fence_acquire"]()
                gpu.barrier()
            first_task = (bid - ATTENTION_CTAS) * PROJECTION_ROWS_PER_CTA
            accumulators0 = [compute["kda_zero"]() for _ in range(len(tile_starts))]
            accumulators1 = [compute["kda_zero"]() for _ in range(len(tile_starts))]
            projection_context = {
                "S": ROWS,
                "lane": lane,
                "wave": wave,
                "matrix_waves": declaration.waves,
                "gemv_tasks": PROJECTION_TASKS,
                "fused_rows": FUSED_ROWS,
                "xrow": declaration.xrow,
                "xl": projection.xl.ptr,
                "args": {"w_in": w_in},
            }
            for slab in range_constexpr(len(slabs)):
                first_group = slabs[slab][0]
                groups = slabs[slab][1]
                operands0 = compute["kda_gemv_loads"](
                    projection_context,
                    first_task,
                    range(first_group, first_group + groups),
                )
                for tile in range_constexpr(len(tile_starts)):
                    projection_context["S"] = tile_rows[tile]
                    stage_x_tile(
                        x,
                        projection.xl.ptr,
                        tid,
                        tile_starts[tile],
                        tile_rows[tile],
                        first_group,
                        groups,
                    )
                    accumulators0[tile] = compute["kda_gemv_slab"](
                        projection_context,
                        operands0,
                        groups,
                        accumulators0[tile],
                    )
                    if const_expr(tile == 0):
                        operands1 = compute["kda_gemv_loads"](
                            projection_context,
                            first_task + 1,
                            range(first_group, first_group + groups),
                        )
                    accumulators1[tile] = compute["kda_gemv_slab"](
                        projection_context,
                        operands1,
                        groups,
                        accumulators1[tile],
                    )
                    gpu.barrier()
            for task_offset in range_constexpr(PROJECTION_ROWS_PER_CTA):
                task = first_task + task_offset
                if task < PROJECTION_TASKS:
                    accumulators = (
                        accumulators0 if const_expr(task_offset == 0) else accumulators1
                    )
                    for tile in range_constexpr(len(tile_starts)):
                        compute["kda_gemv_store"](
                            accumulators[tile],
                            smem.red.ptr,
                            wave,
                            lane,
                        )
                        gpu.barrier()
                        work_items = tile_rows[tile] * ROWS
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
                                global_token = tile_starts[tile] + local_token
                                gfx_backend.mailbox_store_device(
                                    projected,
                                    global_token * NPROJ + task * ROWS + row,
                                    value,
                                    generation,
                                )
                        gpu.barrier()
            device["wait_vector_stores"]()

        recurrence_context = {
            "S": s,
            "L": qlen,
            "SL": spec.state_len,
            "tid": tid,
            "lane": lane,
            "wave": wave,
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
            "slots": recurrence.slots.ptr,
            "head_onrm_lds": recurrence.onrm.ptr,
            "scratch": scratch,
            "proj_offset": layout["proj"][0],
            "qk_offset": layout["qk"][0],
            "onrm_offset": layout["onrm"][0],
            "tag": generation,
            "on": False,
            "xp": 0,
            "lb": spec.lower_bound,
            "onorm_eps": spec.onorm_eps,
            "args": {
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "core_out": core_out,
            },
        }
        first = _task_start(bid, spec.kda0, declaration.blocks)
        if const_expr(qlen == 1):
            head_tasks = (s // qlen) * HEADS
            for task in range(first, head_tasks, declaration.blocks):
                request = task // HEADS
                head = task % HEADS
                compute["kda_head"](
                    recurrence_context,
                    request,
                    head,
                )
                if const_expr(spec.quant_core):
                    device["wait_vector_stores"]()
                    gpu.barrier()
                    if tid == 0:
                        for chunk in range_constexpr(STATE_CHUNKS):
                            gfx_backend.mailbox_store_device(
                                core_ready,
                                (request * HEADS + head) * STATE_CHUNKS + chunk,
                                fx.Int32(1),
                                generation + fx.Int32(1),
                            )
                    gpu.barrier()
        else:
            for task in range(first, task_count, declaration.blocks):
                request = task // (HEADS * STATE_CHUNKS)
                head = (task // STATE_CHUNKS) % HEADS
                chunk = task % STATE_CHUNKS
                conv_values = conv_metadata(
                    recurrence_context,
                    request,
                    head,
                    chunk,
                )
                compute["kda_conv_inputs"](
                    recurrence_context,
                    request,
                    head,
                    chunk,
                )
                compute["kda_conv_compute"](
                    recurrence_context,
                    conv_values,
                    request,
                    head,
                    chunk,
                )
                recurrent_values = load_state_metadata(
                    recurrence_context,
                    gate_metadata(recurrence_context, head),
                    request,
                    head,
                    chunk,
                )
                compute["kda_gates"](
                    recurrence_context,
                    recurrent_values,
                    request,
                    head,
                )
                compute["kda_recur"](
                    recurrence_context,
                    recurrent_values,
                    request,
                    head,
                    chunk,
                )
                compute["kda_norm"](
                    recurrence_context,
                    request,
                    head,
                    chunk,
                )
                if const_expr(spec.quant_core):
                    device["wait_vector_stores"]()
                    gpu.barrier()
                    if tid == 0:
                        gfx_backend.mailbox_store_device(
                            core_ready,
                            (request * HEADS + head) * STATE_CHUNKS + chunk,
                            fx.Int32(1),
                            generation + fx.Int32(1),
                        )
                    gpu.barrier()
        device["wait_vector_stores"]()

        if const_expr(spec.quant_core):  # noqa: SIM102
            if (bid >= ATTENTION_CTAS) & (bid < ATTENTION_CTAS + s):
                token = bid - ATTENTION_CTAS
                request = token // qlen
                if tid < HEADS * STATE_CHUNKS:
                    gfx_backend.mailbox_poll_device(
                        core_ready,
                        request * HEADS * STATE_CHUNKS + tid,
                        generation + fx.Int32(1),
                    )
                gpu.barrier()
                device["device_fence_acquire"]()
                quantize_core(
                    token,
                    tid,
                    lane,
                    wave,
                    smem.red.ptr,
                    core_out,
                    core_q,
                )

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
        core_q: Int64,
        scratch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        with CompilationContext.compile_hints({"waves_per_eu": 2}):
            call = resident(
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
                core_q,
                scratch,
            )
            call.launch(
                grid=(declaration.blocks,),
                block=(declaration.threads,),
                stream=stream,
            )

    LOCAL_ABI.check(resident, launch)
    entry = ResidentK1EntryPoint(
        name,
        launch,
        (declaration.blocks,),
        (declaration.threads,),
    )
    _BUILDS[cache_key] = entry
    return entry
