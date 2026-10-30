# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8 GDN K1/K2 specialization for Qwen3.8-Flash-Next.

K1 computes one rank's six-head GDN shard and publishes its BF16 output through
the symmetric ``attention`` peer region. K2 pulls the eight rank partials in a
fixed order, then runs the accepted TP8 HC+MoE finish and publishes the next
layer's MoE partial. The TP1 GDN emitter stays untouched so this specialization
cannot perturb its ISA.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen4_exp_mono.abi import (
    StructuredAbi,
    argument_values,
)
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels.finish import FinishArgs, build_finish_kernel
from atom.models.qwen4_exp_mono.layout import (
    align,
    finish_scratch_bytes,
)
from atom.models.qwen4_exp_mono.peer_layout import PARTS, peer_offsets
from atom.models.qwen4_exp_mono.plan import MonoBuild
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.collective import (
    publish_pair,
    pull_part_to_bf16,
)
from atom.models.qwen4_exp_mono.stages.gdn_common import (
    norm_head_pair,
    projection_conv_store,
    recurrent_views,
)
from atom.models.qwen4_exp_mono.stages.gdr_decode import create_qwen_gdr_decode
from atom.models.qwen4_exp_mono.stages.hc import (
    _ld_bf,
    _sigmoid,
    _st_bf,
    poll_ready_tiles,
)
from atom.models.qwen4_exp_mono.stages.projection import (
    output_accumulator,
    output_weights,
    projection_accumulator,
)
from atom.mono.device.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    kernel_symbol,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import Mailbox, publish, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import (
    BLOCKS,
    THREADS,
    WAVES,
)


@fx.struct
class GdnTp8K1Args:
    x: Int64
    projection_weight: Int64
    conv_weight: Int64
    conv_state: Int64
    state_read: Int64
    state_write: Int64
    a_log: Int64
    dt_bias: Int64
    state: Int64
    norm_weight: Int64
    output_weight: Int64
    partial_out: Int64
    peer_sym: Int64


K1_ABI = StructuredAbi(GdnTp8K1Args)


@fx.struct
class GdnTp8K2Args:
    __annotations__ = {"peers": Int64, **FinishArgs.__annotations__}


K2_ABI = StructuredAbi(GdnTp8K2Args)

_DIMS = Dims(8)
HIDDEN = _DIMS.hidden
KEY_HEADS = _DIMS.gdn_key_heads
VALUE_HEADS = _DIMS.gdn_value_heads
HEAD_DIM = 128
KEY_DIM = KEY_HEADS * HEAD_DIM
VALUE_DIM = VALUE_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
PROJ_DIM = CONV_DIM + VALUE_DIM + 2 * VALUE_HEADS

ROWS = 16
K_STEP = 32
K_STEPS_PER_WAVE = HIDDEN // (K_STEP * WAVES)
OUT_K_STEPS_PER_WAVE = VALUE_DIM // (K_STEP * WAVES)
PROJECTION_TASKS = (PROJ_DIM + ROWS - 1) // ROWS
CONV_TASKS = CONV_DIM // ROWS
Z_TASKS = VALUE_DIM // ROWS
V_SPLITS = 2
RECURRENCE_TASKS = VALUE_HEADS * V_SPLITS
OUTPUT_TASKS = HIDDEN // ROWS
WEIGHT_BYTES = PROJ_DIM * HIDDEN * 2
OUTPUT_WEIGHT_BYTES = HIDDEN * VALUE_DIM * 2

PAIR_BYTES = 8


K1_BASE = align(finish_scratch_bytes())
PROJECTION_READY_OFFSET = K1_BASE
RECURRENCE_READY_OFFSET = PROJECTION_READY_OFFSET + PAIR_BYTES * PROJECTION_TASKS
HEAD_READY_OFFSET = RECURRENCE_READY_OFFSET + PAIR_BYTES * RECURRENCE_TASKS
NORM_READY_OFFSET = HEAD_READY_OFFSET + PAIR_BYTES * VALUE_HEADS
READY_END = NORM_READY_OFFSET + PAIR_BYTES
PROJECTED_OFFSET = align(READY_END)
CORE_OFFSET = align(PROJECTED_OFFSET + PROJ_DIM * 2)
NORMED_OFFSET = align(CORE_OFFSET + VALUE_DIM * 2)
K1_END = align(NORMED_OFFSET + VALUE_DIM * 2)
K2_PULL_READY_OFFSET = align(K1_END, PAIR_BYTES)
K2_ALL_READY_OFFSET = K2_PULL_READY_OFFSET + PARTS * PAIR_BYTES

assert BLOCKS == 256 and THREADS == 512 and WAVES == 8
assert (KEY_DIM, VALUE_DIM, CONV_DIM, PROJ_DIM) == (256, 768, 1280, 2060)
assert (PROJECTION_TASKS, CONV_TASKS, Z_TASKS) == (129, 80, 48)
assert (RECURRENCE_TASKS, OUTPUT_TASKS) == (12, 160)
assert K_STEPS_PER_WAVE * K_STEP * WAVES == HIDDEN
assert OUT_K_STEPS_PER_WAVE * K_STEP * WAVES == VALUE_DIM


@fx.struct
class GdnSmem:
    activation: fx.Array[fx.Int32, HIDDEN // 2, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


def _projection_views(
    projected, dt_bias, a_log, read_indices, write_indices, state, core
):
    return recurrent_views(
        projected, dt_bias, a_log, read_indices, write_indices, state, core, _DIMS
    )


@traced
def _project_task(
    weight,
    conv_weight,
    conv_state,
    read_slot,
    write_slot,
    projected,
    task,
    tid,
    lane,
    wave,
    activation,
    reduction,
):
    acc = projection_accumulator(
        weight, task, lane, wave, activation, nbytes=WEIGHT_BYTES, rows=PROJ_DIM
    )
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()

    if task < CONV_TASKS:
        projection_conv_store(
            reduction,
            task,
            tid,
            projected,
            conv_weight,
            conv_state,
            read_slot,
            write_slot,
            CONV_DIM,
        )
    elif tid < ROWS // 2:
        row0 = task * ROWS + 2 * tid
        value = bf16_pair(
            bf16_round(row_sum(reduction, 2 * tid, 0)),
            bf16_round(row_sum(reduction, 2 * tid + 1, 0)),
        )
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(projected, nbytes=PROJ_DIM * 2),
            task * (ROWS // 2) + tid,
            mask=row0 + 1 < PROJ_DIM,
            cache_modifier=CM_DEV,
        )
    gpu.barrier()


def _projection_dependencies(task):
    head = task // V_SPLITS
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    q0 = key_head * (HEAD_DIM // ROWS)
    k0 = KEY_DIM // ROWS + q0
    v0 = 2 * KEY_DIM // ROWS + head * (HEAD_DIM // ROWS)
    deps = []
    for offset in range_constexpr(HEAD_DIM // ROWS):
        deps.extend((q0 + offset, k0 + offset, v0 + offset))
    deps.append(PROJECTION_TASKS - 1)
    return deps


@traced
def _wait_projection_dependencies(poll, ready, task, tid):
    deps = _projection_dependencies(task)
    if tid % 64 == 0:
        for owner in range_constexpr(WAVES):
            if tid // 64 == owner:
                poll([(ready, dep, 1) for dep in deps[owner::WAVES]])
    gpu.barrier()


@traced
def _wait_head_dependencies(poll, recurrence_ready, projection_ready, head, tid):
    if tid == 0:
        specs = [
            (recurrence_ready, head * V_SPLITS + split, 1)
            for split in range_constexpr(V_SPLITS)
        ]
        z0 = CONV_TASKS + head * (HEAD_DIM // ROWS)
        specs.extend(
            (projection_ready, z0 + offset, 1)
            for offset in range_constexpr(HEAD_DIM // ROWS)
        )
        poll(specs)
    gpu.barrier()


@traced
def _wait_all_heads(poll, head_ready, tid):
    if tid == 0:
        poll([(head_ready, head, 1) for head in range_constexpr(VALUE_HEADS)])
    gpu.barrier()


@traced
def _wait_norm(poll, norm_ready, tid):
    if tid == 0:
        poll([(norm_ready, 0, 1)])
    gpu.barrier()


@traced
def _norm_heads(core, projected, norm_weight, normed, wave, lane):
    if wave < VALUE_HEADS:
        index0, index1, affine0, affine1 = norm_head_pair(core, norm_weight, wave, lane)
        _st_bf(
            normed,
            index0,
            affine0 * _sigmoid(_ld_bf(projected, CONV_DIM + index0)),
            CM_DEV,
        )
        _st_bf(
            normed,
            index1,
            affine1 * _sigmoid(_ld_bf(projected, CONV_DIM + index1)),
            CM_DEV,
        )


def _output_weights(weight, tile, lane, wave):
    return output_weights(
        weight,
        tile,
        0,
        OUT_K_STEPS_PER_WAVE,
        lane,
        wave,
        width=VALUE_DIM,
        nbytes=OUTPUT_WEIGHT_BYTES,
    )


@traced
def _output_task(
    normed,
    out,
    tile,
    tid,
    lane,
    wave,
    reduction,
    weight_ops,
    mailbox,
    peer_sym,
    peer_region_off,
    layer,
    tokens=1,
    token=0,
):
    acc = output_accumulator(
        normed, 0, lane, wave, weight_ops, cache_modifier=CM_DEV, nbytes=VALUE_DIM * 2
    )
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        value0 = bf16_round(row_sum(reduction, 2 * tid, 0))
        value1 = bf16_round(row_sum(reduction, 2 * tid + 1, 0))
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out, nbytes=HIDDEN * 2),
            tile * (ROWS // 2) + tid,
            cache_modifier=CM_DEV,
        )
        publish_pair(
            mailbox,
            peer_sym,
            peer_region_off,
            "attention",
            layer,
            tokens,
            token,
            tile * ROWS + 2 * tid,
            value0.to(fx.BFloat16),
            value1.to(fx.BFloat16),
        )


@traced
def _run_full_graph(
    emit_gdr_decode_task,
    lds,
    mailbox,
    peer_mailbox,
    scratch,
    args,
    bid,
):
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    projected = scratch + fx.Int64(PROJECTED_OFFSET)
    core = scratch + fx.Int64(CORE_OFFSET)
    normed = scratch + fx.Int64(NORMED_OFFSET)
    projection_ready = sreg(
        scratch, PROJECTION_READY_OFFSET, "tp8_gdn_projection_ready"
    )
    recurrence_ready = sreg(
        scratch, RECURRENCE_READY_OFFSET, "tp8_gdn_recurrence_ready"
    )
    head_ready = sreg(scratch, HEAD_READY_OFFSET, "tp8_gdn_head_ready")
    norm_ready = sreg(scratch, NORM_READY_OFFSET, "tp8_gdn_norm_ready")
    read_slot = fx.Int32(
        bo.buffer_load(rsrc(args["state_read"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot = fx.Int32(
        bo.buffer_load(rsrc(args["state_write"]), 0, vec_width=1, dtype=T.i32)
    )

    if tid < HIDDEN // 8:
        at = tid * 8
        value = fx.Vector(
            bo.buffer_load(
                rsrc(args["x"], nbytes=HIDDEN * 2),
                at // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        fx.ptr_store(value, lds.activation.ptr + at // 2)
    gpu.barrier()

    if bid < PROJECTION_TASKS:
        _project_task(
            args["projection_weight"],
            args["conv_weight"],
            args["conv_state"],
            read_slot,
            write_slot,
            projected,
            bid,
            tid,
            lane,
            wave,
            lds.activation.ptr,
            lds.reduction.ptr,
        )
        publish(
            mailbox.put,
            projection_ready,
            bid,
            fx.Int32(1),
            tid == 0,
        )

    def run_recurrence():
        _wait_projection_dependencies(mailbox.poll, projection_ready, bid, tid)
        emit_gdr_decode_task(
            *_projection_views(
                projected,
                args["dt_bias"],
                args["a_log"],
                args["state_read"],
                args["state_write"],
                args["state"],
                core,
            ),
            fx.Int32(1),
            tid,
            bid,
        )
        publish(
            mailbox.put,
            recurrence_ready,
            bid,
            fx.Int32(1),
            tid == 0,
        )

    if bid < RECURRENCE_TASKS:
        run_recurrence()

    if bid < VALUE_HEADS:
        _wait_head_dependencies(
            mailbox.poll, recurrence_ready, projection_ready, bid, tid
        )
        publish(mailbox.put, head_ready, bid, fx.Int32(1), tid == 0)

    if bid == 0:
        _wait_all_heads(mailbox.poll, head_ready, tid)
        _norm_heads(core, projected, args["norm_weight"], normed, wave, lane)
        publish(mailbox.put, norm_ready, 0, fx.Int32(1), tid == 0)

    if bid < OUTPUT_TASKS:
        weight_ops = _output_weights(args["output_weight"], bid, lane, wave)
        _wait_norm(mailbox.poll, norm_ready, tid)
        _output_task(
            normed,
            args["out"],
            bid,
            tid,
            lane,
            wave,
            lds.reduction.ptr,
            weight_ops,
            peer_mailbox,
            args["peer_sym"],
            args["peer_region_off"],
            args["layer"],
            1,
            0,
        )


def build_gdn_k1(key: MonoBuild):
    """Build rank-local TP8 GDN and publish its attention partial."""

    if key.tokens != 1 or key.tp != 8:
        raise ValueError("Qwen TP8 GDN K1 supports only S=1 TP8")
    keyed = key_tuple(key, SOURCES)
    peer_region_off = peer_offsets(key.tokens, 8)["attention"]

    emit_gdr_decode_task = create_qwen_gdr_decode(_DIMS)

    name = kernel_symbol("qwen_tp8_gdn_k1", s=key.tokens)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(a: GdnTp8K1Args, scratch: Int64, layer: Int32):
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(GdnSmem).peek()
        mailbox = Mailbox(layer)
        args = {
            **argument_values(a),
            "out": a.partial_out,
            "peer_region_off": peer_region_off,
            "layer": layer,
        }
        _run_full_graph(emit_gdr_decode_task, lds, mailbox, mailbox, scratch, args, bid)
        _ = keyed

    launch = K1_ABI.launcher(kernel, name, keyed)
    _ = keyed
    return launch


@traced
def _run_attention_pull(c, bid):
    """Pull each N32 attention part, then expose one local all-ready tag."""

    pull_ready = sreg(c["scratch"], K2_PULL_READY_OFFSET, "tp8_attention_pull_ready")
    all_ready = sreg(c["scratch"], K2_ALL_READY_OFFSET, "tp8_attention_all_ready")
    if bid < PARTS:
        pull_part_to_bf16(
            c["mailbox"],
            c["peers"],
            c["args"]["attn_out"],
            c["attention_region_off"],
            "attention",
            c["layer"],
            1,
            0,
            bid,
            8,
        )
        publish(c["put"], pull_ready, bid, fx.Int32(1), c["tid"] == 0)
    if bid == PARTS - 1:
        poll_ready_tiles(c, pull_ready, 0, PARTS)
        publish(c["put"], all_ready, 0, fx.Int32(1), c["tid"] == 0)
    c["attn_ready"] = all_ready


def build_gdn_k2(key: MonoBuild):
    """Build TP8 attention pull followed by the common HC+MoE finish."""

    if key.tokens != 1 or key.tp != 8:
        raise ValueError("Qwen TP8 GDN K2 supports only S=1 TP8")
    attention_region_off = peer_offsets(1, 8)["attention"]
    name = kernel_symbol("qwen_tp8_gdn_k2", s=1)

    def before_finish(c, a, layer, bid):
        c.update(
            peers=a.peers, attention_region_off=attention_region_off, attn_cm=CM_DEV
        )
        _run_attention_pull(c, bid)

    return build_finish_kernel(key, K2_ABI, name, before_finish)
