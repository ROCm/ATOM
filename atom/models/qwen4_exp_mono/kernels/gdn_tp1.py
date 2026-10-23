# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP1 GDN schedule composed into the single-launch layer kernel.

Projection, convolution, normalization and recurrence math are shared stages;
this module owns TP1 task placement, hand-offs and output reduction.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.layout import (
    CONV_TASKS,
    GDR_A_ROW,
    GDR_B_ROW,
    GDR_FRONT_ROUNDS,
    GDR_FULL_HEAD_READY_OFFSET,
    GDR_FULL_NORM_READY_OFFSET,
    GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
    GDR_FULL_OUTPUT_READY_OFFSET,
    GDR_FULL_PARTIAL_READY_OFFSET,
    GDR_FULL_PROJECTION_READY_OFFSET,
    GDR_FULL_RECURRENCE_READY_OFFSET,
    GDR_GATE_TASKS,
    GDR_NORM_HEADS_PER_TASK,
    GDR_NORM_TASKS,
    GDR_OUT_DIRECT_TASKS,
    GDR_OUT_EARLY_OWNER,
    GDR_OUT_EARLY_TASKS,
    GDR_OUT_FULL_K_STEPS_PER_WAVE,
    GDR_OUT_K_STEPS_PER_WAVE,
    GDR_OUT_LATE_OWNER,
    GDR_OUT_SPLIT_K,
    GDR_OUT_SPLIT_TILES,
    GDR_OUT_TASKS,
    GDR_TASKS,
    GDR_V_SPLITS,
    GDR_Z_ROUNDS,
    GDR_Z_TASKS,
    GDR_Z_WORKERS,
    ROWS,
)
from atom.models.qwen4_exp_mono.shapes import (
    GDN_CONV_DIM,
    GDN_HEAD_DIM,
    GDN_KEY_DIM,
    GDN_KEY_HEADS,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HIDDEN,
)
from atom.models.qwen4_exp_mono.stages.gdn_common import (
    projection_conv_store,
    recurrent_views,
    sigmoid_norm_task,
)
from atom.models.qwen4_exp_mono.stages.memory import load_f32, workspace
from atom.models.qwen4_exp_mono.stages.projection import (
    output_weights,
    projection_accumulator,
)
from atom.mono.device.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish, sreg
from atom.mono.plan.execution import BLOCKS, WAVES


@fx.struct
class GdnSmem:
    activation: fx.Array[fx.Int32, GDN_VALUE_DIM // 2, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


@traced
def load_activation(x, tid, activation):
    """Stage the one BF16 activation row once per resident CTA."""

    if tid < HIDDEN // 8:
        at = tid * 8
        value = fx.Vector(bo.buffer_load(rsrc(x), at // 2, vec_width=4, dtype=T.i32))
        fx.ptr_store(value, activation + at // 2)


@traced
def projection_conv_task(
    weight,
    conv_weight,
    conv_state,
    read_slot,
    write_slot,
    out,
    task,
    tid,
    lane,
    wave,
    activation,
    reduction,
):
    """Project one N16 tile and fold conv into the Q/K/V prefix in place."""

    acc = projection_accumulator(weight, task, lane, wave, activation)
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if task < CONV_TASKS:
        projection_conv_store(
            reduction,
            task,
            tid,
            out,
            conv_weight,
            conv_state,
            read_slot,
            write_slot,
            GDN_CONV_DIM,
        )
    else:
        if tid < ROWS // 2:
            row0 = 2 * tid
            value = bf16_pair(
                row_sum(reduction, row0, 0),
                row_sum(reduction, row0 + 1, 0),
            )
            bo.buffer_store(
                value.bitcast(fx.Int32),
                rsrc(out),
                task * (ROWS // 2) + tid,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()


def output_projection_loads(weight, tile, k_start, k_steps, lane, wave):
    return output_weights(
        weight, tile, k_start, k_steps, lane, wave, width=GDN_VALUE_DIM
    )


def output_projection_accumulator(activation, lane, wave, weight_ops):
    """Consume the staged K range in the original eight-wave MFMA order."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(len(weight_ops)):
        k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // 16)
        x = fx.Vector(
            fx.ptr_load(activation + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(weight_ops[k_pass], x, acc)
    return acc


@traced
def stage_output_activation(normed, k_start, k_size, activation, tid):
    """Reuse input LDS for the normalized BF16 output-projection K range."""
    for part in range_constexpr((k_size + 4095) // 4096):
        at = (part * 512 + tid) * 8
        if at < k_size:
            x = fx.Vector(
                bo.buffer_load(
                    rsrc(normed),
                    (k_start + at) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            fx.ptr_store(x, activation + at // 2)
    gpu.barrier()


@traced
def output_projection_direct_task(
    activation,
    out,
    tile,
    tid,
    lane,
    wave,
    reduction,
    weight_ops,
    after_pair,
):
    """Complete one N16 tile without a global partial hand-off."""

    acc = output_projection_accumulator(activation, lane, wave, weight_ops)
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        after_pair(tile, row0, value0, value1)
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out),
            tile * (ROWS // 2) + tid,
            cache_modifier=CM_DEV,
        )


@traced
def output_projection_partial_task(
    activation,
    partials,
    partial_ready,
    put,
    task,
    tid,
    lane,
    wave,
    reduction,
    weight_ops,
):
    """Produce one N16 x K3072 FP32 partial and publish its completion."""

    acc = output_projection_accumulator(activation, lane, wave, weight_ops)
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        bo.buffer_store(
            row_sum(reduction, tid, 0),
            rsrc(partials),
            task * ROWS + tid,
            cache_modifier=CM_DEV,
        )
    publish(put, partial_ready, task, fx.Int32(1), tid == 0)


@traced
def output_projection_finish(
    partials,
    partial_ready,
    poll,
    out,
    tile,
    tid,
    after_pair,
):
    """Merge the two contiguous-K partials in split order and round to BF16."""

    partial = tile - GDR_OUT_DIRECT_TASKS
    if tid == 0:
        poll(
            [
                (partial_ready, partial, 1),
                (partial_ready, GDR_OUT_SPLIT_TILES + partial, 1),
            ]
        )
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        index0 = partial * ROWS + row0
        index1 = index0 + 1
        split1_base = GDR_OUT_SPLIT_TILES * ROWS
        value0 = bf16_round(
            load_f32(partials, index0) + load_f32(partials, split1_base + index0)
        )
        value1 = bf16_round(
            load_f32(partials, index1) + load_f32(partials, split1_base + index1)
        )
        after_pair(tile, row0, value0, value1)
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out),
            tile * (ROWS // 2) + tid,
            cache_modifier=CM_DEV,
        )


@traced
def _wait_gdr_norms(poll, norm_ready, tid):
    """Hold output-projection CTAs until all eight-head norm tasks finish."""

    if tid == 0:
        poll([(norm_ready, task, 1) for task in range_constexpr(GDR_NORM_TASKS)])
    gpu.barrier()


def _gdr_projection_views(
    projected, dt_bias, A_log, read_indices, write_indices, state, out
):
    return recurrent_views(
        projected, dt_bias, A_log, read_indices, write_indices, state, out, Dims(1)
    )


def _gdr_dependencies(task):
    """Projection N16 tiles required by one value-head recurrence task."""

    head = task // GDR_V_SPLITS
    key_head = head // (GDN_VALUE_HEADS // GDN_KEY_HEADS)
    q0 = key_head * (GDN_HEAD_DIM // ROWS)
    k0 = GDN_KEY_DIM // ROWS + q0
    v0 = 2 * GDN_KEY_DIM // ROWS + head * (GDN_HEAD_DIM // ROWS)
    deps = []
    for offset in range_constexpr(GDN_HEAD_DIM // ROWS):
        deps.extend((q0 + offset, k0 + offset, v0 + offset))
    deps.extend(
        (
            GDR_B_ROW // ROWS + head // ROWS,
            GDR_A_ROW // ROWS + head // ROWS,
        )
    )
    return deps


def _gdr_head_dependencies(head, recurrence_ready, projected_ready):
    """Mailbox inputs required before one complete GDR head can be consumed."""

    specs = [
        (recurrence_ready, head * GDR_V_SPLITS + split, 1)
        for split in range_constexpr(GDR_V_SPLITS)
    ]
    z0 = CONV_TASKS + head * (GDN_HEAD_DIM // ROWS)
    specs.extend(
        (projected_ready, z0 + offset, 1)
        for offset in range_constexpr(GDN_HEAD_DIM // ROWS)
    )
    return specs


@traced
def _wait_gdr_dependencies(poll, projected_ready, task, tid):
    """Wave leaders join the head's disjoint projection tiles."""
    dependencies = _gdr_dependencies(task)
    if tid % 64 == 0:
        for owner in range_constexpr(WAVES):
            if tid // 64 == owner:
                poll([(projected_ready, dep, 1) for dep in dependencies[owner::WAVES]])
    gpu.barrier()


@traced
def _publish_gdr_head_ready(
    poll,
    put,
    recurrence_ready,
    projected_ready,
    head_ready,
    head,
    tid,
):
    """Collapse two recurrence and eight gate tiles into one head-ready tag."""

    if tid == 0:
        poll(_gdr_head_dependencies(head, recurrence_ready, projected_ready))
    gpu.barrier()
    publish(put, head_ready, head, fx.Int32(1), tid == 0)


@traced
def _wait_gdr_norm_heads(poll, head_ready, task, tid):
    """Wait for the eight complete heads consumed by one sigmoid-norm task."""

    head0 = task * GDR_NORM_HEADS_PER_TASK
    if tid == 0:
        poll(
            [
                (head_ready, head0 + offset, 1)
                for offset in range_constexpr(GDR_NORM_HEADS_PER_TASK)
            ]
        )
    gpu.barrier()


@traced
def _publish_gdr_output_groups(poll, put, output_ready, output_group_ready, bid, tid):
    """Join one output group across wave leaders before its publication."""
    for owner, first, count, group in (
        (GDR_OUT_EARLY_OWNER, 0, GDR_OUT_EARLY_TASKS, 0),
        (
            GDR_OUT_LATE_OWNER,
            GDR_OUT_EARLY_TASKS,
            GDR_OUT_TASKS - GDR_OUT_EARLY_TASKS,
            1,
        ),
    ):
        if bid == owner:
            if tid % 64 == 0:
                for wave in range_constexpr(WAVES):
                    if tid // 64 == wave:
                        poll(
                            [
                                (output_ready, first + tile, 1)
                                for tile in range(wave, count, WAVES)
                            ]
                        )
            gpu.barrier()
            publish(put, output_group_ready, group, fx.Int32(1), tid == 0)


def gdn_bindings(a, scratch, layout):
    """Bind the S1 recurrence directly from the complete layer contract."""
    return {
        "x": a.gdn_input,
        "projection_weight": a.gdn_qkvzba,
        "conv_weight": a.gdn_conv,
        "conv_state": a.gdn_conv_state,
        "read_indices": a.gdn_state_read,
        "write_indices": a.gdn_state_write,
        "A_log": a.gdn_a_log,
        "dt_bias": a.gdn_dt_bias,
        "state": a.gdn_recurrent_state,
        "norm_weight": a.gdn_norm,
        "output_weight": a.gdn_out,
        **{
            name: workspace(scratch, layout, "gdn_" + name)
            for name in ("projected", "core", "normed", "out", "partials")
        },
    }


@traced
def run_gdn(
    emit_gdr_decode_task,
    lds,
    mailbox,
    ready,
    args,
    bid,
    after_output_pair,
):
    """Emit the accepted S=1 GDN dependency graph into a caller's kernel.

    Every BF16 output tile is published at device scope. Two owners
    collapse the first 96 and final 64 tiles into staged HC hand-offs.
    """

    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64

    read_slot = fx.Int32(
        bo.buffer_load(rsrc(args["read_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot = fx.Int32(
        bo.buffer_load(rsrc(args["write_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    activation = lds.activation.ptr
    reduction = lds.reduction.ptr
    regions = {
        name: sreg(ready, offset, "gdn_full_" + label)
        for name, offset, label in (
            ("projected", GDR_FULL_PROJECTION_READY_OFFSET, "projection_ready"),
            ("recurrence", GDR_FULL_RECURRENCE_READY_OFFSET, "recurrence_ready"),
            ("head", GDR_FULL_HEAD_READY_OFFSET, "head_ready"),
            ("norm", GDR_FULL_NORM_READY_OFFSET, "norm_ready"),
            ("partial", GDR_FULL_PARTIAL_READY_OFFSET, "output_partial_ready"),
            ("output", GDR_FULL_OUTPUT_READY_OFFSET, "output_ready"),
            ("output_group", GDR_FULL_OUTPUT_GROUP_READY_OFFSET, "output_group_ready"),
        )
    }
    projected_ready, recurrence_ready, head_ready, norm_ready = (
        regions[name] for name in ("projected", "recurrence", "head", "norm")
    )
    partial_ready, output_ready, output_group_ready = (
        regions[name] for name in ("partial", "output", "output_group")
    )

    load_activation(args["x"], tid, activation)
    gpu.barrier()

    def produce(projection_task_id):
        projection_conv_task(
            args["projection_weight"],
            args["conv_weight"],
            args["conv_state"],
            read_slot,
            write_slot,
            args["projected"],
            projection_task_id,
            tid,
            lane,
            wave,
            activation,
            reduction,
        )
        publish(
            mailbox.put,
            projected_ready,
            projection_task_id,
            fx.Int32(1),
            tid == 0,
        )

    # b/a gate rows first, then the convolutional Q/K/V prefix. A recurrence
    # task starts as soon as its own tiles are visible.
    if bid < GDR_GATE_TASKS:
        produce(GDR_B_ROW // ROWS + bid)
    for round_id in range_constexpr(GDR_FRONT_ROUNDS):
        projection_task = bid + BLOCKS * round_id
        if projection_task < CONV_TASKS:
            produce(projection_task)

    recurrence_task = bid
    if recurrence_task < GDR_TASKS:
        _wait_gdr_dependencies(
            mailbox.poll,
            projected_ready,
            recurrence_task,
            tid,
        )
        emit_gdr_decode_task(
            *_gdr_projection_views(
                args["projected"],
                args["dt_bias"],
                args["A_log"],
                args["read_indices"],
                args["write_indices"],
                args["state"],
                args["core"],
            ),
            fx.Int32(1),
            tid,
            recurrence_task,
        )
        publish(
            mailbox.put,
            recurrence_ready,
            recurrence_task,
            fx.Int32(1),
            tid == 0,
        )
    else:
        worker = bid - GDR_TASKS
        for round_id in range_constexpr(GDR_Z_ROUNDS):
            projection_task = CONV_TASKS + worker + GDR_Z_WORKERS * round_id
            if projection_task < CONV_TASKS + GDR_Z_TASKS:
                produce(projection_task)

    # One resident owner per head replaces every norm CTA polling the two
    # recurrence and eight z producers directly.
    if bid < GDN_VALUE_HEADS:
        _publish_gdr_head_ready(
            mailbox.poll,
            mailbox.put,
            recurrence_ready,
            projected_ready,
            head_ready,
            bid,
            tid,
        )

    if bid < GDR_NORM_TASKS:
        _wait_gdr_norm_heads(mailbox.poll, head_ready, bid, tid)
        sigmoid_norm_task(
            args["core"],
            args["projected"] + fx.Int64(2 * GDN_CONV_DIM),
            args["norm_weight"],
            args["normed"],
            bid,
            lane,
            wave,
        )
        publish(mailbox.put, norm_ready, bid, fx.Int32(1), tid == 0)

    # V4.1-style output scheduling: every CTA puts its cold weight batch in
    # flight before waiting for activations. The first 64 tiles are direct;
    # the remaining 96 use two contiguous K halves.
    if bid < GDR_OUT_DIRECT_TASKS:
        weight_ops = output_projection_loads(
            args["output_weight"],
            bid,
            0,
            GDR_OUT_FULL_K_STEPS_PER_WAVE,
            lane,
            wave,
        )
        _wait_gdr_norms(mailbox.poll, norm_ready, tid)
        stage_output_activation(args["normed"], 0, GDN_VALUE_DIM, activation, tid)
        output_projection_direct_task(
            activation,
            args["out"],
            bid,
            tid,
            lane,
            wave,
            reduction,
            weight_ops,
            after_output_pair,
        )
        publish(mailbox.put, output_ready, bid, fx.Int32(1), tid == 0)
    else:
        partial_task = bid - GDR_OUT_DIRECT_TASKS
        split = partial_task // GDR_OUT_SPLIT_TILES
        tile = GDR_OUT_DIRECT_TASKS + partial_task % GDR_OUT_SPLIT_TILES
        weight_ops = output_projection_loads(
            args["output_weight"],
            tile,
            split * GDR_OUT_SPLIT_K,
            GDR_OUT_K_STEPS_PER_WAVE,
            lane,
            wave,
        )
        _wait_gdr_norms(mailbox.poll, norm_ready, tid)
        stage_output_activation(
            args["normed"], split * GDR_OUT_SPLIT_K, GDR_OUT_SPLIT_K, activation, tid
        )
        output_projection_partial_task(
            activation,
            args["partials"],
            partial_ready,
            mailbox.put,
            partial_task,
            tid,
            lane,
            wave,
            reduction,
            weight_ops,
        )
        if bid < GDR_OUT_TASKS:
            output_projection_finish(
                args["partials"],
                partial_ready,
                mailbox.poll,
                args["out"],
                bid,
                tid,
                after_output_pair,
            )
            publish(mailbox.put, output_ready, bid, fx.Int32(1), tid == 0)

    _publish_gdr_output_groups(
        mailbox.poll,
        mailbox.put,
        output_ready,
        output_group_ready,
        bid,
        tid,
    )
