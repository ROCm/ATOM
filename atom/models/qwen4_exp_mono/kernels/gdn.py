# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gated-DeltaNet body composed into the Qwen S=1 persistent layer kernel."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from .gdr_decode import create_vk_gdr_decode_kernel
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.config import (
    GDN_CONV_DIM,
    GDN_CONV_WIDTH,
    GDN_HEAD_DIM,
    GDN_KEY_DIM,
    GDN_KEY_HEADS,
    GDN_PROJ_DIM,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HIDDEN,
)
from atom.mono.device.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    hw_exp2,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    wave_sum,
)
from atom.mono.device.sync import publish, sreg
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES

ROWS = 16
K_STEP = 32
K_STEPS_PER_WAVE = HIDDEN // (K_STEP * WAVES)
PROJECTION_TASKS = GDN_PROJ_DIM // ROWS
CONV_TASKS = GDN_CONV_DIM // ROWS
GDR_V_SPLITS = 2
GDR_TASKS = GDN_VALUE_HEADS * GDR_V_SPLITS
GDR_B_ROW = GDN_CONV_DIM + GDN_VALUE_DIM
GDR_A_ROW = GDR_B_ROW + GDN_VALUE_HEADS
GDR_GATE_TASKS = 2 * GDN_VALUE_HEADS // ROWS
GDR_FRONT_ROUNDS = (CONV_TASKS + BLOCKS - 1) // BLOCKS
GDR_Z_TASKS = GDN_VALUE_DIM // ROWS
GDR_Z_WORKERS = BLOCKS - GDR_TASKS
GDR_Z_ROUNDS = (GDR_Z_TASKS + GDR_Z_WORKERS - 1) // GDR_Z_WORKERS
GDR_NORM_HEADS_PER_TASK = WAVES
GDR_NORM_TASKS = GDN_VALUE_HEADS // GDR_NORM_HEADS_PER_TASK
GDR_OUT_TASKS = HIDDEN // ROWS
GDR_OUT_SPLITS = 2
GDR_OUT_SPLIT_K = GDN_VALUE_DIM // GDR_OUT_SPLITS
GDR_OUT_DIRECT_TASKS = 2 * GDR_OUT_TASKS - BLOCKS
GDR_OUT_SPLIT_TILES = GDR_OUT_TASKS - GDR_OUT_DIRECT_TASKS
GDR_OUT_PARTIAL_TASKS = GDR_OUT_SPLIT_TILES * GDR_OUT_SPLITS
GDR_OUT_K_STEPS_PER_WAVE = GDR_OUT_SPLIT_K // (K_STEP * WAVES)
GDR_OUT_FULL_K_STEPS_PER_WAVE = GDN_VALUE_DIM // (K_STEP * WAVES)
GDR_OUT_EARLY_TASKS = 96
GDR_OUT_READY_GROUPS = 2
GDR_OUT_EARLY_OWNER = GDR_OUT_TASKS + GDR_OUT_EARLY_TASKS - GDR_OUT_DIRECT_TASKS - 1
GDR_OUT_LATE_OWNER = BLOCKS - 1
S4_OUTPUT_PREFETCH_STEPS = 4
MAILBOX_PAIR_BYTES = 8
GDR_FULL_PROJECTION_READY_OFFSET = 0
GDR_FULL_RECURRENCE_READY_OFFSET = (
    GDR_FULL_PROJECTION_READY_OFFSET + MAILBOX_PAIR_BYTES * PROJECTION_TASKS
)
GDR_FULL_HEAD_READY_OFFSET = (
    GDR_FULL_RECURRENCE_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_TASKS
)
GDR_FULL_NORM_READY_OFFSET = (
    GDR_FULL_HEAD_READY_OFFSET + MAILBOX_PAIR_BYTES * GDN_VALUE_HEADS
)
GDR_FULL_PARTIAL_READY_OFFSET = (
    GDR_FULL_NORM_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_NORM_TASKS
)
GDR_FULL_OUTPUT_READY_OFFSET = (
    GDR_FULL_PARTIAL_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_OUT_PARTIAL_TASKS
)
GDR_FULL_OUTPUT_GROUP_READY_OFFSET = (
    GDR_FULL_OUTPUT_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_OUT_TASKS
)
GDR_FULL_READY_BYTES = (
    GDR_FULL_OUTPUT_GROUP_READY_OFFSET + MAILBOX_PAIR_BYTES * GDR_OUT_READY_GROUPS
)
LOG2E = 1.4426950408889634
GDN_NORM_EPS = 1e-6

assert GDN_PROJ_DIM % ROWS == 0
assert GDN_CONV_DIM % ROWS == 0
assert K_STEPS_PER_WAVE * K_STEP * WAVES == HIDDEN
assert GDN_VALUE_HEADS % GDR_NORM_HEADS_PER_TASK == 0
assert GDN_VALUE_DIM % GDR_OUT_SPLITS == 0
assert GDR_OUT_K_STEPS_PER_WAVE * K_STEP * WAVES == GDR_OUT_SPLIT_K
assert GDR_OUT_FULL_K_STEPS_PER_WAVE * K_STEP * WAVES == GDN_VALUE_DIM
assert GDR_OUT_DIRECT_TASKS + GDR_OUT_PARTIAL_TASKS == BLOCKS
assert GDR_OUT_DIRECT_TASKS < GDR_OUT_EARLY_TASKS < GDR_OUT_TASKS
assert GDR_OUT_TASKS <= GDR_OUT_EARLY_OWNER < GDR_OUT_LATE_OWNER < BLOCKS


@fx.struct
class GdnSmem:
    activation: fx.Array[fx.Int32, HIDDEN // 2, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


@fx.struct
class GdnS2Smem:
    activation: fx.Array[fx.Int32, HIDDEN, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


@fx.struct
class GdnS4Smem:
    activation: fx.Array[fx.Int32, 2 * HIDDEN, 16]
    reduction: fx.Array[fx.Float32, WAVES * 64 * 4, 16]


@traced
def load_activation(x, tid, activation):
    """Stage the one BF16 activation row once per resident CTA."""

    if tid < HIDDEN // 8:
        at = tid * 8
        value = fx.Vector(bo.buffer_load(rsrc(x), at // 2, vec_width=4, dtype=T.i32))
        fx.ptr_store(value, activation + at // 2)


@traced
def projection_accumulator(weight, task, lane, wave, activation):
    """One wave's K split for a contiguous N16 projection tile."""

    row = task * ROWS + lane % ROWS
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(K_STEPS_PER_WAVE):
        k0 = (wave + WAVES * k_pass) * K_STEP + 8 * (lane // ROWS)
        w = fx.Vector(
            bo.buffer_load(
                rsrc(weight),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=0,
            )
        ).bitcast(fx.BFloat16)
        x = fx.Vector(
            fx.ptr_load(
                activation + k0 // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(w, x, acc)
    return acc


def projection_accumulator_s2(weight, task, lane, wave, activation0, activation1):
    """Reuse one projection weight stream for two activation rows."""

    row = task * ROWS + lane % ROWS
    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc1 = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(K_STEPS_PER_WAVE):
        k0 = (wave + WAVES * k_pass) * K_STEP + 8 * (lane // ROWS)
        w = fx.Vector(
            bo.buffer_load(
                rsrc(weight),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=0,
            )
        ).bitcast(fx.BFloat16)
        x0 = fx.Vector(
            fx.ptr_load(
                activation0 + k0 // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        x1 = fx.Vector(
            fx.ptr_load(
                activation1 + k0 // 2,
                result_type=T.vec(4, T.i32),
            )
        ).bitcast(fx.BFloat16)
        acc0 = mfma_bf16(w, x0, acc0)
        acc1 = mfma_bf16(w, x1, acc1)
    return acc0, acc1


def projection_accumulator_s4(
    weight,
    task,
    lane,
    wave,
    activation0,
    activation1,
    activation2,
    activation3,
):
    """Reuse one projection weight stream for a four-row decode tile."""

    row = task * ROWS + lane % ROWS
    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc1 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc2 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc3 = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(K_STEPS_PER_WAVE):
        k0 = (wave + WAVES * k_pass) * K_STEP + 8 * (lane // ROWS)
        w = fx.Vector(
            bo.buffer_load(
                rsrc(weight),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=0,
            )
        ).bitcast(fx.BFloat16)
        x0 = fx.Vector(
            fx.ptr_load(activation0 + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        x1 = fx.Vector(
            fx.ptr_load(activation1 + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        x2 = fx.Vector(
            fx.ptr_load(activation2 + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        x3 = fx.Vector(
            fx.ptr_load(activation3 + k0 // 2, result_type=T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        acc0 = mfma_bf16(w, x0, acc0)
        acc1 = mfma_bf16(w, x1, acc1)
        acc2 = mfma_bf16(w, x2, acc2)
        acc3 = mfma_bf16(w, x3, acc3)
    return acc0, acc1, acc2, acc3


def _ld_bf(ptr, index):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), index, vec_width=1, dtype=T.bf16))
    )


def _st_bf(ptr, index, value, cache_modifier=0):
    bo.buffer_store(
        fx.Float32(value).to(fx.BFloat16),
        rsrc(ptr),
        index,
        cache_modifier=cache_modifier,
    )


def _silu(x):
    return fx.Float32(x) / (
        fx.Float32(1.0) + hw_exp2(fx.Float32(x) * fx.Float32(-LOG2E))
    )


def _sigmoid(x):
    return fx.Float32(1.0) / (
        fx.Float32(1.0) + hw_exp2(fx.Float32(x) * fx.Float32(-LOG2E))
    )


def _bf16_round_strict(x):
    """RNE BF16 boundary that remains visible under kernel-wide fast math."""

    bits = fx.Float32(x).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    value = (rounded >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
    return fx.Float32(value)


@traced
def conv_projection_value(
    projected,
    channel,
    conv_weight,
    conv_state,
    read_slot,
    write_slot,
):
    """Apply the width-4 depthwise conv and update its three-value state."""

    state_stride = (GDN_CONV_WIDTH - 1) * GDN_CONV_DIM
    read_base = read_slot * state_stride + channel
    write_base = write_slot * state_stride + channel
    old0 = _ld_bf(conv_state, read_base)
    old1 = _ld_bf(conv_state, read_base + GDN_CONV_DIM)
    old2 = _ld_bf(conv_state, read_base + 2 * GDN_CONV_DIM)
    # Native causal_conv1d_update rounds each BF16 product before FP32 addition.
    acc = _bf16_round_strict(old0 * _ld_bf(conv_weight, channel * GDN_CONV_WIDTH))
    acc = acc + _bf16_round_strict(old1 * _ld_bf(conv_weight, channel * GDN_CONV_WIDTH + 1))
    acc = acc + _bf16_round_strict(old2 * _ld_bf(conv_weight, channel * GDN_CONV_WIDTH + 2))
    acc = acc + _bf16_round_strict(projected * _ld_bf(conv_weight, channel * GDN_CONV_WIDTH + 3))
    _st_bf(conv_state, write_base, old1)
    _st_bf(conv_state, write_base + GDN_CONV_DIM, old2)
    _st_bf(conv_state, write_base + 2 * GDN_CONV_DIM, projected)
    return bf16_round(_silu(acc))


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
        if tid < ROWS:
            channel = task * ROWS + tid
            projected = bf16_round(row_sum(reduction, tid, 0))
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot,
                write_slot,
            )
            _st_bf(out, channel, value, CM_DEV)
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


@traced
def projection_conv_task_s2(
    weight,
    conv_weight,
    conv_state,
    read_slot0,
    write_slot0,
    read_slot1,
    write_slot1,
    out0,
    out1,
    task,
    tid,
    lane,
    wave,
    activation0,
    activation1,
    reduction,
):
    """Project one N16 tile for two tokens with one weight load stream."""

    acc0, acc1 = projection_accumulator_s2(
        weight, task, lane, wave, activation0, activation1
    )
    fx.ptr_store(acc0, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if task < CONV_TASKS:
        if tid < ROWS:
            channel = task * ROWS + tid
            projected = bf16_round(row_sum(reduction, tid, 0))
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot0,
                write_slot0,
            )
            _st_bf(out0, channel, value, CM_DEV)
    else:
        if tid < ROWS // 2:
            row0 = 2 * tid
            value = bf16_pair(
                row_sum(reduction, row0, 0),
                row_sum(reduction, row0 + 1, 0),
            )
            bo.buffer_store(
                value.bitcast(fx.Int32),
                rsrc(out0),
                task * (ROWS // 2) + tid,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()
    fx.ptr_store(acc1, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if task < CONV_TASKS:
        if tid < ROWS:
            channel = task * ROWS + tid
            projected = bf16_round(row_sum(reduction, tid, 0))
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot1,
                write_slot1,
            )
            _st_bf(out1, channel, value, CM_DEV)
    else:
        if tid < ROWS // 2:
            row0 = 2 * tid
            value = bf16_pair(
                row_sum(reduction, row0, 0),
                row_sum(reduction, row0 + 1, 0),
            )
            bo.buffer_store(
                value.bitcast(fx.Int32),
                rsrc(out1),
                task * (ROWS // 2) + tid,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()


@traced
def projection_conv_task_s4(
    weight,
    conv_weight,
    conv_state,
    read_slot0,
    write_slot0,
    read_slot1,
    write_slot1,
    read_slot2,
    write_slot2,
    read_slot3,
    write_slot3,
    out0,
    out1,
    out2,
    out3,
    task,
    tid,
    lane,
    wave,
    activation0,
    activation1,
    activation2,
    activation3,
    reduction,
):
    """Project one N16 tile for four tokens with one weight load stream."""

    acc0, acc1, acc2, acc3 = projection_accumulator_s4(
        weight,
        task,
        lane,
        wave,
        activation0,
        activation1,
        activation2,
        activation3,
    )

    fx.ptr_store(acc0, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if task < CONV_TASKS:
        if tid < ROWS:
            channel = task * ROWS + tid
            projected = bf16_round(row_sum(reduction, tid, 0))
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot0,
                write_slot0,
            )
            _st_bf(out0, channel, value, CM_DEV)
    else:
        if tid < ROWS // 2:
            row0 = 2 * tid
            value = bf16_pair(
                row_sum(reduction, row0, 0), row_sum(reduction, row0 + 1, 0)
            )
            bo.buffer_store(
                value.bitcast(fx.Int32),
                rsrc(out0),
                task * (ROWS // 2) + tid,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()

    fx.ptr_store(acc1, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if task < CONV_TASKS:
        if tid < ROWS:
            channel = task * ROWS + tid
            projected = bf16_round(row_sum(reduction, tid, 0))
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot1,
                write_slot1,
            )
            _st_bf(out1, channel, value, CM_DEV)
    else:
        if tid < ROWS // 2:
            row0 = 2 * tid
            value = bf16_pair(
                row_sum(reduction, row0, 0), row_sum(reduction, row0 + 1, 0)
            )
            bo.buffer_store(
                value.bitcast(fx.Int32),
                rsrc(out1),
                task * (ROWS // 2) + tid,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()

    fx.ptr_store(acc2, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if task < CONV_TASKS:
        if tid < ROWS:
            channel = task * ROWS + tid
            projected = bf16_round(row_sum(reduction, tid, 0))
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot2,
                write_slot2,
            )
            _st_bf(out2, channel, value, CM_DEV)
    else:
        if tid < ROWS // 2:
            row0 = 2 * tid
            value = bf16_pair(
                row_sum(reduction, row0, 0), row_sum(reduction, row0 + 1, 0)
            )
            bo.buffer_store(
                value.bitcast(fx.Int32),
                rsrc(out2),
                task * (ROWS // 2) + tid,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()

    fx.ptr_store(acc3, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if task < CONV_TASKS:
        if tid < ROWS:
            channel = task * ROWS + tid
            projected = bf16_round(row_sum(reduction, tid, 0))
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot3,
                write_slot3,
            )
            _st_bf(out3, channel, value, CM_DEV)
    else:
        if tid < ROWS // 2:
            row0 = 2 * tid
            value = bf16_pair(
                row_sum(reduction, row0, 0), row_sum(reduction, row0 + 1, 0)
            )
            bo.buffer_store(
                value.bitcast(fx.Int32),
                rsrc(out3),
                task * (ROWS // 2) + tid,
                cache_modifier=CM_DEV,
            )
    gpu.barrier()


@traced
def sigmoid_norm_task(
    core,
    gate,
    weight,
    normed,
    task,
    lane,
    wave,
):
    """Normalize and sigmoid-gate eight heads, one head per resident wave."""

    head = task * GDR_NORM_HEADS_PER_TASK + wave
    index0 = head * GDN_HEAD_DIM + lane
    index1 = index0 + 64
    x0 = _ld_bf(core, index0)
    x1 = _ld_bf(core, index1)
    total = wave_sum(x0 * x0 + x1 * x1)
    rrms = hw_rsq(total * fx.Float32(1.0 / GDN_HEAD_DIM) + GDN_NORM_EPS)
    normalized0 = _bf16_round_strict(x0 * rrms)
    normalized1 = _bf16_round_strict(x1 * rrms)
    affine0 = _bf16_round_strict(normalized0 * _ld_bf(weight, lane))
    affine1 = _bf16_round_strict(normalized1 * _ld_bf(weight, lane + 64))
    result0 = affine0 * _sigmoid(_ld_bf(gate, index0))
    result1 = affine1 * _sigmoid(_ld_bf(gate, index1))
    _st_bf(normed, index0, result0, CM_DEV)
    _st_bf(normed, index1, result1, CM_DEV)


def output_projection_loads(weight, tile, k_start, k_steps, lane, wave):
    """Issue one output task's cold weight loads before waiting for its input."""

    row = tile * ROWS + lane % ROWS
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(weight),
                (
                    row * GDN_VALUE_DIM
                    + k_start
                    + (wave + WAVES * k_pass) * K_STEP
                    + 8 * (lane // ROWS)
                )
                // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=0,
            )
        ).bitcast(fx.BFloat16)
        for k_pass in range_constexpr(k_steps)
    ]


def output_projection_accumulator(normed, k_start, lane, wave, weight_ops):
    """Consume prefetched weights for one contiguous K range."""

    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(len(weight_ops)):
        k0 = k_start + (wave + WAVES * k_pass) * K_STEP + 8 * (lane // ROWS)
        x = fx.Vector(
            bo.buffer_load(
                rsrc(normed),
                k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        ).bitcast(fx.BFloat16)
        acc = mfma_bf16(weight_ops[k_pass], x, acc)
    return acc


def output_projection_accumulators_s4(
    normed0,
    normed1,
    normed2,
    normed3,
    weight,
    tile,
    k_start,
    k_steps,
    lane,
    wave,
    first_weight_ops,
):
    """Consume one bounded prefetch window, then stream the remaining weights."""

    row = tile * ROWS + lane % ROWS
    acc0 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc1 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc2 = fx.Vector.filled(4, 0.0, fx.Float32)
    acc3 = fx.Vector.filled(4, 0.0, fx.Float32)
    for k_pass in range_constexpr(k_steps):
        k0 = k_start + (wave + WAVES * k_pass) * K_STEP + 8 * (lane // ROWS)
        if k_pass < S4_OUTPUT_PREFETCH_STEPS:
            w = first_weight_ops[k_pass]
        else:
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(weight),
                    (row * GDN_VALUE_DIM + k0) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=0,
                )
            ).bitcast(fx.BFloat16)
        x0 = fx.Vector(
            bo.buffer_load(
                rsrc(normed0),
                k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        ).bitcast(fx.BFloat16)
        x1 = fx.Vector(
            bo.buffer_load(
                rsrc(normed1),
                k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        ).bitcast(fx.BFloat16)
        x2 = fx.Vector(
            bo.buffer_load(
                rsrc(normed2),
                k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        ).bitcast(fx.BFloat16)
        x3 = fx.Vector(
            bo.buffer_load(
                rsrc(normed3),
                k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        ).bitcast(fx.BFloat16)
        acc0 = mfma_bf16(w, x0, acc0)
        acc1 = mfma_bf16(w, x1, acc1)
        acc2 = mfma_bf16(w, x2, acc2)
        acc3 = mfma_bf16(w, x3, acc3)
    return acc0, acc1, acc2, acc3


@traced
def output_projection_direct_task(
    normed,
    out,
    tile,
    tid,
    lane,
    wave,
    reduction,
    weight_ops,
    output_cache_modifier=0,
    after_pair=None,
):
    """Complete one N16 tile without a global partial hand-off."""

    acc = output_projection_accumulator(normed, 0, lane, wave, weight_ops)
    fx.ptr_store(acc, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        if const_expr(after_pair is not None):
            after_pair(tile, row0, value0, value1)
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )


@traced
def output_projection_direct_task_s2(
    normed0,
    normed1,
    out0,
    out1,
    tile,
    tid,
    lane,
    wave,
    reduction,
    weight_ops,
    output_cache_modifier=0,
    after_pair0=None,
    after_pair1=None,
):
    """Complete one output tile for two tokens from one weight batch."""

    acc0 = output_projection_accumulator(normed0, 0, lane, wave, weight_ops)
    acc1 = output_projection_accumulator(normed1, 0, lane, wave, weight_ops)
    fx.ptr_store(acc0, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        if const_expr(after_pair0 is not None):
            after_pair0(tile, row0, value0, value1)
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out0),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )
    gpu.barrier()
    fx.ptr_store(acc1, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        if const_expr(after_pair1 is not None):
            after_pair1(tile, row0, value0, value1)
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out1),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )


@traced
def output_projection_direct_task_s4(
    normed0,
    normed1,
    normed2,
    normed3,
    out0,
    out1,
    out2,
    out3,
    tile,
    tid,
    lane,
    wave,
    reduction,
    weight,
    first_weight_ops,
    output_cache_modifier=0,
    after_pair0=None,
    after_pair1=None,
    after_pair2=None,
    after_pair3=None,
):
    """Complete one output tile for four tokens from one weight batch."""

    acc0, acc1, acc2, acc3 = output_projection_accumulators_s4(
        normed0,
        normed1,
        normed2,
        normed3,
        weight,
        tile,
        0,
        GDR_OUT_FULL_K_STEPS_PER_WAVE,
        lane,
        wave,
        first_weight_ops,
    )

    fx.ptr_store(acc0, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        if const_expr(after_pair0 is not None):
            after_pair0(tile, row0, value0, value1)
        bo.buffer_store(
            bf16_pair(value0, value1).bitcast(fx.Int32),
            rsrc(out0),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )
    gpu.barrier()

    fx.ptr_store(acc1, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        if const_expr(after_pair1 is not None):
            after_pair1(tile, row0, value0, value1)
        bo.buffer_store(
            bf16_pair(value0, value1).bitcast(fx.Int32),
            rsrc(out1),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )
    gpu.barrier()

    fx.ptr_store(acc2, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        if const_expr(after_pair2 is not None):
            after_pair2(tile, row0, value0, value1)
        bo.buffer_store(
            bf16_pair(value0, value1).bitcast(fx.Int32),
            rsrc(out2),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )
    gpu.barrier()

    fx.ptr_store(acc3, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS // 2:
        row0 = 2 * tid
        value0 = bf16_round(row_sum(reduction, row0, 0))
        value1 = bf16_round(row_sum(reduction, row0 + 1, 0))
        if const_expr(after_pair3 is not None):
            after_pair3(tile, row0, value0, value1)
        bo.buffer_store(
            bf16_pair(value0, value1).bitcast(fx.Int32),
            rsrc(out3),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )


@traced
def output_projection_partial_task(
    normed,
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

    split = task // GDR_OUT_SPLIT_TILES
    k_start = split * GDR_OUT_SPLIT_K
    acc = output_projection_accumulator(normed, k_start, lane, wave, weight_ops)
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
def output_projection_partial_task_s2(
    normed0,
    normed1,
    partials0,
    partials1,
    partial_ready0,
    partial_ready1,
    put,
    task,
    tid,
    lane,
    wave,
    reduction,
    weight_ops,
):
    """Produce the matching FP32 output partial for two tokens."""

    split = task // GDR_OUT_SPLIT_TILES
    k_start = split * GDR_OUT_SPLIT_K
    acc0 = output_projection_accumulator(normed0, k_start, lane, wave, weight_ops)
    acc1 = output_projection_accumulator(normed1, k_start, lane, wave, weight_ops)
    fx.ptr_store(acc0, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        bo.buffer_store(
            row_sum(reduction, tid, 0),
            rsrc(partials0),
            task * ROWS + tid,
            cache_modifier=CM_DEV,
        )
    publish(put, partial_ready0, task, fx.Int32(1), tid == 0)
    gpu.barrier()
    fx.ptr_store(acc1, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        bo.buffer_store(
            row_sum(reduction, tid, 0),
            rsrc(partials1),
            task * ROWS + tid,
            cache_modifier=CM_DEV,
        )
    publish(put, partial_ready1, task, fx.Int32(1), tid == 0)


@traced
def output_projection_partial_task_s4(
    normed0,
    normed1,
    normed2,
    normed3,
    partials0,
    partials1,
    partials2,
    partials3,
    partial_ready0,
    partial_ready1,
    partial_ready2,
    partial_ready3,
    put,
    task,
    tid,
    lane,
    wave,
    reduction,
    weight,
    first_weight_ops,
):
    """Produce matching FP32 output partials for a four-row tile."""

    split = task // GDR_OUT_SPLIT_TILES
    k_start = split * GDR_OUT_SPLIT_K
    acc0, acc1, acc2, acc3 = output_projection_accumulators_s4(
        normed0,
        normed1,
        normed2,
        normed3,
        weight,
        GDR_OUT_DIRECT_TASKS + task % GDR_OUT_SPLIT_TILES,
        k_start,
        GDR_OUT_K_STEPS_PER_WAVE,
        lane,
        wave,
        first_weight_ops,
    )

    fx.ptr_store(acc0, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        bo.buffer_store(
            row_sum(reduction, tid, 0),
            rsrc(partials0),
            task * ROWS + tid,
            cache_modifier=CM_DEV,
        )
    publish(put, partial_ready0, task, fx.Int32(1), tid == 0)
    gpu.barrier()

    fx.ptr_store(acc1, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        bo.buffer_store(
            row_sum(reduction, tid, 0),
            rsrc(partials1),
            task * ROWS + tid,
            cache_modifier=CM_DEV,
        )
    publish(put, partial_ready1, task, fx.Int32(1), tid == 0)
    gpu.barrier()

    fx.ptr_store(acc2, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        bo.buffer_store(
            row_sum(reduction, tid, 0),
            rsrc(partials2),
            task * ROWS + tid,
            cache_modifier=CM_DEV,
        )
    publish(put, partial_ready2, task, fx.Int32(1), tid == 0)
    gpu.barrier()

    fx.ptr_store(acc3, reduction + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS:
        bo.buffer_store(
            row_sum(reduction, tid, 0),
            rsrc(partials3),
            task * ROWS + tid,
            cache_modifier=CM_DEV,
        )
    publish(put, partial_ready3, task, fx.Int32(1), tid == 0)


@traced
def output_projection_finish(
    partials,
    partial_ready,
    poll,
    out,
    tile,
    tid,
    output_cache_modifier=0,
    after_pair=None,
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
            fx.Float32(
                bo.buffer_load(
                    rsrc(partials),
                    index0,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
            + fx.Float32(
                bo.buffer_load(
                    rsrc(partials),
                    split1_base + index0,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
        )
        value1 = bf16_round(
            fx.Float32(
                bo.buffer_load(
                    rsrc(partials),
                    index1,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
            + fx.Float32(
                bo.buffer_load(
                    rsrc(partials),
                    split1_base + index1,
                    vec_width=1,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
        )
        if const_expr(after_pair is not None):
            after_pair(tile, row0, value0, value1)
        value = bf16_pair(value0, value1)
        bo.buffer_store(
            value.bitcast(fx.Int32),
            rsrc(out),
            tile * (ROWS // 2) + tid,
            cache_modifier=output_cache_modifier,
        )


@traced
def _wait_gdr_norms(poll, norm_ready, tid):
    """Hold output-projection CTAs until all eight-head norm tasks finish."""

    if tid == 0:
        poll([(norm_ready, task, 1) for task in range_constexpr(GDR_NORM_TASKS)])
    gpu.barrier()


def _raw_tensor(address, numeric, alignment=16):
    """Flat global tensor view used at the raw-address production ABI."""

    pointer = fx.PointerType.get(
        numeric.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=alignment,
    )
    view = fx.make_view(
        fx.inttoptr(pointer, fx.Int64(address)),
        fx.make_layout((0x7FFFFFFF,), (1,)),
    )
    return fx.Tensor(view)


def _gdr_projection_views(
    projected, dt_bias, A_log, read_indices, write_indices, state, out
):
    """Build the tensor views consumed by AITER's GDR emitter."""

    bf16_bytes = fx.Int64(2)
    query = _raw_tensor(projected, fx.BFloat16)
    key = _raw_tensor(projected + bf16_bytes * fx.Int64(GDN_KEY_DIM), fx.BFloat16)
    value = _raw_tensor(projected + bf16_bytes * fx.Int64(2 * GDN_KEY_DIM), fx.BFloat16)
    b = _raw_tensor(projected + bf16_bytes * fx.Int64(GDR_B_ROW), fx.BFloat16)
    a = _raw_tensor(projected + bf16_bytes * fx.Int64(GDR_A_ROW), fx.BFloat16)
    return (
        query,
        key,
        value,
        a,
        b,
        _raw_tensor(dt_bias, fx.BFloat16),
        _raw_tensor(A_log, fx.BFloat16),
        _raw_tensor(read_indices, fx.Int32, 4),
        _raw_tensor(write_indices, fx.Int32, 4),
        _raw_tensor(state, fx.Float32),
        _raw_tensor(out, fx.BFloat16),
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
    """Hold the recurrence CTA until its own projection tiles are ready."""

    if tid == 0:
        poll([(projected_ready, dep, 1) for dep in _gdr_dependencies(task)])
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
def _publish_gdr_output_groups(
    poll,
    put,
    output_ready,
    output_group_ready,
    bid,
    tid,
):
    """Collapse 160 output tiles into the two passes consumed by HC norm."""

    # Keep the four HC norm CTAs (0..3) free to consume group 0 immediately.
    # These owners finish the high-K partial for each group's final tile, have
    # no downstream finish task, and then collapse the matching ready tags.
    if bid == GDR_OUT_EARLY_OWNER:
        if tid == 0:
            poll(
                [
                    (output_ready, tile, 1)
                    for tile in range_constexpr(GDR_OUT_EARLY_TASKS)
                ]
            )
        publish(put, output_group_ready, 0, fx.Int32(1), tid == 0)
    elif bid == GDR_OUT_LATE_OWNER:
        if tid == 0:
            poll(
                [
                    (output_ready, GDR_OUT_EARLY_TASKS + tile, 1)
                    for tile in range_constexpr(GDR_OUT_TASKS - GDR_OUT_EARLY_TASKS)
                ]
            )
        publish(put, output_group_ready, 1, fx.Int32(1), tid == 0)


def gdn_args(
    x,
    projection_weight,
    conv_weight,
    conv_state,
    read_indices,
    write_indices,
    A_log,
    dt_bias,
    state,
    projected,
    core,
    norm_weight,
    output_weight,
    normed,
    out,
    partials,
):
    """Named arguments consumed by the reusable complete GDN emitter."""

    return {
        "x": x,
        "projection_weight": projection_weight,
        "conv_weight": conv_weight,
        "conv_state": conv_state,
        "read_indices": read_indices,
        "write_indices": write_indices,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state": state,
        "projected": projected,
        "core": core,
        "norm_weight": norm_weight,
        "output_weight": output_weight,
        "normed": normed,
        "out": out,
        "partials": partials,
    }


@traced
def run_gdn(
    emit_gdr_decode_task,
    lds,
    mailbox,
    ready,
    args,
    bid,
    publish_output=False,
    after_output_pair=None,
):
    """Emit the accepted S=1 GDN dependency graph into a caller's kernel.

    ``publish_output`` is a compile-time composition contract. A layer kernel
    sets it so every BF16 output tile is published at device scope. Two owners
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
    projected_ready = sreg(
        ready,
        GDR_FULL_PROJECTION_READY_OFFSET,
        "gdn_full_projection_ready",
    )
    recurrence_ready = sreg(
        ready,
        GDR_FULL_RECURRENCE_READY_OFFSET,
        "gdn_full_recurrence_ready",
    )
    head_ready = sreg(
        ready,
        GDR_FULL_HEAD_READY_OFFSET,
        "gdn_full_head_ready",
    )
    norm_ready = sreg(
        ready,
        GDR_FULL_NORM_READY_OFFSET,
        "gdn_full_norm_ready",
    )
    partial_ready = sreg(
        ready,
        GDR_FULL_PARTIAL_READY_OFFSET,
        "gdn_full_output_partial_ready",
    )
    output_ready = sreg(
        ready,
        GDR_FULL_OUTPUT_READY_OFFSET,
        "gdn_full_output_ready",
    )
    output_group_ready = sreg(
        ready,
        GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
        "gdn_full_output_group_ready",
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
    output_cache_modifier = CM_DEV if publish_output else 0
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
        output_projection_direct_task(
            args["normed"],
            args["out"],
            bid,
            tid,
            lane,
            wave,
            reduction,
            weight_ops,
            output_cache_modifier,
            after_output_pair,
        )
        if const_expr(publish_output):
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
        output_projection_partial_task(
            args["normed"],
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
                output_cache_modifier,
                after_output_pair,
            )
            if const_expr(publish_output):
                publish(mailbox.put, output_ready, bid, fx.Int32(1), tid == 0)

    if const_expr(publish_output):
        _publish_gdr_output_groups(
            mailbox.poll,
            mailbox.put,
            output_ready,
            output_group_ready,
            bid,
            tid,
        )


@traced
def run_gdn_s2(
    emit_gdr_decode_task,
    lds,
    mailbox,
    ready0,
    ready1,
    args0,
    args1,
    bid,
    publish_output=False,
    after_output_pair0=None,
    after_output_pair1=None,
):
    """Run two GDN rows while sharing projection and output weight loads."""

    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64

    read_slot0 = fx.Int32(
        bo.buffer_load(rsrc(args0["read_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot0 = fx.Int32(
        bo.buffer_load(rsrc(args0["write_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    read_slot1 = fx.Int32(
        bo.buffer_load(rsrc(args1["read_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot1 = fx.Int32(
        bo.buffer_load(rsrc(args1["write_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    activation0 = lds.activation.ptr
    activation1 = activation0 + HIDDEN // 2
    reduction = lds.reduction.ptr

    projected_ready0 = sreg(
        ready0,
        GDR_FULL_PROJECTION_READY_OFFSET,
        "gdn_full_projection_ready_0",
    )
    projected_ready1 = sreg(
        ready1,
        GDR_FULL_PROJECTION_READY_OFFSET,
        "gdn_full_projection_ready_1",
    )

    load_activation(args0["x"], tid, activation0)
    load_activation(args1["x"], tid, activation1)
    gpu.barrier()

    def produce(projection_task_id):
        projection_conv_task_s2(
            args0["projection_weight"],
            args0["conv_weight"],
            args0["conv_state"],
            read_slot0,
            write_slot0,
            read_slot1,
            write_slot1,
            args0["projected"],
            args1["projected"],
            projection_task_id,
            tid,
            lane,
            wave,
            activation0,
            activation1,
            reduction,
        )
        publish(
            mailbox.put,
            projected_ready0,
            projection_task_id,
            fx.Int32(1),
            tid == 0,
        )
        publish(
            mailbox.put,
            projected_ready1,
            projection_task_id,
            fx.Int32(1),
            tid == 0,
        )

    if bid < GDR_GATE_TASKS:
        produce(GDR_B_ROW // ROWS + bid)
    for round_id in range_constexpr(GDR_FRONT_ROUNDS):
        projection_task = bid + BLOCKS * round_id
        if projection_task < CONV_TASKS:
            produce(projection_task)

    recurrence_ready0 = sreg(
        ready0,
        GDR_FULL_RECURRENCE_READY_OFFSET,
        "gdn_full_recurrence_ready_0",
    )
    recurrence_ready1 = sreg(
        ready1,
        GDR_FULL_RECURRENCE_READY_OFFSET,
        "gdn_full_recurrence_ready_1",
    )
    recurrence_token = bid // GDR_TASKS
    recurrence_task = bid % GDR_TASKS
    if bid < 2 * GDR_TASKS:
        if recurrence_token == 0:
            _wait_gdr_dependencies(mailbox.poll, projected_ready0, recurrence_task, tid)
            emit_gdr_decode_task(
                *_gdr_projection_views(
                    args0["projected"],
                    args0["dt_bias"],
                    args0["A_log"],
                    args0["read_indices"],
                    args0["write_indices"],
                    args0["state"],
                    args0["core"],
                ),
                fx.Int32(1),
                tid,
                recurrence_task,
            )
            publish(
                mailbox.put,
                recurrence_ready0,
                recurrence_task,
                fx.Int32(1),
                tid == 0,
            )
        else:
            _wait_gdr_dependencies(mailbox.poll, projected_ready1, recurrence_task, tid)
            emit_gdr_decode_task(
                *_gdr_projection_views(
                    args1["projected"],
                    args1["dt_bias"],
                    args1["A_log"],
                    args1["read_indices"],
                    args1["write_indices"],
                    args1["state"],
                    args1["core"],
                ),
                fx.Int32(1),
                tid,
                recurrence_task,
            )
            publish(
                mailbox.put,
                recurrence_ready1,
                recurrence_task,
                fx.Int32(1),
                tid == 0,
            )

    # Keep the 64 CTAs without recurrence work busy while the other 192 CTAs
    # update state. They produce the first half of the paired Z projection;
    # recurrence owners each finish one tile from the second half afterwards.
    if bid >= 2 * GDR_TASKS:
        worker = bid - 2 * GDR_TASKS
        for round_id in range_constexpr(GDR_Z_TASKS // (2 * (BLOCKS - 2 * GDR_TASKS))):
            z_task = worker + (BLOCKS - 2 * GDR_TASKS) * round_id
            produce(CONV_TASKS + z_task)
    else:
        produce(CONV_TASKS + GDR_Z_TASKS // 2 + bid)

    head_ready0 = sreg(
        ready0,
        GDR_FULL_HEAD_READY_OFFSET,
        "gdn_full_head_ready_0",
    )
    head_ready1 = sreg(
        ready1,
        GDR_FULL_HEAD_READY_OFFSET,
        "gdn_full_head_ready_1",
    )
    if bid < 2 * GDN_VALUE_HEADS:
        head_token = bid // GDN_VALUE_HEADS
        head = bid % GDN_VALUE_HEADS
        if head_token == 0:
            _publish_gdr_head_ready(
                mailbox.poll,
                mailbox.put,
                recurrence_ready0,
                projected_ready0,
                head_ready0,
                head,
                tid,
            )
        else:
            _publish_gdr_head_ready(
                mailbox.poll,
                mailbox.put,
                recurrence_ready1,
                projected_ready1,
                head_ready1,
                head,
                tid,
            )

    norm_ready0 = sreg(
        ready0,
        GDR_FULL_NORM_READY_OFFSET,
        "gdn_full_norm_ready_0",
    )
    norm_ready1 = sreg(
        ready1,
        GDR_FULL_NORM_READY_OFFSET,
        "gdn_full_norm_ready_1",
    )
    if bid < 2 * GDR_NORM_TASKS:
        norm_token = bid // GDR_NORM_TASKS
        norm_task = bid % GDR_NORM_TASKS
        if norm_token == 0:
            _wait_gdr_norm_heads(mailbox.poll, head_ready0, norm_task, tid)
            sigmoid_norm_task(
                args0["core"],
                args0["projected"] + fx.Int64(2 * GDN_CONV_DIM),
                args0["norm_weight"],
                args0["normed"],
                norm_task,
                lane,
                wave,
            )
            publish(mailbox.put, norm_ready0, norm_task, fx.Int32(1), tid == 0)
        else:
            _wait_gdr_norm_heads(mailbox.poll, head_ready1, norm_task, tid)
            sigmoid_norm_task(
                args1["core"],
                args1["projected"] + fx.Int64(2 * GDN_CONV_DIM),
                args1["norm_weight"],
                args1["normed"],
                norm_task,
                lane,
                wave,
            )
            publish(mailbox.put, norm_ready1, norm_task, fx.Int32(1), tid == 0)

    partial_ready0 = sreg(
        ready0,
        GDR_FULL_PARTIAL_READY_OFFSET,
        "gdn_full_output_partial_ready_0",
    )
    partial_ready1 = sreg(
        ready1,
        GDR_FULL_PARTIAL_READY_OFFSET,
        "gdn_full_output_partial_ready_1",
    )
    output_ready0 = sreg(
        ready0,
        GDR_FULL_OUTPUT_READY_OFFSET,
        "gdn_full_output_ready_0",
    )
    output_ready1 = sreg(
        ready1,
        GDR_FULL_OUTPUT_READY_OFFSET,
        "gdn_full_output_ready_1",
    )
    output_group_ready0 = sreg(
        ready0,
        GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
        "gdn_full_output_group_ready_0",
    )
    output_group_ready1 = sreg(
        ready1,
        GDR_FULL_OUTPUT_GROUP_READY_OFFSET,
        "gdn_full_output_group_ready_1",
    )
    output_cache_modifier = CM_DEV if publish_output else 0
    if bid < GDR_OUT_DIRECT_TASKS:
        weight_ops = output_projection_loads(
            args0["output_weight"],
            bid,
            0,
            GDR_OUT_FULL_K_STEPS_PER_WAVE,
            lane,
            wave,
        )
        _wait_gdr_norms(mailbox.poll, norm_ready0, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready1, tid)
        output_projection_direct_task_s2(
            args0["normed"],
            args1["normed"],
            args0["out"],
            args1["out"],
            bid,
            tid,
            lane,
            wave,
            reduction,
            weight_ops,
            output_cache_modifier,
            after_output_pair0,
            after_output_pair1,
        )
        if const_expr(publish_output):
            publish(mailbox.put, output_ready0, bid, fx.Int32(1), tid == 0)
            publish(mailbox.put, output_ready1, bid, fx.Int32(1), tid == 0)
    else:
        partial_task = bid - GDR_OUT_DIRECT_TASKS
        split = partial_task // GDR_OUT_SPLIT_TILES
        tile = GDR_OUT_DIRECT_TASKS + partial_task % GDR_OUT_SPLIT_TILES
        weight_ops = output_projection_loads(
            args0["output_weight"],
            tile,
            split * GDR_OUT_SPLIT_K,
            GDR_OUT_K_STEPS_PER_WAVE,
            lane,
            wave,
        )
        _wait_gdr_norms(mailbox.poll, norm_ready0, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready1, tid)
        output_projection_partial_task_s2(
            args0["normed"],
            args1["normed"],
            args0["partials"],
            args1["partials"],
            partial_ready0,
            partial_ready1,
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
                args0["partials"],
                partial_ready0,
                mailbox.poll,
                args0["out"],
                bid,
                tid,
                output_cache_modifier,
                after_output_pair0,
            )
            output_projection_finish(
                args1["partials"],
                partial_ready1,
                mailbox.poll,
                args1["out"],
                bid,
                tid,
                output_cache_modifier,
                after_output_pair1,
            )
            if const_expr(publish_output):
                publish(mailbox.put, output_ready0, bid, fx.Int32(1), tid == 0)
                publish(mailbox.put, output_ready1, bid, fx.Int32(1), tid == 0)

    if const_expr(publish_output):
        _publish_gdr_output_groups(
            mailbox.poll,
            mailbox.put,
            output_ready0,
            output_group_ready0,
            bid,
            tid,
        )
        _publish_gdr_output_groups(
            mailbox.poll,
            mailbox.put,
            output_ready1,
            output_group_ready1,
            bid,
            tid,
        )


@traced
def run_gdn_s4(
    emit_gdr_decode_task,
    lds,
    mailbox,
    ready0,
    ready1,
    ready2,
    ready3,
    args0,
    args1,
    args2,
    args3,
    bid,
    publish_output=False,
    after_output_pair0=None,
    after_output_pair1=None,
    after_output_pair2=None,
    after_output_pair3=None,
):
    """Run one true four-row GDN tile with shared dense weight streams."""

    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64

    read_slot0 = fx.Int32(
        bo.buffer_load(rsrc(args0["read_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot0 = fx.Int32(
        bo.buffer_load(rsrc(args0["write_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    read_slot1 = fx.Int32(
        bo.buffer_load(rsrc(args1["read_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot1 = fx.Int32(
        bo.buffer_load(rsrc(args1["write_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    read_slot2 = fx.Int32(
        bo.buffer_load(rsrc(args2["read_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot2 = fx.Int32(
        bo.buffer_load(rsrc(args2["write_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    read_slot3 = fx.Int32(
        bo.buffer_load(rsrc(args3["read_indices"]), 0, vec_width=1, dtype=T.i32)
    )
    write_slot3 = fx.Int32(
        bo.buffer_load(rsrc(args3["write_indices"]), 0, vec_width=1, dtype=T.i32)
    )

    activation0 = lds.activation.ptr
    activation1 = activation0 + HIDDEN // 2
    activation2 = activation1 + HIDDEN // 2
    activation3 = activation2 + HIDDEN // 2
    reduction = lds.reduction.ptr

    projected_ready0 = sreg(
        ready0, GDR_FULL_PROJECTION_READY_OFFSET, "gdn_s4_projection_ready_0"
    )
    projected_ready1 = sreg(
        ready1, GDR_FULL_PROJECTION_READY_OFFSET, "gdn_s4_projection_ready_1"
    )
    projected_ready2 = sreg(
        ready2, GDR_FULL_PROJECTION_READY_OFFSET, "gdn_s4_projection_ready_2"
    )
    projected_ready3 = sreg(
        ready3, GDR_FULL_PROJECTION_READY_OFFSET, "gdn_s4_projection_ready_3"
    )

    load_activation(args0["x"], tid, activation0)
    load_activation(args1["x"], tid, activation1)
    load_activation(args2["x"], tid, activation2)
    load_activation(args3["x"], tid, activation3)
    gpu.barrier()

    def produce(projection_task_id):
        projection_conv_task_s4(
            args0["projection_weight"],
            args0["conv_weight"],
            args0["conv_state"],
            read_slot0,
            write_slot0,
            read_slot1,
            write_slot1,
            read_slot2,
            write_slot2,
            read_slot3,
            write_slot3,
            args0["projected"],
            args1["projected"],
            args2["projected"],
            args3["projected"],
            projection_task_id,
            tid,
            lane,
            wave,
            activation0,
            activation1,
            activation2,
            activation3,
            reduction,
        )
        publish(
            mailbox.put,
            projected_ready0,
            projection_task_id,
            fx.Int32(1),
            tid == 0,
        )
        publish(
            mailbox.put,
            projected_ready1,
            projection_task_id,
            fx.Int32(1),
            tid == 0,
        )
        publish(
            mailbox.put,
            projected_ready2,
            projection_task_id,
            fx.Int32(1),
            tid == 0,
        )
        publish(
            mailbox.put,
            projected_ready3,
            projection_task_id,
            fx.Int32(1),
            tid == 0,
        )

    if bid < GDR_GATE_TASKS:
        produce(GDR_B_ROW // ROWS + bid)
    for round_id in range_constexpr(GDR_FRONT_ROUNDS):
        projection_task = bid + BLOCKS * round_id
        if projection_task < CONV_TASKS:
            produce(projection_task)

    recurrence_ready0 = sreg(
        ready0, GDR_FULL_RECURRENCE_READY_OFFSET, "gdn_s4_recurrence_ready_0"
    )
    recurrence_ready1 = sreg(
        ready1, GDR_FULL_RECURRENCE_READY_OFFSET, "gdn_s4_recurrence_ready_1"
    )
    recurrence_ready2 = sreg(
        ready2, GDR_FULL_RECURRENCE_READY_OFFSET, "gdn_s4_recurrence_ready_2"
    )
    recurrence_ready3 = sreg(
        ready3, GDR_FULL_RECURRENCE_READY_OFFSET, "gdn_s4_recurrence_ready_3"
    )

    if bid < 2 * GDR_TASKS:
        for recurrence_round in range_constexpr(2):
            recurrence_unit = bid + 2 * GDR_TASKS * recurrence_round
            recurrence_token = recurrence_unit // GDR_TASKS
            recurrence_task = recurrence_unit % GDR_TASKS
            if recurrence_token == 0:
                _wait_gdr_dependencies(
                    mailbox.poll, projected_ready0, recurrence_task, tid
                )
                emit_gdr_decode_task(
                    *_gdr_projection_views(
                        args0["projected"],
                        args0["dt_bias"],
                        args0["A_log"],
                        args0["read_indices"],
                        args0["write_indices"],
                        args0["state"],
                        args0["core"],
                    ),
                    fx.Int32(1),
                    tid,
                    recurrence_task,
                )
                publish(
                    mailbox.put,
                    recurrence_ready0,
                    recurrence_task,
                    fx.Int32(1),
                    tid == 0,
                )
            if recurrence_token == 1:
                _wait_gdr_dependencies(
                    mailbox.poll, projected_ready1, recurrence_task, tid
                )
                emit_gdr_decode_task(
                    *_gdr_projection_views(
                        args1["projected"],
                        args1["dt_bias"],
                        args1["A_log"],
                        args1["read_indices"],
                        args1["write_indices"],
                        args1["state"],
                        args1["core"],
                    ),
                    fx.Int32(1),
                    tid,
                    recurrence_task,
                )
                publish(
                    mailbox.put,
                    recurrence_ready1,
                    recurrence_task,
                    fx.Int32(1),
                    tid == 0,
                )
            if recurrence_token == 2:
                _wait_gdr_dependencies(
                    mailbox.poll, projected_ready2, recurrence_task, tid
                )
                emit_gdr_decode_task(
                    *_gdr_projection_views(
                        args2["projected"],
                        args2["dt_bias"],
                        args2["A_log"],
                        args2["read_indices"],
                        args2["write_indices"],
                        args2["state"],
                        args2["core"],
                    ),
                    fx.Int32(1),
                    tid,
                    recurrence_task,
                )
                publish(
                    mailbox.put,
                    recurrence_ready2,
                    recurrence_task,
                    fx.Int32(1),
                    tid == 0,
                )
            if recurrence_token == 3:
                _wait_gdr_dependencies(
                    mailbox.poll, projected_ready3, recurrence_task, tid
                )
                emit_gdr_decode_task(
                    *_gdr_projection_views(
                        args3["projected"],
                        args3["dt_bias"],
                        args3["A_log"],
                        args3["read_indices"],
                        args3["write_indices"],
                        args3["state"],
                        args3["core"],
                    ),
                    fx.Int32(1),
                    tid,
                    recurrence_task,
                )
                publish(
                    mailbox.put,
                    recurrence_ready3,
                    recurrence_task,
                    fx.Int32(1),
                    tid == 0,
                )
    else:
        worker = bid - 2 * GDR_TASKS
        for round_id in range_constexpr(GDR_Z_TASKS // (2 * (BLOCKS - 2 * GDR_TASKS))):
            z_task = worker + (BLOCKS - 2 * GDR_TASKS) * round_id
            produce(CONV_TASKS + z_task)
    if bid < 2 * GDR_TASKS:
        produce(CONV_TASKS + GDR_Z_TASKS // 2 + bid)

    head_ready0 = sreg(ready0, GDR_FULL_HEAD_READY_OFFSET, "gdn_s4_head_ready_0")
    head_ready1 = sreg(ready1, GDR_FULL_HEAD_READY_OFFSET, "gdn_s4_head_ready_1")
    head_ready2 = sreg(ready2, GDR_FULL_HEAD_READY_OFFSET, "gdn_s4_head_ready_2")
    head_ready3 = sreg(ready3, GDR_FULL_HEAD_READY_OFFSET, "gdn_s4_head_ready_3")
    if bid < 4 * GDN_VALUE_HEADS:
        head_token = bid // GDN_VALUE_HEADS
        head = bid % GDN_VALUE_HEADS
        if head_token == 0:
            _publish_gdr_head_ready(
                mailbox.poll,
                mailbox.put,
                recurrence_ready0,
                projected_ready0,
                head_ready0,
                head,
                tid,
            )
        if head_token == 1:
            _publish_gdr_head_ready(
                mailbox.poll,
                mailbox.put,
                recurrence_ready1,
                projected_ready1,
                head_ready1,
                head,
                tid,
            )
        if head_token == 2:
            _publish_gdr_head_ready(
                mailbox.poll,
                mailbox.put,
                recurrence_ready2,
                projected_ready2,
                head_ready2,
                head,
                tid,
            )
        if head_token == 3:
            _publish_gdr_head_ready(
                mailbox.poll,
                mailbox.put,
                recurrence_ready3,
                projected_ready3,
                head_ready3,
                head,
                tid,
            )

    norm_ready0 = sreg(ready0, GDR_FULL_NORM_READY_OFFSET, "gdn_s4_norm_ready_0")
    norm_ready1 = sreg(ready1, GDR_FULL_NORM_READY_OFFSET, "gdn_s4_norm_ready_1")
    norm_ready2 = sreg(ready2, GDR_FULL_NORM_READY_OFFSET, "gdn_s4_norm_ready_2")
    norm_ready3 = sreg(ready3, GDR_FULL_NORM_READY_OFFSET, "gdn_s4_norm_ready_3")
    if bid < 4 * GDR_NORM_TASKS:
        norm_token = bid // GDR_NORM_TASKS
        norm_task = bid % GDR_NORM_TASKS
        if norm_token == 0:
            _wait_gdr_norm_heads(mailbox.poll, head_ready0, norm_task, tid)
            sigmoid_norm_task(
                args0["core"],
                args0["projected"] + fx.Int64(2 * GDN_CONV_DIM),
                args0["norm_weight"],
                args0["normed"],
                norm_task,
                lane,
                wave,
            )
            publish(mailbox.put, norm_ready0, norm_task, fx.Int32(1), tid == 0)
        if norm_token == 1:
            _wait_gdr_norm_heads(mailbox.poll, head_ready1, norm_task, tid)
            sigmoid_norm_task(
                args1["core"],
                args1["projected"] + fx.Int64(2 * GDN_CONV_DIM),
                args1["norm_weight"],
                args1["normed"],
                norm_task,
                lane,
                wave,
            )
            publish(mailbox.put, norm_ready1, norm_task, fx.Int32(1), tid == 0)
        if norm_token == 2:
            _wait_gdr_norm_heads(mailbox.poll, head_ready2, norm_task, tid)
            sigmoid_norm_task(
                args2["core"],
                args2["projected"] + fx.Int64(2 * GDN_CONV_DIM),
                args2["norm_weight"],
                args2["normed"],
                norm_task,
                lane,
                wave,
            )
            publish(mailbox.put, norm_ready2, norm_task, fx.Int32(1), tid == 0)
        if norm_token == 3:
            _wait_gdr_norm_heads(mailbox.poll, head_ready3, norm_task, tid)
            sigmoid_norm_task(
                args3["core"],
                args3["projected"] + fx.Int64(2 * GDN_CONV_DIM),
                args3["norm_weight"],
                args3["normed"],
                norm_task,
                lane,
                wave,
            )
            publish(mailbox.put, norm_ready3, norm_task, fx.Int32(1), tid == 0)

    partial_ready0 = sreg(
        ready0, GDR_FULL_PARTIAL_READY_OFFSET, "gdn_s4_output_partial_ready_0"
    )
    partial_ready1 = sreg(
        ready1, GDR_FULL_PARTIAL_READY_OFFSET, "gdn_s4_output_partial_ready_1"
    )
    partial_ready2 = sreg(
        ready2, GDR_FULL_PARTIAL_READY_OFFSET, "gdn_s4_output_partial_ready_2"
    )
    partial_ready3 = sreg(
        ready3, GDR_FULL_PARTIAL_READY_OFFSET, "gdn_s4_output_partial_ready_3"
    )
    output_ready0 = sreg(ready0, GDR_FULL_OUTPUT_READY_OFFSET, "gdn_s4_output_ready_0")
    output_ready1 = sreg(ready1, GDR_FULL_OUTPUT_READY_OFFSET, "gdn_s4_output_ready_1")
    output_ready2 = sreg(ready2, GDR_FULL_OUTPUT_READY_OFFSET, "gdn_s4_output_ready_2")
    output_ready3 = sreg(ready3, GDR_FULL_OUTPUT_READY_OFFSET, "gdn_s4_output_ready_3")
    output_group_ready0 = sreg(
        ready0, GDR_FULL_OUTPUT_GROUP_READY_OFFSET, "gdn_s4_output_group_ready_0"
    )
    output_group_ready1 = sreg(
        ready1, GDR_FULL_OUTPUT_GROUP_READY_OFFSET, "gdn_s4_output_group_ready_1"
    )
    output_group_ready2 = sreg(
        ready2, GDR_FULL_OUTPUT_GROUP_READY_OFFSET, "gdn_s4_output_group_ready_2"
    )
    output_group_ready3 = sreg(
        ready3, GDR_FULL_OUTPUT_GROUP_READY_OFFSET, "gdn_s4_output_group_ready_3"
    )
    output_cache_modifier = CM_DEV if publish_output else 0
    if bid < GDR_OUT_DIRECT_TASKS:
        first_weight_ops = output_projection_loads(
            args0["output_weight"],
            bid,
            0,
            S4_OUTPUT_PREFETCH_STEPS,
            lane,
            wave,
        )
        _wait_gdr_norms(mailbox.poll, norm_ready0, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready1, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready2, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready3, tid)
        output_projection_direct_task_s4(
            args0["normed"],
            args1["normed"],
            args2["normed"],
            args3["normed"],
            args0["out"],
            args1["out"],
            args2["out"],
            args3["out"],
            bid,
            tid,
            lane,
            wave,
            reduction,
            args0["output_weight"],
            first_weight_ops,
            output_cache_modifier,
            after_output_pair0,
            after_output_pair1,
            after_output_pair2,
            after_output_pair3,
        )
        if const_expr(publish_output):
            publish(mailbox.put, output_ready0, bid, fx.Int32(1), tid == 0)
            publish(mailbox.put, output_ready1, bid, fx.Int32(1), tid == 0)
            publish(mailbox.put, output_ready2, bid, fx.Int32(1), tid == 0)
            publish(mailbox.put, output_ready3, bid, fx.Int32(1), tid == 0)
    else:
        partial_task = bid - GDR_OUT_DIRECT_TASKS
        split = partial_task // GDR_OUT_SPLIT_TILES
        tile = GDR_OUT_DIRECT_TASKS + partial_task % GDR_OUT_SPLIT_TILES
        first_weight_ops = output_projection_loads(
            args0["output_weight"],
            tile,
            split * GDR_OUT_SPLIT_K,
            S4_OUTPUT_PREFETCH_STEPS,
            lane,
            wave,
        )
        _wait_gdr_norms(mailbox.poll, norm_ready0, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready1, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready2, tid)
        _wait_gdr_norms(mailbox.poll, norm_ready3, tid)
        output_projection_partial_task_s4(
            args0["normed"],
            args1["normed"],
            args2["normed"],
            args3["normed"],
            args0["partials"],
            args1["partials"],
            args2["partials"],
            args3["partials"],
            partial_ready0,
            partial_ready1,
            partial_ready2,
            partial_ready3,
            mailbox.put,
            partial_task,
            tid,
            lane,
            wave,
            reduction,
            args0["output_weight"],
            first_weight_ops,
        )
        if bid < GDR_OUT_TASKS:
            output_projection_finish(
                args0["partials"],
                partial_ready0,
                mailbox.poll,
                args0["out"],
                bid,
                tid,
                output_cache_modifier,
                after_output_pair0,
            )
            output_projection_finish(
                args1["partials"],
                partial_ready1,
                mailbox.poll,
                args1["out"],
                bid,
                tid,
                output_cache_modifier,
                after_output_pair1,
            )
            output_projection_finish(
                args2["partials"],
                partial_ready2,
                mailbox.poll,
                args2["out"],
                bid,
                tid,
                output_cache_modifier,
                after_output_pair2,
            )
            output_projection_finish(
                args3["partials"],
                partial_ready3,
                mailbox.poll,
                args3["out"],
                bid,
                tid,
                output_cache_modifier,
                after_output_pair3,
            )
            if const_expr(publish_output):
                publish(mailbox.put, output_ready0, bid, fx.Int32(1), tid == 0)
                publish(mailbox.put, output_ready1, bid, fx.Int32(1), tid == 0)
                publish(mailbox.put, output_ready2, bid, fx.Int32(1), tid == 0)
                publish(mailbox.put, output_ready3, bid, fx.Int32(1), tid == 0)

    if const_expr(publish_output):
        _publish_gdr_output_groups(
            mailbox.poll,
            mailbox.put,
            output_ready0,
            output_group_ready0,
            bid,
            tid,
        )
        _publish_gdr_output_groups(
            mailbox.poll,
            mailbox.put,
            output_ready1,
            output_group_ready1,
            bid,
            tid,
        )
        _publish_gdr_output_groups(
            mailbox.poll,
            mailbox.put,
            output_ready2,
            output_group_ready2,
            bid,
            tid,
        )
        _publish_gdr_output_groups(
            mailbox.poll,
            mailbox.put,
            output_ready3,
            output_group_ready3,
            bid,
            tid,
        )


def compose_gdn(composition):
    """Give a caller the production GDR task and the complete GDN emitter."""

    def compose_port(
        *,
        module_name,
        emit_gdr_decode_task,
        block_threads,
        tasks_per_batch,
    ):
        if block_threads != THREADS:
            raise ValueError(
                f"GDR task needs {block_threads} threads, expected {THREADS}"
            )
        if tasks_per_batch != GDR_TASKS:
            raise ValueError(
                f"GDR task count is {tasks_per_batch}, expected {GDR_TASKS}"
            )
        return composition(
            module_name=module_name,
            emit_gdr_decode_task=emit_gdr_decode_task,
            run_gdn=run_gdn,
            run_gdn_s2=run_gdn_s2,
            run_gdn_s4=run_gdn_s4,
        )

    return _build_gdr_port(compose_port)


def _build_gdr_port(composition):
    """Specialize AITER's recurrence emitter for the production Qwen shape."""

    q_stride = (GDN_KEY_DIM, GDN_KEY_DIM, GDN_HEAD_DIM, 1)
    v_stride = (GDN_VALUE_DIM, GDN_VALUE_DIM, GDN_HEAD_DIM, 1)
    gate_stride = (GDN_VALUE_HEADS, GDN_VALUE_HEADS, 1)
    # The Mono cache binder stores physical VK and exposes a logical KV view.
    # The raw address here is therefore the contiguous VK base consumed by the
    # AITER emitter, matching gdn_flydsl.decode(state.transpose(-1, -2)).
    state_stride = (
        GDN_VALUE_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM,
        GDN_HEAD_DIM * GDN_HEAD_DIM,
        GDN_HEAD_DIM,
        1,
    )
    return create_vk_gdr_decode_kernel(
        "bf16",
        "bf16",
        "f32",
        1,
        GDN_KEY_HEADS,
        GDN_VALUE_HEADS,
        GDN_HEAD_DIM,
        GDN_HEAD_DIM,
        q_stride,
        q_stride,
        v_stride,
        state_stride,
        gate_stride,
        gate_stride,
        True,
        NUM_BLOCKS_PER_V_DIM=GDR_V_SPLITS,
        NUM_WARPS=WAVES,
        WARP_THREADS_K=16,
        _composition=composition,
        _composition_output_cache_modifier=CM_DEV,
        _composition_round_beta_bf16=True,
    )
