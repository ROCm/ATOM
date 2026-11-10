# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023, F821
"""Architecture-separated gfx1250 Kimi K1 stage and protocol callables.

Its wave32 compute, device-mailbox operations, system epoch operations, and
LDS type emitter are selected through the gfx1250 K1 builder.
"""

import math
from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.kernels.attention.layout import (
    GATE_K,
    HIDDEN,
    ROWS,
    KdaMatrixContract,
)
from atom.models.kimi_k3.mono.kernels.attention.protocol import ATTENTION_CTAS
from atom.models.kimi_k3.mono.kernels.stages.gemv_gfx1250 import (
    ACCUMULATOR_ELEMENTS,
    _fragment,
    accumulator_offset,
)
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    K_PER_CHUNK,
    PACK_ELEMENTS,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    mma_bf16,
    rsrc,
    traced,
    uniform,
    wave_max,
)
from atom.mono.plan.architecture import GFX1250

LOG2E = 1.0 / math.log(2.0)
HEADS = 12
HEAD_DIM = GATE_K
STATE_CHUNKS = 8
VALUE_CHUNK = 16
NPROJ = 6288
SEAM_COLUMNS = 128
SEAM_COLUMN_GROUPS = 32
SEAM_SLICE_GROUPS = 8
SEAM_SLICES_PER_GROUP = 7
PROJ = HEADS * HEAD_DIM
G2_0 = 3 * PROJ
BETA_0 = G2_0 + PROJ
FA_0 = BETA_0 + HEADS

assert SEAM_SLICE_GROUPS * SEAM_SLICES_PER_GROUP == ATTENTION_CTAS


def zero_accumulator():
    return fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32)


def _scratch_region(c, name):
    return c["scratch"] + fx.Int64(c[f"{name}_offset"])


def _load_bf(pointer, index):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(
                rsrc(pointer),
                index,
                vec_width=1,
                dtype=T.bf16,
            )
        )
    )


def gemv_loads(c, row_group, groups):
    """Load selected wave groups of KDA in-projection weight fragments."""

    lane, wave, waves = c["lane"], c["wave"], c["matrix_waves"]
    row = fx.min(row_group, c["gemv_tasks"] - 1) * ROWS + lane % ROWS
    lane_group = lane // ROWS
    resource = rsrc(c["args"]["w_in"], c["fused_rows"] * HIDDEN * 2)
    result = []
    for group in groups:
        chunk = wave + waves * group
        base = (row * HIDDEN + chunk * K_PER_CHUNK) // 2
        result.append(
            [
                _fragment(
                    lambda offset: bo.buffer_load(
                        resource,
                        base + offset,
                        vec_width=PACK_ELEMENTS // 2,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    ),
                    lane_group,
                    step,
                )
                for step in range(4)
            ]
        )
    return result


@traced
def gemv_slab(c, operands, groups: int, accumulator):
    """Advance a WMMA accumulator through one staged KDA projection slab."""

    s, lane, wave, waves = (
        c["S"],
        c["lane"],
        c["wave"],
        c["matrix_waves"],
    )
    lane_group = lane // ROWS
    token = fx.min(lane % ROWS, s - 1)
    row = c["xrow"]
    for group in range_constexpr(groups):
        chunk_k = (wave + waves * group) * K_PER_CHUNK
        for step in range_constexpr(4):
            activation = _fragment(
                lambda offset: fx.ptr_load(
                    c["xl"] + (token * row + chunk_k) // 2 + offset,
                    result_type=fx.Vector.make_type(PACK_ELEMENTS // 2, fx.Int32),
                ),
                lane_group,
                step,
            )
            accumulator = mma_bf16(
                operands[group][step], activation, accumulator, GFX1250
            )
    return accumulator


def store_accumulator(accumulator, red, wave, lane):
    fx.ptr_store(accumulator, red + accumulator_offset(wave, lane))


def row_sum(red, row, token, waves):
    lane = token + ROWS * (row // ACCUMULATOR_ELEMENTS)
    element = row % ACCUMULATOR_ELEMENTS
    total = fx.Float32(0.0)
    for wave in range(waves):
        total = total + fx.ptr_load(red + accumulator_offset(wave, lane) + element)
    return total


def gate_loads(c, head):
    """Load this wave's 16 f_b rows and matching eight-element bias fragment."""

    lane, wave = c["lane"], c["wave"]
    lane_group = lane // ROWS
    resource = rsrc(c["args"]["w_fb"])
    row = head * GATE_K + wave * ROWS + lane % ROWS
    base = row * GATE_K // 2
    fragments = [
        _fragment(
            lambda offset: bo.buffer_load(
                resource,
                base + offset,
                vec_width=PACK_ELEMENTS // 2,
                dtype=T.i32,
            ),
            lane_group,
            step,
        )
        for step in range(4)
    ]
    bias_row = wave * ROWS + lane_group * ACCUMULATOR_ELEMENTS
    bias = [
        fx.Float32(
            bo.buffer_load(
                rsrc(c["args"]["dt_bias"]),
                head * GATE_K + bias_row + element,
                vec_width=1,
                dtype=T.f32,
            )
        )
        for element in range(ACCUMULATOR_ELEMENTS)
    ]
    return fragments, bias


@traced
def gate_compute(c, k):
    """Compute and store this wave's f_b x f_a gate rows with WMMA."""

    L, lane, wave = c["L"], c["lane"], c["wave"]
    lane_group = lane // ROWS
    token = fx.min(lane % ROWS, L - 1)
    accumulator = zero_accumulator()
    for step in range_constexpr(4):
        activation = _fragment(
            lambda offset: fx.Vector.from_elements(
                [
                    fx.ptr_load(c["fa"] + token * GATE_K + 2 * offset + element)
                    for element in range(PACK_ELEMENTS)
                ],
                fx.Float32,
            )
            .to(fx.BFloat16)
            .bitcast(fx.Int32),
            lane_group,
            step,
        )
        accumulator = mma_bf16(k["fb"][step], activation, accumulator, GFX1250)
    for element in range_constexpr(ACCUMULATOR_ELEMENTS):
        output = wave * ROWS + lane_group * ACCUMULATOR_ELEMENTS + element
        value = fx.Float32(accumulator[element]).to(fx.BFloat16)
        g = fx.Float32(value) + k["dtb"][element]
        sigmoid = fx.Float32(1.0) / (
            fx.Float32(1.0) + hw_exp2(k["aa"] * g * fx.Float32(-LOG2E))
        )
        gate = fx.Float32(c["lb"]) * sigmoid
        if lane % ROWS < L:
            fx.ptr_store(
                hw_exp2(gate * fx.Float32(LOG2E)),
                c["decay"] + token * GATE_K + output,
            )


def state_loads(c, k, head, chunk):
    """Load the two state rows serially owned by each wave32 lane."""

    lane, wave = c["lane"], c["wave"]
    args = c["args"]
    base = (
        args["rstate"] + fx.Int64(fx.max(k["slot0"], 0)) * fx.Int64(args["rs_seq"]) * 4
    )
    k["states"] = [
        fx.Vector(
            bo.buffer_load(
                rsrc(base),
                (
                    (
                        head * GATE_K
                        + chunk * ROWS
                        + wave
                        + c["matrix_waves"] * round_index
                    )
                    * GATE_K
                    + lane * 4
                ),
                vec_width=4,
                dtype=T.f32,
            )
        )
        for round_index in range(2)
    ]


@traced
def qk_normalize(c):
    """Normalize 128-column Q/K rows with four columns per wave32 lane.

    ``p0 + p1`` reproduces the gfx950 offset-32 reduction before the remaining
    offset-16 through offset-1 butterfly steps.
    """

    L, lane, wave, waves = (
        c["L"],
        c["lane"],
        c["wave"],
        c["matrix_waves"],
    )
    for round_index in range_constexpr((2 * L + waves - 1) // waves):
        row = fx.min(wave + waves * round_index, 2 * L - 1)
        values = [
            fx.ptr_load(c["qkl"] + row * GATE_K + lane + 32 * group)
            for group in range(4)
        ]
        partial0 = values[0] * values[0] + values[2] * values[2]
        partial1 = values[1] * values[1] + values[3] * values[3]
        square_sum = butterfly(
            partial0 + partial1,
            (16, 8, 4, 2, 1),
            architecture=GFX1250,
        )
        scale = hw_rsq(square_sum + fx.Float32(1e-6))
        scale = (row % 2 == 0).select(scale * fx.Float32(GATE_K**-0.5), scale)
        if wave + waves * round_index < 2 * L:
            for group in range_constexpr(4):
                fx.ptr_store(
                    values[group] * scale,
                    c["qkl"] + row * GATE_K + lane + 32 * group,
                )


@traced
def kda_conv_inputs(c, request, head, chunk):
    """Land projection, f_a, and beta before holding convolution metadata."""

    L, tid = c["L"], c["tid"]
    projected = _scratch_region(c, "proj")
    n_fa = L * GATE_K
    fa_rounds = (n_fa + c["matrix_threads"] - 1) // c["matrix_threads"]
    raw_indices = [
        tid + c["matrix_threads"] * iteration for iteration in range_constexpr(fa_rounds)
    ]
    fa_indices = [
        raw_index
        if const_expr(n_fa % c["matrix_threads"] == 0)
        else fx.min(raw_index, n_fa - 1)
        for raw_index in raw_indices
    ]
    # Clamped so that every thread polls only published pairs; the values
    # are stored below only by the threads that own them.
    beta_token = fx.min(tid, L - 1)
    channel_thread = fx.min(tid, 3 * VALUE_CHUNK - 1)
    channel = (
        (channel_thread // VALUE_CHUNK) * PROJ
        + head * HEAD_DIM
        + chunk * VALUE_CHUNK
        + channel_thread % VALUE_CHUNK
    )
    values = mailbox_poll_many_device(
        [
            (
                projected,
                (request * L + index // GATE_K) * NPROJ + FA_0 + index % GATE_K,
            )
            for index in fa_indices
        ]
        + [(projected, (request * L + beta_token) * NPROJ + BETA_0 + head)]
        + [
            (projected, (request * L + token) * NPROJ + channel)
            for token in range_constexpr(L)
        ],
        c["tag"],
    )
    for iteration in range_constexpr(fa_rounds):
        value = values[iteration]
        if const_expr(n_fa % c["matrix_threads"] == 0):
            fx.ptr_store(_f32_of(value), c["fa"] + fa_indices[iteration])
        else:
            if raw_indices[iteration] < n_fa:
                fx.ptr_store(_f32_of(value), c["fa"] + fa_indices[iteration])
    if tid < L:
        beta = _f32_of(values[fa_rounds])
        fx.ptr_store(
            fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(-beta * LOG2E)),
            c["beta"] + tid,
        )

    if tid < 3 * VALUE_CHUNK:
        for token in range_constexpr(L):
            fx.ptr_store(
                _f32_of(values[fa_rounds + 1 + token]),
                c["qkl"] + tid * L + token,
            )
    gpu.barrier()


@traced
def kda_conv_compute(c, k, request, head, chunk):
    """Consume landed projection with the owned channel's conv metadata."""

    L, tid = c["L"], c["tid"]
    if (tid < 3 * VALUE_CHUNK) & (k["slot_c"] >= 0):
        state = rsrc(k["cs"])
        for token in range_constexpr(c["SL"]):
            if const_expr(token < c["SL"] - L):
                    value = k["old"][token]
            else:
                value = fx.ptr_load(
                    c["qkl"] + tid * L + token - (c["SL"] - L)
                )
            bo.buffer_store(
                fx.Float32(value).to(fx.BFloat16),
                state,
                k["co"] + token * c["args"]["cs_tok"],
            )
    if tid < 3 * VALUE_CHUNK:
        for token in range_constexpr(L):
            value = fx.Float32(0.0)
            for tap in range_constexpr(4):
                sequence_index = token + tap
                if const_expr(sequence_index < 3):
                    sequence_value = k["win"][sequence_index]
                else:
                    sequence_value = fx.ptr_load(
                        c["qkl"] + tid * L + sequence_index - 3
                    )
                value = value + k["cw"][tap] * sequence_value
            output = _bf16_round(
                value
                * (
                    fx.Float32(1.0)
                    / (fx.Float32(1.0) + hw_exp2(-value * LOG2E))
                )
            )
            if k["part"] < 2:
                mailbox_store_device(
                    _scratch_region(c, "qk"),
                    ((request * HEADS + head) * L + token) * 2 * HEAD_DIM
                    + k["part"] * HEAD_DIM
                    + chunk * VALUE_CHUNK
                    + tid % VALUE_CHUNK,
                    output,
                    c["tag"],
                )
            if k["part"] == 2:
                fx.ptr_store(
                    output,
                    c["vloc"] + token * VALUE_CHUNK + tid % VALUE_CHUNK,
                )
    gpu.barrier()


@traced
def kda_head_norm(c, request, head):
    """Finish a whole head from CTA-local partial norms."""

    L, tid, threads = c["L"], c["tid"], c["matrix_threads"]
    outputs = L * HEAD_DIM
    rounds = (outputs + threads - 1) // threads
    items = [
        fx.min(tid + threads * round_index, outputs - 1)
        for round_index in range_constexpr(rounds)
    ]
    gates = mailbox_poll_many_device(
        [
            (
                _scratch_region(c, "proj"),
                (request * L + item // HEAD_DIM) * NPROJ
                + G2_0
                + head * HEAD_DIM
                + item % HEAD_DIM,
            )
            for item in items
        ],
        c["tag"],
    )
    for round_index in range_constexpr(rounds):
        item = items[round_index]
        token = item // HEAD_DIM
        row = item % HEAD_DIM
        square_sum = fx.Float32(0.0)
        for chunk in range_constexpr(STATE_CHUNKS):
            square_sum = square_sum + fx.ptr_load(
                c["head_onrm"] + token * STATE_CHUNKS + chunk
            )
        inverse = hw_rsq(
            square_sum * fx.Float32(1.0 / HEAD_DIM)
            + fx.Float32(c["onorm_eps"])
        )
        weight = fx.Float32(
            bo.buffer_load(
                rsrc(c["args"]["on_w"]),
                row,
                vec_width=1,
                dtype=T.f32,
            )
        )
        output = (
            fx.ptr_load(c["ol"] + token * HEAD_DIM + row)
            * inverse
            * weight
            / (fx.Float32(1.0) + hw_exp2(-_f32_of(gates[round_index]) * LOG2E))
        )
        if tid + threads * round_index < outputs:
            bo.buffer_store(
                output.to(fx.BFloat16),
                rsrc(c["args"]["core_out"]),
                (request * L + token) * PROJ + head * HEAD_DIM + row,
            )
    gpu.barrier()


@traced
def kda_recur_head(c, k, request, head):
    """Advance a whole head: wave ``w`` owns chunk ``w``'s sixteen state rows.

    Every row is an independent recurrence over the head's q/k/decay, so one
    token loop advances all of a wave's rows and their reduction chains
    overlap; partial output norms stay CTA-local."""

    L, lane, wave = c["L"], c["lane"], c["wave"]
    args = c["args"]
    live = k["slot0"] >= 0
    column = lane * 4
    row0 = wave * ROWS
    source = rsrc(
        args["rstate"] + fx.Int64(fx.max(k["slot0"], 0)) * fx.Int64(args["rs_seq"]) * 4
    )
    loaded = [
        fx.Vector(
            bo.buffer_load(
                source,
                (head * GATE_K + row0 + row) * GATE_K + column,
                vec_width=4,
                dtype=T.f32,
            )
        )
        for row in range(ROWS)
    ]
    state = [
        live.select(loaded[row][element], fx.Float32(0.0))
        for row in range(ROWS)
        for element in range(4)
    ]
    token = fx.Int32(0)
    while token < L:
        q = [
            fx.ptr_load(c["qkl"] + (2 * token) * GATE_K + column + element)
            for element in range(4)
        ]
        key = [
            fx.ptr_load(c["qkl"] + (2 * token + 1) * GATE_K + column + element)
            for element in range(4)
        ]
        decay = [
            fx.ptr_load(c["decay"] + token * GATE_K + column + element)
            for element in range(4)
        ]
        beta = fx.ptr_load(c["beta"] + token)
        # Every lane holds the same slot; a VGPR one would put each state
        # store below in its own waterfall loop over the descriptor.
        slot = uniform(fx.ptr_load(c["slots"] + token))
        updated = []
        outputs = []
        for row in range_constexpr(ROWS):
            row_state = [state[4 * row + element] * decay[element] for element in range(4)]
            projection = (
                row_state[0] * key[0]
                + row_state[1] * key[1]
                + row_state[2] * key[2]
                + row_state[3] * key[3]
            )
            projection = butterfly(projection, (16, 8, 4, 2, 1), architecture=GFX1250)
            value = (
                fx.ptr_load(c["vloc"] + token * HEAD_DIM + row0 + row) - projection
            ) * beta
            row_state = [row_state[element] + value * key[element] for element in range(4)]
            output = (
                row_state[0] * q[0]
                + row_state[1] * q[1]
                + row_state[2] * q[2]
                + row_state[3] * q[3]
            )
            output = _bf16_round(
                butterfly(output, (16, 8, 4, 2, 1), architecture=GFX1250)
            )
            updated.extend(row_state)
            outputs.append(live.select(output, fx.Float32(0.0)))
        state = updated
        if (slot >= 0) & live:
            destination = rsrc(
                args["rstate"] + fx.Int64(slot) * fx.Int64(args["rs_seq"]) * 4
            )
            for row in range_constexpr(ROWS):
                bo.buffer_store(
                    fx.Vector.from_elements(state[4 * row : 4 * row + 4], fx.Float32),
                    destination,
                    (head * GATE_K + row0 + row) * GATE_K + column,
                )
        square_sum = fx.Float32(0.0)
        for row in range_constexpr(ROWS):
            square_sum = square_sum + outputs[row] * outputs[row]
        if lane == 0:
            for row in range_constexpr(ROWS):
                fx.ptr_store(outputs[row], c["ol"] + token * HEAD_DIM + row0 + row)
            fx.ptr_store(square_sum, c["head_onrm"] + token * STATE_CHUNKS + wave)
        token = token + fx.Int32(1)
    gpu.barrier()


@traced
def kda_head(c, request, head):
    """One plain-decode request/head without cross-CTA QK/ONRM handoffs."""

    L, tid = c["L"], c["tid"]
    args = c["args"]
    acc_idx = uniform(ld_i32(args["num_acc"], request)) - 1
    slot_c = uniform(ld_i32(args["st_idx"], request * args["st_stride"]))
    slot0 = uniform(
        ld_i32(
            args["st_idx"],
            request * args["st_stride"] + acc_idx,
        )
    )
    if tid < L:
        fx.ptr_store(
            ld_i32(args["st_idx"], request * args["st_stride"] + tid),
            c["slots"] + tid,
        )

    projected = _scratch_region(c, "proj")
    threads = c["matrix_threads"]
    n_fa = L * HEAD_DIM
    fa_rounds = (n_fa + threads - 1) // threads
    channels = 3 * HEAD_DIM
    channel_rounds = (channels + threads - 1) // threads
    fa_raw = [tid + threads * iteration for iteration in range_constexpr(fa_rounds)]
    fa_index = [fx.min(raw, n_fa - 1) for raw in fa_raw]
    channel_raw = [
        tid + threads * round_index for round_index in range_constexpr(channel_rounds)
    ]
    channel_of = [
        (fx.min(raw, channels - 1) // HEAD_DIM) * PROJ
        + head * HEAD_DIM
        + fx.min(raw, channels - 1) % HEAD_DIM
        for raw in channel_raw
    ]
    # Clamped indices keep every polled pair a published one; each value is
    # stored below only by the thread that owns it.
    polled = mailbox_poll_many_device(
        [
            (
                projected,
                (request * L + index // HEAD_DIM) * NPROJ + FA_0 + index % HEAD_DIM,
            )
            for index in fa_index
        ]
        + [(projected, (request * L + fx.min(tid, L - 1)) * NPROJ + BETA_0 + head)]
        + [
            (projected, (request * L + token) * NPROJ + channel)
            for channel in channel_of
            for token in range_constexpr(L)
        ],
        c["tag"],
    )
    for iteration in range_constexpr(fa_rounds):
        if fa_raw[iteration] < n_fa:
            fx.ptr_store(_f32_of(polled[iteration]), c["fa"] + fa_index[iteration])
    if tid < L:
        beta = _f32_of(polled[fa_rounds])
        fx.ptr_store(
            fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(-beta * LOG2E)),
            c["beta"] + tid,
        )

    for round_index in range_constexpr(channel_rounds):
        raw = channel_raw[round_index]
        local_channel = fx.min(raw, channels - 1)
        part = local_channel // HEAD_DIM
        column = local_channel % HEAD_DIM
        channel = part * PROJ + head * HEAD_DIM + column
        state = (
            args["conv_st"]
            + fx.Int64(fx.max(slot_c, 0)) * fx.Int64(args["cs_seq"]) * 2
        )
        state_offset = channel * args["cs_dim"]
        weights = fx.Vector(
            bo.buffer_load(
                rsrc(args["conv_w"]),
                channel * 4,
                vec_width=4,
                dtype=T.f32,
            )
        )
        window = [
            _load_bf(
                state,
                state_offset + (acc_idx + tap) * args["cs_tok"],
            )
            for tap in range(3)
        ]
        new_values = [
            _f32_of(polled[fa_rounds + 1 + round_index * L + token])
            for token in range(L)
        ]
        if (raw < channels) & (slot_c >= 0):
            resource = rsrc(state)
            for token in range_constexpr(c["SL"]):
                value = (
                    _load_bf(
                        state,
                        state_offset
                        + (acc_idx + 1 + token) * args["cs_tok"],
                    )
                    if token < c["SL"] - L
                    else new_values[token - (c["SL"] - L)]
                )
                bo.buffer_store(
                    value.to(fx.BFloat16),
                    resource,
                    state_offset + token * args["cs_tok"],
                )
        sequence = window + new_values
        for token in range_constexpr(L):
            value = fx.Float32(0.0)
            for tap in range_constexpr(4):
                value = value + weights[tap] * sequence[token + tap]
            output = _bf16_round(
                value
                / (fx.Float32(1.0) + hw_exp2(-value * LOG2E))
            )
            if raw < channels:
                if part < 2:
                    fx.ptr_store(
                        output,
                        c["qkl"]
                        + (2 * token + part) * HEAD_DIM
                        + column,
                    )
                if part == 2:
                    fx.ptr_store(
                        output,
                        c["vloc"] + token * HEAD_DIM + column,
                    )
    gpu.barrier()

    metadata = {}
    metadata["fb"], metadata["dtb"] = gate_loads(c, head)
    metadata["aa"] = hw_exp2(
        fx.Float32(
            bo.buffer_load(
                rsrc(args["a_log"]),
                head,
                vec_width=1,
                dtype=T.f32,
            )
        )
        * LOG2E
    )
    gate_compute(c, metadata)
    gpu.barrier()
    qk_normalize(c)
    gpu.barrier()

    metadata["slot0"] = slot0
    c["head_onrm"] = c["head_onrm_lds"]
    kda_recur_head(c, metadata, request, head)
    kda_head_norm(c, request, head)


@traced
def kda_gates(c, k, request, head):
    """Poll complete Q/K rows, then run the wave32 gate and norm backend."""

    L, tid = c["L"], c["tid"]
    elements = L * 2 * HEAD_DIM
    indices = [
        tid + c["matrix_threads"] * iteration
        for iteration in range_constexpr(
            (elements + c["matrix_threads"] - 1) // c["matrix_threads"]
        )
    ]
    qk = _scratch_region(c, "qk")
    base = (request * HEADS + head) * L * 2 * HEAD_DIM
    values = mailbox_poll_many_device(
        [(qk, base + index) for index in indices],
        c["tag"],
    )
    for iteration in range_constexpr(len(indices)):
        fx.ptr_store(_f32_of(values[iteration]), c["qkl"] + indices[iteration])
    gpu.barrier()
    gate_compute(c, k)
    gpu.barrier()
    qk_normalize(c)
    gpu.barrier()


def _bf16_round(value):
    return fx.Float32(fx.Float32(value).to(fx.BFloat16))


@traced
def kda_recur(c, k, req, head, chunk):
    """Run a wave32's two state rows over the tokens with 32-lane reductions.

    The rows are independent recurrences over the same q/k/decay, so one
    token loop advances both and their reduction chains overlap."""

    L, tid, lane = c["L"], c["tid"], c["lane"]
    args = c["args"]
    live = k["slot0"] >= 0
    rows = [c["wave"] + c["matrix_waves"] * row_index for row_index in range(2)]
    head_offsets = [
        (head * GATE_K + chunk * ROWS + state_row) * GATE_K + lane * 4
        for state_row in rows
    ]
    state = [
        live.select(k["states"][row_index][element], fx.Float32(0.0))
        for row_index in range(2)
        for element in range(4)
    ]
    column = lane * 4
    token = fx.Int32(0)
    while token < L:
        q = [
            fx.ptr_load(c["qkl"] + (2 * token) * GATE_K + column + element)
            for element in range(4)
        ]
        key = [
            fx.ptr_load(c["qkl"] + (2 * token + 1) * GATE_K + column + element)
            for element in range(4)
        ]
        decay = [
            fx.ptr_load(c["decay"] + token * GATE_K + column + element)
            for element in range(4)
        ]
        beta = fx.ptr_load(c["beta"] + token)
        # Every lane holds the same slot; a VGPR one would put each state
        # store below in its own waterfall loop over the descriptor.
        slot = uniform(fx.ptr_load(c["slots"] + token))
        updated = []
        outputs = []
        for row_index in range_constexpr(2):
            row_state = [
                state[4 * row_index + element] * decay[element]
                for element in range(4)
            ]
            projection = (
                row_state[0] * key[0]
                + row_state[1] * key[1]
                + row_state[2] * key[2]
                + row_state[3] * key[3]
            )
            projection = butterfly(
                projection,
                (16, 8, 4, 2, 1),
                architecture=GFX1250,
            )
            value = (
                fx.ptr_load(c["vloc"] + token * ROWS + rows[row_index]) - projection
            ) * beta
            row_state = [
                row_state[element] + value * key[element] for element in range(4)
            ]
            output = (
                row_state[0] * q[0]
                + row_state[1] * q[1]
                + row_state[2] * q[2]
                + row_state[3] * q[3]
            )
            output = _bf16_round(
                butterfly(
                    output,
                    (16, 8, 4, 2, 1),
                    architecture=GFX1250,
                )
            )
            updated.extend(row_state)
            outputs.append(live.select(output, fx.Float32(0.0)))
        state = updated
        if (slot >= 0) & live:
            destination = rsrc(
                args["rstate"] + fx.Int64(slot) * fx.Int64(args["rs_seq"]) * 4
            )
            for row_index in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(
                        state[4 * row_index : 4 * row_index + 4], fx.Float32
                    ),
                    destination,
                    head_offsets[row_index],
                )
        if lane == 0:
            for row_index in range_constexpr(2):
                fx.ptr_store(
                    outputs[row_index], c["ol"] + token * ROWS + rows[row_index]
                )
        token = token + fx.Int32(1)
    gpu.barrier()
    if tid < L:
        square_sum = fx.Float32(0.0)
        for element in range_constexpr(ROWS):
            value = fx.ptr_load(c["ol"] + tid * ROWS + element)
            square_sum = square_sum + value * value
        if const_expr(c.get("head_onrm") is not None):
            fx.ptr_store(
                square_sum,
                c["head_onrm"] + tid * STATE_CHUNKS + chunk,
            )
        else:
            mailbox_store_device(
                _scratch_region(c, "onrm"),
                ((req * HEADS + head) * L + tid) * STATE_CHUNKS + chunk,
                square_sum,
                c["tag"],
            )
    gpu.barrier()


@traced
def _kda_norm_row(c, request, head, chunk, token, element):
    L = c["L"]
    row = chunk * VALUE_CHUNK + element
    onrm = _scratch_region(c, "onrm")
    values = mailbox_poll_many_device(
        [
            (onrm, ((request * HEADS + head) * L + token) * STATE_CHUNKS + other)
            for other in range_constexpr(STATE_CHUNKS)
        ]
        + [
            (
                _scratch_region(c, "proj"),
                (request * L + token) * NPROJ + G2_0 + head * HEAD_DIM + row,
            )
        ],
        c["tag"],
    )
    square_sum = fx.Float32(0.0)
    for other in range_constexpr(STATE_CHUNKS):
        square_sum = square_sum + _f32_of(values[other])
    gate = _f32_of(values[STATE_CHUNKS])
    inverse = hw_rsq(
        square_sum * fx.Float32(1.0 / HEAD_DIM)
        + fx.Float32(c["onorm_eps"])
    )
    weight = fx.Float32(
        bo.buffer_load(
            rsrc(c["args"]["on_w"]),
            row,
            vec_width=1,
            dtype=T.f32,
        )
    )
    output = (
        fx.ptr_load(c["ol"] + token * VALUE_CHUNK + element)
        * inverse
        * weight
        * (fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(-gate * LOG2E)))
    )
    bo.buffer_store(
        output.to(fx.BFloat16),
        rsrc(c["args"]["core_out"]),
        (request * L + token) * PROJ + head * HEAD_DIM + row,
    )


@traced
def kda_norm(c, request, head, chunk):
    """Finish one wave32 chunk from tagged partial norms and output gates."""

    L, tid = c["L"], c["tid"]
    if const_expr(L == 16):
        _kda_norm_row(
            c,
            request,
            head,
            chunk,
            tid // VALUE_CHUNK,
            tid % VALUE_CHUNK,
        )
    else:
        if tid < L * VALUE_CHUNK:
            _kda_norm_row(
                c,
                request,
                head,
                chunk,
                tid // VALUE_CHUNK,
                tid % VALUE_CHUNK,
            )
    gpu.barrier()


def attention_reduce32(value):
    """Sum one complete wave32 in the gfx950 subgroup order."""

    return butterfly(value, (16, 8, 4, 2, 1), architecture=GFX1250)


def workgroup_barrier():
    """Architecture-neutral workgroup barrier used by gfx1250 AITER kernels."""

    gpu.barrier()


def wait_vector_stores():
    """Drain gfx12 stores before publishing a mailbox tag."""

    rocdl.s_wait_storecnt(0)


def device_fence_acquire():
    """Order bulk device-scope payload loads after a ready-tag observation."""

    fx.memory_fence(
        ordering=fx.AtomicOrdering.Acquire,
        syncscope="agent",
    )


def mailbox_store_device(pointer, index, payload, tag):
    """Publish one payload/tag pair through gfx12's L2 coherence point."""

    bits = (
        payload.bitcast(fx.Int32)
        if isinstance(payload, fx.Float32)
        else fx.Int32(payload)
    )
    pair = fx.Vector.from_elements([bits, fx.Int32(tag)], fx.Int32)
    bo.buffer_store(
        pair,
        rsrc(pointer),
        fx.Int32(index) * 2,
        cache_modifier=CM_DEV,
    )


def mailbox_load_device(pointer, index):
    """Load one sibling-CTA payload/tag pair at gfx12 device scope."""

    return fx.Vector(
        bo.buffer_load(
            rsrc(pointer),
            fx.Int32(index) * 2,
            vec_width=2,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )


@traced
def mailbox_poll_device(pointer, index, expected_tag):
    """Poll one device-scope pair until its publication tag is current."""

    pair = mailbox_load_device(pointer, index)
    while pair[1] != fx.Int32(expected_tag):
        rocdl.s_sleep(1)
        pair = mailbox_load_device(pointer, index)
    return pair[0]


@traced
def mailbox_poll_many_device(items, expected_tag):
    """Poll ``(pointer, index)`` pairs with every load in flight at once.

    Polling them one by one costs one dependent L2 round trip per pair even
    when all of them are already published."""

    def stale(pairs):
        flags = [pair[1] != fx.Int32(expected_tag) for pair in pairs]
        result = flags[0]
        for flag in flags[1:]:
            result = result | flag
        return result

    pairs = [mailbox_load_device(pointer, index) for pointer, index in items]
    while stale(pairs):
        rocdl.s_sleep(1)
        pairs = [mailbox_load_device(pointer, index) for pointer, index in items]
    return [pair[0] for pair in pairs]


def mailbox_reset_device(pointer, index):
    """Reset one device-scope mailbox pair; caller owns synchronization."""

    mailbox_store_device(pointer, index, fx.Int32(0), fx.Int32(0))


@traced
def mailbox_publish_device(pointer, index, payload, tag, who):
    """Drain CTA data stores, rendezvous, then publish from one thread."""

    wait_vector_stores()
    workgroup_barrier()
    if who:
        mailbox_store_device(pointer, index, payload, tag)


def epoch_store_system_release(address, value):
    """System-scope release store used for an epoch handoff."""

    comm_ops.store_i32_global_system_release(address, value)


def epoch_load_system_acquire(address):
    """System-scope acquire load used to observe an epoch handoff."""

    return comm_ops.load_i32_global_system(address, acquire=True)


def epoch_fence_system_release():
    """Release prior stores at the checked-in gfx1250 system scope."""

    comm_ops.fence_system_release()


def epoch_fence_system_acquire():
    """Acquire system-visible stores at the checked-in gfx1250 scope."""

    comm_ops.fence_system_acquire()


def epoch_arrive_system_atomic(address, value):
    """System-scope atomic arrival, returning the old counter value."""

    return comm_ops.atomic_add_system(address, value)


@dataclass(frozen=True)
class RoleLdsTypes:
    """FlyDSL classes emitted from a host-validated K1 LDS type spec."""

    smem: object
    role: object
    attention: object
    projection: object
    recurrence: object


def emit_role_lds_types(spec):
    """Emit gfx1250 K1 LDS types from ``K1LdsTypeSpec``.

    The type spec is validated against the byte planner before this device
    module is imported.  Keep these declarations separate from the gfx950
    builder so its source and type layout remain unchanged.
    """

    smem = {field.name: field for field in spec.smem}
    attention = {field.name: field for field in spec.variant("attention")}
    projection = {field.name: field for field in spec.variant("projection")}
    recurrence = {field.name: field for field in spec.variant("recurrence")}
    expected = (
        set(smem),
        set(attention),
        set(projection),
        set(recurrence),
    )
    if expected != (
        {"red", "tls"},
        {"vals"},
        {"xl"},
        {
            "fa",
            "qkl",
            "decay",
            "vloc",
            "ol",
            "beta",
            "slots",
            "onrm",
        },
    ):
        raise ValueError("unexpected gfx1250 K1 LDS type specification")

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, attention["vals"].elements, 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, projection["xl"].elements, 16]

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, recurrence["fa"].elements, 16]
        qkl: fx.Array[fx.Float32, recurrence["qkl"].elements, 16]
        decay: fx.Array[fx.Float32, recurrence["decay"].elements, 16]
        vloc: fx.Array[fx.Float32, recurrence["vloc"].elements, 16]
        ol: fx.Array[fx.Float32, recurrence["ol"].elements, 16]
        beta: fx.Array[fx.Float32, recurrence["beta"].elements, 16]
        slots: fx.Array[fx.Int32, recurrence["slots"].elements, 16]
        onrm: fx.Array[fx.Float32, recurrence["onrm"].elements, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds
        kda: KdaLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, smem["red"].elements, 16]
        tls: fx.Array[fx.Int64, smem["tls"].elements, 16]

    return RoleLdsTypes(Smem, RoleLds, AttnLds, GemvLds, KdaLds)


def _f32_of(word):
    return word.bitcast(fx.Float32)


def _sum_polled(values):
    total = _f32_of(values[0][0])
    for index in range(1, len(values)):
        total = total + _f32_of(values[index][0])
    return total


def _load_bf4(pointer, index):
    words = fx.Vector(
        bo.buffer_load(rsrc(pointer), index // 2, vec_width=2, dtype=T.i32)
    )
    packed = words.bitcast(fx.BFloat16)
    return [fx.Float32(packed[element]) for element in range(4)]


def _bf4_words(values):
    return fx.Vector.from_elements(values, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


@traced
def _attention_gate(c, token, token0, total_tokens, resident):
    """Wave32 softmax over one tile's at most 32 AttnRes sources."""

    sources, tid, lane, wave, red = (
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    global_token = token0 + token
    pairs = sources * 2
    if tid < SEAM_SLICE_GROUPS * pairs:
        group = tid // pairs
        pair = tid % pairs
        values = []
        for slice_index in range_constexpr(SEAM_SLICES_PER_GROUP):
            index = (
                (
                    (group * SEAM_SLICES_PER_GROUP + slice_index)
                    * total_tokens
                    + global_token
                )
                * sources
                * 2
                + pair
            )
            if const_expr(resident):
                word = mailbox_poll_device(c["part"], index, c["tag"])
            else:
                word = c["poll"]([(c["part"], index, 1)])[0][0]
            values.append(word.bitcast(fx.Float32))
        fx.ptr_store(sum(values), red + group * pairs + pair)
    gpu.barrier()
    if wave == 0:
        source = fx.min(lane, sources - 1)
        square_sum = fx.ptr_load(red + source * 2)
        dot = fx.ptr_load(red + source * 2 + 1)
        for group in range_constexpr(1, SEAM_SLICE_GROUPS):
            square_sum = square_sum + fx.ptr_load(red + group * pairs + source * 2)
            dot = dot + fx.ptr_load(red + group * pairs + source * 2 + 1)
        inverse_norm = hw_rsq(
            square_sum * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"])
        )
        logit = dot * inverse_norm * fx.Float32(LOG2E)
        logit = (lane < sources).select(logit, fx.Float32(-3.0e38))
        maximum = wave_max(logit, GFX1250)
        exponential = (lane < sources).select(hw_exp2(logit - maximum), fx.Float32(0.0))
        denominator = butterfly(
            exponential,
            (16, 8, 4, 2, 1),
            architecture=GFX1250,
        )
        if lane < sources:
            index = global_token * (sources + 1) + lane
            if const_expr(resident):
                mailbox_store_device(c["wgt"], index, exponential, c["tag"])
            else:
                c["put"](c["wgt"], index, exponential)
        if lane == 0:
            index = global_token * (sources + 1) + sources
            if const_expr(resident):
                mailbox_store_device(c["wgt"], index, denominator, c["tag"])
            else:
                c["put"](c["wgt"], index, denominator)
    gpu.barrier()


@traced
def attention_gate(c, token):
    _attention_gate(c, token, 0, c["S"], False)


@traced
def attention_gate_tile(c, token):
    _attention_gate(c, token, c["token0"], c["total_S"], True)


@traced
def _attention_mix(c, task, token0, total_tokens, tile, resident):
    """Mix one tile's values with token-global mailbox and output indices."""

    s, sources, tid, lane, wave, waves, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["matrix_waves"],
        c["red"],
    )
    args = c["args"]
    column0 = task * SEAM_COLUMNS
    threads = c["matrix_threads"]
    weight_count = s * (sources + 1)
    for iteration in range_constexpr((weight_count + threads - 1) // threads):
        item = tid + threads * iteration
        if item < weight_count:
            local_token = item // (sources + 1)
            pair = item % (sources + 1)
            index = (token0 + local_token) * (sources + 1) + pair
            if const_expr(resident):
                value = mailbox_poll_device(c["wgt"], index, c["tag"])
            else:
                value = c["poll"]([(c["wgt"], index, 1)])[0][0]
            fx.ptr_store(_f32_of(value), red + item)
    gpu.barrier()

    rounds = (s + waves - 1) // waves
    tokens = []
    lives = []
    mixed_rounds = []
    for round_index in range_constexpr(rounds):
        token = fx.min(wave + waves * round_index, s - 1)
        live = wave + waves * round_index < s
        column = lane * 4
        mixed = []
        for element in range_constexpr(4):
            total = fx.Float32(0.0)
            for source in range_constexpr(sources):
                total = total + fx.ptr_load(
                    red + token * (sources + 1) + source
                ) * fx.ptr_load(
                    c["vals"]
                    + (token * sources + source) * SEAM_COLUMNS
                    + column
                    + element
                )
            mixed.append(total / fx.ptr_load(red + token * (sources + 1) + sources))
        square_sum = (
            mixed[0] * mixed[0]
            + mixed[1] * mixed[1]
            + mixed[2] * mixed[2]
            + mixed[3] * mixed[3]
        )
        square_sum = attention_reduce32(square_sum)
        if live & (lane == 0):
            index = task * total_tokens + token0 + token
            if const_expr(resident):
                mailbox_store_device(c["msq"], index, square_sum, c["tag"])
            else:
                c["put"](c["msq"], index, square_sum)
        if const_expr(c["mix_lds_handoff"]):
            for element in range_constexpr(4):
                fx.ptr_store(
                    mixed[element],
                    c["vals"] + (token * sources) * SEAM_COLUMNS + lane * 4 + element,
                )
        else:
            tokens.append(token)
            lives.append(live)
            mixed_rounds.append(mixed)
    gpu.barrier()

    group_work = SEAM_SLICE_GROUPS * s
    for iteration in range_constexpr((group_work + threads - 1) // threads):
        item = tid + threads * iteration
        if item < group_work:
            group = item // s
            token = item % s
            values = []
            for slice_index in range_constexpr(SEAM_SLICES_PER_GROUP):
                index = (
                    (group * SEAM_SLICES_PER_GROUP + slice_index)
                    * total_tokens
                    + token0
                    + token
                )
                if const_expr(resident):
                    word = mailbox_poll_device(c["msq"], index, c["tag"])
                else:
                    word = c["poll"]([(c["msq"], index, 1)])[0][0]
                values.append(word.bitcast(fx.Float32))
            fx.ptr_store(
                sum(values),
                red + 128 + group * s + token,
            )
    gpu.barrier()

    for round_index in range_constexpr(rounds):
        if const_expr(c["mix_lds_handoff"]):
            token = fx.min(wave + waves * round_index, s - 1)
            live = wave + waves * round_index < s
            mixed = [
                fx.ptr_load(
                    c["vals"] + (token * sources) * SEAM_COLUMNS + lane * 4 + element
                )
                for element in range_constexpr(4)
            ]
        else:
            token = tokens[round_index]
            live = lives[round_index]
            mixed = mixed_rounds[round_index]
        total = fx.ptr_load(red + 128 + token)
        for group in range_constexpr(1, SEAM_SLICE_GROUPS):
            total = total + fx.ptr_load(red + 128 + group * s + token)
        inverse_norm = hw_rsq(
            total * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"])
        )
        norm_weight = _load_bf4(args["in_nw"], column0 + lane * 4)
        output = _bf4_words(
            [
                mixed[element] * inverse_norm * norm_weight[element]
                for element in range(4)
            ]
        )
        if live:
            at = ((token0 + token) * HIDDEN + column0 + lane * 4) // 2
            bo.buffer_store(output, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
            if const_expr(c["debug"]):
                bo.buffer_store(output, rsrc(args["x_dbg"]), at)
    wait_vector_stores()
    workgroup_barrier()
    if tid == 0:
        index = tile * ATTENTION_CTAS + task
        if const_expr(resident):
            mailbox_store_device(c["xrdy"], index, fx.Int32(1), c["tag"])
        else:
            c["put"](c["xrdy"], index, fx.Int32(1))
    gpu.barrier()


@traced
def attention_mix(c, task):
    _attention_mix(c, task, 0, c["S"], 0, False)


@traced
def attention_mix_tile(c, task):
    _attention_mix(c, task, c["token0"], c["total_S"], c["tile"], True)


@dataclass(frozen=True)
class StageBackend:
    """KDA matrix API bound to an explicit eight-wave gfx1250 contract."""

    contract: KdaMatrixContract

    def __post_init__(self):
        if self.contract.architecture is not GFX1250:
            raise ValueError("gfx1250 KDA backend needs GFX1250")
        if self.contract.waves != 8:
            raise ValueError("gfx1250 KDA backend currently needs eight waves")

    @property
    def waves(self):
        return self.contract.waves

    @staticmethod
    def zero_accumulator():
        return zero_accumulator()

    @staticmethod
    def gemv_loads(c, row_group, groups):
        return gemv_loads(c, row_group, groups)

    @staticmethod
    def gemv_slab(c, operands, groups, accumulator):
        return gemv_slab(c, operands, groups, accumulator)

    @staticmethod
    def store_accumulator(accumulator, red, wave, lane):
        store_accumulator(accumulator, red, wave, lane)

    def row_sum(self, red, row, token):
        return row_sum(red, row, token, self.waves)

    @staticmethod
    def gate_loads(c, head):
        return gate_loads(c, head)

    @staticmethod
    def gate_compute(c, k):
        gate_compute(c, k)

    @staticmethod
    def state_loads(c, k, head, chunk):
        state_loads(c, k, head, chunk)

    @staticmethod
    def qk_normalize(c):
        qk_normalize(c)

    @staticmethod
    def kda_conv_inputs(c, request, head, chunk):
        kda_conv_inputs(c, request, head, chunk)

    @staticmethod
    def kda_conv_compute(c, k, request, head, chunk):
        kda_conv_compute(c, k, request, head, chunk)

    @staticmethod
    def kda_head(c, request, head):
        kda_head(c, request, head)

    @staticmethod
    def kda_gates(c, k, request, head):
        kda_gates(c, k, request, head)

    @staticmethod
    def kda_recur(c, k, req, head, chunk):
        kda_recur(c, k, req, head, chunk)

    @staticmethod
    def kda_norm(c, request, head, chunk):
        kda_norm(c, request, head, chunk)

    @staticmethod
    def attention_reduce32(value):
        return attention_reduce32(value)

    @staticmethod
    def attention_gate(c, token):
        attention_gate(c, token)

    @staticmethod
    def attention_gate_tile(c, token):
        attention_gate_tile(c, token)

    @staticmethod
    def attention_mix(c, task):
        attention_mix(c, task)

    @staticmethod
    def attention_mix_tile(c, task):
        attention_mix_tile(c, task)

    @staticmethod
    def workgroup_barrier():
        workgroup_barrier()

    @staticmethod
    def wait_vector_stores():
        wait_vector_stores()

    @staticmethod
    def device_fence_acquire():
        device_fence_acquire()

    @staticmethod
    def mailbox_store_device(pointer, index, payload, tag):
        mailbox_store_device(pointer, index, payload, tag)

    @staticmethod
    def mailbox_load_device(pointer, index):
        return mailbox_load_device(pointer, index)

    @staticmethod
    def mailbox_poll_device(pointer, index, expected_tag):
        return mailbox_poll_device(pointer, index, expected_tag)

    @staticmethod
    def mailbox_reset_device(pointer, index):
        mailbox_reset_device(pointer, index)

    @staticmethod
    def mailbox_publish_device(pointer, index, payload, tag, who):
        mailbox_publish_device(pointer, index, payload, tag, who)

    @staticmethod
    def epoch_store_system_release(address, value):
        epoch_store_system_release(address, value)

    @staticmethod
    def epoch_load_system_acquire(address):
        return epoch_load_system_acquire(address)

    @staticmethod
    def epoch_fence_system_release():
        epoch_fence_system_release()

    @staticmethod
    def epoch_fence_system_acquire():
        epoch_fence_system_acquire()

    @staticmethod
    def epoch_arrive_system_atomic(address, value):
        return epoch_arrive_system_atomic(address, value)

    @staticmethod
    def emit_role_lds_types(spec):
        return emit_role_lds_types(spec)
