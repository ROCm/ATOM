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
    mma_bf16,
    rsrc,
    traced,
    wave_max,
)
from atom.mono.plan.architecture import GFX1250

LOG2E = 1.0 / math.log(2.0)
HEADS = 12
STATE_CHUNKS = 8
SEAM_COLUMNS = 128
SEAM_COLUMN_GROUPS = 32
SEAM_SLICE_GROUPS = 8
SEAM_SLICES_PER_GROUP = 7


def zero_accumulator():
    return fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32)


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
    k["state_rows"] = [
        wave + c["matrix_waves"] * round_index for round_index in range(2)
    ]
    k["head_offs"] = [
        ((head * GATE_K + chunk * ROWS + state_row) * GATE_K + lane * 4)
        for state_row in k["state_rows"]
    ]
    k["states"] = [
        fx.Vector(bo.buffer_load(rsrc(base), offset, vec_width=4, dtype=T.f32))
        for offset in k["head_offs"]
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


def _bf16_round(value):
    return fx.Float32(fx.Float32(value).to(fx.BFloat16))


@traced
def kda_recur(c, k, req, head, chunk):
    """Run two serial state rows per wave32 with 32-lane reductions."""

    L, tid, lane = c["L"], c["tid"], c["lane"]
    args = c["args"]
    live = k["slot0"] >= 0
    for row_index in range_constexpr(2):
        state_row = k["state_rows"][row_index]
        state = [
            live.select(k["states"][row_index][element], fx.Float32(0.0))
            for element in range(4)
        ]
        column = lane * 4
        for token in range_constexpr(L):
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
            state = [state[element] * decay[element] for element in range(4)]
            projection = (
                state[0] * key[0]
                + state[1] * key[1]
                + state[2] * key[2]
                + state[3] * key[3]
            )
            projection = butterfly(
                projection,
                (16, 8, 4, 2, 1),
                architecture=GFX1250,
            )
            value = (
                fx.ptr_load(c["vloc"] + token * ROWS + state_row) - projection
            ) * fx.ptr_load(c["beta"] + token)
            state = [state[element] + value * key[element] for element in range(4)]
            output = (
                state[0] * q[0] + state[1] * q[1] + state[2] * q[2] + state[3] * q[3]
            )
            output = _bf16_round(
                butterfly(
                    output,
                    (16, 8, 4, 2, 1),
                    architecture=GFX1250,
                )
            )
            output = live.select(output, fx.Float32(0.0))
            slot = k["slots"][token]
            if (slot >= 0) & live:
                destination = rsrc(
                    args["rstate"] + fx.Int64(slot) * fx.Int64(args["rs_seq"]) * 4
                )
                bo.buffer_store(
                    fx.Vector.from_elements(state, fx.Float32),
                    destination,
                    k["head_offs"][row_index],
                )
            if lane == 0:
                fx.ptr_store(output, c["ol"] + token * ROWS + state_row)
    gpu.barrier()
    if tid < L:
        square_sum = fx.Float32(0.0)
        for element in range_constexpr(ROWS):
            value = fx.ptr_load(c["ol"] + tid * ROWS + element)
            square_sum = square_sum + value * value
        c["put"](
            c["onrm"],
            ((req * HEADS + head) * L + tid) * STATE_CHUNKS + chunk,
            square_sum,
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
        {"fa", "qkl", "decay", "vloc", "ol", "beta"},
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
def attention_gate(c, token):
    """Wave32 softmax over at most 32 AttnRes sources."""

    s, sources, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    pairs = sources * 2
    if tid < SEAM_SLICE_GROUPS * pairs:
        group = tid // pairs
        pair = tid % pairs
        values = c["poll"](
            [
                (
                    c["part"],
                    (
                        ((group * SEAM_SLICES_PER_GROUP + slice_index) * s + token)
                        * sources
                    )
                    * 2
                    + pair,
                    1,
                )
                for slice_index in range(SEAM_SLICES_PER_GROUP)
            ]
        )
        fx.ptr_store(_sum_polled(values), red + group * pairs + pair)
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
            c["put"](c["wgt"], token * (sources + 1) + lane, exponential)
        if lane == 0:
            c["put"](c["wgt"], token * (sources + 1) + sources, denominator)
    gpu.barrier()


@traced
def attention_mix(c, task):
    """Mix values with one or more token rows serially owned by each wave32."""

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
            value = c["poll"]([(c["wgt"], item, 1)])[0][0]
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
            c["put"](c["msq"], task * s + token, square_sum)
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
            values = c["poll"](
                [
                    (
                        c["msq"],
                        (group * SEAM_SLICES_PER_GROUP + slice_index) * s + token,
                        1,
                    )
                    for slice_index in range(SEAM_SLICES_PER_GROUP)
                ]
            )
            fx.ptr_store(
                _sum_polled(values),
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
            at = (token * HIDDEN + column0 + lane * 4) // 2
            bo.buffer_store(output, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
            if const_expr(c["debug"]):
                bo.buffer_store(output, rsrc(args["x_dbg"]), at)
    wait_vector_stores()
    workgroup_barrier()
    if tid == 0:
        c["put"](c["xrdy"], task, fx.Int32(1))
    gpu.barrier()


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
    def kda_recur(c, k, req, head, chunk):
        kda_recur(c, k, req, head, chunk)

    @staticmethod
    def attention_reduce32(value):
        return attention_reduce32(value)

    @staticmethod
    def attention_gate(c, token):
        attention_gate(c, token)

    @staticmethod
    def attention_mix(c, task):
        attention_mix(c, task)

    @staticmethod
    def workgroup_barrier():
        workgroup_barrier()

    @staticmethod
    def wait_vector_stores():
        wait_vector_stores()

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
