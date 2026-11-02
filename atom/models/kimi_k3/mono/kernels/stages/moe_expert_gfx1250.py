# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 scaled-WMMA Kimi K2 expert gate/up and down callables."""

import math
from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.kernels.stages.moe_expert_layout import (
    GFX1250_EXPERT,
    K_STEP,
    LATENT,
    ROUTED_INTERMEDIATE,
    TOPK,
    ExpertTaskContract,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_layout import UG_GROUPS
from atom.mono.device.ops import (
    bf16_round,
    hw_exp2,
    hw_rcp,
    rsrc,
    traced,
    uniform,
)
from atom.mono.plan.architecture import GFX1250

LOG2E = 1.0 / math.log(2.0)


def _vector(parts):
    return fx.Vector.from_elements(parts, fx.Int32)


def _load_v4(resource, index):
    return fx.Vector(
        bo.buffer_load(
            resource,
            index,
            vec_width=4,
            dtype=T.i32,
        )
    )


def _weight_fragment(pointer, expert, rows, k, row0, step, lane):
    """Load the v16-i32 WMMA weight operand from shuffle_weight bytes."""

    lane16 = lane % 16
    k_parity = lane // 16
    base = expert * (rows * k // 8)
    resource = rsrc(pointer)
    parts = []
    for output_half in range_constexpr(2):
        row_group = (row0 // 16) + output_half
        tile = base + row_group * (2 * k) + step * 256
        for k64_half in range_constexpr(2):
            offset = tile + k64_half * 128 + k_parity * 64 + lane16 * 4
            value = _load_v4(resource, offset)
            parts.extend([value[index] for index in range_constexpr(4)])
    return _vector(parts)


def _weight_scale_word(pointer, expert, rows, k, row, step):
    """Load one gfx1250 n32k4 dword of four K32 weight scales."""

    groups = k // 32
    byte_index = (
        ((expert * (rows // 32) + row // 32) * (groups // 4) + step) * 32 + row % 32
    ) * 4
    return fx.Int32(
        bo.buffer_load(
            rsrc(pointer),
            byte_index // 4,
            vec_width=1,
            dtype=T.i32,
        )
    )


def _activation_fragment(pointer, row_words, token, step, lane):
    """Load row-major FP4 groups (parity, parity+2) into v8-i32."""

    parity = lane // 16
    base = token * row_words + step * 16 + parity * 4
    first = fx.Vector(
        fx.ptr_load(
            pointer + base,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    second = fx.Vector(
        fx.ptr_load(
            pointer + base + 8,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    return _vector(
        [first[index] for index in range_constexpr(4)]
        + [second[index] for index in range_constexpr(4)]
    )


def _activation_scale_word(pointer, row_scales, token, step):
    packed = fx.Int32(0)
    for byte in range_constexpr(4):
        code = fx.ptr_load(pointer + token * row_scales + step * 4 + byte)
        packed = packed | ((code & 0xFF) << (byte * 8))
    return packed


def _zero_accumulator():
    return fx.Vector.filled(16, 0.0, fx.Float32)


def _wmma(weight, activation, accumulator, weight_scale, activation_scale):
    """Exact AITER gfx1250 a4w4 scaled-WMMA intrinsic boundary."""

    return fx.Vector(
        rocdl.wmma_scale_f32_32x16x128_f4(
            T.vec(16, T.f32),
            weight.ir_value(),
            activation.ir_value(),
            accumulator.ir_value(),
            weight_scale,
            activation_scale,
            scaleAType=0,
            scaleBType=0,
        )
    )


def _red_offset(slot, lane):
    return (slot * 32 + lane) * 16


def _tanh(value):
    exponential = hw_exp2(abs(value) * fx.Float32(-2.0 * LOG2E))
    result = (fx.Float32(1.0) - exponential) * hw_rcp(fx.Float32(1.0) + exponential)
    return (value > fx.Float32(0.0)).select(result, -result)


def _situ(gate, up):
    sigmoid = hw_rcp(fx.Float32(1.0) + hw_exp2(gate * fx.Float32(-LOG2E)))
    activated = (fx.Float32(4.0) * _tanh(gate * fx.Float32(0.25))) * sigmoid
    return (activated * fx.Float32(25.0)) * _tanh(up * fx.Float32(0.04))


def _e8m0_roundup(amax):
    bits = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    code = fx.Int32(fx.Uint32(bits + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)) & 0xFF
    return fx.min(code, fx.Int32(254))


def _emit_f32_to_e2m1(value):
    """RNE E2M1 conversion using FlyDSL Numeric types.

    Some pinned AITER images still spell the bitcast operand as the MLIR
    ``T.i32`` type, which current FlyDSL rejects while tracing a newly fused
    caller.  This is the same validated algorithm with the public Numeric
    classes required by that compiler API.
    """

    denorm_bias = fx.Int32(0x4A800000)
    bits = fx.Float32(value).bitcast(fx.Int32)
    sign = bits & fx.Int32(-2147483648)
    absolute = bits & fx.Int32(0x7FFFFFFF)
    absolute_u = fx.Uint32(absolute)
    denormal = absolute_u < fx.Uint32(0x3F800000)
    normal = (absolute_u < fx.Uint32(0x40C00000)) & (
        absolute_u >= fx.Uint32(0x3F800000)
    )
    denormal_f32 = absolute.bitcast(fx.Float32) + denorm_bias.bitcast(fx.Float32)
    denormal_x = denormal_f32.bitcast(fx.Int32) - denorm_bias
    mantissa_odd = fx.Int32((absolute_u >> fx.Uint32(22)) & fx.Uint32(1))
    normal_x = fx.Uint32(absolute + fx.Int32(-1054867457) + mantissa_odd) >> fx.Uint32(
        22
    )
    e2m1 = normal.select(normal_x, fx.Uint32(7))
    e2m1 = denormal.select(fx.Uint32(denormal_x), e2m1)
    return fx.Int32((fx.Uint32(sign) >> fx.Uint32(28)) | fx.Uint32(e2m1))


def _pack8(values, reciprocal_scale):
    packed = fx.Int32(0)
    for element in range_constexpr(8):
        nibble = _emit_f32_to_e2m1(values[element] * reciprocal_scale)
        packed = packed | (nibble << (element * 4))
    return packed


def ug_loads(c, task):
    """Load this wave's gate/up N32 weight operands and n32k4 scales."""

    wave, lane = c["wave"], c["lane"]
    slot = task // UG_GROUPS
    group = task % UG_GROUPS
    expert = uniform(fx.ptr_load(c["rt"]["expert"] + slot))
    half = wave // 4
    steps = [
        half * (LATENT // K_STEP // 2) + index for index in range(LATENT // K_STEP // 2)
    ]
    gate_row = group * 32
    row = ((wave % 4) >= 2).select(
        fx.Int32(ROUTED_INTERMEDIATE + gate_row),
        fx.Int32(gate_row),
    )
    weights = [
        _weight_fragment(
            c["args"]["w13"],
            expert,
            2 * ROUTED_INTERMEDIATE,
            LATENT,
            row,
            step,
            lane,
        )
        for step in steps
    ]
    scales = [
        _weight_scale_word(
            c["args"]["w13s"],
            expert,
            2 * ROUTED_INTERMEDIATE,
            LATENT,
            row + lane,
            step,
        )
        for step in steps
    ]
    return weights, scales


@traced
def stage_ug(c, task, operands):
    """N32 gate/up WMMA, SiTUv2, and one MXFP4 output group."""

    tokens, lane, wave = c["S"], c["lane"], c["wave"]
    slot = task // UG_GROUPS
    group = task % UG_GROUPS
    token = fx.min(lane % 16, tokens - 1)
    live = lane % 16 < tokens
    weights, scales = operands
    half = wave // 4
    steps = [
        half * (LATENT // K_STEP // 2) + index for index in range(LATENT // K_STEP // 2)
    ]
    if wave % 2 == 0:
        accumulator = _zero_accumulator()
        for index in range_constexpr(len(steps)):
            step = steps[index]
            activation = _activation_fragment(c["latq"], LATENT // 8, token, step, lane)
            activation_scale = _activation_scale_word(
                c["latsc"], LATENT // 32, token, step
            )
            accumulator = _wmma(
                weights[index],
                activation,
                accumulator,
                scales[index],
                activation_scale,
            )
        fx.ptr_store(
            accumulator,
            c["expert_red"] + _red_offset(wave // 2, lane),
        )
    gpu.barrier()

    if wave == 0:
        values = []
        for element in range_constexpr(16):
            gate = fx.ptr_load(
                c["expert_red"] + _red_offset(0, lane) + element
            ) + fx.ptr_load(c["expert_red"] + _red_offset(2, lane) + element)
            up = fx.ptr_load(
                c["expert_red"] + _red_offset(1, lane) + element
            ) + fx.ptr_load(c["expert_red"] + _red_offset(3, lane) + element)
            values.append(_situ(gate, up))

        maximum = fx.Float32(0.0)
        for element in range_constexpr(len(values)):
            maximum = fx.max(maximum, abs(values[element]))
        maximum = fx.max(maximum, maximum.shuffle_xor(16, GFX1250.wave_size))
        code = _e8m0_roundup(maximum)
        reciprocal = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        topk = fx.ptr_load(c["rt"]["kof"] + slot * tokens + token)
        if live & (topk >= 0):
            pick = token * TOPK + topk
            parity = lane // 16
            for half in range_constexpr(2):
                packed = _pack8(values[half * 8 : half * 8 + 8], reciprocal)
                location = pick * 48 + group * 4 + parity + 2 * half
                if const_expr(c.get("interq_out") is not None):
                    bo.buffer_store(
                        packed,
                        rsrc(c["interq_out"]),
                        location,
                    )
                else:
                    c["put_words"](
                        c["inter"],
                        location,
                        [packed],
                    )
            if lane < 16:
                location = pick * UG_GROUPS + group
                if const_expr(c.get("intsc_out") is not None):
                    bo.buffer_store(
                        code,
                        rsrc(c["intsc_out"]),
                        location,
                    )
                else:
                    c["put"](c["isc"], location, code)
    gpu.barrier()


@traced
def stage_down(c, task, union_size, waves):
    """N32 down projection; each wave owns disjoint expert-union slots."""

    tokens, lane, wave = c["S"], c["lane"], c["wave"]
    token = fx.min(lane % 16, tokens - 1)
    live_token = lane % 16 < tokens
    row0 = task * 32
    slot = wave
    while slot < union_size:
        expert = uniform(fx.ptr_load(c["rt"]["expert"] + slot))
        topk = fx.ptr_load(c["rt"]["kof"] + slot * tokens + token)
        picked = live_token & (topk >= 0)
        pick = token * TOPK + fx.max(topk, 0)
        accumulator = _zero_accumulator()
        for step in range_constexpr(ROUTED_INTERMEDIATE // K_STEP):
            weight = _weight_fragment(
                c["args"]["w2"],
                expert,
                LATENT,
                ROUTED_INTERMEDIATE,
                row0,
                step,
                lane,
            )
            weight_scale = _weight_scale_word(
                c["args"]["w2s"],
                expert,
                LATENT,
                ROUTED_INTERMEDIATE,
                row0 + lane,
                step,
            )
            activation = _activation_fragment(
                c["interq"],
                ROUTED_INTERMEDIATE // 8,
                pick,
                step,
                lane,
            )
            activation_scale = _activation_scale_word(
                c["intsc"],
                ROUTED_INTERMEDIATE // 32,
                pick,
                step,
            )
            activation = fx.Vector.from_elements(
                [
                    picked.select(activation[element], fx.Int32(0))
                    for element in range_constexpr(8)
                ],
                fx.Int32,
            )
            activation_scale = picked.select(
                activation_scale,
                fx.Int32(0x7F7F7F7F),
            )
            accumulator = _wmma(
                weight,
                activation,
                accumulator,
                weight_scale,
                activation_scale,
            )
        if picked:
            route_weight = fx.ptr_load(
                c["rt"]["route"] + token * 2 * TOPK + TOPK + topk
            ).bitcast(fx.Float32)
            parity = lane // 16
            for element in range_constexpr(16):
                output_half = element // 8
                column = output_half * 16 + parity * 8 + element % 8
                fx.ptr_store(
                    bf16_round(accumulator[element] * route_weight),
                    c["contrib"] + (pick * GFX1250_EXPERT.output_rows + column),
                )
        slot = slot + waves
    gpu.barrier()

    if c["tid"] < tokens * 16:
        output_token = c["tid"] // 16
        pair = c["tid"] % 16
        low = fx.Float32(0.0)
        high = fx.Float32(0.0)
        for topk_index in range_constexpr(TOPK):
            base = (
                output_token * TOPK + topk_index
            ) * GFX1250_EXPERT.output_rows + 2 * pair
            low = bf16_round(low + fx.ptr_load(c["contrib"] + base))
            high = bf16_round(high + fx.ptr_load(c["contrib"] + base + 1))
        if const_expr(c.get("routed_partial") is not None):
            word = (
                fx.Vector.from_elements([low, high], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
            bo.buffer_store(
                word,
                rsrc(c["routed_partial"]),
                (output_token * LATENT + row0 + 2 * pair) // 2,
            )
        else:
            for peer in range_constexpr(8):
                c["put_bf"](
                    c["peer"]["latr"][peer],
                    (c["rank"] * tokens + output_token) * LATENT + row0 + 2 * pair,
                    [low, high],
                )
    gpu.barrier()


@dataclass(frozen=True)
class ExpertBackend:
    tasks: ExpertTaskContract
    token_tiling: object

    def __post_init__(self):
        if self.tasks.waves != 8:
            raise ValueError("gfx1250 expert backend requires eight waves")

    def ug_loads(self, c, task):
        return ug_loads(c, task)

    @staticmethod
    def _require_tile_context(c):
        if c["S"] > 16:
            raise ValueError(
                "gfx1250 expert callables require an M16 tile context " "for S > 16"
            )

    def stage_ug(self, c, task, operands):
        self._require_tile_context(c)
        stage_ug(c, task, operands)

    def stage_down(self, c, task, union_size):
        self._require_tile_context(c)
        stage_down(c, task, union_size, self.tasks.waves)
