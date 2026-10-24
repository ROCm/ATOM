# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K2a output consumers, with caller-owned scratch and publications.

Attention supplies BF16 [rows, heads, 512]. Inverse RoPE operates in FP32
with the model's FP32 tables, and publishes contiguous group32 MXFP8 rows.
wo_a preserves its BF16 result before quantizing it for wo_b. These device
bodies are also usable after a fused attention PV/combine stage.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.kernels.gfx1250_gemv import (
    projection_load,
    projection_ready,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_mhc import _typed
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import bf_hi, bf_lo, fp8_pack4, traced

HEAD_DIM, GROUP_K, O_RANK, HIDDEN = 512, 4096, 1024, 5120
WOA_TILE = 32
# Current AITER inverse_rope_group_quant uses kAmaxFloor=1e-10; ATOM's
# downstream quantize_fp8 still uses 1e-4. Do not copy the older gfx950 IRQ
# floor or assume the two native quantizers share a floor.
IRQ_FLOOR, MID_FLOOR = 1e-10, 1e-4


def woa_tasks(heads):
    return heads // 8 * O_RANK // WOA_TILE


def inverse_rope_pair(even, odd, column, position, cos, sin):
    """FP32 inverse rotation of one aligned BF16 attention pair."""
    table = position * 32 + fx.max(column - (HEAD_DIM - 64), 0) // 2
    c = fx.ptr_load(_typed(cos, fx.Float32) + table)
    s = fx.ptr_load(_typed(sin, fx.Float32) + table)
    rotary = column >= HEAD_DIM - 64
    return (
        rotary.select(fx.fma(odd, s, even * c), even),
        rotary.select(fx.fma(-even, s, odd * c), odd),
    )


@traced
def inverse_rope_quant_device(
    attention, positions, cos, sin, quant, codes, tid, task, rows, heads
):
    """One CTA per token/head; caller validates positions before entering.

    Each of the first 128 threads owns four consecutive elements. Groups of
    eight lanes reduce one group32 amax. Non-rotary values retain their BF16
    input; rotary values are quantized from FP32, without an extra BF16 round.
    The final workgroup barrier permits one release publication per head.
    """
    row = task // heads
    source = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(attention), num_records_bytes=rows * heads * HEAD_DIM * 2
    )
    target = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(quant), num_records_bytes=rows * heads * HEAD_DIM
    )
    scales = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(codes), num_records_bytes=rows * heads * HEAD_DIM // 32
    )
    position = fx.ptr_load(_typed(positions, fx.Int64) + row)
    if tid < HEAD_DIM // 4:
        column = tid * 4
        at = task * HEAD_DIM + column
        words = [
            fx.Int32(bo.buffer_load(source, at // 2 + j, vec_width=1, dtype=T.i32))
            for j in range(2)
        ]
        values = [bf_lo(words[0]), bf_hi(words[0]), bf_lo(words[1]), bf_hi(words[1])]
        for j in range_constexpr(2):
            values[j * 2], values[j * 2 + 1] = inverse_rope_pair(
                values[j * 2], values[j * 2 + 1], column + j * 2, position, cos, sin
            )
        maximum = fx.Float32(IRQ_FLOOR)
        for value in values:
            maximum = fx.max(maximum, abs(value))
        for offset in (1, 2, 4):
            maximum = fx.max(maximum, maximum.shuffle_xor(offset, 32))
        code = code_ceil(maximum * fx.Float32(1.0 / FP8_MAX))
        multiplier = rcp_pow2(code)
        bo.buffer_store(
            fp8_pack4(*[clamp_fp8(value * multiplier) for value in values]),
            target,
            at // 4,
        )
        if tid % 8 == 0:
            bo.buffer_store(code.to(fx.Int8), scales, task * 16 + tid // 8)
    fx.gpu.barrier()


def woa_weight_load(weight, scale, tid, task):
    group, tile = task // (O_RANK // WOA_TILE), task % (O_RANK // WOA_TILE)
    w = _typed(weight, fx.Uint8) + group * O_RANK * GROUP_K
    ws = _typed(scale, fx.Uint8) + group * (O_RANK * GROUP_K // 1024)
    return projection_load(w, ws, tid, tile * 2, GROUP_K, O_RANK)


@traced
def woa_quant_device(
    quant, codes, weight, scale, mid, mid_quant, mid_codes, lds, tid, task, rows, heads, prefetched=None
):
    """Produce one complete N32 group, then publish its BF16/group32 outputs.

    The group-specific views keep the complete input/output token strides.
    Weight FP8 bytes are native (16,16)-preshuffled; weight scales remain raw
    32x32 blocks. Two N16 computations reuse LDS before the N32 quantizer.
    """
    groups = heads // 8
    group, tile = task // (O_RANK // WOA_TILE), task % (O_RANK // WOA_TILE)
    x = _typed(quant, fx.Uint8) + group * GROUP_K
    xs = _typed(codes, fx.Uint8) + group * (GROUP_K // 32)
    w = _typed(weight, fx.Uint8) + group * O_RANK * GROUP_K
    ws = _typed(scale, fx.Uint8) + group * (O_RANK * GROUP_K // 1024)
    y = _typed(mid, fx.BFloat16) + group * O_RANK
    for half in range_constexpr(2):
        tile16 = tile * 2 + half
        ops = (prefetched if const_expr(half == 0 and prefetched is not None)
               else projection_load(w, ws, tid, tile16, GROUP_K, O_RANK))
        projection_ready(
            x,
            xs,
            w,
            ws,
            mid,
            mid,
            y,
            y,
            ops,
            lds,
            tid,
            tile16,
            rows,
            GROUP_K,
            O_RANK,
            input_stride=groups * GROUP_K,
            output_stride=groups * O_RANK,
        )
    source = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(mid), num_records_bytes=rows * groups * O_RANK * 2
    )
    target = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(mid_quant), num_records_bytes=rows * groups * O_RANK
    )
    scales = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(mid_codes), num_records_bytes=rows * groups * O_RANK // 32
    )
    if tid < rows * 8:
        row, lane = tid // 8, tid % 8
        at = row * groups * O_RANK + task * WOA_TILE + lane * 4
        words = [
            fx.Int32(bo.buffer_load(source, at // 2 + j, vec_width=1, dtype=T.i32))
            for j in range(2)
        ]
        values = [bf_lo(words[0]), bf_hi(words[0]), bf_lo(words[1]), bf_hi(words[1])]
        maximum = fx.Float32(MID_FLOOR)
        for value in values:
            maximum = fx.max(maximum, abs(value))
        for offset in (1, 2, 4):
            maximum = fx.max(maximum, maximum.shuffle_xor(offset, 32))
        code = code_ceil(maximum * fx.Float32(1.0 / FP8_MAX))
        multiplier = rcp_pow2(code)
        bo.buffer_store(
            fp8_pack4(*[clamp_fp8(value * multiplier) for value in values]),
            target,
            at // 4,
        )
        if lane == 0:
            bo.buffer_store(code.to(fx.Int8), scales, row * woa_tasks(heads) + task)
    fx.gpu.barrier()


@traced
def wob_device(quant, codes, weight, scale, out, lds, tid, task, rows, heads, prefetched=None):
    inner = heads // 8 * O_RANK
    ops = (prefetched if const_expr(prefetched is not None)
           else projection_load(weight, scale, tid, task, inner, HIDDEN))
    projection_ready(
        quant,
        codes,
        weight,
        scale,
        out,
        out,
        out,
        out,
        ops,
        lds,
        tid,
        task,
        rows,
        inner,
        HIDDEN,
    )
