# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 wave32/WMMA BF16 GEMV fragments for Kimi mono.

Callers supply the explicit wave count used by their local K1/K2 stage.

For a row-major 16x32 operand tile, lane ``l`` owns row ``l % 16``.  Its half
``l // 16`` selects an eight-BF16 pack, and a second pack 16 K elements later
forms the v16 BF16 operand consumed by ``v_wmma_f32_16x16x32_bf16``.  The v8
f32 result maps lane ``l`` element ``e`` to output
``[8 * (l // 16) + e, l % 16]``.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX1250_GEMV,
    K_PER_CHUNK,
    K_PER_MMA,
    PACK_ELEMENTS,
    ROWS,
)
from atom.mono.device.ops import CM_NT, mma_bf16, rsrc
from atom.mono.plan.architecture import GFX1250

ACCUMULATOR_ELEMENTS = GFX1250_GEMV.accumulator_elements_per_lane


def chunks(k: int, waves: int) -> tuple[int, int]:
    """``(maximum chunks a wave owns, total chunks)`` for K columns."""

    return GFX1250_GEMV.chunks_per_wave(k, waves)


def const_live(per: int, n: int, waves: int) -> bool:
    """Whether a wave's final scheduled chunk may be outside K."""

    return per * waves != n


def _fragment(load_pack, lane_group, mma_step, live=None):
    """Assemble one WMMA v16 BF16 operand from two v4-i32 loads."""

    words = []
    for pack in range_constexpr(GFX1250_GEMV.operand_packs_per_lane):
        element = (
            mma_step * K_PER_MMA
            + lane_group * PACK_ELEMENTS
            + pack * GFX1250_GEMV.operand_pack_stride
        )
        raw = fx.Vector(load_pack(element // 2))
        words.extend(raw[word] for word in range_constexpr(PACK_ELEMENTS // 2))
    if live is not None:
        words = [live.select(word, fx.Int32(0)) for word in words]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def loads(lane, wave, waves, w, row0, k, nt=True, ldk=None, k0=0):
    """Load this wave's row-major weight operands as v16 BF16 fragments.

    The return shape is ``[chunk owned by wave][four K32 WMMA steps]``.
    Clamped chunks are loaded in range; ``mfmas`` masks their activation
    operands to zero when that scheduled chunk is past K.
    """

    per, n = chunks(k, waves)
    ldk = k if ldk is None else ldk
    resource = rsrc(w)
    row = row0 + lane % ROWS
    lane_group = lane // ROWS
    out = []
    for index in range(per):
        chunk = fx.min(wave + waves * index, n - 1)
        base = (row * ldk + k0 + chunk * K_PER_CHUNK) // 2
        out.append(
            [
                _fragment(
                    lambda offset: bo.buffer_load(
                        resource,
                        base + offset,  # noqa: B023 - consumed immediately
                        vec_width=PACK_ELEMENTS // 2,
                        dtype=T.i32,
                        cache_modifier=CM_NT if nt else 0,
                    ),
                    lane_group,
                    mma_step,
                )
                for mma_step in range(GFX1250_GEMV.mma_steps_per_chunk)
            ]
        )
    return out


def fp8x8_to_bf16(words, exponent):
    """Eight FP8 values (two dwords) times the E8M0 ``exponent``'s power of
    two, as BF16: exact, since the scale only moves the exponent."""

    return fx.Vector(
        rocdl.cvt_scale_pk8_bf16_fp8(
            T.vec(PACK_ELEMENTS, T.bf16), as_ir_value(words), as_ir_value(exponent), 0
        )
    )


def fp8_loads(lane, wave, waves, w, scale, row0, k, nt=True):
    """``loads`` of ATOM's online per_1x128 FP8 weight: AITER's gfx1250
    preshuffle (``[rows / 16][k / 32][2][16 rows][16 B]``) with its
    ``[rows / 128, k / 128]`` E8M0 block scales. A wave's chunk is one scale
    block, and ``row0``'s 16 rows share it. The BF16 fragments equal ``loads``'
    of the dequantized weight."""

    assert K_PER_CHUNK == 128
    per, n = chunks(k, waves)
    weights, scales = rsrc(w), rsrc(scale)
    row = lane % ROWS
    lane_group = lane // ROWS
    out = []
    for index in range(per):
        chunk = fx.min(wave + waves * index, n - 1)
        block = row0 // 128 * (k // K_PER_CHUNK) + chunk
        word = bo.buffer_load(scales, block >> 2, vec_width=1, dtype=T.i32)
        exponent = (fx.Int32(word) >> ((block & 3) * 8)) & 0xFF
        steps = []
        for mma_step in range(GFX1250_GEMV.mma_steps_per_chunk):
            k32 = chunk * GFX1250_GEMV.mma_steps_per_chunk + mma_step
            elements = []
            for half in range_constexpr(2):
                byte = (
                    row0 * k
                    + (k32 * 2 + half) * ROWS * 16
                    + row * 16
                    + lane_group * PACK_ELEMENTS
                )
                raw = bo.buffer_load(
                    weights,
                    byte // 4,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_NT if nt else 0,
                )
                converted = fp8x8_to_bf16(raw, exponent)
                elements.extend(converted[i] for i in range_constexpr(PACK_ELEMENTS))
            steps.append(fx.Vector.from_elements(elements, fx.BFloat16))
        out.append(steps)
    return out


def mfmas(lane, wave, waves, xl, xrow, k, s, operands, acc=None):
    """Multiply ``loads``' weights by token rows staged in LDS.

    ``acc`` may carry a previous K slab.  It is a v8 f32 WMMA accumulator, not
    the gfx950 backend's v4 MFMA accumulator.
    """

    per, n = chunks(k, waves)
    lane_group = lane // ROWS
    token = fx.min(lane % ROWS, s - 1)
    if acc is None:
        acc = fx.Vector.filled(ACCUMULATOR_ELEMENTS, 0.0, fx.Float32)
    for index in range_constexpr(per):
        chunk = wave + waves * index
        live = chunk < n
        chunk_k = fx.min(chunk, n - 1) * K_PER_CHUNK
        for mma_step in range_constexpr(GFX1250_GEMV.mma_steps_per_chunk):
            activation = _fragment(
                lambda offset: fx.ptr_load(
                    xl + (token * xrow + chunk_k) // 2 + offset,  # noqa: B023
                    result_type=fx.Vector.make_type(PACK_ELEMENTS // 2, fx.Int32),
                ),
                lane_group,
                mma_step,
                live if const_live(per, n, waves) else None,
            )
            acc = mma_bf16(operands[index][mma_step], activation, acc, GFX1250)
    return acc


def accumulator_offset(wave, lane):
    """First f32 of one lane's v8 accumulator in a wave-major LDS array."""

    return (wave * GFX1250.wave_size + lane) * ACCUMULATOR_ELEMENTS


def store_accumulator(acc, red, wave, lane):
    """Store one lane's v8 accumulator to the architecture-specific LDS layout."""

    fx.ptr_store(acc, red + accumulator_offset(wave, lane))


def row_sum(red, row, token, waves: int):
    """Sum output ``[row, token]`` from every wave's partial accumulator."""

    lane = token + ROWS * (row // ACCUMULATOR_ELEMENTS)
    element = row % ACCUMULATOR_ELEMENTS
    total = fx.Float32(0.0)
    for wave in range(waves):
        total = total + fx.ptr_load(red + accumulator_offset(wave, lane) + element)
    return total
