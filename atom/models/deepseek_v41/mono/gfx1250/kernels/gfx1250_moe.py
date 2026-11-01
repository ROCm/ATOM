# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 K2b compute with native gfx1250 FP4 and n32k4 scale layouts.

The shared FP8 expert uses raw 32x32 weight scales. Routed FP4 experts use
per-output-row group32 scales, GGUU or gfx1250 row-interleaved GUGU weights.
Intermediate BF16 rounding, limited SwiGLU and MXFP8 publication follow the
gfx950 stage contracts. No launch wrapper or wave64 mailbox is called here.
"""

import os

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_attention import _load_bf16x16, _wmma
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_mhc import _typed
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import bf16_round, fp8_pack4, hw_exp2, lane_gather, traced

HIDDEN, EXPERTS, TOPK, WAVES = 5120, 384, 6, 8
LOG2E, LIMIT = 1.4426950408889634, 10.0
GU_PROJECTIONS = int(os.environ.get("ATOM_DSV41_GFX1250_GU_REUSE", "0"))
if GU_PROJECTIONS not in (0, 2, 4):
    raise ValueError("[gfx1250 GU] GU_REUSE must be 0, 2 (pair), or 4 (quad)")
DOWN_PARALLEL = int(os.environ.get("ATOM_DSV41_GFX1250_DOWN_PARALLEL", "0"))
if DOWN_PARALLEL not in (0, 1, 2):
    raise ValueError("gfx1250 down mode must be 0, 1 (parallel), or 2 (prefetch)")


GU_WAVE_GROUPS = os.environ.get("ATOM_DSV41_GFX1250_GU_WAVE_GROUPS", "0") == "1"
if GU_WAVE_GROUPS and GU_PROJECTIONS != 2:
    raise ValueError("gfx1250 GU wave groups require GU_REUSE=2 (pair)")


def moe_widths(tp):
    if tp not in (2, 4):
        raise ValueError("[gfx1250 K2b] requires TP2/TP4")
    real = 2304 // tp
    return real, (real + 127) // 128 * 128


def fp4_weight_offset(tile, step, half, lane16, piece, inner):
    """Native shuffle_weight(..., layout=(16,16)) on packed FP4 bytes."""
    return tile * inner * 8 + step * 1024 + half * 256 + lane16 * 16 + piece * 512


def fp4_scale_offset(tile, step, lane16, inner):
    """Native gfx1250 shuffle_scale_n32k4; one four-code word per row."""
    return (tile // 2) * inner + step * 128 + (tile % 2 * 16 + lane16) * 4


@traced
def quantize_row_device(source, quant, codes, tid, row, width, floor):
    """One group32 per eight lanes, once per token; no shared/routed ABI mix."""
    for chunk in range_constexpr((width + 1023) // 1024):
        col = chunk * 1024 + tid * 4
        if col < width:
            at = row * width + col
            values = [
                fx.Float32(fx.ptr_load(_typed(source, fx.BFloat16) + at + j))
                for j in range(4)
            ]
            maximum = fx.Float32(floor)
            for v in values:
                maximum = fx.max(maximum, abs(v))
            for off in (1, 2, 4):
                maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
            code = code_ceil(maximum * fx.Float32(1.0 / FP8_MAX))
            inv = rcp_pow2(code)
            fx.ptr_store(
                fp8_pack4(*[clamp_fp8(v * inv) for v in values]),
                _typed(quant, fx.Int32) + at // 4,
            )
            if tid % 8 == 0:
                fx.ptr_store(code.to(fx.Uint8), _typed(codes, fx.Uint8) + at // 32)
    fx.gpu.barrier()


@traced
def router_device(normed, weight, logits, local, tid, task, rows):
    """BF16 WMMA of 16 router rows, K distributed over eight waves."""
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    row = fx.min(lane16, rows - 1)
    xa = fx.Int64(fx.ptrtoint(normed)) + fx.Int64(row * HIDDEN) * 2
    wa = fx.Int64(fx.ptrtoint(weight)) + fx.Int64((task * 16 + lane16) * HIDDEN) * 2
    accum = fx.make_rmem_tensor(8, fx.Float32)
    accum.store(fx.full(8, 0.0, fx.Float32))
    for iteration in range_constexpr(HIDDEN // 32 // WAVES):
        column = (wave + iteration * WAVES) * 32 + half * 16
        _wmma(
            accum,
            _load_bf16x16(xa + fx.Int64(column) * 2),
            _load_bf16x16(wa + fx.Int64(column) * 2),
        )
    red = _typed(local, fx.Float32)
    if lane16 < rows:
        fx.ptr_store(accum.load(), red + (wave * rows + lane16) * 16 + half * 8)
    fx.gpu.barrier()
    if tid < rows * 16:
        token, column = tid // 16, tid % 16
        value = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            value = value + fx.ptr_load(red + (w * rows + token) * 16 + column)
        fx.ptr_store(
            value.to(fx.BFloat16),
            _typed(logits, fx.BFloat16) + token * EXPERTS + task * 16 + column,
        )
    fx.gpu.barrier()


def _score(value):
    softplus = (value > 20.0).select(
        value,
        fx.Float32(fmath.log(fx.Float32(1.0) + hw_exp2(value * fx.Float32(LOG2E)))),
    )
    # fmath.log is natural log; overflow branch uses x as in native topk.
    score = fx.Float32(fmath.sqrt(softplus))
    return fx.isnan(score).select(fx.Float32(0.0), score)


@traced
def route_device(logits, bias, ids, weights, tid, row):
    """sqrtsoftplus, selection-only bias, top6, renormalization times1.5."""
    lane = tid % 32
    if tid < 32:
        original, corrected = [], []
        for i in range_constexpr(EXPERTS // 32):
            expert = lane + i * 32
            value = fx.Float32(
                fx.ptr_load(_typed(logits, fx.BFloat16) + row * EXPERTS + expert)
            )
            score = _score(value)
            b = fx.ptr_load(_typed(bias, fx.Float32) + expert)
            original.append(score)
            corrected.append(fx.max(score + b, fx.Float32(float("-inf"))))
        total, my_weight, my_id = fx.Float32(0.0), fx.Float32(0.0), fx.Int32(0)
        for pick in range_constexpr(TOPK):
            best, best_id, best_orig = (
                fx.Float32(float("-inf")),
                fx.Int32(EXPERTS),
                fx.Float32(0.0),
            )
            for i in range_constexpr(EXPERTS // 32):
                expert = lane + i * 32
                wins = (corrected[i] > best) | (
                    (corrected[i] == best) & (expert < best_id)
                )
                best_orig = wins.select(original[i], best_orig)
                best_id = wins.select(expert, best_id)
                best = fx.max(best, corrected[i])
            maximum = best
            for off in (16, 8, 4, 2, 1):
                maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
            selected = (best == maximum).select(best_id, fx.Int32(EXPERTS))
            for off in (16, 8, 4, 2, 1):
                selected = fx.min(selected, selected.shuffle_xor(off, 32))
            selected_orig = lane_gather(
                best_orig.bitcast(fx.Int32), selected % 32
            ).bitcast(fx.Float32)
            total = total + selected_orig
            my_weight = (lane == pick).select(selected_orig, my_weight)
            my_id = (lane == pick).select(selected, my_id)
            for i in range_constexpr(EXPERTS // 32):
                corrected[i] = (lane + i * 32 == selected).select(
                    fx.Float32(float("-inf")), corrected[i]
                )
        if lane < TOPK:
            fx.ptr_store(my_id, _typed(ids, fx.Int32) + row * TOPK + lane)
            fx.ptr_store(
                my_weight * (fx.Float32(1.5) / fx.max(total, fx.Float32(1e-20))),
                _typed(weights, fx.Float32) + row * TOPK + lane,
            )
    fx.gpu.barrier()


@traced
def union_device(ids, union_ids, count, pick_slots, local, diag, tid, rows):
    """Ascending expert union and token/pick -> slot, bounded to twelve picks."""
    masks = _typed(local, fx.Int32)
    if tid < EXPERTS // 32:
        fx.ptr_store(fx.Int32(0), masks + tid)
    fx.gpu.barrier()
    if tid < rows * TOPK:
        expert = fx.ptr_load(_typed(ids, fx.Int32) + tid)
        if (expert >= 0) & (expert < EXPERTS):
            fx.llvm.atomic_or(
                masks + expert // 32,
                fx.Int32(1) << (expert % 32),
                syncscope="workgroup",
            )
        else:
            fx.atomic_cas(
                diag,
                fx.Int32(0),
                fx.Int32(19_000_000),
                success_ordering=fx.AtomicOrdering.Release,
                failure_ordering=fx.AtomicOrdering.Monotonic,
                syncscope="agent",
            )
    fx.gpu.barrier()
    if tid < rows * TOPK:
        expert = fx.ptr_load(_typed(ids, fx.Int32) + tid)
        if (expert >= 0) & (expert < EXPERTS):
            slot = fx.Int32(0)
            for word in range_constexpr(EXPERTS // 32):
                mask = fx.ptr_load(masks + word)
                below = (fx.Int32(1) << (expert % 32)) - 1
                mask = (word < expert // 32).select(
                    mask, (word == expert // 32).select(mask & below, fx.Int32(0))
                )
                slot = slot + fx.Int32(fmath.ctpop(mask))
            fx.ptr_store(slot, _typed(pick_slots, fx.Int32) + tid)
            first = fx.Boolean(True)
            for pick in range_constexpr(rows * TOPK):
                first = first & (
                    (pick >= tid)
                    | (fx.ptr_load(_typed(ids, fx.Int32) + pick) != expert)
                )
            if first:
                fx.ptr_store(expert, _typed(union_ids, fx.Int32) + slot)
    if tid == 0:
        n = fx.Int32(0)
        for word in range_constexpr(EXPERTS // 32):
            n = n + fx.Int32(fmath.ctpop(fx.ptr_load(masks + word)))
        fx.ptr_store(n, _typed(count, fx.Int32))
    fx.gpu.barrier()


@traced
def project_device(
    x,
    xs,
    weight,
    scale,
    dest,
    local,
    tid,
    tile,
    rows,
    inner,
    output,
    fp4=False,
    contribution=False,
    slot=None,
    pick_slots=None,
    route_weights=None,
):
    """K-strided WMMA -> BF16; FP8 K32 tails zero invalid groups explicitly.

    FP4 buffers point to one expert, keeping resource extents below 4GiB.
    The routed down epilogue applies routing weight before the BF16 round,
    then publishes one contribution for every pick of this union slot.
    """
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    active_row = fx.min(lane16, rows - 1)
    rw = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(weight), num_records_bytes=output * inner // (2 if fp4 else 1)
    )
    rx = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(x), num_records_bytes=rows * inner
    )
    accum = fx.make_rmem_tensor(8, fx.Float32)
    accum.store(fx.full(8, 0.0, fx.Float32))
    atom = fx.make_mma_atom(
        fx.rocdl.WMMAScale(
            16,
            16,
            128,
            fx.Float4E2M1FN if fp4 else fx.Float8E4M3FN,
            fx.Float8E4M3FN,
            fx.Float32,
        )
    )
    for iteration in range_constexpr(((inner + 127) // 128 + WAVES - 1) // WAVES):
        step = wave + iteration * WAVES
        st = fx.min(step, (inner - 1) // 128)
        live = step * 128 < inner
        av, bv = [], []
        for j in range_constexpr(4):
            valid = live & (st * 128 + half * 16 + j * 32 < inner)
            safe_col = fx.min(st * 128 + half * 16 + j * 32, inner - 16)
            got = fx.Vector(
                bo.buffer_load(
                    rx, (active_row * inner + safe_col) // 4, vec_width=4, dtype=T.i32
                )
            )
            av.extend([valid.select(got[k], fx.Int32(0)) for k in range(4)])
        for j in range_constexpr(2 if fp4 else 4):
            valid = live & (
                st * 128 + half * (32 if fp4 else 16) + j * (64 if fp4 else 32) < inner
            )
            offset = (
                fp4_weight_offset(tile, st, half, lane16, j, inner)
                if fp4
                else tile * inner * 16 + st * 2048 + half * 256 + lane16 * 16 + j * 512
            )
            # A safe in-resource fallback is loaded only for inactive K groups.
            offset = valid.select(offset, fx.Int32(0))
            got = fx.Vector(bo.buffer_load(rw, offset // 4, vec_width=4, dtype=T.i32))
            bv.extend([valid.select(got[k], fx.Int32(0)) for k in range(4)])
        sa, sb = fx.Int32(0), fx.Int32(0)
        for j in range_constexpr(4):
            valid = live & (st * 4 + j < inner // 32)
            acol = fx.min(st * 4 + j, inner // 32 - 1)
            ac = fx.Int32(
                fx.ptr_load(_typed(xs, fx.Uint8) + active_row * (inner // 32) + acol)
            )
            if const_expr(fp4):
                wc = fx.Int32(
                    fx.ptr_load(
                        _typed(scale, fx.Uint8)
                        + fp4_scale_offset(tile, st, lane16, inner)
                        + j
                    )
                )
            else:
                wc = fx.Int32(
                    fx.ptr_load(
                        _typed(scale, fx.Uint8) + (tile // 2) * (inner // 32) + acol
                    )
                )
            sa = sa | (valid.select(ac, fx.Int32(127)) << (j * 8))
            sb = sb | (valid.select(wc, fx.Int32(127)) << (j * 8))
        a, b = fx.make_rmem_tensor(16, fx.Int32), fx.make_rmem_tensor(
            8 if fp4 else 16, fx.Int32
        )
        a.store(fx.Vector.from_elements(av, fx.Int32))
        b.store(fx.Vector.from_elements(bv, fx.Int32))
        fx.gemm(atom, accum, b, a, accum, scale_a=sb, scale_b=sa)
    red = _typed(local, fx.Float32)
    if lane16 < rows:
        fx.ptr_store(accum.load(), red + (wave * rows + lane16) * 16 + half * 8)
    fx.gpu.barrier()
    if tid < rows * 16:
        row, column = tid // 16, tid % 16
        value = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            value = value + fx.ptr_load(red + (w * rows + row) * 16 + column)
        if const_expr(contribution):
            for pick in range_constexpr(TOPK):
                at = row * TOPK + pick
                if fx.ptr_load(_typed(pick_slots, fx.Int32) + at) == slot:
                    weighted = value * fx.ptr_load(
                        _typed(route_weights, fx.Float32) + at
                    )
                    fx.ptr_store(
                        weighted.to(fx.BFloat16),
                        _typed(dest, fx.BFloat16) + at * output + tile * 16 + column,
                    )
        else:
            fx.ptr_store(
                value.to(fx.BFloat16),
                _typed(dest, fx.BFloat16) + row * output + tile * 16 + column,
            )
    fx.gpu.barrier()


def gu_projection_tiles(group, width, interleaved):
    """Four physical N16 tiles covering one logical N32 gate/up group.

    GGUU pairs gate/up for each half; GUGU pairs adjacent physical tiles,
    whose rows already alternate gate/up. Every tile retains its original
    destination row and its own native weight-scale addressing.
    """
    if interleaved:
        return tuple(group * 4 + i for i in range(4))
    return tuple(
        part * (width // 16) + group * 2 + half
        for half in range(2)
        for part in range(2)
    )


@traced
def project_multi_device(
    x, xs, weight, scale, dest, local, tid, tiles, rows, inner, output, fp4=False
):
    """Pair/quad N16 projections sharing A and its scale across each K step.

    Each projection has an independent accumulator and the same K ordering
    as project_device. Private LDS partials occupy at most the existing 4KiB
    FFN arena. Both CTA barriers are unconditional, including inactive lanes.
    There is no publication here: the N32 middle producer still owns it.
    """
    count = len(tiles)
    assert count in (2, 4) and rows in (1, 2)
    assert inner % 32 == 0 and output % 16 == 0
    assert count * WAVES * rows * 16 * 4 <= 4096
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    active_row = fx.min(lane16, rows - 1)
    rw = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(weight), num_records_bytes=output * inner // (2 if fp4 else 1)
    )
    rx = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(x), num_records_bytes=rows * inner
    )
    accums = [fx.make_rmem_tensor(8, fx.Float32) for _ in range(count)]
    for projection in range_constexpr(count):
        accums[projection].store(fx.full(8, 0.0, fx.Float32))
    atom = fx.make_mma_atom(
        fx.rocdl.WMMAScale(
            16, 16, 128,
            fx.Float4E2M1FN if fp4 else fx.Float8E4M3FN,
            fx.Float8E4M3FN, fx.Float32,
        )
    )
    for iteration in range_constexpr(((inner + 127) // 128 + WAVES - 1) // WAVES):
        step = wave + iteration * WAVES
        st = fx.min(step, (inner - 1) // 128)
        live = step * 128 < inner
        av = []
        for j in range_constexpr(4):
            valid = live & (st * 128 + half * 16 + j * 32 < inner)
            safe_col = fx.min(st * 128 + half * 16 + j * 32, inner - 16)
            got = fx.Vector(bo.buffer_load(
                rx, (active_row * inner + safe_col) // 4, vec_width=4, dtype=T.i32
            ))
            av.extend([valid.select(got[k], fx.Int32(0)) for k in range(4)])
        sa = fx.Int32(0)
        for j in range_constexpr(4):
            valid = live & (st * 4 + j < inner // 32)
            acol = fx.min(st * 4 + j, inner // 32 - 1)
            ac = fx.Int32(fx.ptr_load(
                _typed(xs, fx.Uint8) + active_row * (inner // 32) + acol
            ))
            sa = sa | (valid.select(ac, fx.Int32(127)) << (j * 8))
        a = fx.make_rmem_tensor(16, fx.Int32)
        a.store(fx.Vector.from_elements(av, fx.Int32))
        for projection in range_constexpr(count):
            tile = tiles[projection]
            bv = []
            for j in range_constexpr(2 if fp4 else 4):
                valid = live & (
                    st * 128 + half * (32 if fp4 else 16)
                    + j * (64 if fp4 else 32) < inner
                )
                offset = (
                    fp4_weight_offset(tile, st, half, lane16, j, inner)
                    if fp4 else tile * inner * 16 + st * 2048
                    + half * 256 + lane16 * 16 + j * 512
                )
                offset = valid.select(offset, fx.Int32(0))
                got = fx.Vector(bo.buffer_load(rw, offset // 4, vec_width=4, dtype=T.i32))
                bv.extend([valid.select(got[k], fx.Int32(0)) for k in range(4)])
            sb = fx.Int32(0)
            for j in range_constexpr(4):
                valid = live & (st * 4 + j < inner // 32)
                acol = fx.min(st * 4 + j, inner // 32 - 1)
                scale_at = (
                    fp4_scale_offset(tile, st, lane16, inner) + j
                    if fp4 else (tile // 2) * (inner // 32) + acol
                )
                wc = fx.Int32(fx.ptr_load(_typed(scale, fx.Uint8) + scale_at))
                sb = sb | (valid.select(wc, fx.Int32(127)) << (j * 8))
            b = fx.make_rmem_tensor(8 if fp4 else 16, fx.Int32)
            b.store(fx.Vector.from_elements(bv, fx.Int32))
            acc = accums[projection]
            fx.gemm(atom, acc, b, a, acc, scale_a=sb, scale_b=sa)
    red = _typed(local, fx.Float32)
    for projection in range_constexpr(count):
        if lane16 < rows:
            at = ((projection * WAVES + wave) * rows + lane16) * 16 + half * 8
            fx.ptr_store(accums[projection].load(), red + at)
    fx.gpu.barrier()
    if tid < rows * 16:
        row, column = tid // 16, tid % 16
        for projection in range_constexpr(count):
            value = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                at = ((projection * WAVES + w) * rows + row) * 16 + column
                value = value + fx.ptr_load(red + at)
            fx.ptr_store(
                value.to(fx.BFloat16),
                _typed(dest, fx.BFloat16) + row * output + tiles[projection] * 16 + column,
            )
    fx.gpu.barrier()


@traced
def project_wave_groups_middle_device(
    x, xs, weight, scale, mid, quant, codes, local, tid, tiles, rows, inner, output,
    group, real, padded, interleaved=False, fp4=False
):
    """Two four-wave K groups compute an N32 G/U task in parallel.

    Each wave owns two N16 accumulators. Four K partials per projection are
    reduced in the original native GGUU/GUGU row layout, followed by BF16
    G/U rounding, limited SwiGLU and group32 MXFP8 quantization. The existing
    4KiB LDS arena and N32 publication ABI are retained. All 256 threads
    join both barriers; the optional path does not use the gup scratch.
    """
    count = 2
    assert len(tiles) == 4
    assert count == 2 and rows in (1, 2)
    assert inner % 32 == 0 and output % 16 == 0
    assert count * WAVES * rows * 16 * 4 <= 4096
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    wave_group, k_wave = wave // 4, wave % 4
    active_row = fx.min(lane16, rows - 1)
    rw = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(weight), num_records_bytes=output * inner // (2 if fp4 else 1)
    )
    rx = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(x), num_records_bytes=rows * inner
    )
    accums = [fx.make_rmem_tensor(8, fx.Float32) for _ in range(count)]
    for projection in range_constexpr(count):
        accums[projection].store(fx.full(8, 0.0, fx.Float32))
    atom = fx.make_mma_atom(
        fx.rocdl.WMMAScale(
            16, 16, 128,
            fx.Float4E2M1FN if fp4 else fx.Float8E4M3FN,
            fx.Float8E4M3FN, fx.Float32,
        )
    )
    for iteration in range_constexpr(((inner + 127) // 128 + 3) // 4):
        step = k_wave + iteration * 4
        st = fx.min(step, (inner - 1) // 128)
        live = step * 128 < inner
        av = []
        for j in range_constexpr(4):
            valid = live & (st * 128 + half * 16 + j * 32 < inner)
            safe_col = fx.min(st * 128 + half * 16 + j * 32, inner - 16)
            got = fx.Vector(bo.buffer_load(
                rx, (active_row * inner + safe_col) // 4, vec_width=4, dtype=T.i32
            ))
            av.extend([valid.select(got[k], fx.Int32(0)) for k in range(4)])
        sa = fx.Int32(0)
        for j in range_constexpr(4):
            valid = live & (st * 4 + j < inner // 32)
            acol = fx.min(st * 4 + j, inner // 32 - 1)
            ac = fx.Int32(fx.ptr_load(
                _typed(xs, fx.Uint8) + active_row * (inner // 32) + acol
            ))
            sa = sa | (valid.select(ac, fx.Int32(127)) << (j * 8))
        a = fx.make_rmem_tensor(16, fx.Int32)
        a.store(fx.Vector.from_elements(av, fx.Int32))
        for projection in range_constexpr(count):
            tile = (wave_group == 0).select(tiles[projection], tiles[projection + 2])
            bv = []
            for j in range_constexpr(2 if fp4 else 4):
                valid = live & (
                    st * 128 + half * (32 if fp4 else 16)
                    + j * (64 if fp4 else 32) < inner
                )
                offset = (
                    fp4_weight_offset(tile, st, half, lane16, j, inner)
                    if fp4 else tile * inner * 16 + st * 2048
                    + half * 256 + lane16 * 16 + j * 512
                )
                offset = valid.select(offset, fx.Int32(0))
                got = fx.Vector(bo.buffer_load(rw, offset // 4, vec_width=4, dtype=T.i32))
                bv.extend([valid.select(got[k], fx.Int32(0)) for k in range(4)])
            sb = fx.Int32(0)
            for j in range_constexpr(4):
                valid = live & (st * 4 + j < inner // 32)
                acol = fx.min(st * 4 + j, inner // 32 - 1)
                scale_at = (
                    fp4_scale_offset(tile, st, lane16, inner) + j
                    if fp4 else (tile // 2) * (inner // 32) + acol
                )
                wc = fx.Int32(fx.ptr_load(_typed(scale, fx.Uint8) + scale_at))
                sb = sb | (valid.select(wc, fx.Int32(127)) << (j * 8))
            b = fx.make_rmem_tensor(8 if fp4 else 16, fx.Int32)
            b.store(fx.Vector.from_elements(bv, fx.Int32))
            acc = accums[projection]
            fx.gemm(atom, acc, b, a, acc, scale_a=sb, scale_b=sa)
    red = _typed(local, fx.Float32)
    for projection in range_constexpr(count):
        if lane16 < rows:
            at = ((projection * WAVES + wave) * rows + lane16) * 16 + half * 8
            fx.ptr_store(accums[projection].load(), red + at)
    fx.gpu.barrier()
    if tid < rows * 32:
        row, column = tid // 32, tid % 32
        logical_column, group_half = column % 16, column // 16
        gate, up = fx.Float32(0.0), fx.Float32(0.0)
        for w in range_constexpr(4):
            wave_at = group_half * 4 + w
            if const_expr(interleaved):
                projection, physical_column = logical_column // 8, (logical_column % 8) * 2
                gate_at = ((projection * WAVES + wave_at) * rows + row) * 16 + physical_column
                up_at = gate_at + 1
            else:
                gate_at = (wave_at * rows + row) * 16 + logical_column
                up_at = ((WAVES + wave_at) * rows + row) * 16 + logical_column
            gate = gate + fx.ptr_load(red + gate_at)
            up = up + fx.ptr_load(red + up_at)
        col = group * 32 + column
        value = (col < real).select(
            bf16_round(_silu_mul(bf16_round(gate), bf16_round(up))), fx.Float32(0.0)
        )
        at = row * padded + col
        fx.ptr_store(value.to(fx.BFloat16), _typed(mid, fx.BFloat16) + at)
        maximum = fx.max(abs(value), fx.Float32(1e-10 if fp4 else 1e-4))
        for off in (1, 2, 4, 8, 16):
            maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
        code = code_ceil(maximum * fx.Float32(1.0 / FP8_MAX))
        inv = rcp_pow2(code)
        values = [lane_gather(value.bitcast(fx.Int32), (column // 4) * 4 + j).bitcast(fx.Float32)
                  for j in range(4)]
        if column % 4 == 0:
            fx.ptr_store(fp8_pack4(*[clamp_fp8(v * inv) for v in values]),
                         _typed(quant, fx.Int32) + at // 4)
        if column == 0:
            fx.ptr_store(code.to(fx.Uint8), _typed(codes, fx.Uint8) + at // 32)
    fx.gpu.barrier()


def _silu_mul(gate, up):
    gate = fx.min(gate, fx.Float32(LIMIT))
    up = fx.max(fx.min(up, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
    return gate / (fx.Float32(1.0) + fx.Float32(fmath.exp(-gate))) * up


@traced
def middle_device(
    gup, mid, quant, codes, tid, group, rows, real, padded, interleaved, floor
):
    """One N32 SwiGLU/quant producer; explicitly writes padded zero columns."""
    if tid < rows * 8:
        row, lane = tid // 8, tid % 8
        column = group * 32 + lane * 4
        values = []
        for j in range_constexpr(4):
            col = column + j
            gate_col = col * 2 if interleaved else col
            up_col = col * 2 + 1 if interleaved else padded + col
            gate = fx.Float32(
                fx.ptr_load(_typed(gup, fx.BFloat16) + row * 2 * padded + gate_col)
            )
            up = fx.Float32(
                fx.ptr_load(_typed(gup, fx.BFloat16) + row * 2 * padded + up_col)
            )
            value = (col < real).select(
                bf16_round(_silu_mul(gate, up)), fx.Float32(0.0)
            )
            values.append(value)
            fx.ptr_store(
                value.to(fx.BFloat16), _typed(mid, fx.BFloat16) + row * padded + col
            )
        maximum = fx.Float32(floor)
        for v in values:
            maximum = fx.max(maximum, abs(v))
        for off in (1, 2, 4):
            maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
        code = code_ceil(maximum * fx.Float32(1.0 / FP8_MAX))
        inv = rcp_pow2(code)
        at = row * padded + column
        fx.ptr_store(
            fp8_pack4(*[clamp_fp8(v * inv) for v in values]),
            _typed(quant, fx.Int32) + at // 4,
        )
        if lane == 0:
            fx.ptr_store(code.to(fx.Uint8), _typed(codes, fx.Uint8) + at // 32)
    fx.gpu.barrier()


@traced
def shared_up_device(
    x, xs, weight, scale, gup, mid, quant, codes, local, tid, task, rows, real
):
    if const_expr(GU_WAVE_GROUPS):
        assert GU_PROJECTIONS == 2
        tiles = gu_projection_tiles(task, real, False)
        project_wave_groups_middle_device(
            x, xs, weight, scale, mid, quant, codes, local, tid, tiles,
            rows, HIDDEN, 2*real, task, real, real,
        )
    else:
        if const_expr(GU_PROJECTIONS == 0):
            for part in range_constexpr(2):
                for half in range_constexpr(2):
                    tile = part * (real // 16) + task * 2 + half
                    project_device(
                        x, xs, weight, scale, gup, local, tid, tile, rows, HIDDEN, 2 * real
                    )
        else:
            tiles = gu_projection_tiles(task, real, False)
            for start in range_constexpr(0, 4, GU_PROJECTIONS):
                project_multi_device(
                    x, xs, weight, scale, gup, local, tid,
                    tiles[start:start + GU_PROJECTIONS], rows, HIDDEN, 2 * real,
                )
        middle_device(gup, mid, quant, codes, tid, task, rows, real, real, False, 1e-4)


@traced
def routed_up_device(
    x,
    xs,
    weight,
    scale,
    gup,
    mid,
    quant,
    codes,
    union_ids,
    local,
    tid,
    task,
    rows,
    real,
    padded,
    interleaved,
):
    slot, group = task // (padded // 32), task % (padded // 32)
    expert = fx.ptr_load(_typed(union_ids, fx.Int32) + slot)
    w = _typed(weight, fx.Uint8) + fx.Int64(expert) * (2 * padded * HIDDEN // 2)
    s = _typed(scale, fx.Uint8) + fx.Int64(expert) * (2 * padded * HIDDEN // 32)
    gu = _typed(gup, fx.BFloat16) + slot * rows * 2 * padded
    m = _typed(mid, fx.BFloat16) + slot * rows * padded
    q = _typed(quant, fx.Uint8) + slot * rows * padded
    qs = _typed(codes, fx.Uint8) + slot * rows * (padded // 32)
    if const_expr(GU_WAVE_GROUPS):
        assert GU_PROJECTIONS == 2
        tiles = gu_projection_tiles(group, padded, interleaved)
        project_wave_groups_middle_device(
            x, xs, w, s, m, q, qs, local, tid, tiles,
            rows, HIDDEN, 2*padded, group, real, padded, interleaved, True,
        )
    else:
        if const_expr(GU_PROJECTIONS == 0):
            for part in range_constexpr(2):
                for half in range_constexpr(2):
                    tile = (
                        group * 4 + part * 2 + half
                        if interleaved
                        else part * (padded // 16) + group * 2 + half
                    )
                    project_device(
                        x, xs, w, s, gu, local, tid, tile, rows, HIDDEN, 2 * padded, fp4=True
                    )
        else:
            tiles = gu_projection_tiles(group, padded, interleaved)
            for start in range_constexpr(0, 4, GU_PROJECTIONS):
                project_multi_device(
                    x, xs, w, s, gu, local, tid,
                    tiles[start:start + GU_PROJECTIONS], rows, HIDDEN, 2 * padded, fp4=True,
                )
        middle_device(gu, m, q, qs, tid, group, rows, real, padded, interleaved, 1e-10)


@traced
def routed_down_device(
    mid_q,
    mid_s,
    weight,
    scale,
    contributions,
    union_ids,
    pick_slots,
    route_weights,
    local,
    tid,
    task,
    slot,
    rows,
    padded,
):
    expert = fx.ptr_load(_typed(union_ids, fx.Int32) + slot)
    w = _typed(weight, fx.Uint8) + fx.Int64(expert) * (HIDDEN * padded // 2)
    s = _typed(scale, fx.Uint8) + fx.Int64(expert) * (HIDDEN * padded // 32)
    q = _typed(mid_q, fx.Uint8) + slot * rows * padded
    qs = _typed(mid_s, fx.Uint8) + slot * rows * (padded // 32)
    project_device(
        q,
        qs,
        w,
        s,
        contributions,
        local,
        tid,
        task,
        rows,
        padded,
        HIDDEN,
        fp4=True,
        contribution=True,
        slot=slot,
        pick_slots=pick_slots,
        route_weights=route_weights,
    )


def _down_wave_weight(rw, scale, tile, step, half, lane16, inner):
    values = []
    for piece in range_constexpr(2):
        offset = fp4_weight_offset(tile, step, half, lane16, piece, inner)
        got = fx.Vector(bo.buffer_load(rw, offset // 4, vec_width=4, dtype=T.i32))
        values.extend(got[i] for i in range(4))
    code = fx.ptr_load(_typed(scale, fx.Int32) + fp4_scale_offset(tile, step, lane16, inner) // 4)
    return values, code


@traced
def routed_down_parallel_device(
    mid_q, mid_s, weight, scale, contributions, union_ids, pick_slots,
    route_weights, local, tid, task, count, rows, padded,
    before_wait=None, prefetch=True, mark_body=None,
):
    """One expert per wave; every live pick gets its weighted BF16 output.

    The caller acquired the union before entry. Weight addresses are legal
    before UG completes. Only the first K tile is retained across the wait;
    activation and its scales are read after the per-wave acquire. All waves
    join every round, including those beyond the live expert union.
    """
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    active_row = fx.min(lane16, rows - 1)
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
        16, 16, 128, fx.Float4E2M1FN, fx.Float8E4M3FN, fx.Float32,
    ))
    for round in range((count + WAVES - 1) // WAVES):
        slot = round * WAVES + wave
        live = slot < count
        safe_slot = fx.min(slot, count - 1)
        expert = fx.ptr_load(_typed(union_ids, fx.Int32) + safe_slot)
        w = _typed(weight, fx.Uint8) + fx.Int64(expert) * (HIDDEN * padded // 2)
        s = _typed(scale, fx.Uint8) + fx.Int64(expert) * (HIDDEN * padded // 32)
        rw = bo.create_buffer_resource_from_addr(fx.ptrtoint(w), num_records_bytes=HIDDEN*padded//2)
        first_values, first_scale = None, None
        if const_expr(prefetch):
            first_values, first_scale = _down_wave_weight(rw,s,task,0,half,lane16,padded)
        acquired = before_wait(safe_slot,live) if const_expr(before_wait is not None) else fx.Boolean(True)
        if const_expr(mark_body is not None):
            mark_body(False)
        if live & acquired:
            x = _typed(mid_q, fx.Uint8) + safe_slot * rows * padded
            xs = _typed(mid_s, fx.Uint8) + safe_slot * rows * (padded // 32)
            rx = bo.create_buffer_resource_from_addr(fx.ptrtoint(x),num_records_bytes=rows*padded)
            accum = fx.make_rmem_tensor(8,fx.Float32)
            accum.store(fx.full(8,0.0,fx.Float32))
            for step in range_constexpr(padded//128):
                if const_expr(prefetch and step==0):
                    bv,sb = first_values,first_scale
                else:
                    bv,sb = _down_wave_weight(rw,s,task,step,half,lane16,padded)
                av=[]
                for part in range_constexpr(4):
                    got=fx.Vector(bo.buffer_load(rx,
                        (active_row*padded+step*128+half*16+part*32)//4,
                        vec_width=4,dtype=T.i32))
                    av.extend(got[i] for i in range(4))
                sa=fx.ptr_load(_typed(xs,fx.Int32)+active_row*(padded//128)+step)
                a,b=fx.make_rmem_tensor(16,fx.Int32),fx.make_rmem_tensor(8,fx.Int32)
                a.store(fx.Vector.from_elements(av,fx.Int32))
                b.store(fx.Vector.from_elements(bv,fx.Int32))
                fx.gemm(atom,accum,b,a,accum,scale_a=sb,scale_b=sa)
            if lane16 < rows:
                values=accum.load()
                for pick in range_constexpr(TOPK):
                    at=lane16*TOPK+pick
                    if fx.ptr_load(_typed(pick_slots,fx.Int32)+at)==slot:
                        routing=fx.ptr_load(_typed(route_weights,fx.Float32)+at)
                        for column in range_constexpr(8):
                            fx.ptr_store((values[column]*routing).to(fx.BFloat16),
                                _typed(contributions,fx.BFloat16)+at*HIDDEN+task*16+half*8+column)
        fx.gpu.barrier()
        if const_expr(mark_body is not None):
            mark_body(True)

@traced
def combine_device(contributions, shared_out, out, tid, task, rows):
    if tid < rows * 16:
        row, column = tid // 16, tid % 16
        col = task * 16 + column
        value = fx.Float32(0.0)
        for pick in range_constexpr(TOPK):
            v = fx.Float32(
                fx.ptr_load(
                    _typed(contributions, fx.BFloat16)
                    + (row * TOPK + pick) * HIDDEN
                    + col
                )
            )
            value = bf16_round(value + v)
        shared = fx.Float32(
            fx.ptr_load(_typed(shared_out, fx.BFloat16) + row * HIDDEN + col)
        )
        fx.ptr_store(
            (value + shared).to(fx.BFloat16),
            _typed(out, fx.BFloat16) + row * HIDDEN + col,
        )
    fx.gpu.barrier()


@traced
def compact_union_c1(ids, union_ids, count, pick_slots, diag, tid):
    """Preserve six-pick order and ascending union, with strict ID validation.

    The route owner calls this after its full CTA join, before publishing the
    union flag. Invalid or duplicate IDs publish the original union diagnostic
    and count0, so consumers retain the existing sticky-failure path.
    """
    lane = tid % 32
    if tid < 32:
        expert = fx.ptr_load(_typed(ids, fx.Int32) + fx.min(lane, fx.Int32(5)))
        slot = fx.Int32(0)
        valid = (expert >= 0) & (expert < EXPERTS)
        for pick in range_constexpr(TOPK):
            other = lane_gather(expert, pick)
            slot = slot + (other < expert).select(fx.Int32(1), fx.Int32(0))
            valid = valid & ((pick == lane) | (other != expert))
        failed = ((lane < TOPK) & ~valid).select(fx.Int32(1), fx.Int32(0))
        for off in (16, 8, 4, 2, 1):
            failed = failed | failed.shuffle_xor(off, 32)
        if failed == 0:
            if lane < TOPK:
                fx.ptr_store(expert, _typed(union_ids, fx.Int32) + slot)
                fx.ptr_store(slot, _typed(pick_slots, fx.Int32) + lane)
            if lane == 0:
                fx.ptr_store(fx.Int32(TOPK), _typed(count, fx.Int32))
        else:
            if lane == 0:
                fx.atomic_cas(diag, fx.Int32(0), fx.Int32(19_000_000),
                    success_ordering=fx.AtomicOrdering.Release,
                    failure_ordering=fx.AtomicOrdering.Monotonic, syncscope="agent")
                fx.ptr_store(fx.Int32(0), _typed(count, fx.Int32))
    fx.gpu.barrier()
