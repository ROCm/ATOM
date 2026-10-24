# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replicated BF16 index weights, per-head FP8 quant and safe ring insertion."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from atom.mono.device.mx import clamp_fp8
from atom.mono.device.ops import bf16_round, fp8_pack4, traced
from atom.models.deepseek_v41.mono.kernels.gfx1250_mhc import _typed
from atom.models.deepseek_v41.mono.kernels.gfx1250_gemv import grouped_partials
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops


@traced
def iw_device(normed, weight, out, local, tid, head, rows):
    normed, weight, out = [_typed(p, fx.BFloat16) for p in (normed, weight, out)]
    lane, wave = tid % 32, tid // 32
    red = fx.recast_iter(
        fx.PointerType.get(fx.Float32.ir_type, local.address_space, 4), local
    )
    for row in range_constexpr(rows):
        value = fx.Float32(0.0)
        for step in range_constexpr(20):
            col = tid + step * 256
            x = fx.Float32(fx.ptr_load(normed + row * 5120 + col))
            w = fx.Float32(fx.ptr_load(weight + head * 5120 + col))
            value = fx.fma(x, w, value)
        for off in (16, 8, 4, 2, 1):
            value = value + value.shuffle_xor(off, 32)
        if lane == 0:
            fx.ptr_store(value, red + wave)
        fx.gpu.barrier()
        if tid == 0:
            total = fx.Float32(0.0)
            for w in range_constexpr(8):
                total = total + fx.ptr_load(red + w)
            fx.ptr_store(total.to(fx.BFloat16), out + row * 32 + head)
        fx.gpu.barrier()


@traced
def iquant_device(query, iw, quant, scales, scaled_iw, tid, head, rows):
    query, iw = [_typed(p, fx.BFloat16) for p in (query, iw)]
    scales, scaled_iw = [_typed(p, fx.Float32) for p in (scales, scaled_iw)]
    if tid < 32:
        for row in range_constexpr(rows):
            at = (row * 32 + head) * 128 + tid * 4
            vals = [fx.Float32(fx.ptr_load(query + at + j)) for j in range(4)]
            amax = fx.Float32(1e-10)
            for v in vals:
                amax = fx.max(amax, abs(v))
            for off in (16, 8, 4, 2, 1):
                amax = fx.max(amax, amax.shuffle_xor(off, 32))
            scale = amax * fx.Float32(1.0 / 448.0)
            inv = fx.Float32(1.0) / scale
            word = fp8_pack4(*[clamp_fp8(v * inv) for v in vals])
            raw = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, quant.address_space, 4), quant
            )
            fx.ptr_store(word, raw + at // 4)
            if tid == 0:
                fx.ptr_store(scale, scales + row * 32 + head)
                w = fx.Float32(fx.ptr_load(iw + row * 32 + head))
                fx.ptr_store(
                    (w * scale) * fx.Float32(1.0 / 64.0), scaled_iw + row * 32 + head
                )
    fx.gpu.barrier()


@traced
def iq_projection_quant_device(
    q, qs, cos, sin, query, quant, scales, iw, scaled_iw, ops, lds, tid, head, rows
):
    # An entire N128 head is available in LDS. One wave per row reduces four
    # columns per lane and quantizes those registers, without an IQ readback.
    grouped_partials(q, qs, ops, lds, tid, head, rows, 4096, 8)
    if tid < rows * 32:
        row, lane = tid // 32, tid % 32
        col = lane * 4
        group = col // 16
        inside = col % 16
        ld32, st32 = make_lds_copy_ops(32)
        sums = [fx.Float32(0.0) for _ in range(4)]
        if const_expr(rows == 1):
            ld128, st128 = make_lds_copy_ops(128)
            for part in range_constexpr(8):
                loaded = ld128(
                    lds, ((group * 8 + part) * rows * 16 + row * 16 + inside) * 4
                )
                for j in range_constexpr(4):
                    sums[j] = sums[j] + fx.Int32(loaded[j]).bitcast(fx.Float32)
        else:
            for part in range_constexpr(8):
                for j in range_constexpr(4):
                    off = ((group * 8 + part) * rows * 16 + row * 16 + inside + j) * 4
                    value = fx.Int32(ld32(lds, off)[0]).bitcast(fx.Float32)
                    sums[j] = sums[j] + value
        values = [bf16_round(v) for v in sums]
        cb, sb = _typed(cos, fx.BFloat16), _typed(sin, fx.BFloat16)
        rotated = []
        for pair in range_constexpr(2):
            dim = col + pair * 2
            at = row * 32 + fx.max(dim - 64, 0) // 2
            cs = fx.Float32(fx.ptr_load(cb + at))
            sn = fx.Float32(fx.ptr_load(sb + at))
            x, y = values[pair * 2], values[pair * 2 + 1]
            rotated.append(bf16_round((dim >= 64).select(x * cs - y * sn, x)))
            rotated.append(bf16_round((dim >= 64).select(x * sn + y * cs, y)))
        iq = _typed(query, fx.BFloat16)
        at = (row * 32 + head) * 128 + col
        for j in range_constexpr(4):
            fx.ptr_store(rotated[j].to(fx.BFloat16), iq + at + j)
        amax = fx.Float32(1e-10)
        for value in rotated:
            amax = fx.max(amax, abs(value))
        for off in (16, 8, 4, 2, 1):
            amax = fx.max(amax, amax.shuffle_xor(off, 32))
        qscale = amax * fx.Float32(1.0 / 448.0)
        inv = fx.Float32(1.0) / qscale
        word = fp8_pack4(*[clamp_fp8(v * inv) for v in rotated])
        fx.ptr_store(word, _typed(quant, fx.Int32) + at // 4)
        if lane == 0:
            fx.ptr_store(qscale, _typed(scales, fx.Float32) + row * 32 + head)
            w = fx.Float32(fx.ptr_load(_typed(iw, fx.BFloat16) + row * 32 + head))
            fx.ptr_store(
                (w * qscale) * fx.Float32(1.0 / 64.0),
                _typed(scaled_iw, fx.Float32) + row * 32 + head,
            )
    fx.gpu.barrier()


@traced
def ring_device(kv, ring, ring_rows, diag, tid, row, rows, pool_rows, ring_off):
    kv, ring = [_typed(p, fx.BFloat16) for p in (kv, ring)]
    ring_rows, diag = [_typed(p, fx.Int32) for p in (ring_rows, diag)]
    relative = fx.Int32(fx.ptr_load(ring_rows + row))
    destination = fx.Int64(relative) + fx.Int64(ring_off)
    legal = (relative == -1) | (
        (relative >= 0) & (destination >= 0) & (destination < pool_rows)
    )
    for other in range_constexpr(rows):
        r = fx.Int32(fx.ptr_load(ring_rows + other))
        legal = legal & ((row == other) | (relative == -1) | (r != relative))
    if legal:
        if relative >= 0:
            for j in range_constexpr(2):
                # A global pointer with Int64 element arithmetic handles rings
                # larger than 4 GiB without a buffer's 32-bit offset overflow.
                fx.ptr_store(
                    fx.ptr_load(kv + row * 512 + tid * 2 + j),
                    ring + destination * fx.Int64(512) + fx.Int64(tid * 2 + j),
                )
    else:
        if tid == 0:
            fx.atomic_cas(
                diag,
                fx.Int32(0),
                fx.Int32(6_000_000) + row,
                success_ordering=fx.AtomicOrdering.Release,
                failure_ordering=fx.AtomicOrdering.Monotonic,
                syncscope="agent",
            )
    fx.gpu.barrier()
