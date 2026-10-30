# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 delayed mHC device bodies for the local gfx1250 attention seam.

160 column slices retain FP32 checkpoint weights as BF16 hi plus BF16 lo.
Incoming pre is used for collapse; the gates produced here belong to the next
seam. All helpers are CTA collective and use at most 2080 bytes of caller LDS.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T
from atom.mono.device.ops import bf_lo, bf16_round, hw_exp2, hw_rcp, hw_rsq, traced

HIDDEN = 5120
SLICES = 160


def _typed(p, dtype):
    return fx.recast_iter(
        fx.PointerType.get(dtype.ir_type, p.address_space, dtype.width // 8), p
    )


def _seam_bf16(value):
    # Native Triton first rounds its FMA result to FP32, then converts to
    # BF16. Keep the conversion opaque to forbid a mixed-BF16 final FMA.
    bits = fx.Int32(
        llvm.inline_asm(
            T.i32,
            [value.ir_value()],
            "v_cvt_pk_bf16_f32 $0, $1, $1",
            "=v,v",
            has_side_effects=False,
        )
    )
    return bf_lo(bits)


def _sum32(v):
    for off in (16, 8, 4, 2, 1):
        v = v + v.shuffle_xor(off, 32)
    return v


@traced
def slice_device(
    residual,
    y,
    pre,
    post,
    comb,
    fn,
    rout,
    collapse,
    partial,
    local,
    tid,
    task,
    rows,
    fold,
):
    residual, y, rout, collapse = [
        _typed(p, fx.BFloat16) for p in (residual, y, rout, collapse)
    ]
    pre, post, comb, fn, partial = [
        _typed(p, fx.Float32) for p in (pre, post, comb, fn, partial)
    ]
    lane, wave = tid % 32, tid // 32
    rl = fx.recast_iter(
        fx.PointerType.get(fx.Float32.ir_type, local.address_space, 4), local
    )
    for t in range_constexpr(rows):
        col = task * 32 + lane
        if wave < 4:
            old = [
                fx.Float32(fx.ptr_load(residual + (t * 4 + h) * HIDDEN + col))
                for h in range(4)
            ]
            if const_expr(fold):
                z = fx.Float32(fx.ptr_load(post + t * 4 + wave)) * fx.Float32(
                    fx.ptr_load(y + t * HIDDEN + col)
                )
                for h in range_constexpr(4):
                    z = fx.fma(
                        fx.Float32(fx.ptr_load(comb + t * 16 + h * 4 + wave)), old[h], z
                    )
                z = _seam_bf16(z)
                fx.ptr_store(z.to(fx.BFloat16), rout + (t * 4 + wave) * HIDDEN + col)
            else:
                z = fx.Float32(fx.ptr_load(residual + (t * 4 + wave) * HIDDEN + col))
            fx.ptr_store(z, rl + t * 128 + wave * 32 + lane)
    fx.gpu.barrier()
    for t in range_constexpr(rows):
        if wave == 0:
            rs = [fx.ptr_load(rl + t * 128 + h * 32 + lane) for h in range(4)]
            # gfx1250's native Triton seam contracts the later products into
            # the running FP32 sum. Explicit FMA preserves its halfway cases.
            z = fx.Float32(fx.ptr_load(pre + t * 4)) * rs[0]
            for h in range_constexpr(1, 4):
                z = fx.fma(fx.Float32(fx.ptr_load(pre + t * 4 + h)), rs[h], z)
            z = _seam_bf16(z)
            fx.ptr_store(z.to(fx.BFloat16), collapse + t * HIDDEN + task * 32 + lane)
            sq = rs[0] * rs[0]
            for h in range_constexpr(1, 4):
                sq = sq + rs[h] * rs[h]
            sq = _sum32(sq)
            # Native TILE_K32 has eight consecutive values per thread:
            # register bits 4/2/1, then column-thread bits 16/8.
            sx = z * z
            for off in (4, 2, 1, 16, 8):
                sx = sx + sx.shuffle_xor(off, 32)
            if lane == 0:
                fx.ptr_store(sq, partial + (t * SLICES + task) * 26 + 24)
                fx.ptr_store(sx, partial + (t * SLICES + task) * 26 + 25)
        for j0 in range_constexpr(3):
            j = wave + j0 * 8
            hi, lo = fx.Float32(0.0), fx.Float32(0.0)
            for h in range_constexpr(4):
                r = fx.ptr_load(rl + t * 128 + h * 32 + lane)
                w = fx.Float32(
                    fx.ptr_load(fn + j * (4 * HIDDEN) + h * HIDDEN + task * 32 + lane)
                )
                wh = bf16_round(w)
                wl = bf16_round(w - wh)
                hi = fx.fma(r, wh, hi)
                lo = fx.fma(r, wl, lo)
            total = _sum32(hi) + _sum32(lo)
            if lane == 0:
                fx.ptr_store(total, partial + (t * SLICES + task) * 26 + j)
    fx.gpu.barrier()


@traced
def gate_device(
    partial, scale, base, pre_out, post_out, comb_out, local, tid, row, rows
):
    partial, scale, base, pre_out, post_out, comb_out = [
        _typed(p, fx.Float32)
        for p in (partial, scale, base, pre_out, post_out, comb_out)
    ]
    red = _typed(local, fx.Float32)
    # Twenty groups of eight slices replace 160 serial global loads on one
    # wave. This changes only the FP32 gate reduction tree, not the collapse
    # or attention/index numerical path. All 520 partials fit in 2080B LDS.
    for pas in range_constexpr(3):
        index = tid + pas * 256
        if index < 20 * 26:
            group, j = index // 26, index % 26
            acc = fx.Float32(0.0)
            for k in range_constexpr(8):
                acc = acc + fx.ptr_load(
                    partial + (row * SLICES + group * 8 + k) * 26 + j
                )
            fx.ptr_store(acc, red + index)
    fx.gpu.barrier()
    if tid < 32:
        lane = tid
        j = (lane < 16).select(lane % 8, lane - 8)
        mix, sq = fx.Float32(0.0), fx.Float32(0.0)
        for group in range_constexpr(20):
            mix = mix + fx.ptr_load(red + group * 26 + j)
            sq = sq + fx.ptr_load(red + group * 26 + 24)
        rstd = hw_rsq(fx.fma(sq, fx.Float32(1.0 / (4 * HIDDEN)), fx.Float32(1e-20)))
        s = fx.ptr_load(scale + (lane < 4).select(0, (lane < 8).select(1, 2)))
        v = fx.fma(mix * rstd, s, fx.ptr_load(base + j))
        sig = hw_rcp(fx.Float32(1.0) + hw_exp2(-v * fx.Float32(1.4426950408889634)))
        if lane < 4:
            fx.ptr_store(sig + fx.Float32(1e-6), pre_out + row * 4 + lane)
        if (lane >= 4) & (lane < 8):
            fx.ptr_store(sig * fx.Float32(2.0), post_out + row * 4 + lane - 4)
        m = fx.max(v, v.shuffle_xor(1, 32))
        m = fx.max(m, m.shuffle_xor(2, 32))
        p = hw_exp2((v - m) * fx.Float32(1.4426950408889634))
        srow = p + p.shuffle_xor(1, 32)
        srow = srow + srow.shuffle_xor(2, 32)
        p = fx.fma(p, hw_rcp(srow), fx.Float32(1e-6))
        scol = p + p.shuffle_xor(4, 32)
        scol = scol + scol.shuffle_xor(8, 32)
        p = p * hw_rcp(scol + fx.Float32(1e-6))
        for _ in range_constexpr(19):
            srow = p + p.shuffle_xor(1, 32)
            srow = srow + srow.shuffle_xor(2, 32)
            p = p * hw_rcp(srow + fx.Float32(1e-6))
            scol = p + p.shuffle_xor(4, 32)
            scol = scol + scol.shuffle_xor(8, 32)
            p = p * hw_rcp(scol + fx.Float32(1e-6))
        if lane >= 16:
            fx.ptr_store(p, comb_out + row * 16 + lane - 16)
    fx.gpu.barrier()
