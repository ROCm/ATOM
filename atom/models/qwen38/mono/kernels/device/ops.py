# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ATOM atom/mono/device/ops.py (MIT, Copyright (C) 2026 Advanced
# Micro Devices, Inc.), itself ported from the FlyDSL GLM-5 layer kernel.
"""L0 device primitives for gfx950 monokernels: buffer resources, global loads / stores, opaque copies,
torch-order reductions, f64 math and MFMA, the kernel symbol.

The register-only primitives (lane shuffles and reductions, hardware math, bf16 / fp8 packing, the bf16 MFMA, the
uniform / opaque moves, ``global_load``, ``spin_pause``, ``traced``) are main's (``atom.mono.device.ops``); ``rsrc``
and ``ld_i32`` stay here, on this package's ``buffer_ops`` (FlyDSL's typed-pointer generation).

Every function except ``traced`` and ``kernel_symbol`` is called inside a traced
``@flyc.kernel`` body. Functions with dynamic control flow are ``@traced``
(FlyDSL's AST rewrite) so their ``if`` / ``while`` lower to scf ops.

Tracer rule: inside dynamic control flow, never call a method on,
or assign an attribute / item of, a *local* object; the rewriter would carry it
as loop state. These helpers are free functions for that reason.
"""

from __future__ import annotations

import re

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.mono.device.ops import (  # noqa: F401 - re-exported: the blocks import them from here
    CM_DEV,
    CM_NT,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fresh,
    global_load,
    hw_exp2,
    hw_rsq,
    lane_gather,
    memrealtime,
    mfma_bf16,
    permlane_swap,
    readlane,
    spin_pause,
    traced,
    uniform,
    wave_max,
    wave_sum,
    xred,
    xshfl,
)


def kernel_symbol(stem: str, **params) -> str:
    """A kernel symbol that profilers show as ``mono::<stem>_<k><v>...``
    (Itanium-mangled: the assembler rejects ``::``)."""
    tail = "_".join(f"{k}{v}" for k, v in params.items())
    name = re.sub(r"[^0-9A-Za-z_]", "_", f"{stem}_{tail}" if tail else stem)
    return f"_ZN4mono{len(name)}{name}E"


# ------------------------------------------------------------------ memory


def rsrc(addr, nbytes=None):
    """A buffer resource over the bytes at device address ``addr``."""
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def ld_i32(ptr, i):
    """Word ``i`` of a buffer, a plain Int32 load."""
    return fx.Int32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32))


def global_store(addr, value):
    """Store ``value`` (an i32 / bf16 scalar or vector) at a 64-bit global address."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr)))
    v = as_ir_value(value)
    t = v.type
    nbytes = (t.shape[0] if isinstance(t, ir.VectorType) else 1) * (
        (t.element_type if isinstance(t, ir.VectorType) else t).width // 8
    )
    llvm.StoreOp(v, ptr.result, alignment=nbytes)


# ------------------------------------------------------------------ opaque copies


def fresh_s(v):
    """A copy of the uniform Int32 ``v`` that LLVM cannot merge with another copy."""
    asm = llvm.InlineAsmOp(
        T.i32, [fx.Int32(v).ir_value()], "; fresh $0", "=s,0", has_side_effects=True
    )
    return fx.Int32(asm.result)


def fresh_s64(v):
    asm = llvm.InlineAsmOp(
        T.i64, [fx.Int64(v).ir_value()], "; fresh $0", "=s,0", has_side_effects=True
    )
    return fx.Int64(asm.result)


# ------------------------------------------------------------------ rsqrt as torch rounds it


def rsqrt_rn(x):
    """``torch.rsqrt`` of an fp32 value as torch computes it on the GPU: correctly rounded,
    ``f32(1 / sqrt(f64(x)))`` (bit-equal on 10M random inputs, torch 2.14 / ROCm 10.1, MI355X).
    ``hw_rsq`` (``v_rsq_f32``) is ~1 ulp off on some inputs: reference norms use this.
    """
    from flydsl.expr import math as fmath

    return (fx.Float64(1.0) / fx.Float64(fmath.sqrt(fx.Float32(x).to(fx.Float64)))).to(
        fx.Float32
    )


def rsqrt_rn_pos(x):
    """:func:`rsqrt_rn` of a positive finite fp32 ``x`` (a norm's ``mean + eps``), the same value
    without LLVM's f64-sqrt range scaling and class checks: they never apply to an fp32 input
    above 0 (the scaling starts below 2^-767), but their f64 threshold is a constant LLVM hoists
    into an SGPR pair held across a whole persistent kernel (spills). The sqrt is LLVM's own sequence (``v_rsq_f64``,
    then Newton steps to the correctly rounded root), the division the IEEE f64 one."""
    from flydsl.expr import math as fmath

    xd = fx.Float32(x).to(fx.Float64)
    y = fx.Float64(
        llvm.call_intrinsic(T.f64, "llvm.amdgcn.rsq.f64", [xd.ir_value()], [], [])
    )
    g = xd * y
    h = y * fx.Float64(0.5)
    r = fx.Float64(fmath.fma(-h, g, fx.Float64(0.5)))
    g = fx.Float64(fmath.fma(g, r, g))
    h = fx.Float64(fmath.fma(h, r, h))
    for _ in range(2):
        d = fx.Float64(fmath.fma(-g, g, xd))
        g = fx.Float64(fmath.fma(d, h, g))
    return (fx.Float64(1.0) / g).to(fx.Float32)


# ------------------------------------------------------------------ torch-order reductions


def row_sum_torch(vals, lane=None):
    """The fp32 sum of a row of 256 elements held by one wave (lane ``l`` holds elements ``4 l ..``, in order) in
    **torch's GPU reduction order** for ``.sum(-1)`` / ``.mean(-1)`` of a contiguous fp32 row, so a kernel matches the
    reference's ``rms_norm`` (``x.square().mean(-1)``) bit for bit. The rule (torch 2.14 / ROCm 10.1, MI355X; every
    output count from 1 to 65536 rows tested): 64 threads each sum 4 consecutive elements in order, ``((a + b) + c)
    + d``, then the threads pair **adjacent first** (xor offsets 1 .. 32). Every lane gets the sum. (``wave_sum``
    pairs 32 first: ~40 % of rows end 1 ulp off.) ``lane``: unused."""
    if len(vals) != 4:
        raise ValueError(
            f"row_sum_torch: rows of 256 (4 a lane), got {len(vals)} a lane"
        )
    s = ((vals[0] + vals[1]) + vals[2]) + vals[3]
    return butterfly(s, (1, 2, 4, 8, 16, 32))


def torch_sum_lanes(
    width: int, outputs: int, max_threads: int = 512, warp: int = 64
) -> int:
    """Threads per output row of **torch's GPU sum / mean** of an fp32 tensor's contiguous last
    dimension ``width`` (> 128: its loads vectorized by 4) when the call reduces ``outputs``
    rows (torch's ``setReduceConfig`` / ``set_block_dimension``, ROCm: 512 threads, 64-wide
    warps): the block is ``lanes`` x ``block_height`` threads, ``lanes = min(p0, 512 /
    min(pow2(outputs), 8))`` with ``p0 = min(pow2(width / 4), 512)``. Width 2560:
    128 lanes at 4 outputs, 64 at 8 or more (tested on every count up to 4096 and on
    real rows). Raises where torch would split a row across warps or blocks (not implemented).
    """

    def pow2(n):
        return 1 << (int(n).bit_length() - 1)

    if width <= 128 or width % 4:
        raise ValueError(
            f"torch_sum_lanes: rows of width > 128 and a multiple of 4, got {width}"
        )
    dim0 = width // 4
    p0 = min(pow2(dim0), max_threads)
    p1 = min(pow2(outputs), max_threads)
    lanes = min(p0, warp)
    height = min(p1, max_threads // lanes)
    lanes = min(p0, max_threads // height)
    per = -(-dim0 // lanes)
    if per >= height * 16 or per >= 256:
        raise ValueError(
            f"torch_sum_lanes: torch splits a {width}-wide row across warps at {outputs} outputs"
        )
    return lanes


def torch_thread_sum(term4, width: int, lanes: int, lane, m: int):
    """Torch's thread ``t = lane + 64 m``'s part of a row sum: its 4
    running sums over its vectors ``t, t + lanes, ...`` (in that order), combined ``((a0 + a1) +
    a2) + a3``. ``term4(v, c)``: the 4 fp32 terms of the row's elements ``4 v .. 4 v + 3``, ``v =
    lane + 64 c`` (``c`` a Python int: the lane's ``c``-th vector, for operands in registers).
    """
    nvec = width // 4
    acc = None
    for j in range(-(-nvec // lanes)):
        v0 = m * 64 + j * lanes  # this lane's vector: v0 + lane
        if v0 >= nvec:
            break
        if v0 + 64 > nvec:
            raise ValueError(
                f"torch_thread_sum: a row of {nvec} vectors ends inside a wave's {v0} .."
            )
        v = lane + v0
        t = term4(v, v0 // 64)
        acc = t if acc is None else [a + x for a, x in zip(acc, t, strict=True)]
    return ((acc[0] + acc[1]) + acc[2]) + acc[3]


def torch_tree(vals):
    """The rest of a torch row sum from the threads' parts ``vals`` (``vals[m]``: thread ``lane +
    64 m``'s): threads above 64 fold down by halving (``t += t + off``, ``off = lanes / 2 .. 64``:
    torch's shared memory), then lanes pair adjacent first (``shfl_down`` 1 .. 32, here the xor
    butterfly: every lane gets lane 0's value)."""
    while len(vals) > 1:
        half = len(vals) // 2
        vals = [vals[i] + vals[i + half] for i in range(half)]
    return butterfly(vals[0], (1, 2, 4, 8, 16, 32))


# ------------------------------------------------------------------ f64 MFMA


def _no_neg():
    return ir.Attribute.parse("#rocdl<mfma_neg_modifier none>")


def mfma_f64_16x16x4(a, b, c):
    """``D = A B + C`` in f64 (IEEE products and sums), 16 x 16 x 4. Lane ``l`` holds ``A[l % 16][l
    // 16]`` (``a``), ``B[l // 16][l % 16]`` (``b``) and ``D[l // 16 + 4 e][l % 16]`` (element ``e``
    of ``c`` / the result, a vector of 4 f64). Throughput ~64 cycles a SIMD (gfx950: f64 matrix
    rate = f64 vector rate), dependent latency ~180."""
    r = rocdl.mfma_f64_16x16x4f64(
        T.vec(4, T.f64),
        fx.Float64(a).ir_value(),
        fx.Float64(b).ir_value(),
        c.ir_value(),
        0,
        0,
        _no_neg(),
    )
    return fx.Vector(r.result)


def mfma_f64_4x4x4(a, b, c):
    """Four independent ``D_g = A_g B_g + C_g`` in f64, 4 x 4 x 4 each (``g`` = ``(l // 4) % 4``):
    lane ``16 k + 4 g + i`` holds ``A_g[i][k]``, lane ``16 k + 4 g + j`` holds ``B_g[k][j]``, lane
    ``16 i + 4 g + j`` holds ``D_g[i][j]`` (``c`` / the result, one f64). Throughput ~16 cycles a
    SIMD, dependent latency ~160 (several accumulation chains needed)."""
    r = rocdl.mfma_f64_4x4x4f64(
        T.f64,
        fx.Float64(a).ir_value(),
        fx.Float64(b).ir_value(),
        fx.Float64(c).ir_value(),
        0,
        0,
        _no_neg(),
    )
    return fx.Float64(r.result)
