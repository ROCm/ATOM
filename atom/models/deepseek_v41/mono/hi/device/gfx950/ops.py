# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ATOM atom/mono/device/ops.py (MIT, Copyright (C) 2026 Advanced
# Micro Devices, Inc.), itself ported from the FlyDSL GLM-5 layer kernel.
"""L0 device primitives for gfx950 monokernels: buffer resources, wave / CTA
reductions and shuffles, hardware math, bf16 / fp8 packing, MFMA wrappers.

The primitives ATOM's mono kernels share and this package's code calls unchanged (``traced``, the uniform / opaque
moves, ``memrealtime``, the hardware math, lane permutes and ballots, bf16 / fp8 packing, the bf16 MFMA) are
``atom.mono.device.ops``'s, re-exported here; the rest bind this package's ``buffer_ops`` / ``dpp_utils`` or differ.

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
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from atom.mono.device.ops import (  # noqa: F401 - re-exported: the blocks import them from here
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_log2,
    hw_rcp,
    hw_rsq,
    memrealtime,
    mfma_bf16,
    permlane_swap,
    popcount,
    readlane,
    traced,
    uniform,
)

from ...plan import trace
from ...target import MI355X
from . import buffer_ops as bo
from . import dual
from .dpp_utils import update_dpp_i32

THREADS = MI355X.threads
WAVES = MI355X.waves

# gfx950 cache-policy bits (LLVM CPol): SC0 = 1, SC1 = 16, NT = 2. SC1 alone is
# device scope (past the per-XCD L2s); SC0 | SC1 is system scope (peers).
CM_DEV = 16
CM_SYS = 17
CM_NT = 2  # nontemporal: weights a step streams once


def kernel_symbol(stem: str, **params) -> str:
    """A kernel symbol that profilers show as ``dsv41hi::<stem>_<k><v>...``
    (Itanium-mangled: the assembler rejects ``::``)."""
    tail = "_".join(f"{k}{v}" for k, v in params.items())
    name = re.sub(r"[^0-9A-Za-z_]", "_", f"{stem}_{tail}" if tail else stem)
    return f"_ZN7dsv41hi{len(name)}{name}E"


# ------------------------------------------------------------------ memory


def rsrc(addr, nbytes=None):
    """A buffer resource over the bytes at device address ``addr``."""
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def slot_base(ptr, b, slot_bytes: int):
    """Request ``b``'s slot (``slot_bytes`` a slot) of a per-request cache at device address ``ptr``: its address."""
    return fx.Int64(ptr) + fx.Int64(b) * slot_bytes


def ld_i32(ptr, i, cm=0):
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


def rope_sin(name: str) -> str:
    """The pointer-table entry of a RoPE table's sine half (ATOM's ``RotaryEmbedding.sin_cache``) beside ``name``, its
    cosine half (``cos_cache``)."""
    return name + ".sin"


def rope_cs(k, name: str, pos, pair, n: int = 1):
    """(cos, sin) as f32 of the rotary pairs ``pair .. pair + n - 1`` at position ``pos``, interleaved (c0, s0, c1,
    ..): read in place from ATOM's RoPE caches (``RotaryEmbedding``: bf16 [positions, 32], the table ``name`` and
    ``rope_sin(name)``; a bf16 widens to f32 exactly). ``n`` 2: ``pair`` even (one word a half).
    """
    rc, rs = rsrc(k.args[name]), rsrc(k.args[rope_sin(name)])
    i = pos * 32 + pair
    if n == 2:
        wc = fx.Int32(bo.buffer_load(rc, i // 2, vec_width=1, dtype=T.i32))
        ws = fx.Int32(bo.buffer_load(rs, i // 2, vec_width=1, dtype=T.i32))
        lo, hi = (lambda w: (w << 16).bitcast(fx.Float32)), (
            lambda w: (w & fx.Int32(-65536)).bitcast(fx.Float32)
        )
        return fx.Vector.from_elements([lo(wc), lo(ws), hi(wc), hi(ws)], fx.Float32)
    if n != 1:
        raise ValueError(f"rope_cs: {n} pairs (1 or 2)")
    odd = (i & 1) == 1
    vals = []
    for r in (rc, rs):
        w = fx.Int32(bo.buffer_load(r, i // 2, vec_width=1, dtype=T.i32))
        vals.append(odd.select(w & fx.Int32(-65536), w << 16).bitcast(fx.Float32))
    return fx.Vector.from_elements(vals, fx.Float32)


def load_ptr64(table, i):
    """Entry ``i`` of an int64 table in memory, wave-uniform."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def global_load(addr, words=1, nt=False):
    """``words`` i32 at a 64-bit global address (reaches past a buffer's 4 GiB)."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    trace.mem(
        "load", addr, CM_NT if nt else 0, 4 * words
    )  # the memory-model record (no IR)
    dual.global_access(addr)  # a dual build: not on a dual region (no IR)
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr)))
    return llvm.LoadOp(
        ty, ptr.result, alignment=4 * words, nontemporal=nt or None
    ).result


# ------------------------------------------------------------------ uniformity / ordering


def fresh_after(v, deps):
    """``fresh(v)``, also after the per-lane f32 / Int32 values ``deps`` exist: math on ``v`` (e.g. a poll's address)
    waits for them, so a chain of polls and sums runs poll, sum, poll (the sums not sunk below every poll, their
    loaded words all live)."""
    ins = [fx.Int32(v).ir_value()] + [(d.bitcast(fx.Int32) if isinstance(d, fx.Float32) else fx.Int32(d)).ir_value()
                                       for d in deps]  # fmt: skip
    refs = " ".join(f"${i + 2}" for i in range(len(deps)))
    cons = "=v," + ",".join(["v"] * len(ins))
    return fx.Int32(
        llvm.inline_asm(
            T.i32, ins, f"v_mov_b32 $0, $1 ; {refs}", cons, has_side_effects=True
        )
    )


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


def spin_pause():
    """An opaque, side-effecting op inside a poll loop: keeps its loads from being hoisted."""
    llvm.InlineAsmOp(None, [], "s_nop 0", "", has_side_effects=True)


# ------------------------------------------------------------------ hardware math


def rsqrt_rn_pos(x):
    """The correctly rounded ``1 / sqrt(x)`` of a positive finite fp32 ``x`` (a norm's ``mean + eps``),
    without LLVM's f64-sqrt range scaling and class checks: they never apply to an fp32 input
    above 0 (the scaling starts below 2^-767), but their f64 threshold is a constant LLVM hoists
    into an SGPR pair held across a whole persistent kernel (HCIn's norm: 2 SGPR spills seen in a
    composed program). The sqrt is LLVM's own sequence (``v_rsq_f64``,
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


def div_rn(x, d, r):
    """``x / d`` correctly rounded, given ``r = 1 / d`` from one true division."""
    from flydsl.expr import math as fmath

    q = x * r
    return fx.Float32(fmath.fma(fx.Float32(fmath.fma(-q, d, x)), r, q))


# ------------------------------------------------------------------ lanes


def lds_read_tr16(ptr):
    """gfx950 ``ds_read_b64_tr_b16`` at the LDS pointer ``ptr`` (each lane's address of 4 consecutive b16) -> 2 i32
    (b16 pairs, low half first). The transpose works on each 16-lane group: lane ``4 a + b`` gets element ``b`` of the
    4-element rows addressed by the group's lanes ``a``, ``a + 4``, ``a + 8``, ``a + 12``, in that order (measured on
    MI355X). So for an MFMA B operand (lane ``n`` the column, element ``e`` the K index) from a K-major LDS matrix
    (a row a K index): group lane ``j`` addresses row ``K0 + j // 4`` at column ``4 (j % 4)``.
    """
    v = fx.Vector(
        rocdl.ds_read_tr16_b64(T.vec(4, T.i16), fx.to_llvm_ptr(ptr)).result
    ).bitcast(fx.Int32)
    return [fx.Int32(v[0]), fx.Int32(v[1])]


def xshfl(v, off):
    """Lane ``lane ^ off``'s v: DPP within a 16-lane row, permlane swaps across rows."""
    assert off in (1, 2, 4, 8, 16, 32), off
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 8:
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def xred(v, off, op):
    """op(v, v of lane ^ off) for a symmetric op; offsets 32 / 16 use both halves of one swap."""
    if off < 16:
        return op(v, xshfl(v, off))
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    a, b = permlane_swap(off, x, x)
    if is_f:
        return op(a.bitcast(fx.Float32), b.bitcast(fx.Float32))
    return op(type(v)(a), type(v)(b))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over ``offsets`` in that order: the order is the numerics contract."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def torch_sum_lanes(
    width: int, outputs: int, max_threads: int = 512, warp: int = 64
) -> int:
    """Threads per output row of **torch's GPU sum / mean** of an fp32 tensor's contiguous last
    dimension ``width`` (> 128: its loads vectorized by 4) when the call reduces ``outputs``
    rows (torch's ``setReduceConfig`` / ``set_block_dimension``, ROCm: 512 threads, 64-wide
    warps): the block is ``lanes`` x ``block_height`` threads, ``lanes = min(p0, 512 /
    min(pow2(outputs), 8))`` with ``p0 = min(pow2(width / 4), 512)``. Width 2560 (the hc measurements
    H-numbers): 128 lanes at 4 outputs, 64 at 8 or more (tested on every count up to 4096 and on
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
    """Torch's thread ``t = lane + 64 m``'s part of a row sum (the rest: :func:`torch_tree`): its 4
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
        tail = v0 + 64 > nvec  # some lanes past the row: their terms are 0 (exact)
        v = fx.min(lane + v0, fx.Int32(nvec - 1)) if tail else lane + v0
        t = term4(v, v0 // 64)
        if tail:
            live = (lane + v0) < nvec
            t = [live.select(x, fx.Float32(0.0)) for x in t]
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


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def wave_max(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)


def _xpartner(v, off, lane):
    """Lane ``lane ^ off``'s Int32 v: DPP within a row, v_permlane*_swap across rows."""
    if off < 16:
        return xshfl(v, off)
    a, b = permlane_swap(off, v, v)
    return ((lane & off) != 0).select(a, b)


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order."""
    blk = v
    for off in (1, 2, 4, 8, 16, 32):
        o = _xpartner(blk, off, lane)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


@traced
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of the Int32 v over the CTA in thread order, CTA total); ``buf``: WAVES
    words of LDS."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == 63:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


# ------------------------------------------------------------------ bf16 / fp8 packing


# ------------------------------------------------------------------ MFMA
