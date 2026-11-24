# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The all-reduce half of the fused TP4 all-reduce + add + Gemma RMSNorm kernels: AITER's ``cross_device_reduce_2stage``
bits (chunk c = rows [c q, (c + 1) q) summed by rank c in fp32 in the order c, c + 1, c + 2, c + 3, denormals flushed,
one bf16 rounding), and the flag protocol of ``ar_add_rmsnorm`` (rows % 4 == 0).

There a workgroup owns one row b. On the rank c that owns b's chunk (the reducer) it pulls row b of the 4 inputs, sums
it and publishes it in its buffer; on the 3 other ranks (the pullers) it pulls the reduced row from rank c. Every rank
owns one uncached buffer mapped by the peers: the flag words (a 128-byte line per row: polls on shared lines slowed the
flag stores), then its reduced rows (chunk ``me``). A flag holds the row's call count (kept on the device, the same on
every rank), so the handshake replays in graphs: ARR (b, s) on the reducer = row b's puller on rank s has started (its
previous read of the slot is done and rank s's input is complete), RED (b) on a puller = row b is published."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.common import H, pk_bf16

WORLD = 4
SYS = 0x11  # sc0 | sc1: system-coherent loads, write-through stores
ROW_DW = H // 2  # dwords per bf16 row
MAX_ROWS = 1024
MAX_Q = MAX_ROWS // WORLD
LINE = 32  # flag words per row: each row's flags in their own 128-byte line
ARR = 0  # flag word b * LINE + s
RED = MAX_ROWS * LINE  # flag word RED + b * LINE
FLAG_BYTES = 2 * MAX_ROWS * LINE * 4
RED_OFF = FLAG_BYTES
BUF_BYTES = RED_OFF + MAX_Q * H * 2


def _gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def _flag_store(addr, v):
    """An atomic exchange at system scope (write-through)."""
    _llvm.AtomicRMWOp(
        _llvm.AtomicBinOp.xchg,
        _gptr(addr).llvm_ptr,
        fx.Int32(v).ir_value(),
        _llvm.AtomicOrdering.monotonic,
        syncscope="",
        alignment=4,
    )


def sel4(p, a0, a1, a2, a3):
    return (p == 0).select(a0, (p == 1).select(a1, (p == 2).select(a2, a3)))


def _flag_read(addr):
    """An atomic add of 0 at system scope (a spin on it is never hoisted)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _gptr(addr).llvm_ptr,
            fx.Int32(0).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


@flyc.jit
def _spin(a, e):
    f = _flag_read(a)
    while (f - e) < 0:  # signed difference: the counts may wrap
        f = _flag_read(a)


@flyc.jit
def arrive(b0, b1, b2, b3, c, me, b, e, puller):
    """A puller's lane 0 stores e into ARR (b, me) on the reducer c."""
    if puller & (fx.thread_idx.x == 0):
        _flag_store(sel4(c, b0, b1, b2, b3) + fx.Int64((ARR + b * LINE + me) * 4), e)


@flyc.jit
def wait_pullers(mine, me, b, e, reducer):
    """The reducer's lanes 0-2 spin until ARR (b, s) of the 3 pullers reach e, then the barrier. A poll waits for the
    wave's outstanding vector loads (vmcnt): issue none before it."""
    lane = fx.thread_idx.x
    if reducer & (lane < WORLD - 1):
        _spin(mine + fx.Int64((ARR + b * LINE + (me + 1 + lane) % WORLD) * 4), e)
    fx.gpu.barrier()


@flyc.jit
def publish(b0, b1, b2, b3, me, b, e, reducer):
    """The reducer's lanes 0-2 store e into RED (b) on the 3 pullers."""
    lane = fx.thread_idx.x
    if reducer & (lane < WORLD - 1):
        p = (me + 1 + lane) % WORLD
        _flag_store(sel4(p, b0, b1, b2, b3) + fx.Int64((RED + b * LINE) * 4), e)


@flyc.jit
def wait_reduced(mine, b, e):
    """A puller's lane 0 spins until RED (b) reaches e; then the barrier."""
    if fx.thread_idx.x == 0:
        _spin(mine + fx.Int64((RED + b * LINE) * 4), e)
    fx.gpu.barrier()


def load_ptrs(arg_xs):
    """The 4 ranks' input addresses from the int64[4] at arg_xs (scalar loads)."""
    tr = buffer_ops.create_buffer_resource_from_addr(arg_xs, num_records_bytes=32)
    out = []
    for h in range_constexpr(2):
        v = Vec(buffer_ops.buffer_load(tr, h * 4, vec_width=4, is_scalar=True))
        for i in range_constexpr(2):
            lo = fx.Int64(fx.Int32(v[2 * i])) & fx.Int64(0xFFFFFFFF)
            out.append(lo | (fx.Int64(fx.Int32(v[2 * i + 1])) << 32))
    return out


def unrotate(rot, vs):
    """The 3 packs loaded in rotated order (slot t = peer 1 + (t + rot) % 3) back in peer order 1, 2, 3."""
    out = []
    for i in range_constexpr(3):
        a, b, c = vs[i], vs[(i + 2) % 3], vs[(i + 1) % 3]
        out.append(
            Vec.from_elements(
                [
                    (rot == 0).select(
                        fx.Int32(a[d]),
                        (rot == 1).select(fx.Int32(b[d]), fx.Int32(c[d])),
                    )
                    for d in range_constexpr(4)
                ],
                fx.Int32,
            )
        )
    return out


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _ftz(u):
    """fp32 bits with a denormal flushed to a zero of its sign."""
    return ((u & 0x7F800000) == 0).select(u & fx.Int32(-0x80000000), u)


def _add_ftz(a, b):
    return _ftz((_as_f32(a) + _as_f32(_ftz(b))).bitcast(fx.Int32))


def sum4(packs):
    """bf16(((p0 + p1) + p2) + p3) per element of four 16 B packs in fp32, with AITER's denormal flush
    (-fgpu-flush-denormals-to-zero: operands and sums)."""
    out = []
    for i in range_constexpr(4):
        ws = [fx.Int32(p[i]) for p in packs]
        lo = _ftz(ws[0] << 16)
        hi = _ftz(ws[0] & fx.Int32(-65536))
        for r in range_constexpr(1, WORLD):
            lo = _add_ftz(lo, ws[r] << 16)
            hi = _add_ftz(hi, ws[r] & fx.Int32(-65536))
        out.append(pk_bf16(_as_f32(lo), _as_f32(hi)))
    return Vec.from_elements(out, fx.Int32)


def _enter_fp32_ftz(packs):
    """Save the wave's FP32 denormal mode and flush denormals (operands and sums) until the restore.

    Every source pack crosses this opaque identity asm, so LLVM cannot move a rank-sum op above the mode change."""
    n = len(packs)
    ty = ir.Type.parse(
        "!llvm.struct<(" + ", ".join(["vector<4xi32>"] * n + ["i32"]) + ")>"
    )
    asm = (
        f"s_getreg_b32 ${n}, hwreg(HW_REG_MODE,4,2)\n"
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE,4,2), 0"
    )
    constraints = ",".join(["=v"] * n + ["=s"] + [str(i) for i in range(n)])
    result = _llvm.inline_asm(
        ty, [v.ir_value() for v in packs], asm, constraints, has_side_effects=True
    )
    values = [
        Vec(_llvm.extractvalue(T.vec(4, T.i32), result, [i])) for i in range(n)
    ]
    return values, fx.Int32(_llvm.extractvalue(T.i32, result, [n]))


def _restore_fp32_denorm_mode(packs, mode):
    """Restore the saved mode after all rank sums: the sums cross this identity asm, so the residual add and the norm
    run in the restored mode."""
    n = len(packs)
    ty = ir.Type.parse(
        "!llvm.struct<(" + ", ".join(["vector<4xi32>"] * n) + ")>"
    )
    constraints = ",".join(["=v"] * n + ["s"] + [str(i) for i in range(n)])
    result = _llvm.inline_asm(
        ty,
        [mode.ir_value()] + [v.ir_value() for v in packs],
        f"s_setreg_b32 hwreg(HW_REG_MODE,4,2), ${n}",
        constraints,
        has_side_effects=True,
    )
    return [Vec(_llvm.extractvalue(T.vec(4, T.i32), result, [i])) for i in range(n)]


def _sum4_hardware_ftz(packs):
    """The fixed-order BF16 rank sum; hardware FP32 FTZ is active only here."""
    out = []
    for i in range_constexpr(4):
        ws = [fx.Int32(p[i]) for p in packs]
        lo = (ws[0] << 16).bitcast(fx.Float32)
        hi = (ws[0] & fx.Int32(-65536)).bitcast(fx.Float32)
        for r in range_constexpr(1, WORLD):
            lo = lo + (ws[r] << 16).bitcast(fx.Float32)
            hi = hi + (ws[r] & fx.Int32(-65536)).bitcast(fx.Float32)
        out.append(pk_bf16(lo, hi))
    return Vec.from_elements(out, fx.Int32)
