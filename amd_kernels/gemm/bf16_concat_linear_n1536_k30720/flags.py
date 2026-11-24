# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""System-scope epoch flags, copied from the all-reduce family."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

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


