# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The publish epilogue: the GEMM output lands in an IPC slot of all_to_all_bf16_n50016_ranks4 and the last
workgroup publishes it, so the exchange runs its pull only (no stage copy).

The same protocol and words as that entry's stage_publish kernel: the slot is counter[0] & 1, the payload goes out
with system-scope stores (no dirty XCD L2 lines for the peers' system loads), every workgroup waits for its stores
before lane 0 takes a ticket on flag word 1, and the last arriver resets the ticket, increments the counter and
stores it to flag word 0 with a system-scope store. Tickets and the counter are counters, not output adds.
The tail itself sits in each kernel_publish body (FlyDSL rewrites dynamic control flow only there).
"""
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

SYS_CM = 0x11  # sc0 sc1: system scope


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def flag_add(addr_i64, v):
    """Atomic add at system scope on an uncached word; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(v).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )
