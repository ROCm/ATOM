# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Kernarg preload hints, raw-pointer and LDS-atomic helpers and the DPP wave scan."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

LOG2E = 1.4426950408889634
# launch.compile_hints: the kernel arguments (first 56 bytes) preloaded into SGPRs instead of s_load'ed
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _tensor_addr(t):
    """A launch's tensor argument as a device address (a tensor kernel argument ends the kernarg preload)."""
    return fx.Int64(fx.ptrtoint(fx.get_iter(t)))


def _global_i32_ptr(addr_i64):
    """Typed global i32 pointer at a raw device address (``p[i]`` loads / stores)."""
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _lds_atomic_add_i32(ptr, value):
    """``old = *ptr; *ptr += value`` on an LDS i32 pointer (workgroup scope), returns ``old``; fx has no atomic
    wrapper, so this is a raw LLVM op."""
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


def _global_i32_at(addr_i64, idx):
    return _global_i32_ptr(addr_i64)[idx]


_WAVE_SCAN_DPP = (
    (0x111, 0xF, True),
    (0x112, 0xF, True),
    (0x114, 0xF, True),
    (0x118, 0xF, True),
    (0x142, 0xA, False),
    (0x143, 0xC, False),
)


def wave_incl_scan_i32(x):
    """Inclusive prefix sum of an i32 over the 64 lanes of a wave (DPP, no LDS)."""
    from flydsl._mlir.dialects import rocdl as _rocdl_ops

    zero = _raw(fx.Int32(0))
    for ctrl, row_mask, bound in _WAVE_SCAN_DPP:
        t = _rocdl_ops.update_dpp(T.i32, zero, _raw(x), ctrl, row_mask, 0xF, bound)
        x = x + fx.Int32(t)
    return x
