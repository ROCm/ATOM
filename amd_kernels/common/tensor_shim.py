# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Launch helpers: ``ptr_arg`` (a tensor as a raw kernel pointer) and ``_run_compiled`` (compile on the first call,
dispatch the cached function after)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir


def ptr_arg(t: torch.Tensor, dtype=None):
    """Wrap a torch.Tensor as an fx.Pointer (PointerJitArg) for kernel launch."""
    if dtype is None:
        dtype = fx.Uint8
    type_name = type(t).__name__
    module_name = type(t).__module__
    if type_name == "FakeTensor" or "fake_tensor" in module_name:
        return flyc.from_c_void_p(dtype, 0)
    return flyc.from_c_void_p(dtype, t.data_ptr())


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise
