# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton launches that skip the JIT dispatch for a specialization already seen.

`kernel[grid](*args)` runs all of `JITFunction.run` on every call: it derives
each argument's specialization, formats the compile options into the cache
key, checks the kernel's globals and builds launch metadata, and only then
calls the compiled kernel's launcher. With Triton 3.8 on ROCm that host time
is several times the launch itself (ROCm/ATOM#3429).

`CachedLauncher` sends the first launch of each specialization through
`kernel[grid]` and keeps the compiled kernel under a key made of the facts
Triton's own cache key is made of; a later launch with the same key calls the
compiled kernel's launcher directly. The key holds:

* the current device, and the knobs that enter the compile options
  (`runtime.debug`, `compilation.instrumentation_mode`,
  `compilation.fpsan_homomorphic_casts`) or the specialization
  (`amd.use_buffer_ops`);
* per tensor: its dtype, whether its data pointer is 16-byte aligned, and,
  with buffer ops, whether its storage fits in 2 GiB
  (`HIPBackend.get_tensor_specialization`);
* per int, which must fit in int32: whether it is 1, else whether it is a
  multiple of 16;
* per float, bool or None: the type;
* per constexpr: its type and value.

A type annotation on a runtime parameter (`n: tl.int32`) fixes its type in
Triton's key and keeps (ints, pointers) or drops (floats, bools) the value
specialization above, so the same key still covers it.

A launch this key cannot describe goes through `kernel[grid]` as before:
launch, pre-run or stages-inspection hooks; a kernel with checked globals or
unspecialized parameters; torch.compile tracing; a callable grid; keyword
arguments out of signature order; an int outside int32; a constexpr that is
not an int, float, bool, str or None; any other argument kind.
"""

from operator import itemgetter

import torch
from triton import knobs
from triton.compiler import CompiledKernel
from triton.runtime import driver

_Tensor = torch.Tensor
_INT32 = range(-(2**31), 2**31)
# `HIPBackend.is_within_2gb`: the storage size up to which buffer ops apply.
_BUFFER_OPS_MAX_BYTES = 2**31 - 1
_CONSTEXPR_TYPES = frozenset((int, float, bool, str, type(None)))


def _hooked(hook) -> bool:
    """Whether a launch hook would run; an empty `HookChain` does not."""
    if hook is None:
        return False
    calls = getattr(hook, "calls", None)
    return calls is None or len(calls) > 0


def _picker(positions):
    """`args -> tuple(args[i] for i in positions)`."""
    if len(positions) > 1:
        return itemgetter(*positions)
    if positions:
        (i,) = positions
        return lambda args: (args[i],)
    return lambda args: ()


class CachedLauncher:
    """`kernel[grid](*args, **kwargs, **options)` without Triton's per-call
    dispatch once the specialization has been launched.

    Pass the kernel's arguments positionally; the trailing ones may be
    keywords, in signature order."""

    def __init__(self, kernel, **options) -> None:
        self.kernel = kernel
        self.options = options
        params = getattr(kernel, "params", None) or ()
        self._names = tuple(p.name for p in params)
        self._constexpr = tuple(p.is_constexpr for p in params)
        self._runtime = next(
            (i for i, constexpr in enumerate(self._constexpr) if constexpr),
            len(params),
        )
        # Constexprs ahead of a runtime parameter: pick both out by position.
        self._split = None
        if not all(self._constexpr[self._runtime :]):
            self._split = (
                _picker([i for i, c in enumerate(self._constexpr) if not c]),
                _picker([i for i, c in enumerate(self._constexpr) if c]),
            )
        self._enabled = bool(params) and not (
            kernel.do_not_specialize or kernel.do_not_specialize_on_alignment
        )
        self._kernels = {}
        self._get_stream = None

    def __call__(self, grid, *args, **kwargs) -> None:
        full = args + tuple(kwargs.values()) if kwargs else args
        key = None
        if (
            self._enabled
            and len(full) == len(self._names)
            and (not kwargs or tuple(kwargs) == self._names[len(args) :])
            and type(grid) is tuple
            and not torch.compiler.is_compiling()
            and not self.kernel.pre_run_hooks
            and knobs.runtime.add_stages_inspection_hook is None
            and not _hooked(knobs.runtime.launch_enter_hook)
            and not _hooked(knobs.runtime.launch_exit_hook)
        ):
            key = self._key(full)
            entry = self._kernels.get(key) if key is not None else None
            if entry is not None:
                run, function, metadata, _ = entry
                n = len(grid)
                run(
                    grid[0],
                    grid[1] if n > 1 else 1,
                    grid[2] if n > 2 else 1,
                    self._get_stream(key[0]),
                    function,
                    metadata,
                    None,
                    None,
                    None,
                    *full,
                )
                return
        compiled = self.kernel[grid](*args, **kwargs, **self.options)
        if (
            key is not None
            and isinstance(compiled, CompiledKernel)
            and not self.kernel.used_global_vals
        ):
            self._get_stream = driver.active.get_current_stream
            # The CompiledKernel stays referenced so its module is never
            # unloaded while its function handle is cached.
            self._kernels[key] = (
                compiled.run,
                compiled.function,
                compiled.packed_metadata,
                compiled,
            )

    def _key(self, args) -> tuple | None:
        """Flat: a tensor adds (dtype, aligned, buffer-ops range), any other
        runtime argument one string or type, so the layout is unambiguous."""
        buffer_ops = knobs.amd.use_buffer_ops
        if self._split is None:
            runtime, constexprs = args[: self._runtime], tuple(args[self._runtime :])
        else:
            runtime, constexprs = self._split[0](args), self._split[1](args)
        constexpr_types = tuple(map(type, constexprs))
        if not _CONSTEXPR_TYPES.issuperset(constexpr_types):
            return None
        key = [
            torch.cuda.current_device(),
            knobs.runtime.debug,
            knobs.compilation.instrumentation_mode,
            knobs.compilation.fpsan_homomorphic_casts,
            buffer_ops,
            constexprs,
            constexpr_types,
        ]
        append = key.append
        for arg in runtime:
            cls = arg.__class__
            if cls is _Tensor or isinstance(arg, _Tensor):
                append(arg.dtype)
                append(not arg.data_ptr() & 15)
                append(
                    buffer_ops
                    and arg.untyped_storage().nbytes() <= _BUFFER_OPS_MAX_BYTES
                )
            elif cls is int:
                if arg not in _INT32:
                    return None
                append("1" if arg == 1 else "" if arg & 15 else "D")
            elif cls is float or cls is bool or arg is None:
                append(cls)
            else:
                return None
        return tuple(key)
