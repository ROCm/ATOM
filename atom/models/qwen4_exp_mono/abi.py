# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Argument groups shared by Qwen stages and complete launch ABIs.

Structured arguments flatten into the existing scalar GPU ABI. The runtime tail
stays explicit; no pointer table or device-side indirection is introduced.
"""

import inspect

from atom.mono.runtime.abi import KernelAbi


class StructuredAbi:
    """One typed field declaration for device arguments and host packing."""

    def __init__(self, argument_type):
        self.argument_type = argument_type
        self.fields = tuple(argument_type.__annotations__)
        self.names = (*self.fields, "scratch", "layer")
        self.flat = KernelAbi(self.names)

    def check(self, kernel, launcher):
        KernelAbi(("a", "scratch", "layer")).check(kernel, launcher)
        for fn in (kernel, launcher):
            py = getattr(fn, "func", None) or getattr(fn, "_func", None)
            annotation = inspect.signature(py).parameters["a"].annotation
            if annotation is not self.argument_type:
                raise TypeError(f"argument type {annotation} != {self.argument_type}")

    def launcher(self, kernel, name, keyed):
        """Build the common persistent launch from this typed argument contract."""
        import flydsl.expr as fx
        from flydsl.expr.typing import Int32, Int64

        from atom.mono.device.ops import launcher
        from atom.mono.plan.execution import BLOCKS, THREADS, WAVES

        argument_type = self.argument_type
        current_stream = fx.Stream(None)
        # Qwen's polling CTAs must not acquire a second resident CTA after a
        # register-lifetime change. This occupancy cap belongs to this kernel ABI.
        value_attrs = {
            "rocdl.waves_per_eu": WAVES // 4,
            "rocdl.flat_work_group_size": f"{THREADS},{THREADS}",
        }

        @launcher(name)
        def launch(
            a: argument_type,
            scratch: Int64,
            layer: Int32,
            stream: fx.Stream = current_stream,
        ):
            _ = keyed
            kernel(a, scratch, layer, value_attrs=value_attrs).launch(
                grid=(BLOCKS,), block=(THREADS,), stream=stream
            )

        self.check(kernel, launch)
        return launch

    def pack(self, values):
        packed = self.flat.pack(values)
        return [self.argument_type(*packed[:-2]), *packed[-2:]]

    def zeros(self):
        return [self.argument_type(*([0] * len(self.fields))), 0, 0]


def argument_values(a):
    """Build a compile-time stage context from the typed launch contract."""
    return {name: getattr(a, name) for name in type(a).__annotations__}


def finish_bindings(a, **overrides):
    values = argument_values(a)
    values.update(overrides)
    return values
