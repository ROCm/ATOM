# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A built, checked, launchable layer program."""

from dataclasses import dataclass, field

import torch

from atom.models.qwen38.mono.kernels.compile import (
    ResourceError,
    compile_traced,
    gate,
    resources,
)
from atom.models.qwen38.mono.kernels.device.hazards import srcc_hazards
from atom.models.qwen38.mono.kernels.emit import build
from atom.models.qwen38.mono.kernels.isa import code_object, objdump
from atom.models.qwen38.mono.kernels.program import (
    DUAL_PAGE,
    LAYER_SLOTS,
    Program,
    plan,
)
from atom.models.qwen38.mono.kernels.target import device_cus


class TagError(ValueError):
    """A launch would reuse a tag: its pairs could be satisfied by an earlier launch's."""


@dataclass
class Buffers:
    """Per-launch-sequence state: scratch (zeroed once, never cleared again) and the step counter.

    A launch's tag is ``step * LAYER_SLOTS + layer + 1``; it must differ from the tags still in scratch (in
    practice: the previous launch's), so between two launches with the same ``layer`` the step must be bumped
    (:meth:`bump`, graph-capturable). ``launched`` holds the layer ids launched (or captured) since the last
    host-side :meth:`bump`: ``MonoKernel.launch`` refuses one twice. A graph replay launches no Python, so whoever
    replays a graph bumps the step for it."""

    scratch: torch.Tensor
    step: torch.Tensor
    launched: set = field(default_factory=set)
    # launched inside a graph capture (its step tensor is in the graph)
    captured: bool = False

    def bump(self) -> None:
        """A new step: every layer id is fresh again (one device add, capturable)."""
        self.step.add_(1)
        self.launched.clear()


class MonoKernel:
    """``program`` planned, emitted, compiled, contract-checked and resource-gated for MI355X.

    Construction raises ``ContractError`` (a hand-off could deadlock) or ``ResourceError`` (scratch, spills, LDS,
    residency, MFMA hazards in the built code) before anything runs."""

    def __init__(self, program: Program):
        self.program = plan(program)
        t = program.target
        self.launcher = build(program)
        self.record = compile_traced(self.launcher, program)
        self.resources = resources(self.launcher, t.arch)[0]
        gate(
            self.resources,
            t.lds_bytes,
            sgpr_spill_ok=program.sgpr_spill_ok,
            grid=t.grid,
            cus=(
                device_cus()
                if self.resources["max_blocks_per_cu"] is not None
                else None
            ),
        )
        # the built code's MFMA srcC hazards
        bad = srcc_hazards(objdump(code_object(self), t.mcpu))
        if bad:
            raise ResourceError(
                f"{program.name}: {len(bad)} code hazard(s), e.g. {bad[0]}"
            )

    def buffers(self, device="cuda") -> Buffers:
        """Zeroed scratch and a step counter for launches. A dual-copy program's run area starts a 2 MiB page and
        its XCD copy word follows it (``runtime.dual.scratch_word``, calibrated on this allocation).
        """
        p = self.program
        if p.dual_on:
            from atom.models.qwen38.mono.kernels.dual import scratch_word

            raw = torch.zeros(
                p.scratch_bytes + DUAL_PAGE, dtype=torch.uint8, device=device
            )
            shift = -raw.data_ptr() % DUAL_PAGE
            scratch = raw[shift : shift + p.scratch_bytes]
            word = scratch_word(scratch, p.dual_bytes)
            # (as int32 bits: page 3's XCD 7 is bit 31)
            word = word - (1 << 32) if word >= 1 << 31 else word
            scratch[p.dual_bytes : p.dual_bytes + 4].view(torch.int32)[0] = word
        else:
            scratch = torch.zeros(
                max(p.scratch_bytes, 256), dtype=torch.uint8, device=device
            )
        return Buffers(
            scratch=scratch, step=torch.zeros(1, dtype=torch.int32, device=device)
        )

    def table(self, pointers: dict, device="cuda") -> torch.Tensor:
        """The pointer table: ``pointers[name]`` (a tensor or an int address) per ``program.args`` entry, int64, on
        ``device``. It keeps the tensors it addresses alive; keep the table alive while launching.
        """
        missing = set(self.program.args) - set(pointers)
        if missing:
            raise KeyError(f"pointer table needs {sorted(missing)}")
        addr = [
            p.data_ptr() if isinstance(p, torch.Tensor) else int(p)
            for p in (pointers[n] for n in self.program.args)
        ]
        table = torch.tensor(addr + [0], dtype=torch.int64, device=device)
        table._mono_refs = [p for p in pointers.values() if isinstance(p, torch.Tensor)]
        return table

    def launch(
        self, table: torch.Tensor, bufs: Buffers, layer: int = 0, stream=None
    ) -> None:
        """One launch. Each launch needs a fresh tag: a new ``layer`` (0 .. ``LAYER_SLOTS - 1``) within a step, or
        ``bufs.bump()`` between steps; raises ``TagError`` on a layer id already launched since the last bump.
        """
        if not 0 <= layer < LAYER_SLOTS:
            raise TagError(
                f"layer id {layer} outside 0..{LAYER_SLOTS - 1} (it would alias another step's tag)"
            )
        if layer in bufs.launched:
            raise TagError(
                f"{self.program.name}: layer id {layer} launched twice in one step (bump the step)"
            )
        bufs.launched.add(layer)
        if torch.cuda.is_current_stream_capturing():
            bufs.captured = True
        self.launcher(
            bufs.step.data_ptr(),
            layer,
            table.data_ptr(),
            bufs.scratch.data_ptr(),
            0,  # (the kernel's timeline argument: no timeline)
            stream=stream or torch.cuda.current_stream(),
        )
