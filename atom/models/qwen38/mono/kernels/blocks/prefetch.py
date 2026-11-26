# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``Prefetch``: touch the bytes of a buffer a *later launch* reads (the next layer's attention-HC down weight), one
dword load a 128-byte line with the LRU cache policy, so that launch's own (nontemporal) loads find them in the
Infinity Cache instead of HBM. It reads no channel and writes nothing (the loads are kept alive by an empty asm use).

Args: ``<name>.w`` the buffer, ``<name>.n`` an int32 [1] tensor holding its byte count (loads past it read 0 without
touching memory). One stage, ``tasks`` tasks from CTA ``base``; task t touches lines ``t, t + tasks, ..`` of the first
``nbytes``.
"""

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device.ops import fresh, rsrc
from atom.models.qwen38.mono.kernels.program import Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X

THREADS = MI355X.threads
LINE = 128


@dataclass(frozen=True)
class _PPlan:
    name: str
    nbytes: int
    tasks: int


def _keep(vals):
    """An empty asm use of the loaded words: the loads stay (their values are never used otherwise)."""
    ins = [fx.Int32(v).ir_value() for v in vals]
    refs = " ".join(f"${i}" for i in range(len(ins)))
    llvm.InlineAsmOp(
        None, ins, f"; keep {refs}", ",".join(["v"] * len(ins)), has_side_effects=True
    )


def _task(k, t, p: _PPlan):
    n = fx.Int32(
        bo.buffer_load(rsrc(k.args[f"{p.name}.n"]), 0, vec_width=1, dtype=T.i32)
    )
    r = rsrc(k.args[f"{p.name}.w"], n)
    lines = -(-p.nbytes // LINE)
    per = -(-lines // (p.tasks * THREADS))
    q = []
    for u in range(per):
        line = (
            fresh(k.tid) + u * THREADS
        ) * p.tasks + t  # line-interleaved over the tasks
        q.append(
            bo.buffer_load(
                r, line * (LINE // 4), vec_width=1, dtype=T.i32, cache_modifier=0
            )
        )
    _keep(q)  # (every touch in flight: 4 a thread at 8 MiB over 32 tasks)


class Prefetch(Block):
    """A later launch's buffer touched into the caches (module docstring)."""

    def __init__(self, name: str, nbytes: int, tasks: int, base: int | None = None):
        if nbytes <= 0 or not 0 < tasks <= MI355X.cus:
            raise ValueError(f"Prefetch {name}: nbytes > 0, 0 < tasks <= {MI355X.cus}")
        self.name, self.nbytes, self.tasks, self.base = name, nbytes, tasks, base

    def stages(self, sc: Scenario) -> list[Stage]:
        p = _PPlan(self.name, self.nbytes, self.tasks)
        return [Stage(self.name, self.tasks, partial(_task, p=p), args=(f"{self.name}.w", f"{self.name}.n"),
                      base=self.base, key=("prefetch", p))]  # fmt: skip
