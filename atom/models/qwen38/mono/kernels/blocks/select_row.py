# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``SelectRow``: rows of a tagged channel, chosen at run time, as a channel of ``out_rows`` rows; ``AtRows``: a block
planned for another scenario than its program's.

The MTP drafter's first step (spec 4) runs its attention on every row of the step and the MLP half on each request's
last accepted row: ``SelectRow`` copies those rows of the attention combine's streams (``h1``), and the MLP half's
blocks plan for them (``AtRows(block, Scenario(batch, 1))``).

``SelectRow(name, src, dst, out_rows)``: ``dst[b, :] = src[sel[b], :]`` for ``b < out_rows`` (``sel`` the int32 arg
``sel``: ATOM's ``selected_rows``, clamped to the source's rows), the tagged pairs' payload words copied as they are.
``TASKS`` tasks split each output row; a thread copies two pairs (one 16-byte poll, one 16-byte put).
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
from functools import partial

import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device.ops import rsrc, traced
from atom.models.qwen38.mono.kernels.program import Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X

THREADS = MI355X.threads
ARG = "sel"  # the selected rows' int32 arg
TASKS = 8  # tasks an output row


@dataclass(frozen=True)
class _SP:
    name: str
    src: str
    dst: str
    arg: str
    rows: int  # the source's rows
    pairs: int  # pairs a row
    tasks: int  # tasks an output row
    out_rows: int = 1


@traced
def _put_if(ok, a, i, words, tag):
    if ok:
        chan.put_words(a, i, words, tag)


def _task(k, t0, p: _SP):
    b = t0 // p.tasks if p.out_rows > 1 else 0
    t = t0 % p.tasks if p.out_rows > 1 else t0
    sel = fx.Int32(bo.buffer_load(rsrc(k.args[p.arg]), b, vec_width=1, dtype=T.i32))
    sel = fx.max(fx.min(sel, fx.Int32(p.rows - 1)), fx.Int32(0))
    duo = p.pairs // 2  # two-pair units a row
    per = -(-duo // p.tasks)
    for ps in range_constexpr(-(-per // THREADS)):
        j = k.tid + ps * THREADS
        ok = (j < per) & (t * per + j < duo)
        u = fx.min(t * per + j, fx.Int32(duo - 1))
        w = chan.poll_words([(k.mb[p.src], sel * p.pairs + 2 * u, 2)], k.tag)[0]
        _put_if(ok, k.mb[p.dst], b * p.pairs + 2 * u, [w[0], w[1]], k.tag)


class SelectRow(Block):
    """Rows ``sel[b]`` of ``src`` into ``dst`` (module docstring)."""

    def __init__(self, name: str, src: Channel, dst: Channel, out_rows: int = 1):
        if src.fmt is not dst.fmt or src.width != dst.width or src.pairs_per_row() % 2:
            raise ValueError(
                f"SelectRow {name}: tagged channels of one format and width (an even number of pairs)"
            )
        self.name, self.src, self.dst, self.out_rows = name, src, dst, out_rows

    def stages(self, sc: Scenario) -> list[Stage]:
        p = _SP(
            self.name,
            self.src.name,
            self.dst.name,
            ARG,
            sc.rows,
            self.src.pairs_per_row(),
            TASKS,
            self.out_rows,
        )
        return [
            Stage(
                self.name,
                TASKS * self.out_rows,
                partial(_task, p=p),
                writes=tuple(self.dst.regions(self.out_rows, (self.name,))),
                args=(ARG,),
                key=astuple(p),
            )
        ]


class AtRows:
    """``block`` planned for scenario ``sc`` whatever its program's (``stages``); everything else is the block's."""

    def __init__(self, block, sc: Scenario):
        self._block, self._sc = block, sc

    def stages(self, sc: Scenario) -> list[Stage]:
        return self._block.stages(self._sc)
