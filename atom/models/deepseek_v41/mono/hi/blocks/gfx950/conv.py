# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ConvPrep``: the DFlash2 drafter's block-local dynamic 2-tap conv before a projection, fused with MXFP8
quantization (the projection's input). The output may be bulk + flag (``Proto.FLAG``: dense words, one flag per task).

Per row ``r`` of the step (fp32 math, rows of one request only: row ``r - 1`` must be the same request's)::

    c[r] = bf16((b0 + d0[r]) x[r] + [r is not its request's first row] (b1 + d1[r]) x[r - 1])
    q, s = MXFP8(c[r])                                  (group 32, E8M0 scales, as ``NormQuant``)

``x``: a BF16 channel (e.g. ``NormQuant``'s ``bf16_out``); ``d0[r]`` / ``d1[r]`` per 16 features ``f .. f +
15``: elements ``d0 + f // 16`` / ``d1 + f // 16`` of row ``r`` of the BF16 channel ``coef`` (the kernel
projection's output); ``b0`` / ``b1``: rows 0 / 1 of the plain bf16 [2, width] arg ``base``. The conv after
a projection is ``linear.ConvFinish`` (an output option of ``Linear``).

Stage: ``rows * split`` tasks; task ``t`` handles part ``t % split`` of row ``t // split``, thread ``i <
width / split / 16`` the 16 features ``16 i ..`` of the part (a 32-group = two adjacent lanes). The
quantization is ``NormQuant``'s (``norm.e8m0_code`` of the bf16 amax, ``norm.fp8x2_scaled``) and the
stores are its ``_put_slot``: the same channel bytes ``quantize_mxfp8`` gives for the bf16 ``c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from . import norm as nq
from ..base import Block, Channel, Fmt, Search
from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import channels as chan
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import bf2_f32, bf16_pair, bf16_round, butterfly, rsrc, traced
from ...plan.program import Scenario, Stage

FEATS = 16  # features per thread (8 bf16 words): one coefficient pair


@dataclass(frozen=True)
class _Cfg:
    width: int
    split: int
    seg: int  # rows per request
    x: str
    coef: str
    coef_w: int
    d0: int
    d1: int
    base: str
    out: str
    dense: bool = False  # the MXFP8 output is bulk + flag (``Proto.FLAG``): one flag per task

    @property
    def threads(self) -> int:
        return self.width // self.split // FEATS


def _words(v):
    return [v[e] for e in range(4)]


@traced
def _task(k, row, part, c: _Cfg):
    if k.tid < c.threads:
        _body(k, row, part, c)


def _body(k, row, part, c: _Cfg):
    """Thread ``tid``'s 16 features of (row, part): base taps loaded, rows r / r - 1 and the
    coefficient pair polled (one batch), the conv, the MXFP8 group (two lanes), the stores."""
    col = part * (c.width // c.split) + k.tid * FEATS
    r_b = rsrc(k.args[c.base])
    b0 = [
        w for j in range(2) for w in _words(fx.Vector(bo.buffer_load(r_b, col // 2 + 4 * j, vec_width=4, dtype=T.i32)))
    ]
    b1 = [
        w
        for j in range(2)
        for w in _words(fx.Vector(bo.buffer_load(r_b, (c.width + col) // 2 + 4 * j, vec_width=4, dtype=T.i32)))
    ]
    first = (row % c.seg) == 0
    prow = first.select(row, row - 1)
    i0 = row * c.coef_w + c.d0 + col // FEATS
    i1 = row * c.coef_w + c.d1 + col // FEATS
    xa, ca = k.mb[c.x], k.mb[c.coef]
    specs = [(xa, (row * c.width + col) // 2 + 2 * j, 2) for j in range(4)]
    specs += [(xa, (prow * c.width + col) // 2 + 2 * j, 2) for j in range(4)]
    specs += [(ca, i0 // 2, 1), (ca, i1 // 2, 1)]
    got = mb.poll(specs, k.tag)
    cur = [v for g in got[0:4] for w in g for v in bf2_f32(w)]
    prv = [v for g in got[4:8] for w in g for v in bf2_f32(w)]
    d0 = (i0 % 2 == 1).select(bf2_f32(got[8][0])[1], bf2_f32(got[8][0])[0])
    d1 = (i1 % 2 == 1).select(bf2_f32(got[9][0])[1], bf2_f32(got[9][0])[0])
    bv0 = [v for w in b0 for v in bf2_f32(w)]
    bv1 = [v for w in b1 for v in bf2_f32(w)]
    out = []
    for i in range(FEATS):
        a = (bv0[i] + d0) * cur[i]
        out.append(bf16_round(first.select(a, a + (bv1[i] + d1) * prv[i])))
    y = [bf16_pair(out[2 * j], out[2 * j + 1]).bitcast(fx.Int32) for j in range(FEATS // 2)]
    am = fmath.absf(out[0])
    for v in out[1:]:
        am = fx.max(am, fmath.absf(v))
    am = butterfly(am, (1,), fx.max)
    code = nq.e8m0_code(am)
    scale = (code << 23).bitcast(fx.Float32)
    words = [
        nq.fp8x2_scaled(y[j + 1], scale, nq.fp8x2_scaled(y[j], scale, fx.Int32(0), False), True)
        for j in range(0, FEATS // 2, 2)
    ]
    wbase = (row * c.width + part * (c.width // c.split)) // 2
    nq._put_slot(
        k.mb[f"{c.out}.q"], k.mb[f"{c.out}.s"], None, wbase, wbase >> 4, k.tid * (FEATS // 2),
        c.threads, 2, k.tid, k.tag, words, code, y, c.dense,
    )  # fmt: skip


def _emit(k, t, c: _Cfg):
    if c.split == 1:
        _task(k, t, fx.Int32(0), c)
    else:
        _task(k, t // c.split, t % c.split, c)
    if c.dense:  # this task's words drained, then its flag (every thread of the CTA calls it)
        chan.put_flag(k, f"{c.out}.f", t)


class ConvPrep(Block):
    """Conv-prepare of BF16 channel ``x`` -> MXFP8 channel ``out`` (module docstring). ``coef``: the
    BF16 coefficient channel, ``d0`` / ``d1`` the elements of features 0 .. 15's taps in its rows;
    ``base``: the plain bf16 [2, width] arg (``pack``: from ``layer.retained[weight]``)."""

    # what the tuner may change (``Block.SEARCH``)
    SEARCH = {"split": Search((1, 2, 4), "bitwise", "shape", "drafter T5: conv split noise at B1S8; shape-dependent")}

    def __init__(
        self,
        name: str,
        x: Channel,
        coef: Channel,
        d0: int,
        d1: int,
        base: str,
        out: Channel,
        weight: str | None = None,
        split: int = 1,
    ):
        if x.fmt is not Fmt.BF16 or coef.fmt is not Fmt.BF16 or x.flagged or coef.flagged:
            raise ValueError("ConvPrep polls tagged BF16 channels")
        if out.fmt is not Fmt.MXFP8 or out.width != x.width:
            raise ValueError(f"out must be an MXFP8 channel of width {x.width}")
        if x.width % (split * 2 * FEATS) or not 0 < x.width // split // FEATS <= 512:
            raise ValueError(f"ConvPrep {name}: split {split} must cut width {x.width} into whole groups")
        self.name, self.x, self.coef, self.d0, self.d1, self.base, self.out = name, x, coef, d0, d1, base, out
        self.weight, self.split = weight, split

    def cfg(self, sc: Scenario) -> _Cfg:
        return _Cfg(
            self.x.width, self.split, sc.spec, self.x.name, self.coef.name, self.coef.width, self.d0, self.d1,
            self.base, self.out.name, self.out.dense,
        )  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        c = self.cfg(sc)
        return [
            Stage(
                self.name,
                sc.rows * self.split,
                partial(_emit, c=c),
                writes=self.out.regions(sc.rows, (self.name,), flags=sc.rows * self.split),
                args=(self.base,),
                key=("conv_prep", c),
            )
        ]

    def pack(self, layer) -> dict[str, torch.Tensor]:
        return {self.base: layer.retained[self.weight].contiguous()} if self.weight else {}
