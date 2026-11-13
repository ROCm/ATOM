# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hyper-connections: ``HCIn`` (stream norm, low-rank mix, injection weights) and ``HCWrite``,
the combine of a ``Linear``'s output (or any block's) into every residual stream.

The numerics contract is the checkpoint's hyper-connections (any ``HCGeometry``: ``streams``
bf16 streams of ``hidden``, the flat row ``width = streams * hidden`` stream-major, a
low-rank path of ``lowrank`` columns). Every rounding point below is the contract's.

**HCIn** (``mix``) runs as two stages:

- ``<name>.down`` (``(blocks + 1) * slices`` tasks): K slice ``s`` (``slices = streams *
  split``; slice ``s`` is part ``s % split`` of stream ``s // split``) of the low-rank down
  projection (and, in its last ``slices`` tasks, of the injection projection) for every row.
  A task loads its stream of every row (the whole stream: the norm's statistics), normalizes
  its part (``normed = bf16((x * rstd) * (1 + w))``, ``rstd = rsqrt_rn_pos(sum(x^2) * fp32(1 /
  hidden) + eps)``, bit for bit the contract's ``rms_norm``: the sum of squares in **torch's GPU
  order** for the contract's call, which reduces ``rows x streams`` rows at once
  (``ops.torch_sum_lanes``: 128 threads a row at 4 rows, 64 from 8; ``ops.torch_thread_sum`` /
  ``torch_tree``), the correctly rounded reciprocal root (``ops.rsqrt_rn_pos``): each wave
  fetches its rows (at 128 threads a row, its half of the row's threads) in torch's lane layout
  (loads from a plain source, a poll of a channel), sums them, one barrier for every row's
  rstd) into LDS as the bf16 B operand, and multiplies it by its weights on the
  bf16 MFMA (``device.gfx950.gemv16``: the down weights are MXFP4 values, packed
  losslessly as MXFP4, or with ``weights_fmt="bf16"`` plain bf16 units; the injection weights
  are plain bf16). It publishes its fp32 partial sums (``<name>.part`` [slices][rows][lowrank +
  streams]); the tasks of block 0 also publish each stream's ``rstd`` (``<name>.rstd`` [rows][streams]).
- ``<name>.up`` (``hidden / 32`` tasks): task ``t`` owns columns ``32 t .. 32 t + 31`` of the
  mixed row, i.e. those columns of every stream's up projection (``2 streams`` row groups).
  It sums the partials in slice order (``d = bf16(sum)``), applies the scaled SiLU (``g =
  bf16(d / streams)``, ``g = bf16(g * sigmoid(g))``) into LDS, multiplies by its up weights
  (MXFP4 values or, ``weights_fmt="bf16"``, bf16; K padded to 128), and per element ``u = bf16(acc)``, ``term_s =
  sigmoid(u_s) * normed_s`` (``normed`` recomputed from the stream, the gamma and the
  published ``rstd``); ``mixed = bf16((term_0 + term_1 + ...) / streams)`` (stream order) goes
  out as an MXFP8 channel (group 32 = the task's columns) and optionally a BF16 channel.
  Task 0 also publishes the injection multipliers ``inj[r, s] = 2 sigmoid(bf16(bf16(d_inj) /
  streams))`` (an F32 channel [rows, streams]): what ``combine`` multiplies the block by.

Sigmoid and SiLU are torch's fp32 formulas (accurate exp, IEEE division).
Sums are fp32 in a fixed order (MFMA, then K units per wave, waves in order, slices in
order); the contract's torch GEMMs sum in another order, so a bf16 rounding of ``d`` or
``u`` can differ in its last bit where the fp32 sum sits on a rounding boundary.

**HCWrite** (``combine``): ``out[r, s * hidden + c] = bf16(res[r, s * hidden + c] +
bf16(y[r, c]) * inj[r, s])`` for every stream ``s`` (fp32 product, fp32 sum, one rounding),
``y`` the producer's fp32 output. A ``Linear`` output kind (``Linear(..., out=HCWrite(...))``);
other producers call :func:`write_issue` / :func:`write_store` from their epilogue.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import channels as chan
from ...device.gfx950 import gemv, gemv16
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    fresh,
    fresh_after,
    rsqrt_rn_pos,
    rsrc,
    spin_pause,
    torch_sum_lanes,
    torch_thread_sum,
    torch_tree,
    traced,
)
from ...geometry import HCGeometry
from ...plan.program import PlanError, Region, Scenario, Stage
from ...plan.trace import note
from ...target import MI355X
from ..base import Block, Channel, Fmt
from . import norm as nq

WAVES = MI355X.waves
THREADS = MI355X.threads
NORM_ROWS = 4  # rows normalized per norm call (two CTA barriers each)
COLS = 32  # mixed columns per up task: one MXFP8 group
TILE_ROWS = 16  # the MFMA N dimension: past it, rows run in 16-row tiles on the same weight operands
MAX_ROWS = 64  # the mixed row's lanes (rows x 8 <= THREADS)
SENTINEL_ROWS = 2  # up tasks wait on sentinels above this many rows (E8)
TORCH_EARLY = 1  # passes of the stream norm's sum loads (8 rows each at 2+ rows) issued with the weights


# ------------------------------------------------------------------ shared device helpers


def sigmoid(x):
    """torch's fp32 sigmoid: 1 / (1 + exp(-x)) (accurate exp, IEEE division)."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - x))


def _per_stream(x, p):
    """``x / streams`` (fp32, IEEE): a multiplication by the exact reciprocal when ``streams`` is a
    power of two (the same correctly rounded quotient), else a division."""
    if p.streams & (p.streams - 1) == 0:
        return x * (1.0 / p.streams)
    return x / fx.Float32(p.streams)


def _attempt(specs):
    """One poll attempt of ``specs`` [(Addr, pair index, npairs)], issued now, not waited for:
    the raw (value, tag) words of every spec, in order (device scope, as ``mailbox.poll``).
    """
    out = []
    for a, i, n in specs:
        note(a.region, a.space, "poll")
        v = fx.Vector(mb.load(a, fx.Int32(i) * 2, 2 * n))
        out += [v[e] for e in range(2 * n)]
    return out


@traced
def _spin_from(v, load_all, pending):
    while pending(v):
        spin_pause()
        v = load_all()
    return v


def settle(k, specs, raw):
    """The value words of ``specs`` given an attempt's ``raw`` words (``_attempt``): the
    attempt's if every tag is current, else polled again until they are."""
    if not specs:
        return []

    def load_all():
        words = []
        for a, i, n in specs:
            v = fx.Vector(mb.load(a, fx.Int32(i) * 2, 2 * n))
            words += [v[e] for e in range(2 * n)]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = v[1] != k.tag
        for e in range_constexpr(3, len(raw), 2):
            bad = bad | (v[e] != k.tag)
        return bad

    v0 = fx.Vector.from_elements(raw, fx.Int32)
    v = _spin_from(v0, load_all, pending)
    return [v[e] for e in range(0, len(raw), 2)]


def _words(ptr, w, n):
    """``n`` (1, 2 or 4) plain words at word ``w``."""
    v = bo.buffer_load(rsrc(ptr), w, vec_width=n, dtype=T.i32)
    if n == 1:
        return [fx.Int32(v)]
    v = fx.Vector(v)
    return [v[e] for e in range(n)]


# ------------------------------------------------------------------ HCWrite: the combine


@dataclass(frozen=True)
class HCWrite:
    """The hyper-connections' combine as an output spec: the producer's row
    ``y`` [rows, hidden] (fp32, rounded to bf16 here) goes into each of ``streams`` residual
    streams with its injection multiplier: ``out[r, s * hidden + c] = bf16(res[r, s * hidden
    + c] + bf16(y[r, c]) * inj[r, s])``.

    - ``inj``: the multipliers, an F32 channel [rows, streams] (``HCIn``'s ``inj``);
    - the residual streams [rows, streams * hidden] (bf16): a plain arg (``arg``, written
      before the launch, e.g. the layer input) or a BF16 channel (``channel``, written earlier
      in the launch, e.g. the attention combine's output);
    - the result: a plain bf16 arg ``out`` [rows, streams * hidden] and / or a BF16 channel
      ``publish`` (the next HC's and the next combine's input).

    ``streams = 1`` with ``inj = 1`` is a plain residual add (``linear.Residual``'s)."""

    streams: int
    inj: Channel
    out: str | None = None
    arg: str | None = None
    channel: Channel | None = None
    publish: Channel | None = None

    def __post_init__(self):
        if (self.arg is None) == (self.channel is None):
            raise ValueError("HCWrite needs exactly one of arg= and channel=")
        if self.out is None and self.publish is None:
            raise ValueError(
                "HCWrite writes a plain arg (out=), a channel (publish=) or both"
            )
        if (
            self.inj.fmt is not Fmt.F32
            or self.inj.width != self.streams
            or self.inj.flagged
        ):
            raise ValueError(
                f"HCWrite: inj must be a tagged F32 channel of width {self.streams}"
            )
        for ch in (self.channel, self.publish):
            if ch is not None and (
                ch.fmt is not Fmt.BF16 or ch.width % self.streams or ch.flagged
            ):
                raise ValueError(
                    f"HCWrite: {ch.name} must be a tagged BF16 channel of streams x hidden"
                )

    def check(self, hidden: int) -> None:
        """Raise unless the channels fit a producer row of ``hidden``."""
        for ch in (self.channel, self.publish):
            if ch is not None and ch.width != self.streams * hidden:
                raise ValueError(
                    f"HCWrite: {ch.name} has width {ch.width}, not {self.streams} x {hidden}"
                )
        if hidden % 4:
            raise ValueError("HCWrite: hidden must be a multiple of 4")

    def args(self) -> tuple[str, ...]:
        """Pointer-table args the write reads or writes."""
        return tuple(a for a in (self.arg, self.out) if a)

    def writes(self, rows: int, writer: str) -> tuple[Region, ...]:
        """Regions a producer stage ``writer`` declares for the write (the published result)."""
        return self.publish.regions(rows, (writer,)) if self.publish is not None else ()

    def sources(self) -> tuple[Channel, ...]:
        """The channels the write polls (stand-in producers for standalone builds)."""
        return (self.inj,) + ((self.channel,) if self.channel is not None else ())


# ------------------------------------------------------------------ HCIn: the plan


def _inj_specs(k, w: HCWrite, row):
    a = k.mb[w.inj.name]
    base = row * w.streams
    return [(a, base + s, min(2, w.streams - s)) for s in range(0, w.streams, 2)]


def write_issue(k, w: HCWrite, hidden: int, row, col):
    """Issue the loads elements ``col .. col + 3`` (``col % 4 == 0``) of row ``row`` need:
    the residual of every stream (a plain arg's words, or one poll attempt of the channel's
    pairs) and an attempt at the row's injection multipliers. Nothing is waited for; pass
    the result to :func:`write_store`. Indices must be in range (clamp them)."""
    width = w.streams * hidden
    specs = _inj_specs(k, w, row)
    res = []
    for s in range(w.streams):
        e = row * width + s * hidden + col
        if w.channel is not None:
            specs.append((k.mb[w.channel.name], e // 2, 2))
        else:
            res += _words(k.args[w.arg], e // 2, 2)
    return specs, _attempt(specs), res


@traced
def write_store(k, w: HCWrite, hidden: int, ok, row, col, y, issued):
    """Elements ``col .. col + 3`` of row ``row`` of every stream := bf16(res + bf16(y) * inj)
    (``y``: 4 fp32), from :func:`write_issue`'s ``issued``; nothing stored unless ``ok``.
    """
    specs, raw, res = issued
    got = settle(k, specs, raw)
    if ok:
        _write(k, w, hidden, row, col, y, got, res)


def _write(k, w, hidden, row, col, y, got, res):
    width = w.streams * hidden
    inj = [v.bitcast(fx.Float32) for v in got[: w.streams]]
    if w.channel is not None:
        res = got[w.streams :]
    block = [bf16_round(v) for v in y]
    for s in range(w.streams):
        r = list(bf2_f32(res[2 * s])) + list(bf2_f32(res[2 * s + 1]))
        o = [bf16_round(r[q] + block[q] * inj[s]) for q in range(4)]
        e = row * width + s * hidden + col
        if w.out is not None:
            words = [
                bf16_pair(o[0], o[1]).bitcast(fx.Int32),
                bf16_pair(o[2], o[3]).bitcast(fx.Int32),
            ]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32), rsrc(k.args[w.out]), e // 2
            )
        if w.publish is not None:
            chan.put_bf16(k.mb[w.publish.name], e, o, k.tag)


@dataclass(frozen=True)
class HCNames:
    """``layer.retained`` names of one HC's weights: the norm gamma [width], the
    low-rank down [lowrank, width], up [width, lowrank] and injection [streams, width].
    """

    norm: str
    down: str
    up: str
    inject: str


@dataclass(frozen=True)
class _Plan:
    """Every Python constant the emit closes over (also the stages' build key)."""

    name: str
    rows: int
    streams: int
    hidden: int
    lowrank: int
    eps: float
    src: str  # source arg or channel name
    src_ch: bool  # the source is a BF16 channel (polled)
    g: int  # down row groups per down task
    split: int  # K slices per stream
    out: str  # mixed: MXFP8 channel
    bf: str  # mixed: BF16 channel ("" if none)
    inj: str  # injection multipliers: F32 channel ("": a mixer without injection)
    norm_from: str = (
        "stream"  # the norm sums' source: "stream" (a second, torch-layout fetch) or "lds"
    )
    wait: str = (
        "direct"  # how up tasks wait for the partials: "direct" (every thread spins on its items) or
    )
    # "sentinel" (one pair of every producer task, a CTA barrier, then the items: less fabric load)
    # always False (a removed debug probe); the field stays because the stage keys (Program.digest, the kernel
    # symbol) include it
    probe: bool = False
    # the low-rank down / up weights: "mxfp4" (MXFP4 values packed losslessly, decoded to bf16) or "bf16" (plain bf16
    # units, ``gemv16.pack_bf16``: a checkpoint whose HC weights are real bf16, e.g. Quark Flash's)
    wfmt: str = "mxfp4"
    # past one tile: the rows a down task normalizes and multiplies (0: every row). Row tiles replace the K split there:
    # a down task's work stays a tile's, and the up tasks sum ``streams`` partials a column, not ``streams x split``
    # (the hc measurements H-rowtasks)
    rt: int = 0
    rows_all: int = (
        0  # a down task's view (``down_view``): the stage's rows (the strides of the source, part, rstd)
    )
    # the low-rank activation from a stage of its own (``<name>.lr``: each item's slice sums and scaled SiLU once, out
    # as bf16 words), polled by the up tasks; else every up task sums every partial itself, in spins of ``POLL_MAX``
    # specs (the hc measurements H-lr)
    lr: bool = False

    @property
    def width(self):
        return self.streams * self.hidden

    @property
    def tiles(
        self,
    ):  # 16-row tiles (the MFMA's N in the down stage, its M in the up stage): one at <= 16 rows
        return -(-self.rows // TILE_ROWS)

    @property
    def rg_down(self):  # low-rank row groups
        return self.lowrank // 16

    @property
    def blocks(self):  # down blocks of g row groups (the injection block follows)
        return -(-self.rg_down // self.g)

    @property
    def slices(self):
        return self.streams * self.split

    @property
    def row_tiles(self):  # the down tasks' row tiles (``rt``)
        return self.rows // self.rt if self.rt else 1

    @property
    def down_tasks(self):
        return (self.blocks + (1 if self.inj else 0)) * self.slices * self.row_tiles

    @property
    def down_view(self):
        """The plan a down task runs with: ``rt`` rows (strided over the stage's), or the plan itself."""
        return (
            replace(self, rows=self.rt, rows_all=self.rows, rt=0) if self.rt else self
        )

    @property
    def stride_rows(self):  # the rows of the source / part / rstd layouts
        return self.rows_all or self.rows

    @property
    def kse(self):  # K per slice
        return self.hidden // self.split

    @property
    def us(self):  # K units per slice
        return self.kse // 128

    @property
    def c(self):  # K units per wave (the last waves may have fewer)
        return -(-self.us // WAVES)

    @property
    def k_units(self):  # K units of a full row
        return self.width // 128

    @property
    def nt(self):  # partials per row and slice: low-rank columns, then injection logits
        return self.lowrank + (self.streams if self.inj else 0)

    @property
    def spec(self) -> nq.NormSpec:
        words = self.kse // 2
        wps = 4
        threads = min(THREADS, words // wps)
        return nq.NormSpec(
            self.hidden, self.eps, threads=threads, wps=wps, split=self.split
        )

    @property
    def norm_rows(self):
        return min(self.rows, NORM_ROWS)

    @property
    def sum_lanes(
        self,
    ):  # torch's threads a row for the contract's stream norm (rows x streams rows)
        return torch_sum_lanes(self.hidden, self.stride_rows * self.streams)

    @property
    def xch(self):  # LDS word of the norm sums' thread parts (at 128 threads a row)
        # past the rstds: a row each at ``red + row`` (``_direct_rstds``); the norm's words alone held 32 rows, so past
        # 32 rows the rstds ran past the stage's LDS onto a composed program's stashed launch context (H-sregs)
        return self.red + max(self.spec.lds_words(self.norm_rows), self.rows)

    # down-stage LDS (words): the normed slice [rows][pitch] bf16 (later the wave partials), the norm's sums
    @property
    def pitch(self):
        return self.kse // 2 + 4

    @property
    def red(self):
        part = WAVES * self.g * self.rows * 16
        return -(-max(self.rows * self.pitch, part) // 4) * 4

    @property
    def down_lds(self):
        xch = self.rows * self.sum_lanes if self.sum_lanes > 64 else 0
        return self.xch + xch

    # up stage
    @property
    def ku(self):  # K units of the up projection (low-rank padded to 128)
        return -(-self.lowrank // 128)

    @property
    def rgu(self):  # up row groups per task: 2 per stream (32 columns)
        return 2 * self.streams

    @property
    def er(
        self,
    ):  # live rows per lane and tile in the up epilogue (rows on the MFMA's M side)
        return min(4, self.rows)

    @property
    def lr_tasks(
        self,
    ):  # the low-rank stage's tasks (``lr``): a thread an item, two adjacent low-rank columns of a row
        return -(-self.rows * (self.lowrank // 2) // THREADS)

    @property
    def up_tasks(self):
        return self.hidden // COLS

    @property
    def upitch(self):  # LDS words per row of the low-rank activation (bf16, padded K)
        return self.ku * 64 + 4

    @property
    def u_terms(self):  # LDS word: terms [rows][streams][32] f32
        return self.rows * self.upitch

    @property
    def u_rstd(self):  # LDS word: rstd [rows][streams]
        return self.u_terms + self.rows * self.streams * COLS

    @property
    def nx(
        self,
    ):  # extra words per thread polled with the partials: the stream words of a channel source
        return self.er * self.tiles * -(-self.rgu // WAVES) if self.src_ch else 0

    @property
    def u_x(self):  # LDS word: the extra words [threads][nx]
        return self.u_rstd + self.rows * self.streams

    @property
    def up_lds(self):
        return -(-(self.u_x + THREADS * self.nx) // 4) * 4


class HCIn(Block):
    """One hyper-connection's input side (module docstring): the streams ``src`` (a plain bf16
    [rows, width] arg such as the layer input, or a BF16 channel such as a combine's output)
    -> the mixed row ``out`` (MXFP8 [rows, hidden]; ``bf16_out``: also BF16) and the injection
    multipliers ``inj`` (F32 [rows, streams], for ``HCWrite``).

    Knobs: task shape ``group_rows`` (low-rank rows per down task, a multiple
    of 16; default 16) and ``split`` (K slices per stream; default 1); the up tasks' hand-off
    ``wait`` ("direct": every thread spins on its partials; "sentinel": one pair per producer
    task first, then the partials, less load on the fabric; default by rows,
    ``SENTINEL_ROWS``); ``row_tasks`` (past one tile: down tasks of a row tile each, split 1, instead of the K split the
    LDS forces; default off); ``lowrank_stage`` (the low-rank activation computed once by a stage of its own,
    ``<name>.lr``, and polled by the up tasks as bf16 words; default off: every up task sums every partial).
    """

    def __init__(
        self,
        name: str,
        src: str | Channel,
        geo: HCGeometry,
        out: Channel,
        inj: Channel | None,
        weights: HCNames,
        bf16_out: Channel | None = None,
        group_rows: int | None = None,
        split: int | None = None,
        wait: str | None = None,
        norm_from: str = "stream",
        weights_fmt: str = "mxfp4",
        row_tasks: bool = False,
        lowrank_stage: bool = False,
    ):
        h = geo.hidden
        if isinstance(src, Channel) and (
            src.fmt is not Fmt.BF16 or src.width != geo.width or src.flagged
        ):
            raise ValueError(
                f"HCIn {name}: src must be a plain arg or a tagged BF16 channel of width {geo.width}"
            )
        if out.fmt is not Fmt.MXFP8 or out.width != h or out.flagged:
            raise ValueError(
                f"HCIn {name}: out must be a tagged MXFP8 channel of width {h}"
            )
        if bf16_out is not None and (
            bf16_out.fmt is not Fmt.BF16 or bf16_out.width != h or bf16_out.flagged
        ):
            raise ValueError(
                f"HCIn {name}: bf16_out must be a tagged BF16 channel of width {h}"
            )
        if inj is not None and (
            inj.fmt is not Fmt.F32 or inj.width != geo.streams or inj.flagged
        ):
            raise ValueError(
                f"HCIn {name}: inj must be a tagged F32 channel of width {geo.streams}"
            )
        self.split_given = split is not None
        split = split or 1
        g = (group_rows or 16) // 16
        if (
            (group_rows or 16) % 16
            or h % (128 * split)
            or h % COLS
            or geo.lowrank % 32
            or geo.streams > 16
            or geo.streams % 2
        ):
            raise ValueError(
                f"HCIn {name}: unsupported shape {geo} (group_rows {group_rows}, split {split})"
            )
        self.name, self.src, self.geo, self.out, self.inj, self.weights = (
            name,
            src,
            geo,
            out,
            inj,
            weights,
        )
        if wait not in (None, "direct", "sentinel"):
            raise ValueError(f"HCIn {name}: wait {wait!r} is not direct or sentinel")
        self.bf16_out, self.g, self.split, self.wait = bf16_out, g, split, wait
        if norm_from not in ("stream", "lds") or (norm_from == "lds" and split != 1):
            raise ValueError(
                f'HCIn {name}: norm_from {norm_from!r} ("lds" needs split 1: the whole stream landed)'
            )
        self.norm_from = norm_from
        # weights_fmt: the down / up weights' format, "mxfp4" (their values must be MXFP4 values: packed losslessly,
        # decoded to bf16 operands) or "bf16" (plain bf16 units fed to the bf16 MFMA as the injection rows are)
        if weights_fmt not in ("mxfp4", "bf16"):
            raise ValueError(
                f"HCIn {name}: weights_fmt {weights_fmt!r} is not mxfp4 or bf16"
            )
        self.weights_fmt = weights_fmt
        self.row_tasks = row_tasks
        self.lowrank_stage = lowrank_stage

    @property
    def down_name(self):
        return f"{self.name}.down"

    @property
    def up_name(self):
        return f"{self.name}.up"

    @property
    def lr_name(self):
        return f"{self.name}.lr"

    def _args(self):
        n = self.name
        return {
            "norm": f"{n}.norm",
            "dw": f"{n}.dw",
            "ds": f"{n}.ds",
            "iw": f"{n}.iw",
            "uw": f"{n}.uw",
            "us": f"{n}.us",
        }

    def plan(self, sc: Scenario) -> _Plan:
        if sc.rows > MAX_ROWS:
            raise PlanError(f"HCIn {self.name}: at most {MAX_ROWS} rows")
        if sc.rows <= TILE_ROWS or self.split_given:
            return self._plan(sc, self.split)
        if (
            self.row_tasks
        ):  # the fewest equal row tiles of at most 16 rows (24 rows: 2 x 12; 48: 3 x 16)
            n = -(-sc.rows // TILE_ROWS)
            while sc.rows % n:
                n += 1
            return self._plan(sc, 1, rt=sc.rows // n)
        # past one tile with no split given: the fewest K slices a stream whose LDS fits (the normed slice grows with
        # the rows; 2 at 32 rows, 4 at 64 for a 2560-wide stream), where none built before
        for split in (1, 2, 4, 8):
            if self.geo.hidden % (128 * split) == 0:
                try:
                    return self._plan(sc, split)
                except PlanError:
                    continue
        raise PlanError(
            f"HCIn {self.name}: LDS at {sc.rows} rows does not fit at any split"
        )

    def _plan(self, sc: Scenario, split: int, rt: int = 0) -> _Plan:
        geo = self.geo
        if split != 1 and self.norm_from == "lds":
            raise PlanError(f'HCIn {self.name}: norm_from "lds" needs split 1')
        p = _Plan(
            name=self.name,
            rows=sc.rows,
            streams=geo.streams,
            hidden=geo.hidden,
            lowrank=geo.lowrank,
            eps=geo.eps,
            src=self.src.name if isinstance(self.src, Channel) else self.src,
            src_ch=isinstance(self.src, Channel),
            g=self.g,
            split=split,
            out=self.out.name,
            bf=self.bf16_out.name if self.bf16_out is not None else "",
            inj=self.inj.name if self.inj is not None else "",
            wait=self.wait or ("sentinel" if sc.rows > SENTINEL_ROWS else "direct"),
            norm_from=self.norm_from,
            wfmt=self.weights_fmt,
            rt=rt,
            lr=self.lowrank_stage,
        )
        if (p.hidden // 4) % p.sum_lanes:
            raise PlanError(
                f"HCIn {self.name}: torch's norm order over {p.hidden} needs a tail (not implemented)"
            )
        if (
            p.wait == "sentinel" and p.blocks * p.slices > THREADS
        ):  # ``_lowrank``: a producer task's sentinel a thread
            raise PlanError(f"HCIn {self.name}: wait 'sentinel' polls a producer task a thread: blocks x slices "
                            f"{p.blocks * p.slices} > {THREADS}")  # fmt: skip
        for lds in (p.down_view.down_lds, p.up_lds):
            if lds * 4 > MI355X.lds_bytes - 1024:
                raise PlanError(
                    f"HCIn {self.name}: LDS {lds * 4} B at {sc.rows} rows (raise split)"
                )
        return p

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        a = self._args()
        src_arg = () if p.src_ch else (p.src,)
        part, rstd = p.slices * p.rows * p.nt, p.rows * p.streams
        down_writes = (
            Region(f"{p.name}.part", part, (self.down_name,)),
            Region(f"{p.name}.rstd", rstd, (self.down_name,)),
        )
        up_writes = self.out.regions(p.rows, (self.up_name,))
        if self.inj is not None:
            up_writes += self.inj.regions(p.rows, (self.up_name,))
        if self.bf16_out is not None:
            up_writes += self.bf16_out.regions(p.rows, (self.up_name,))
        lr = (
            [
                Stage(
                    self.lr_name,
                    p.lr_tasks,
                    partial(_lr, p=p),
                    writes=(
                        Region(
                            f"{p.name}.lr", p.rows * (p.lowrank // 2), (self.lr_name,)
                        ),
                    ),
                    key=("hc.lr",) + (p,),
                )
            ]
            if p.lr
            else []
        )
        return [
            Stage(
                self.down_name,
                p.down_tasks,
                partial(_down, p=p),
                lds_words=p.down_view.down_lds,
                writes=down_writes,
                args=src_arg
                + (a["norm"], a["dw"])
                + ((a["ds"],) if p.wfmt == "mxfp4" else ())
                + ((a["iw"],) if p.inj else ()),
                key=("hc.down",) + (p,),
            ),
            *lr,
            Stage(
                self.up_name,
                p.up_tasks,
                partial(_up, p=p),
                lds_words=p.up_lds,
                writes=up_writes,
                args=src_arg
                + (a["norm"], a["uw"])
                + ((a["us"],) if p.wfmt == "mxfp4" else ()),
                key=("hc.up",) + (p,),
            ),
        ]

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """The HC's weights from ``layer.retained`` (bf16): the gamma as stored, the down and up
        projections as MXFP4 (``pack_mxfp4_exact``: raises unless their values are MXFP4
        values) or, ``weights_fmt`` "bf16", as bf16 units (``gemv16.pack_bf16``), the injection
        rows as bf16 units (padded to 16 rows)."""
        geo, a, r = self.geo, self._args(), layer.retained
        down, up = r[self.weights.down], r[self.weights.up]
        if down.shape != (geo.lowrank, geo.width) or up.shape != (
            geo.width,
            geo.lowrank,
        ):
            raise ValueError(
                f"HCIn {self.name}: down {tuple(down.shape)} / up {tuple(up.shape)} do not fit {geo}"
            )
        kp = -(-geo.lowrank // 128) * 128
        # up rows in task order: task t's row groups are (stream s, columns 32 t .. 32 t + 31)
        upr = up.reshape(geo.streams, geo.hidden // COLS, COLS, geo.lowrank).permute(
            1, 0, 2, 3
        )
        upr = torch.nn.functional.pad(
            upr.reshape(geo.width, geo.lowrank), (0, kp - geo.lowrank)
        )
        out = {a["norm"]: r[self.weights.norm].contiguous()}
        if self.weights_fmt == "bf16":
            out[a["dw"]] = gemv16.pack_bf16(down.to(torch.bfloat16).contiguous())
            out[a["uw"]] = gemv16.pack_bf16(upr.to(torch.bfloat16).contiguous())
        else:
            out[a["dw"]], out[a["ds"]] = gemv.pack_weight(
                *pack_mxfp4_exact(down, f"{self.name} down")
            )
            out[a["uw"]], out[a["us"]] = gemv.pack_weight(
                *pack_mxfp4_exact(upr, f"{self.name} up")
            )
        if self.inj is not None:
            inject = r[self.weights.inject]
            if inject.shape != (geo.streams, geo.width):
                raise ValueError(
                    f"HCIn {self.name}: inject {tuple(inject.shape)} does not fit {geo}"
                )
            inj = torch.zeros(16, geo.width, dtype=torch.bfloat16, device=inject.device)
            inj[: geo.streams] = inject
            out[a["iw"]] = gemv16.pack_bf16(inj)
        return out


def pack_mxfp4_exact(
    w: torch.Tensor, what: str = "weight"
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 (packed, codes) of a bf16 ``w`` [N, K] whose values are MXFP4 values (groups of
    32 along K); raises if any value would change."""
    from ...weights.mx import dequantize_mxfp4, quantize_mxfp4

    f = w.float()
    packed, codes = quantize_mxfp4(f)
    if not torch.equal(dequantize_mxfp4(packed, codes), f):
        raise ValueError(
            f"{what}: values are not MXFP4 values (this engine stores them losslessly as MXFP4)"
        )
    return packed, codes


# ------------------------------------------------------------------ device: the down stage


def _down(k, t, p):
    r0 = None
    if (
        p.rt
    ):  # row tiles: tile t // per's rows (``down_view``), its tasks' (block, slice) t % per
        per = p.down_tasks // p.row_tiles
        r0 = fresh((t // per) * p.rt)
        t, p = t % per, p.down_view
    blk, s = t // p.slices, t % p.slices
    if p.inj:
        _down_either(k, p, blk, s, r0)
    else:
        _down_task(k, p, "fp4", blk, s, r0)


@traced
def _down_either(k, p, blk, s, r0=None):
    if blk == p.blocks:
        _down_task(k, p, "bf16", blk, s, r0)
    else:
        _down_task(k, p, "fp4", blk, s, r0)


def _down_units(k, p, fmt, blk, s):
    """Issue this wave's weight units (and fp4 scale codes): [row group][i] for K units
    ``wave + 8 i`` of slice ``s``."""
    lane, w = k.lane, k.wave
    out = []
    groups = p.g if fmt == "fp4" else 1
    for gl in range(groups):
        per = []
        for i in range(p.c):
            ku = w + WAVES * i
            unit = s * p.us + ku
            valid = None if p.us % WAVES == 0 else ku < p.us
            if (
                fmt == "fp4"
            ):  # the low-rank rows (``p.wfmt``: MXFP4 units and codes, or bf16 units)
                rg = blk * p.g + gl
                if p.rg_down % p.g:
                    ok = rg < p.rg_down
                    valid = ok if valid is None else valid & ok
                u = fx.min(rg, fx.Int32(p.rg_down - 1)) * p.k_units + unit
                if p.wfmt == "bf16":
                    per.append(
                        gemv16.load_bf16(
                            rsrc(k.args[f"{p.name}.dw"], p.lowrank * p.width * 2),
                            u,
                            lane,
                            valid,
                        )
                    )
                else:
                    ws = gemv16.load_fp4(
                        rsrc(k.args[f"{p.name}.dw"], p.lowrank * p.width // 2),
                        u,
                        lane,
                        valid,
                    )
                    sc = gemv16.load_scale(
                        rsrc(k.args[f"{p.name}.ds"], p.lowrank * p.width // 32),
                        u,
                        lane,
                        valid,
                    )
                    per.append((ws, sc))
            else:  # the injection rows: lanes of rows >= streams read nothing
                ok = (lane % 16) < p.streams
                valid = ok if valid is None else valid & ok
                per.append(
                    gemv16.load_bf16(
                        rsrc(k.args[f"{p.name}.iw"], 16 * p.width * 2),
                        unit,
                        lane,
                        valid,
                    )
                )
        out.append(per)
    return out


def _row_issue(k, p, rows, st, part, r0=None):
    """Issue stream ``st``'s words of ``rows`` (the part quantized, and the whole stream when
    split): a plain arg's words, or one poll attempt of the channel's pairs. ``r0``: the rows are a down task's row
    tile's, from row ``r0`` on."""
    spec = p.spec
    out = []
    # past one tile: the task's stream offset once, in a VGPR (an opaque copy), each row's constant on it; as scalars
    # every row's base was its own SGPR, CSE'd with the norm sums' fetch and held between them (64 rows: 17 SGPR
    # spills, the hc measurements H-sregs)
    sbase = fresh(st * (p.hidden // 2)) if p.tiles > 1 else None
    for r in rows:
        base = (
            r * (p.width // 2) + st * (p.hidden // 2)
            if sbase is None
            else sbase + r * (p.width // 2)
        )
        if r0 is not None:
            base = base + r0 * (p.width // 2)
        if not p.src_ch:
            q = nq._map_load(
                spec.qmap, k.args[p.src], base + part * spec.qmap.words, k.tid
            )
            sm = (
                None
                if spec.shared
                else nq._map_load(spec.smap, k.args[p.src], base, k.tid)
            )
            out.append((None, (q, sm)))
        else:
            specs = nq._map_specs(
                spec.qmap, k.mb[p.src], base + part * spec.qmap.words, k.tid
            )
            if not spec.shared:
                specs += nq._map_specs(spec.smap, k.mb[p.src], base, k.tid)
            out.append((specs, _attempt(specs)))
    return out


def _row_land(k, p, issued):
    """``issued`` rows -> per row (qmap chunks, smap chunks or None) of words. A channel
    source's rows settle in one spin (one round trip after the last producer, not one per row).
    """
    spec = p.spec
    if issued[0][0] is None:
        return [raw for _, raw in issued]
    vals = settle(
        k,
        [sp for specs, _ in issued for sp in specs],
        [w for _, raw in issued for w in raw],
    )
    out = []
    for specs, _ in issued:
        mine, vals = vals[: 2 * len(specs)], vals[2 * len(specs) :]
        got = [mine[2 * i : 2 * i + 2] for i in range(len(specs))]  # per 2-pair spec
        q, got = nq._map_words(spec.qmap, got)
        sm = None
        if not spec.shared:
            sm, got = nq._map_words(spec.smap, got)
        out.append((q, sm))
    return out


@traced
def _put_rstd(k, p, ok, r, st, rstd):
    if ok:
        mb.put(k.mb[f"{p.name}.rstd"], r * p.streams + st, rstd, k.tag)


@traced
def _place(k, p, r, v, words):
    """Slot ``v`` of this thread (``wps`` packed bf16 of normed row ``r``) into the LDS operand."""
    m = p.spec.qmap
    if k.tid < m.limit(v):
        slot = k.tid + v * m.threads
        fx.ptr_store(
            fx.Vector.from_elements(words, fx.Int32),
            k.lds + (r * p.pitch + slot * m.wps),
        )


@traced
def _store_rstd(k, ok, i, rstd):
    if ok:
        fx.ptr_store(rstd, k.ldsf + i)


def _sum_units(p):
    """The waves' norm-sum work in passes: unit ``(row, m)`` is torch's threads ``lane + 64 m`` of
    row ``row`` (``p.sum_lanes`` threads a row); wave ``w`` takes the ``w``-th unit of a pass.
    """
    units = [(r, m) for r in range(p.rows) for m in range(p.sum_lanes // 64)]
    return [units[i : i + WAVES] for i in range(0, len(units), WAVES)]


def _torch_issue(k, p, units, st, r0=None):
    """Wave ``i`` (< ``len(units)``) fetches unit ``units[i] = (row, m)``'s vectors of stream
    ``st`` (torch's thread ``lane + 64 m`` of the row: the row's vectors ``lane + 64 m + sum_lanes
    j``, 2 words each). A plain source: loads, the other waves' offsets past the source (no
    access). A channel: one poll attempt (``_attempt``; the other waves repeat the last unit),
    settled by :func:`_direct_rstds`."""
    wave, lane = fresh(k.wave), fresh(
        k.lane
    )  # the selects in VGPRs (SGPRs bind in composed programs)

    def word(u):
        r, m = u
        return r * (p.width // 2) + st * (p.hidden // 2) + 64 * m * 2

    if p.tiles > 1 and all(u == (units[0][0] + i, 0) for i, u in enumerate(units)):
        # past one tile (one unit a row, consecutive rows): the wave's row offset as arithmetic; the select chain's
        # constants were materialized in VGPRs at the stage's top for every pass (64 rows: 64 held through the
        # stage, the hc measurements H-sregs)
        base = fx.Int32(word(units[0])) + fx.min(wave, fx.Int32(len(units) - 1)) * (
            p.width // 2
        )
    else:
        base = fx.Int32(word(units[-1]))
        for i in range(len(units) - 1):
            base = (wave == i).select(fx.Int32(word(units[i])), base)
    if r0 is not None:  # a row tile's rows
        base = base + r0 * (p.width // 2)
    vecs = p.hidden // 4 // p.sum_lanes
    if p.src_ch:
        base = base + lane * 2
        specs = [(k.mb[p.src], base + p.sum_lanes * 2 * j, 2) for j in range(vecs)]
        return specs, _attempt(specs)
    if len(units) < WAVES:
        base = (wave < len(units)).select(base, fx.Int32(gemv.OOB // 4))
    base = base + lane * 2
    res = rsrc(k.args[p.src], p.stride_rows * p.width * 2)
    out = []
    for j in range(vecs):
        v = fx.Vector(
            bo.buffer_load(res, base + p.sum_lanes * 2 * j, vec_width=2, dtype=T.i32)
        )
        out.append([v[0], v[1]])
    return out


def _direct_rstds(k, p, tws, landed=False):
    """Before the caller's barrier: each wave's unit (``_sum_units``) from its own fetch (``tws``,
    one per pass; a channel's attempt settled here). One unit a row (``sum_lanes`` 64): the wave finishes the row
    (torch's tree, the reciprocal root) into LDS ``red + row``. Several (128 threads a row, at 4
    outputs): its threads' parts into LDS ``xch``, finished by :func:`_direct_post`."""
    ms = p.sum_lanes // 64
    for units, tw in zip(_sum_units(p), tws, strict=True):
        if callable(tw):  # a later pass: issued now (TORCH_EARLY)
            tw = tw()
        if p.src_ch and not landed:  # the poll attempt settled: 2 words a vector
            words = settle(k, *tw)
            tw = [words[2 * j : 2 * j + 2] for j in range(len(words) // 2)]

        def squares(v, c, tw=tw):
            return _squares(tw[c])

        # lane / wave through opaque copies: masks on k.lane / k.wave would be hoisted to the
        # kernel top and hold SGPR pairs across every stage (spills seen in composed programs)
        lane, wave = fresh(k.lane), fresh(k.wave)
        part = torch_thread_sum(
            squares, 256 * len(tw), 64, lane, 0
        )  # the unit's vectors, in order
        live = wave < len(units)
        w = fx.min(wave, fx.Int32(len(units) - 1))
        if ms == 1:
            rstd = rsqrt_rn_pos(torch_tree([part]) * (1.0 / p.hidden) + p.eps)
            _store_rstd(k, (lane == 0) & live, p.red + units[0][0] + w, rstd)
        else:
            _store_rstd(
                k, live, p.xch + (units[0][0] * ms + units[0][1] + w) * 64 + lane, part
            )


def _direct_post(k, p):
    """After the barrier: every row's rstd (``red``; or every wave finishes the rows from the
    threads' parts in ``xch``)."""
    ms = p.sum_lanes // 64
    if ms == 1:
        return None
    out, lane = [], fresh(k.lane)
    for r in range(p.rows):
        parts = [
            fx.Float32(fx.ptr_load(k.ldsf + (p.xch + (r * ms + m) * 64 + lane)))
            for m in range(ms)
        ]
        out.append(rsqrt_rn_pos(torch_tree(parts) * (1.0 / p.hidden) + p.eps))
    return out


def _squares(words):
    out = []
    for w in words:
        lo, hi = bf2_f32(w)
        out += [lo * lo, hi * hi]  # exact (bf16 x): torch's square kernel, then its sum
    return out


def _rstds(k, p, rows):
    """Each row's ``rstd`` (``_direct_rstds``: in LDS ``red + row``)."""
    if p.tiles == 1:
        return [fx.Float32(fx.ptr_load(k.ldsf + (p.red + r))) for r in rows]
    red = fresh(
        fx.Int32(p.red)
    )  # (past one tile: the rows' constant addresses not hoisted and held, as above)
    return [fx.Float32(fx.ptr_load(k.ldsf + (red + r))) for r in rows]


def _normalize(
    k, p, issued_rows, rows, gw, st, publish_rstd, gi=0, rstds=None, r0=None
):
    """Normalize ``rows`` (their landed words, norm group ``gi``) into the LDS operand; publish
    each row's rstd (``rstds``: given; ``r0``: a row tile's rows, from row ``r0`` on).
    """
    xs = _row_land(k, p, issued_rows)
    if rstds is None:
        rstds = _rstds(k, p, rows)
    g1 = nq.gamma1(gw)
    for r, (q, _), rstd in zip(rows, xs, rstds, strict=True):
        for v, (words, g) in enumerate(zip(q, g1, strict=True)):
            y = []
            for j, wd in enumerate(words):
                lo, hi = bf2_f32(wd)
                y.append(
                    bf16_pair(
                        (lo * rstd) * g[2 * j], (hi * rstd) * g[2 * j + 1]
                    ).bitcast(fx.Int32)
                )
            _place(k, p, r, v, y)
        _put_rstd(
            k, p, publish_rstd & (k.tid == 0), r if r0 is None else r0 + r, st, rstd
        )


def _stage_raw(k, p, issued_rows, rows):
    """``norm_from`` "lds": ``rows``' landed words, raw, into the LDS operand (each slot where ``_place`` puts it)."""
    for r, (q, _) in zip(rows, _row_land(k, p, issued_rows), strict=True):
        for v, words in enumerate(q):
            _place(k, p, r, v, list(words))


def _lds_vectors(k, p, units):
    """``_torch_issue``'s vectors from the landed stream in LDS: wave ``i``'s unit ``units[i] = (row, m)``, torch's
    thread ``lane + 64 m``: vectors ``lane + 64 m + sum_lanes j`` (2 words each; the other waves read the last unit).
    """
    wave, lane = fresh(k.wave), fresh(k.lane)

    def word(u):
        r, m = u
        return r * p.pitch + 64 * m * 2

    base = fx.Int32(word(units[-1]))
    for i in range(len(units) - 1):
        base = (wave == i).select(fx.Int32(word(units[i])), base)
    base = base + lane * 2
    out = []
    for j in range(p.hidden // 4 // p.sum_lanes):
        v = fx.Vector(
            fx.ptr_load(
                k.lds + (base + p.sum_lanes * 2 * j),
                result_type=fx.Vector.make_type(2, fx.Int32),
            )
        )
        out.append([v[0], v[1]])
    return out


def _normalize_lds(k, p, rows, gw, st, publish, rstds, r0=None):
    """``norm_from`` "lds": each thread's slots of ``rows`` normalized in place (the raw words it staged)."""
    m = p.spec.qmap
    g1 = nq.gamma1(gw)
    for r, rstd in zip(rows, rstds, strict=True):
        for v, g in enumerate(g1):
            slot = fx.min(k.tid + v * m.threads, fx.Int32(m.slots - 1))
            raw = fx.Vector(
                fx.ptr_load(
                    k.lds + (r * p.pitch + slot * m.wps),
                    result_type=fx.Vector.make_type(m.wps, fx.Int32),
                )
            )
            y = []
            for j in range(m.wps):
                lo, hi = bf2_f32(raw[j])
                y.append(
                    bf16_pair(
                        (lo * rstd) * g[2 * j], (hi * rstd) * g[2 * j + 1]
                    ).bitcast(fx.Int32)
                )
            _place(k, p, r, v, y)
        _put_rstd(k, p, publish & (k.tid == 0), r if r0 is None else r0 + r, st, rstd)


def _down_task(k, p, fmt, blk, s, r0=None):
    st, part = s // p.split, s % p.split
    lane = fresh(
        k.lane
    )  # masks on it are not CSE'd with other stages' and held across them (spills seen in a composed program)
    # 1. the weights first, then the gamma, the rows' first group and (a plain source) the waves'
    # own loads of their rows in torch's layout for the norm's sums (vmcnt is in order)
    units = _down_units(k, p, fmt, blk, s)
    rocdl.sched_barrier(0)
    gw = nq._map_load(
        p.spec.qmap,
        k.args[f"{p.name}.norm"],
        st * (p.hidden // 2) + part * p.spec.qmap.words,
        k.tid,
    )
    groups = [
        list(range(r0, min(r0 + p.norm_rows, p.rows)))
        for r0 in range(0, p.rows, p.norm_rows)
    ]
    early = _row_issue(k, p, groups[0], st, part, r0)
    # the first TORCH_EARLY passes' loads go out now, later ones when their pass runs (each pass
    # holds ~40 VGPRs from issue to its sums: at 9-16 rows two passes spilled, the hc measurements H-cover)
    tws = (
        []
        if p.norm_from == "lds"
        else [
            (
                _torch_issue(k, p, us, st, r0)
                if i < TORCH_EARLY
                else partial(_torch_issue, k, p, us, st, r0)
            )
            for i, us in enumerate(_sum_units(p))
        ]
    )
    rocdl.sched_barrier(0)
    # the weights decoded to bf16 MFMA operands while the rows are still in flight
    mx = fmt == "fp4" and p.wfmt == "mxfp4"
    ops = [
        [gemv16.decode_fp4(*u) if mx else gemv16.bf16_operands(u) for u in per]
        for per in units
    ]
    # 2. the norm of every row's part into LDS (rstd published by block 0, part 0)
    publish = (blk == 0) & (part == 0) if p.split > 1 else blk == 0
    if p.norm_from == "lds":
        # the stream lands once, raw, in the operand; the sums read it there in torch's layout; normalized in place
        for gi, rows in enumerate(groups):
            _stage_raw(
                k, p, early if gi == 0 else _row_issue(k, p, rows, st, part, r0), rows
            )
        gpu.barrier()
        _direct_rstds(
            k, p, [_lds_vectors(k, p, us) for us in _sum_units(p)], landed=True
        )
        gpu.barrier()
        post = _direct_post(k, p)
        for rows in groups:
            rstds = _rstds(k, p, rows) if post is None else [post[r] for r in rows]
            _normalize_lds(k, p, rows, gw, st, publish, rstds, r0)
    else:
        # every row's rstd at once: one barrier, none per norm group
        _direct_rstds(k, p, tws)
        gpu.barrier()
        post = _direct_post(k, p)
        for gi, rows in enumerate(groups):
            issued = early if gi == 0 else _row_issue(k, p, rows, st, part, r0)
            _normalize(
                k,
                p,
                issued,
                rows,
                gw,
                st,
                publish,
                gi,
                None if post is None else [post[r] for r in rows],
                r0,
            )
    gpu.barrier()
    # 3. MFMAs: K units wave + 8 i, every row group; past 16 rows an MFMA a 16-row tile on the same weight operands
    # (a row's products and sums do not depend on its tile)
    rows = [
        fx.min(_tile_row(lane % 16, tl), fx.Int32(p.rows - 1)) for tl in range(p.tiles)
    ]
    j = lane // 16
    accs = []
    for per in ops:
        acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.tiles)]
        for i, a in enumerate(per):
            ku = k.wave + WAVES * i
            if p.us % WAVES:
                ku = fx.min(ku, fx.Int32(p.us - 1))
            for tl in range(p.tiles):
                acc[tl] = gemv16.mfma_unit(
                    acc[tl], a, gemv16.b_operands(k.lds, rows[tl], j, ku, p.pitch)
                )
        accs.append(acc)
    # 4. wave partials through LDS (over the operand), summed in wave order, published
    gpu.barrier()
    for tl in range(p.tiles):
        _red_store(k, p, [acc[tl] for acc in accs], tl)
    gpu.barrier()
    _publish_part(k, p, fmt, blk, s, len(units), r0)
    gpu.barrier()  # the next task reuses the LDS


def _tile_row(r, tl):
    """Row ``r`` of 16-row tile ``tl`` (the first tile's rows as before: no offset)."""
    return r if tl == 0 else r + TILE_ROWS * tl


@traced
def _red_store(k, p, accs, tl=0):
    """Wave partials of 16-row tile ``tl`` -> LDS [wave][row group][row][16] f32 (useful lanes only)."""
    n = _tile_row(fresh(k.lane) % 16, tl)
    if n < p.rows:
        for gl in range_constexpr(len(accs)):
            off = ((k.wave * p.g + gl) * p.rows + n) * 16 + (k.lane // 16) * 4
            fx.ptr_store(accs[gl], k.ldsf + off)


def _publish_part(k, p, fmt, blk, s, groups, r0=None):
    items = groups * p.rows * 4
    for c in range(-(-items // THREADS)):
        e = fx.Int32(c * THREADS) + k.tid
        ok = e < items
        e = fx.min(e, fx.Int32(items - 1))
        gl, n, f4 = e // (p.rows * 4), (e // 4) % p.rows, e % 4
        ng = n if r0 is None else r0 + n  # the row in the stage's layout
        tot = None
        for w in range(WAVES):
            off = ((w * p.g + gl) * p.rows + n) * 16 + f4 * 4
            v = fx.Vector(
                fx.ptr_load(
                    k.ldsf + off, result_type=fx.Vector.make_type(4, fx.Float32)
                )
            )
            vals = [v[q] for q in range(4)]
            tot = vals if tot is None else [tot[q] + vals[q] for q in range(4)]
        if fmt == "fp4":
            rg = blk * p.g + gl
            if p.rg_down % p.g:
                ok = ok & (rg < p.rg_down)
            col = rg * 16 + f4 * 4
            _put_part4(k, p, ok, s, ng, col, tot)
        else:
            _put_part_inj(k, p, ok & (f4 * 4 < p.streams), s, ng, f4 * 4, tot)


@traced
def _put_part4(k, p, ok, s, n, col, tot):
    if ok:
        i = (s * p.stride_rows + n) * p.nt + col
        mb.put2(k.mb[f"{p.name}.part"], i, tot[0], tot[1], k.tag)
        mb.put2(k.mb[f"{p.name}.part"], i + 2, tot[2], tot[3], k.tag)


@traced
def _put_part_inj(k, p, ok, s, n, c0, tot):
    if ok:
        i = (s * p.stride_rows + n) * p.nt + p.lowrank + c0
        for q in range_constexpr(4):
            if c0 + q < p.streams:
                mb.put(k.mb[f"{p.name}.part"], i + q, tot[q], k.tag)


# ------------------------------------------------------------------ device: the up stage


def _up(k, t, p):
    lane, wave = fresh(k.lane), k.wave  # (see _down_task)
    j = lane // 16
    ns = [
        fx.min(_tile_row(lane % 16, tl), fx.Int32(p.rows - 1)) for tl in range(p.tiles)
    ]
    # this wave's row groups (stream, half) and their weights; the lane's stream / gamma words
    mine = []
    for i in range(-(-p.rgu // WAVES)):
        r = fx.min(wave + WAVES * i, fx.Int32(p.rgu - 1))
        valid = None if p.rgu % WAVES == 0 else (wave + WAVES * i) < p.rgu
        units = []
        for ku in range(p.ku):
            u = (t * p.rgu + r) * p.ku + ku
            if p.wfmt == "bf16":
                units.append(
                    gemv16.load_bf16(
                        rsrc(k.args[f"{p.name}.uw"], p.width * p.ku * 256),
                        u,
                        lane,
                        valid,
                    )
                )
                continue
            units.append(
                (
                    gemv16.load_fp4(
                        rsrc(k.args[f"{p.name}.uw"], p.width * p.ku * 64),
                        u,
                        lane,
                        valid,
                    ),
                    gemv16.load_scale(
                        rsrc(k.args[f"{p.name}.us"], p.width * p.ku * 4), u, lane, valid
                    ),
                )
            )
        # the MFMA puts rows on the result's M side: lane l holds feature l % 16 of rows
        # 4 (l // 16) + e, e < ``p.er``, of each 16-row tile: only live rows' epilogue math is issued
        st, half = r // 2, r % 2
        col = t * COLS + half * 16 + lane % 16  # within the stream
        gword = _words(k.args[f"{p.name}.norm"], (st * p.hidden + col) // 2, 1)[0]
        es = [
            (
                fx.min(_tile_row(j * 4 + e, tl), fx.Int32(p.rows - 1)) * p.width
                + st * p.hidden
                + col
            )
            // 2
            for tl in range(p.tiles)
            for e in range(p.er)
        ]
        if p.src_ch:
            hspecs = [(k.mb[p.src], w, 1) for w in es]
            hraw = _attempt(hspecs)
        else:
            hspecs, hraw = [], [_words(k.args[p.src], w, 1)[0] for w in es]
        mine.append((r, valid, units, st, half, gword, hspecs, hraw))
    rocdl.sched_barrier(0)
    # the weights decoded to bf16 MFMA operands while the partials are in flight
    ops = [
        [
            gemv16.bf16_operands(u) if p.wfmt == "bf16" else gemv16.decode_fp4(*u)
            for u in m[2]
        ]
        for m in mine
    ]
    # the partials (slice order), the rstds -> the scaled SiLU of the low-rank row in LDS; the
    # streams' words (a channel source) settle in the same spin (``p.lr``: the low-rank stage's words instead)
    xspecs, xraw = [sp for m in mine for sp in m[6]], [
        w for m in mine if m[6] for w in m[7]
    ]
    if p.lr:
        _lowrank_words(k, p, xspecs, xraw)
    else:
        _lowrank(k, p, t, xspecs, xraw)
    gpu.barrier()
    for i, (r, valid, units, st, half, gwords, hspecs, hraw) in enumerate(mine):
        acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.tiles)]
        for ku, a in enumerate(ops[i]):
            for tl in range(p.tiles):
                acc[tl] = gemv16.mfma_unit(
                    acc[tl], gemv16.b_operands(k.lds, ns[tl], j, ku, p.upitch), a
                )
        for tl in range(p.tiles):
            at = p.er * (i * p.tiles + tl)
            hw = (
                [
                    fx.ptr_load(k.lds + (p.u_x + k.tid * p.nx + at + e))
                    for e in range(p.er)
                ]
                if hspecs
                else hraw[p.er * tl : p.er * (tl + 1)]
            )
            _terms(k, p, valid, st, half, acc[tl], gwords, hw, tl)
    gpu.barrier()
    _mixed(k, p, t)
    _inject(k, p, t)
    gpu.barrier()  # the next task reuses the LDS


def _lowrank(k, p, t, xspecs, xraw):
    """Every row's low-rank activation (bf16, K padded with zeros) and the rstds into LDS (and
    the words of ``xspecs``, attempted as ``xraw``, into LDS ``u_x``). Thread items: pairs of
    adjacent low-rank columns, each the sum of its slices' partials; a batch of at most
    ``mb.POLL_MAX`` specs settles in one spin (one round trip after the last producer). Threads
    without an item poll only the rstd and ``xspecs``: the spinning is the load on the fabric
    the producers' stores must cross (an up task's 512 threads each polling 5 pairs ~ 3 TB/s).
    """
    nr = p.rows * p.streams
    rspecs = [(k.mb[f"{p.name}.rstd"], fx.min(k.tid, fx.Int32(nr - 1)), 1)]
    rraw = _attempt(rspecs)
    items = p.rows * (p.lowrank // 2)
    rounds = -(-items // THREADS)
    per = max(1, mb.POLL_MAX // p.slices)  # item rounds per spin (its specs: registers)
    extras = (rspecs + xspecs, rraw + xraw)
    if (
        p.tiles > 1 and xspecs
    ):  # past one tile: the stream words and rstds settle in a spin of their own (their words
        # and a round's partials together held 66 words in the spin at 64 rows, the hc measurements H-sregs)
        _extras(k, p, settle(k, *extras))
        extras = ([], [])
    if (
        p.wait == "sentinel"
    ):  # one pair of every low-rank producer task first, then the items
        ns = p.blocks * p.slices
        sidx = fx.min(k.tid, fx.Int32(ns - 1))
        col = (sidx // p.slices) * (
            16 * p.g
        )  # the first column of the task's first row group
        sspec = [(k.mb[f"{p.name}.part"], (sidx % p.slices) * p.rows * p.nt + col, 1)]
        _sentinel(k, ns, sspec)
    # past one tile (many rounds): each spin's polls after the previous spin's sums, or the scheduler issues every
    # round's polls up front and holds their words (64 rows: 20 rounds, the stage at 256 VGPRs, the hc measurements H-sregs)
    after = [] if p.tiles > 1 else None
    for c0 in range(0, rounds, per):
        cn = min(rounds, c0 + per)
        ex = extras if c0 == 0 else ([], [])
        if (
            c0 + 1
        ) * THREADS > items:  # threads past the batch's items: only the extras
            _batch_or_extras(k, p, c0, cn, ex, after)
        else:
            after = _batch(k, p, c0, cn, ex, after)
    _zero_pad_all(k, p)


@traced
def _sentinel(k, ns, sspec):
    """Threads < ``ns`` spin on one pair each, then the CTA meets."""
    if k.tid < ns:
        settle(k, sspec, _attempt(sspec))
    gpu.barrier()


@traced
def _batch_or_extras(k, p, c0, cn, ex, after=None):
    if fx.Int32(c0 * THREADS) + k.tid < p.rows * (p.lowrank // 2):
        _batch(k, p, c0, cn, ex, after)
    else:
        _extras(k, p, settle(k, *ex))


def _batch(k, p, c0, cn, ex, after=None):
    """Item rounds ``c0 .. cn - 1`` of this thread (and the extras ``ex``): one spin, then the
    scaled SiLU of every item into LDS. ``after`` (a list): the polls' addresses wait for those values (the previous
    spin's sums); returns this spin's sums for the next (None: the code of before, returns None).
    """
    half_nt = p.lowrank // 2
    items = p.rows * half_nt
    part = k.mb[f"{p.name}.part"]
    tid = k.tid if not after else fresh_after(k.tid, after)
    work = []
    for c in range(c0, cn):
        idx = fx.Int32(c * THREADS) + tid
        ok = idx < items
        e = ok.select(idx, fx.Int32(0)) if (c + 1) * THREADS > items else idx
        n, col = e // half_nt, (e % half_nt) * 2
        specs = [(part, (s * p.rows + n) * p.nt + col, 2) for s in range(p.slices)]
        work.append((ok, n, col, specs, _attempt(specs)))
    got = settle(
        k,
        [sp for w in work for sp in w[3]] + ex[0],
        [x for w in work for x in w[4]] + ex[1],
    )
    _extras(k, p, got[2 * p.slices * len(work) :])
    sums = []
    for i, (ok, n, col, _, _) in enumerate(work):
        g = got[2 * p.slices * i : 2 * p.slices * (i + 1)]
        tot = [g[q].bitcast(fx.Float32) for q in range(2)]
        for s in range(1, p.slices):
            tot = [tot[q] + g[2 * s + q].bitcast(fx.Float32) for q in range(2)]
        _lowrank_item(k, p, ok, n, col, tot)
        sums += tot
    return None if after is None else sums[-1:]


def _extras(k, p, words):
    """The rstd (thread < rows * streams) and the extra words (LDS ``u_x``, per thread)."""
    if not words:
        return
    _store_word(k, k.tid < p.rows * p.streams, p.u_rstd + k.tid, words[0])
    for i, w in enumerate(words[1:]):
        fx.ptr_store(w, k.lds + (p.u_x + k.tid * p.nx + i))


def _lowrank_item(k, p, ok, n, col, tot):
    _store_g(k, p, ok, n, col, _scaled_silu(p, tot))


def _scaled_silu(p, tot):
    """An item's two low-rank columns from their slice sums: ``g = bf16(bf16(d) / streams)``, ``g * sigmoid(g)``."""
    d = [bf16_round(v) for v in tot]
    gs = [bf16_round(_per_stream(v, p)) for v in d]
    return [v * sigmoid(v) for v in gs]


# ------------------------------------------------------------------ device: the low-rank stage (``lowrank_stage``)


def _lr(k, t, p):
    """Task ``t``'s items, a thread each (``_batch``'s item ``t * THREADS + tid``: two adjacent low-rank columns of a
    row): the slices' partials summed in slice order, the scaled SiLU, out as one bf16 word (``<name>.lr`` [rows]
    [lowrank / 2])."""
    half_nt = p.lowrank // 2
    items = p.rows * half_nt
    idx = t * THREADS + k.tid
    e = fx.min(idx, fx.Int32(items - 1))
    n, col = e // half_nt, (e % half_nt) * 2
    specs = [
        (k.mb[f"{p.name}.part"], (s * p.rows + n) * p.nt + col, 2)
        for s in range(p.slices)
    ]
    got = settle(k, specs, _attempt(specs))
    tot = [got[q].bitcast(fx.Float32) for q in range(2)]
    for s in range(1, p.slices):
        tot = [tot[q] + got[2 * s + q].bitcast(fx.Float32) for q in range(2)]
    g = _scaled_silu(p, tot)
    _put_lr(k, p, idx < items, e, bf16_pair(g[0], g[1]))


@traced
def _put_lr(k, p, ok, e, word):
    if ok:
        mb.put(k.mb[f"{p.name}.lr"], e, word, k.tag)


def _lowrank_words(k, p, xspecs, xraw):
    """``p.lr``: the up task's low-rank operand from the low-rank stage's words (two a spec, in spins of at most
    ``POLL_MAX`` specs) into LDS, as ``_lowrank`` leaves it; the rstds and ``xspecs`` settle in a spin of their own.
    """
    nr = p.rows * p.streams
    rspecs = [(k.mb[f"{p.name}.rstd"], fx.min(k.tid, fx.Int32(nr - 1)), 1)]
    _extras(k, p, settle(k, rspecs + xspecs, _attempt(rspecs) + xraw))
    half_nt = (
        p.lowrank // 2
    )  # words a row (even: lowrank % 32 == 0), so a spec's two words are one row's
    pairs = p.rows * half_nt // 2
    lr = k.mb[f"{p.name}.lr"]
    work = []
    for c in range(-(-pairs // THREADS)):
        idx = fx.Int32(c * THREADS) + k.tid
        e = fx.min(idx, fx.Int32(pairs - 1))
        work.append((idx < pairs, e))
    for c0 in range(0, len(work), mb.POLL_MAX):
        batch = work[c0 : c0 + mb.POLL_MAX]
        specs = [(lr, e * 2, 2) for _, e in batch]
        got = settle(k, specs, _attempt(specs))
        for i, (ok, e) in enumerate(batch):
            _store_lr2(
                k,
                p,
                ok,
                (e * 2) // half_nt,
                (e * 2) % half_nt,
                got[2 * i],
                got[2 * i + 1],
            )
    _zero_pad_all(k, p)


@traced
def _store_lr2(k, p, ok, n, w, w0, w1):
    if ok:
        fx.ptr_store(
            fx.Vector.from_elements([w0, w1], fx.Int32), k.lds + (n * p.upitch + w)
        )


def _inject(k, p, t):
    """Task 0, after its mixed row: the injection multipliers (off the mixed row's path: only
    here are the injection tasks' partials polled)."""
    if p.inj:
        _inject_task0(k, p, t)


@traced
def _inject_task0(k, p, t):
    if t == 0:
        _inject_rows(k, p)


def _inject_rows(k, p):
    part = k.mb[f"{p.name}.part"]
    items = p.rows * p.streams
    # past one tile: the thread id through an opaque copy, or the slices' addresses are hoisted to the stage's top and
    # held through it (64 rows: 16 VGPRs, the hc measurements H-sregs)
    tid = fresh(k.tid) if p.tiles > 1 else k.tid
    for c in range(-(-items // THREADS)):
        idx = fx.Int32(c * THREADS) + tid
        ok = idx < items
        e = fx.min(idx, fx.Int32(items - 1))
        n, i = e // p.streams, e % p.streams
        got = mb.poll(
            [
                (part, (s * p.rows + n) * p.nt + p.lowrank + i, 1)
                for s in range(p.slices)
            ],
            k.tag,
        )
        tot = got[0][0].bitcast(fx.Float32)
        for s in range(1, p.slices):
            tot = tot + got[s][0].bitcast(fx.Float32)
        v = bf16_round(_per_stream(bf16_round(tot), p))
        _put_inj(k, p, ok, e, fx.Float32(2.0) * sigmoid(v))


def _zero_pad_all(k, p):
    """Zero the K padding of the low-rank operand (the weights there are zero too; LDS
    garbage could be NaN)."""
    pad = (p.ku * 128 - p.lowrank) // 2
    if pad:
        for c in range(-(-p.rows * pad // THREADS)):
            e = fx.Int32(c * THREADS) + k.tid
            _zero_pad(k, p, e < p.rows * pad, e // pad, p.lowrank // 2 + e % pad)


@traced
def _store_g(k, p, ok, n, col, g):
    if ok:
        fx.ptr_store(bf16_pair(g[0], g[1]), k.ldsf + (n * p.upitch + col // 2))


@traced
def _zero_pad(k, p, ok, n, w):
    if ok:
        fx.ptr_store(fx.Int32(0), k.lds + (n * p.upitch + w))


@traced
def _store_word(k, ok, at, v):
    if ok:
        fx.ptr_store(v, k.lds + at)


@traced
def _put_inj(k, p, ok, e, v):
    if ok:
        mb.put(k.mb[p.inj], e, v, k.tag)


def _terms(k, p, valid, st, half, acc, gword, hw, tl=0):
    """term = sigmoid(bf16(acc)) * normed of the lane's live rows (``acc[e]``: row 4 (lane // 16)
    + e of 16-row tile ``tl``, feature lane % 16) -> LDS [row][stream][32]."""
    lane = (
        fresh(k.lane) if p.tiles > 1 else k.lane
    )  # (past one tile: the tiles' addresses not hoisted, ``_inject_rows``)
    f = lane % 16
    odd = (f % 2) == 1  # the column's half of its bf16 word (t * 32 + 16 half is even)
    w1 = odd.select(bf2_f32(gword)[1], bf2_f32(gword)[0]) + 1.0
    for e in range(p.er):
        row = _tile_row((lane // 16) * 4 + e, tl)
        ok = row < p.rows
        if valid is not None:
            ok = ok & valid
        n = fx.min(row, fx.Int32(p.rows - 1))
        rstd = fx.ptr_load(k.ldsf + (p.u_rstd + n * p.streams + st))
        x = odd.select(bf2_f32(hw[e])[1], bf2_f32(hw[e])[0])
        term = sigmoid(bf16_round(acc[e])) * bf16_round((x * rstd) * w1)
        _store_f32(k, ok, p.u_terms + (n * p.streams + st) * COLS + half * 16 + f, term)


@traced
def _store_f32(k, ok, off, v):
    if ok:
        fx.ptr_store(v, k.ldsf + off)


def _mixed(k, p, t):
    """Lanes q < rows * 8 (row q // 8, columns 4 (q % 8) ..): the stream sum, the mixed row's
    MXFP8 group (8 lanes) and bf16 words."""
    q = fx.min(k.tid, fx.Int32(p.rows * 8 - 1))
    n, c4 = q // 8, q % 8
    tot = None
    for s in range(p.streams):
        v = fx.Vector(
            fx.ptr_load(
                k.ldsf + (p.u_terms + (n * p.streams + s) * COLS + c4 * 4),
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        )
        vals = [v[e] for e in range(4)]
        tot = vals if tot is None else [tot[e] + vals[e] for e in range(4)]
    mixed = [bf16_round(_per_stream(v, p)) for v in tot]
    code = gemv.activation_code(gemv.group_amax(mixed, 8))
    word = gemv.quant_word(mixed, code)
    _put_mixed(
        k, p, k.tid < p.rows * 8, n, t * COLS + c4 * 4, c4 == 0, word, code, mixed
    )


@traced
def _put_mixed(k, p, ok, n, col, lead, word, code, mixed):
    if ok:
        e = n * p.hidden + col
        chan.put_mxfp8_word(k.mb[f"{p.out}.q"], e // 4, word, k.tag)
        if lead:
            chan.put_mxfp8_scale(k.mb[f"{p.out}.s"], e // 32, code, k.tag)
        if const_expr(p.bf != ""):
            chan.put_bf16(k.mb[p.bf], e, mixed, k.tag)
