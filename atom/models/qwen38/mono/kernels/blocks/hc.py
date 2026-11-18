# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hyper-connections: ``HCIn`` (stream norm, low-rank mix, injection weights) and ``HCWrite``,
the combine of a ``Linear``'s output (or any block's) into every residual stream.

Any ``HCGeometry``: ``streams`` bf16 streams of ``hidden``, the flat row ``width = streams *
hidden`` stream-major, a low-rank path of ``lowrank`` columns. Every rounding point below is
part of the numerics contract.

**HCIn** (``mix``) runs as two stages:

- ``<name>.down`` (``(blocks + 1) * streams`` tasks): stream ``s`` of the low-rank down
  projection (and, in its last ``streams`` tasks, of the injection projection) for every row.
  A task loads its stream of every row, normalizes it (``normed = bf16((x * rstd) * (1 + w))``,
  ``rstd = rsqrt_rn_pos(sum(x^2) * fp32(1 / hidden) + eps)``, bit for bit torch's ``rms_norm``: the sum of
  squares in **torch's GPU order** for a call that reduces ``rows x streams`` rows at once
  (``ops.torch_sum_lanes``: 128 threads a row at 4 rows, 64 from 8; ``ops.torch_thread_sum`` /
  ``torch_tree``), the correctly rounded reciprocal root (``ops.rsqrt_rn_pos``): each wave
  fetches its rows (at 128 threads a row, its half of the row's threads) in torch's lane layout
  (``norm_from`` "stream": loads from a plain source, a poll of a channel; "lds": from the stream landed in LDS),
  sums them, one barrier for every row's rstd) into LDS as the bf16 B operand, and multiplies it by its weights on
  the bf16 MFMA (``device.gemv16``: the down, up and injection weights are plain bf16
  units). It publishes its fp32 partial sums (``<name>.part`` [streams][rows][lowrank + streams]); the tasks of
  block 0 also publish each stream's ``rstd`` (``<name>.rstd`` [rows][streams]).
- ``<name>.up`` (``hidden / 32`` tasks): task ``t`` owns columns ``32 t .. 32 t + 31`` of the
  mixed row, i.e. those columns of every stream's up projection (``2 streams`` row groups).
  It sums the partials in stream order (``d = bf16(sum)``), applies the scaled SiLU (``g =
  bf16(d / streams)``, ``g = bf16(g * sigmoid(g))``) into LDS, multiplies by its up weights
  (K padded to 128), and per element ``u = bf16(acc)``, ``term_s =
  sigmoid(u_s) * normed_s`` (``normed`` recomputed from the stream, the gamma and the
  published ``rstd``); ``mixed = bf16((term_0 + term_1 + ...) / streams)`` (stream order) goes
  out as an MXFP8 channel (group 32 = the task's columns) and a BF16 channel.
  Task 0 also publishes the injection multipliers ``inj[r, s] = 2 sigmoid(bf16(bf16(d_inj) /
  streams))`` (an F32 channel [rows, streams]): what ``combine`` multiplies the block by.

Sigmoid and SiLU are torch's fp32 formulas (accurate exp, IEEE division; as in ``gdn``).
Sums are fp32 in a fixed order (MFMA, then K units per wave, waves in order, streams in
order); torch's GEMMs sum in another order, so a bf16 rounding of ``d`` or
``u`` can differ in its last bit where the fp32 sum sits on a rounding boundary.

**HCWrite** (``combine``): ``out[r, s * hidden + c] = bf16(res[r, s * hidden + c] +
bf16(y[r, c]) * inj[r, s])`` for every stream ``s`` (fp32 product, fp32 sum, one rounding),
``y`` the producer's fp32 output. A ``Linear`` output kind (``Linear(..., out=HCWrite(...))``);
other producers call :func:`write_issue` / :func:`write_store` from their epilogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device import gemv, gemv16
from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    fresh,
    rsqrt_rn_pos,
    rsrc,
    spin_pause,
    torch_sum_lanes,
    torch_thread_sum,
    torch_tree,
    traced,
)
from atom.models.qwen38.mono.kernels.geometry import HCGeometry
from atom.models.qwen38.mono.kernels.program import PlanError, Region, Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X
from atom.mono.plan.trace import note

WAVES = MI355X.waves
THREADS = MI355X.threads
NORM_ROWS = 4  # rows normalized per norm call (two CTA barriers each)
COLS = 32  # mixed columns per up task: one MXFP8 group
MAX_ROWS = 16  # the MFMA N dimension
SENTINEL_ROWS = 2  # up tasks wait on sentinels above this many rows
GROUP_WORDS = 16  # packed bf16 words of a 32-element MX group


# ------------------------------------------------------------------ shared device helpers


def sigmoid(x):
    """torch's fp32 sigmoid: 1 / (1 + exp(-x)) (accurate exp, IEEE division)."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(fx.Float32(0.0) - x))


def _per_stream(x, p):
    """``x / streams`` (fp32, IEEE): a multiplication by the exact reciprocal (``streams`` is a power of two: the
    same correctly rounded quotient)."""
    return x * (1.0 / p.streams)


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


# ------------------------------------------------------------------ the stream norm's row mapping


@dataclass(frozen=True)
class WordMap:
    """Slots of ``wps`` packed bf16 words over ``words`` words, interleaved over ``threads``: slot ``s = tid + v *
    threads`` (chunk ``v < chunks``) is words ``[s * wps, s * wps + wps)`` of the range, valid iff ``tid <
    limit(v)`` (an invalid slot loads a clamped duplicate). A 32-group (16 words) spans ``16 / wps`` adjacent
    lanes."""

    words: int
    threads: int
    wps: int

    def __post_init__(self):
        lanes = GROUP_WORDS // self.wps if self.wps <= GROUP_WORDS else 0
        if (
            self.wps not in (2, 4, 8, 16)
            or self.words % GROUP_WORDS
            or self.threads % lanes
        ):
            raise ValueError(f"bad {self}")
        if not 0 < self.threads <= THREADS:
            raise ValueError(f"{self}: threads must be in 1..{THREADS}")

    @property
    def slots(self) -> int:
        return self.words // self.wps

    @property
    def chunks(self) -> int:
        return -(-self.slots // self.threads)

    def limit(self, v: int) -> int:
        """Threads ``tid < limit(v)`` hold a valid chunk ``v``."""
        return min(self.threads, self.slots - v * self.threads)


def _slot_word(m: WordMap, tid, v):
    """First word (within the map's range) of chunk ``v``'s slot, clamped into the range."""
    s = tid + v * m.threads
    top = THREADS - 1 + v * m.threads  # the largest s any thread computes
    return (s if top < m.slots else fx.min(s, m.slots - 1)) * m.wps


def _load(ptr, word, n):
    """``n`` (2, 4, 8 or 16) i32 words at word ``word``: 8- or 16-byte loads."""
    out = []
    for j in range(0, n, 4):
        width = min(4, n - j)
        v = fx.Vector(bo.buffer_load(rsrc(ptr), word + j, vec_width=width, dtype=T.i32))
        out += [v[e] for e in range(width)]
    return out


def _map_load(m: WordMap, ptr, base, tid):
    return [_load(ptr, base + _slot_word(m, tid, v), m.wps) for v in range(m.chunks)]


def _map_specs(m: WordMap, addr, base, tid):
    specs = []
    for v in range(m.chunks):
        w = base + _slot_word(m, tid, v)
        specs += [(addr, w + j, 2) for j in range(0, m.wps, 2)]
    return specs


def _map_words(m: WordMap, got):
    per = m.wps // 2
    return [
        [w for g in got[v * per : (v + 1) * per] for w in g] for v in range(m.chunks)
    ], got[m.chunks * per :]


def _gamma1(gw):
    """(1 + w) in f32 per chunk, from the gamma's packed bf16 words (``_map_load``)."""
    out = []
    for words in gw:
        vals = []
        for w in words:
            lo, hi = bf2_f32(w)
            vals += [lo + 1.0, hi + 1.0]
        out.append(vals)
    return out


# ------------------------------------------------------------------ HCWrite: the combine


@dataclass(frozen=True)
class HCWrite:
    """The HC combine as an output spec: the producer's row
    ``y`` [rows, hidden] (fp32, rounded to bf16 here) goes into each of ``streams`` residual
    streams with its injection multiplier: ``out[r, s * hidden + c] = bf16(res[r, s * hidden
    + c] + bf16(y[r, c]) * inj[r, s])``.

    - ``inj``: the multipliers, an F32 channel [rows, streams] (``HCIn``'s ``inj``);
    - the residual streams [rows, streams * hidden] (bf16): a plain arg (``arg``, written
      before the launch, e.g. the layer input) or a BF16 channel (``channel``, written earlier
      in the launch, e.g. the attention combine's output);
    - the result: a plain bf16 arg ``out`` [rows, streams * hidden] and / or a BF16 channel
      ``publish`` (the next HC's and the next combine's input).

    ``streams = 1`` with ``inj = 1`` is a plain residual add."""

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
        if self.inj.fmt is not Fmt.F32 or self.inj.width != self.streams:
            raise ValueError(
                f"HCWrite: inj must be a tagged F32 channel of width {self.streams}"
            )
        for ch in (self.channel, self.publish):
            if ch is not None and (ch.fmt is not Fmt.BF16 or ch.width % self.streams):
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


# ------------------------------------------------------------------ HCIn: the plan


@dataclass(frozen=True)
class HCNames:
    """``QuantizedLayer.retained`` names of one HC's weights: the norm gamma [width], the
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
    out: str  # mixed: MXFP8 channel
    bf: str  # mixed: BF16 channel
    inj: str  # injection multipliers: F32 channel
    norm_from: (
        str  # the norm sums' source: "stream" (a second, torch-layout fetch) or "lds"
    )
    wait: str  # how up tasks wait for the partials: "direct" (every thread spins on its items) or
    # "sentinel" (one pair of every producer task, a CTA barrier, then the items: less fabric load)
    g = 1  # down row groups per down task (class constants, not fields)
    split = 1  # K slices per stream: a down task normalizes its whole stream

    @property
    def width(self):
        return self.streams * self.hidden

    @property
    def rg_down(self):  # low-rank row groups
        return self.lowrank // 16

    @property
    def blocks(self):  # down blocks of one row group (the injection block follows)
        return self.rg_down

    @property
    def slices(self):  # K slices: the streams
        return self.streams

    @property
    def down_tasks(self):
        return (self.blocks + 1) * self.slices

    @property
    def us(self):  # K units per slice
        return self.hidden // 128

    @property
    def c(self):  # K units per wave (the last waves may have fewer)
        return -(-self.us // WAVES)

    @property
    def k_units(self):  # K units of a full row
        return self.width // 128

    @property
    def nt(self):  # partials per row and slice: low-rank columns, then injection logits
        return self.lowrank + self.streams

    @property
    def qmap(self) -> WordMap:  # a stream's words over the threads, 4 a slot
        words = self.hidden // 2
        return WordMap(words, min(THREADS, words // 4), 4)

    @property
    def norm_rows(self):
        return min(self.rows, NORM_ROWS)

    @property
    def sum_lanes(
        self,
    ):  # torch's threads a row for the contract's stream norm (rows x streams rows)
        return torch_sum_lanes(self.hidden, self.rows * self.streams)

    @property
    def xch(self):  # LDS word of the norm sums' thread parts (at 128 threads a row)
        return self.red + self.norm_rows * WAVES

    # down-stage LDS (words): the normed stream [rows][pitch] bf16 (later the wave partials), the norm's sums
    @property
    def pitch(self):
        return self.hidden // 2 + 4

    @property
    def red(self):
        part = WAVES * self.rows * 16
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
    def er(self):  # live rows per lane in the up epilogue (rows on the MFMA's M side)
        return min(4, self.rows)

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
        return self.er * -(-self.rgu // WAVES) if self.src_ch else 0

    @property
    def u_x(self):  # LDS word: the extra words [threads][nx]
        return self.u_rstd + self.rows * self.streams

    @property
    def up_lds(self):
        return -(-(self.u_x + THREADS * self.nx) // 4) * 4


class HCIn(Block):
    """One hyper-connection's input side (module docstring): the streams ``src`` (a plain bf16
    [rows, width] arg such as the layer input, or a BF16 channel such as a combine's output)
    -> the mixed row ``out`` (MXFP8 [rows, hidden]) and ``bf16_out`` (BF16), and the injection
    multipliers ``inj`` (F32 [rows, streams], for ``HCWrite``). The weights are bf16 (``weights_fmt``
    "bf16": the Quark checkpoint family's).

    Options: the up tasks' hand-off ``wait`` ("direct": every thread spins on its partials; "sentinel": one pair
    per producer task first, then the partials, less load on the fabric; default by rows, ``SENTINEL_ROWS``) and
    ``norm_from`` (the norm sums' source: "stream" or "lds")."""

    def __init__(
        self,
        name: str,
        src: str | Channel,
        geo: HCGeometry,
        out: Channel,
        inj: Channel,
        weights: HCNames,
        bf16_out: Channel,
        wait: str | None = None,
        norm_from: str = "stream",
        weights_fmt: str = "mxfp4",
    ):
        h = geo.hidden
        if (
            h % 128
            or h % COLS
            or geo.lowrank % 32
            or geo.streams > 16
            or geo.streams & (geo.streams - 1)
        ):
            raise ValueError(f"HCIn {name}: unsupported shape {geo}")
        if wait not in (None, "direct", "sentinel") or norm_from not in (
            "stream",
            "lds",
        ):
            raise ValueError(f"HCIn {name}: wait {wait!r} / norm_from {norm_from!r}")
        self.name, self.src, self.geo, self.out, self.inj, self.weights = (
            name,
            src,
            geo,
            out,
            inj,
            weights,
        )
        self.bf16_out, self.wait, self.norm_from, self.weights_fmt = (
            bf16_out,
            wait,
            norm_from,
            weights_fmt,
        )

    @property
    def down_name(self):
        return f"{self.name}.down"

    @property
    def up_name(self):
        return f"{self.name}.up"

    def _args(self):
        n = self.name
        return {"norm": f"{n}.norm", "dw": f"{n}.dw", "iw": f"{n}.iw", "uw": f"{n}.uw"}

    def plan(self, sc: Scenario) -> _Plan:
        if sc.rows > MAX_ROWS:
            raise PlanError(f"HCIn {self.name}: at most {MAX_ROWS} rows")
        geo = self.geo
        p = _Plan(
            name=self.name,
            rows=sc.rows,
            streams=geo.streams,
            hidden=geo.hidden,
            lowrank=geo.lowrank,
            eps=geo.eps,
            src=self.src.name if isinstance(self.src, Channel) else self.src,
            src_ch=isinstance(self.src, Channel),
            out=self.out.name,
            bf=self.bf16_out.name,
            inj=self.inj.name,
            wait=self.wait or ("sentinel" if sc.rows > SENTINEL_ROWS else "direct"),
            norm_from=self.norm_from,
        )
        if (p.hidden // 4) % p.sum_lanes:
            raise PlanError(
                f"HCIn {self.name}: torch's norm order over {p.hidden} needs a tail (not implemented)"
            )
        if (
            p.norm_from == "stream" and len(_sum_units(p)) > 1
        ):  # (its sums' loads go out with the weights, one pass)
            raise PlanError(
                f'HCIn {self.name}: norm_from "stream" sums {p.rows} rows in more than one pass'
            )
        for lds in (p.down_lds, p.up_lds):
            if lds * 4 > MI355X.lds_bytes - 1024:
                raise PlanError(f"HCIn {self.name}: LDS {lds * 4} B at {sc.rows} rows")
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
        up_writes += self.inj.regions(p.rows, (self.up_name,))
        up_writes += self.bf16_out.regions(p.rows, (self.up_name,))
        return [
            Stage(
                self.down_name,
                p.down_tasks,
                partial(_down, p=p),
                lds_words=p.down_lds,
                writes=down_writes,
                args=src_arg + (a["norm"], a["dw"], a["iw"]),
                key=("hc.down",) + (p,),
            ),
            Stage(
                self.up_name,
                p.up_tasks,
                partial(_up, p=p),
                lds_words=p.up_lds,
                writes=up_writes,
                args=src_arg + (a["norm"], a["uw"]),
                key=("hc.up",) + (p,),
            ),
        ]

    def weight_specs(self) -> tuple:
        from atom.models.qwen38.mono.kernels.weights.source import WeightSpec

        geo, a = self.geo, self._args()
        ku = -(-geo.lowrank // 128)
        bf16 = "mono.bf16.gemv16"
        return (
            WeightSpec(a["norm"], "bf16", (geo.width,), (torch.bfloat16,), align=4),
            WeightSpec(a["dw"], bf16, (geo.lowrank * geo.width,), (torch.bfloat16,)),
            WeightSpec(a["uw"], bf16, (geo.width * ku * 128,), (torch.bfloat16,)),
            WeightSpec(a["iw"], bf16, (16 * geo.width,), (torch.bfloat16,), align=16),
        )

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """The HC's weights from ``layer.retained`` (bf16): the gamma as stored, the down and up
        projections and the injection rows (padded to 16 rows) as bf16 units (``gemv16.pack_bf16``).
        """
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
        if self.weights_fmt != "bf16":
            raise ValueError(f"HCIn {self.name}: bf16 hyper-connection weights only")
        out[a["dw"]] = gemv16.pack_bf16(down.to(torch.bfloat16).contiguous())
        out[a["uw"]] = gemv16.pack_bf16(upr.to(torch.bfloat16).contiguous())
        inject = r[self.weights.inject]
        if inject.shape != (geo.streams, geo.width):
            raise ValueError(
                f"HCIn {self.name}: inject {tuple(inject.shape)} does not fit {geo}"
            )
        inj = torch.zeros(16, geo.width, dtype=torch.bfloat16, device=inject.device)
        inj[: geo.streams] = inject
        out[a["iw"]] = gemv16.pack_bf16(inj)
        return out


# ------------------------------------------------------------------ device: the down stage


def _down(k, t, p):
    blk, s = t // p.slices, t % p.slices
    _down_either(k, p, blk, s)


@traced
def _down_either(k, p, blk, s):
    if blk == p.blocks:
        _down_task(k, p, "bf16", blk, s)
    else:
        _down_task(k, p, "fp4", blk, s)


def _down_units(k, p, fmt, blk, s):
    """Issue this wave's weight units: [row group][i] for K units ``wave + 8 i`` of slice ``s`` (``fmt`` "fp4": the
    low-rank rows, "bf16": the injection rows; both bf16 units)."""
    lane, w = k.lane, k.wave
    out = []
    for gl in range(p.g):
        per = []
        for i in range(p.c):
            ku = w + WAVES * i
            unit = s * p.us + ku
            valid = None if p.us % WAVES == 0 else ku < p.us
            if fmt == "fp4":  # the low-rank rows
                rg = blk * p.g + gl
                u = fx.min(rg, fx.Int32(p.rg_down - 1)) * p.k_units + unit
                per.append(
                    gemv16.load_bf16(
                        rsrc(k.args[f"{p.name}.dw"], p.lowrank * p.width * 2),
                        u,
                        lane,
                        valid,
                    )
                )
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


def _row_issue(k, p, rows, st, part):
    """Issue stream ``st``'s words of ``rows``: a plain arg's words, or one poll attempt of the channel's pairs."""
    m = p.qmap
    out = []
    for r in rows:
        base = r * (p.width // 2) + st * (p.hidden // 2)
        if not p.src_ch:
            out.append(
                (None, _map_load(m, k.args[p.src], base + part * m.words, k.tid))
            )
        else:
            specs = _map_specs(m, k.mb[p.src], base + part * m.words, k.tid)
            out.append((specs, _attempt(specs)))
    return out


def _row_land(k, p, issued):
    """``issued`` rows -> per row the ``qmap`` chunks of words. A channel source's rows settle in one spin (one
    round trip after the last producer, not one per row)."""
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
        out.append(_map_words(p.qmap, got)[0])
    return out


@traced
def _put_rstd(k, p, ok, r, st, rstd):
    if ok:
        mb.put(k.mb[f"{p.name}.rstd"], r * p.streams + st, rstd, k.tag)


@traced
def _place(k, p, r, v, words):
    """Slot ``v`` of this thread (``wps`` packed bf16 of normed row ``r``) into the LDS operand."""
    m = p.qmap
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


def _torch_issue(k, p, units, st):
    """Wave ``i`` (< ``len(units)``) fetches unit ``units[i] = (row, m)``'s vectors of stream
    ``st`` (torch's thread ``lane + 64 m`` of the row: the row's vectors ``lane + 64 m + sum_lanes
    j``, 2 words each). A plain source: loads, the other waves' offsets past the source (no
    access). A channel: one poll attempt (``_attempt``; the other waves repeat the last unit),
    settled by :func:`_direct_rstds`."""
    # the selects in VGPRs (SGPRs bind in composed programs)
    wave, lane = fresh(k.wave), fresh(k.lane)

    def word(u):
        r, m = u
        return r * (p.width // 2) + st * (p.hidden // 2) + 64 * m * 2

    base = fx.Int32(word(units[-1]))
    for i in range(len(units) - 1):
        base = (wave == i).select(fx.Int32(word(units[i])), base)
    vecs = p.hidden // 4 // p.sum_lanes
    if p.src_ch:
        base = base + lane * 2
        specs = [(k.mb[p.src], base + p.sum_lanes * 2 * j, 2) for j in range(vecs)]
        return specs, _attempt(specs)
    if len(units) < WAVES:
        base = (wave < len(units)).select(base, fx.Int32(gemv.OOB // 4))
    base = base + lane * 2
    res = rsrc(k.args[p.src], p.rows * p.width * 2)
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
        if p.src_ch and not landed:  # the poll attempt settled: 2 words a vector
            words = settle(k, *tw)
            tw = [words[2 * j : 2 * j + 2] for j in range(len(words) // 2)]

        def squares(v, c, tw=tw):
            return _squares(tw[c])

        # lane / wave through opaque copies: masks on k.lane / k.wave would be hoisted to the
        # kernel top and hold SGPR pairs across every stage
        lane, wave = fresh(k.lane), fresh(k.wave)
        # the unit's vectors, in order
        part = torch_thread_sum(squares, 256 * len(tw), 64, lane, 0)
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
    return [fx.Float32(fx.ptr_load(k.ldsf + (p.red + r))) for r in rows]


def _normalize(k, p, issued_rows, rows, gw, st, publish_rstd, gi=0, rstds=None):
    """Normalize ``rows`` (their landed words, norm group ``gi``) into the LDS operand; publish
    each row's rstd (``rstds``: given)."""
    xs = _row_land(k, p, issued_rows)
    if rstds is None:
        rstds = _rstds(k, p, rows)
    g1 = _gamma1(gw)
    for r, q, rstd in zip(rows, xs, rstds, strict=True):
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
        _put_rstd(k, p, publish_rstd & (k.tid == 0), r, st, rstd)


def _stage_raw(k, p, issued_rows, rows):
    """``norm_from`` "lds": ``rows``' landed words, raw, into the LDS operand (each slot where ``_place`` puts it)."""
    for r, q in zip(rows, _row_land(k, p, issued_rows), strict=True):
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


def _normalize_lds(k, p, rows, gw, st, publish, rstds):
    """``norm_from`` "lds": each thread's slots of ``rows`` normalized in place (the raw words it staged)."""
    m = p.qmap
    g1 = _gamma1(gw)
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
        _put_rstd(k, p, publish & (k.tid == 0), r, st, rstd)


def _down_task(k, p, fmt, blk, s):
    st, part = s // p.split, s % p.split
    # masks on it are not CSE'd with other stages' and held across them
    lane = fresh(k.lane)
    # 1. the weights first, then the gamma, the rows' first group and (a plain source) the waves'
    # own loads of their rows in torch's layout for the norm's sums (vmcnt is in order)
    units = _down_units(k, p, fmt, blk, s)
    rocdl.sched_barrier(0)
    gw = _map_load(
        p.qmap,
        k.args[f"{p.name}.norm"],
        st * (p.hidden // 2) + part * p.qmap.words,
        k.tid,
    )
    groups = [
        list(range(r0, min(r0 + p.norm_rows, p.rows)))
        for r0 in range(0, p.rows, p.norm_rows)
    ]
    early = _row_issue(k, p, groups[0], st, part)
    # "stream": the sums' loads go out now too (one pass, ``HCIn.plan``)
    tws = (
        []
        if p.norm_from == "lds"
        else [_torch_issue(k, p, us, st) for us in _sum_units(p)]
    )
    rocdl.sched_barrier(0)
    # the weights decoded to bf16 MFMA operands while the rows are still in flight
    ops = [[gemv16.bf16_operands(u) for u in per] for per in units]
    # 2. the norm of every row into LDS (rstd published by block 0)
    publish = blk == 0
    if p.norm_from == "lds":
        # the stream lands once, raw, in the operand; the sums read it there in torch's layout; normalized in place
        for gi, rows in enumerate(groups):
            _stage_raw(
                k, p, early if gi == 0 else _row_issue(k, p, rows, st, part), rows
            )
        gpu.barrier()
        _direct_rstds(
            k, p, [_lds_vectors(k, p, us) for us in _sum_units(p)], landed=True
        )
        gpu.barrier()
        post = _direct_post(k, p)
        for rows in groups:
            _normalize_lds(
                k,
                p,
                rows,
                gw,
                st,
                publish,
                _rstds(k, p, rows) if post is None else [post[r] for r in rows],
            )
    else:
        # every row's rstd at once: one barrier, none per norm group
        _direct_rstds(k, p, tws)
        gpu.barrier()
        post = _direct_post(k, p)
        for gi, rows in enumerate(groups):
            issued = early if gi == 0 else _row_issue(k, p, rows, st, part)
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
            )
    gpu.barrier()
    # 3. MFMAs: K units wave + 8 i, every row group
    row = fx.min(lane % 16, fx.Int32(p.rows - 1))
    j = lane // 16
    accs = []
    for per in ops:
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i, a in enumerate(per):
            ku = k.wave + WAVES * i
            if p.us % WAVES:
                ku = fx.min(ku, fx.Int32(p.us - 1))
            b = gemv16.b_operands(k.lds, row, j, ku, p.pitch)
            acc = gemv16.mfma_unit(acc, a, b)
        accs.append(acc)
    # 4. wave partials through LDS (over the operand), summed in wave order, published
    gpu.barrier()
    _red_store(k, p, accs)
    gpu.barrier()
    _publish_part(k, p, fmt, blk, s, len(units))
    gpu.barrier()  # the next task reuses the LDS


@traced
def _red_store(k, p, accs):
    """Wave partials -> LDS [wave][row group][row][16] f32 (useful lanes only)."""
    n = fresh(k.lane) % 16
    if n < p.rows:
        for gl in range_constexpr(len(accs)):
            off = ((k.wave * p.g + gl) * p.rows + n) * 16 + (k.lane // 16) * 4
            fx.ptr_store(accs[gl], k.ldsf + off)


def _publish_part(k, p, fmt, blk, s, groups):
    items = groups * p.rows * 4
    for c in range(-(-items // THREADS)):
        e = fx.Int32(c * THREADS) + k.tid
        ok = e < items
        e = fx.min(e, fx.Int32(items - 1))
        gl, n, f4 = e // (p.rows * 4), (e // 4) % p.rows, e % 4
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
            col = rg * 16 + f4 * 4
            _put_part4(k, p, ok, s, n, col, tot)
        else:
            _put_part_inj(k, p, ok & (f4 * 4 < p.streams), s, n, f4 * 4, tot)


@traced
def _put_part4(k, p, ok, s, n, col, tot):
    if ok:
        i = (s * p.rows + n) * p.nt + col
        mb.put2(k.mb[f"{p.name}.part"], i, tot[0], tot[1], k.tag)
        mb.put2(k.mb[f"{p.name}.part"], i + 2, tot[2], tot[3], k.tag)


@traced
def _put_part_inj(k, p, ok, s, n, c0, tot):
    if ok:
        i = (s * p.rows + n) * p.nt + p.lowrank + c0
        for q in range_constexpr(4):
            if c0 + q < p.streams:
                mb.put(k.mb[f"{p.name}.part"], i + q, tot[q], k.tag)


# ------------------------------------------------------------------ device: the up stage


def _up(k, t, p):
    lane, wave = fresh(k.lane), k.wave  # (see _down_task)
    j = lane // 16
    n = fx.min(lane % 16, fx.Int32(p.rows - 1))
    # this wave's row groups (stream, half) and their weights; the lane's stream / gamma words
    mine = []
    for i in range(-(-p.rgu // WAVES)):
        r = fx.min(wave + WAVES * i, fx.Int32(p.rgu - 1))
        valid = None if p.rgu % WAVES == 0 else (wave + WAVES * i) < p.rgu
        units = []
        for ku in range(p.ku):
            u = (t * p.rgu + r) * p.ku + ku
            units.append(
                gemv16.load_bf16(
                    rsrc(k.args[f"{p.name}.uw"], p.width * p.ku * 256), u, lane, valid
                )
            )
        # the MFMA puts rows on the result's M side: lane l holds feature l % 16 of rows
        # 4 (l // 16) + e, e < ``p.er``: only live rows' epilogue math is issued
        st, half = r // 2, r % 2
        col = t * COLS + half * 16 + lane % 16  # within the stream
        gword = _words(k.args[f"{p.name}.norm"], (st * p.hidden + col) // 2, 1)[0]
        es = [
            (fx.min(j * 4 + e, fx.Int32(p.rows - 1)) * p.width + st * p.hidden + col)
            // 2
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
    ops = [[gemv16.bf16_operands(u) for u in m[2]] for m in mine]
    # the partials (slice order), the rstds -> the scaled SiLU of the low-rank row in LDS; the
    # streams' words (a channel source) settle in the same spin
    _lowrank(
        k,
        p,
        t,
        [sp for m in mine for sp in m[6]],
        [w for m in mine if m[6] for w in m[7]],
    )
    gpu.barrier()
    for i, (r, valid, units, st, half, gwords, hspecs, hraw) in enumerate(mine):
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for ku, a in enumerate(ops[i]):
            acc = gemv16.mfma_unit(acc, gemv16.b_operands(k.lds, n, j, ku, p.upitch), a)
        hw = (
            [
                fx.ptr_load(k.lds + (p.u_x + k.tid * p.nx + p.er * i + e))
                for e in range(p.er)
            ]
            if hspecs
            else hraw
        )
        _terms(k, p, valid, st, half, acc, gwords, hw)
    gpu.barrier()
    _mixed(k, p, t)
    _inject_task0(k, p, t)  # (after its mixed row: off the mixed row's path)
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
        p.wait == "sentinel"
    ):  # one pair of every low-rank producer task first, then the items
        ns = p.blocks * p.slices
        sidx = fx.min(k.tid, fx.Int32(ns - 1))
        # the first column of the task's first row group
        col = (sidx // p.slices) * (16 * p.g)
        sspec = [(k.mb[f"{p.name}.part"], (sidx % p.slices) * p.rows * p.nt + col, 1)]
        _sentinel(k, ns, sspec)
    for c0 in range(0, rounds, per):
        cn = min(rounds, c0 + per)
        ex = extras if c0 == 0 else ([], [])
        if (
            c0 + 1
        ) * THREADS > items:  # threads past the batch's items: only the extras
            _batch_or_extras(k, p, c0, cn, ex)
        else:
            _batch(k, p, c0, cn, ex)
    _zero_pad_all(k, p)


@traced
def _sentinel(k, ns, sspec):
    """Threads < ``ns`` spin on one pair each, then the CTA meets."""
    if k.tid < ns:
        settle(k, sspec, _attempt(sspec))
    gpu.barrier()


@traced
def _batch_or_extras(k, p, c0, cn, ex):
    if fx.Int32(c0 * THREADS) + k.tid < p.rows * (p.lowrank // 2):
        _batch(k, p, c0, cn, ex)
    else:
        _extras(k, p, settle(k, *ex))


def _batch(k, p, c0, cn, ex):
    """Item rounds ``c0 .. cn - 1`` of this thread (and the extras ``ex``): one spin, then the
    scaled SiLU of every item into LDS."""
    half_nt = p.lowrank // 2
    items = p.rows * half_nt
    part = k.mb[f"{p.name}.part"]
    work = []
    for c in range(c0, cn):
        idx = fx.Int32(c * THREADS) + k.tid
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
    for i, (ok, n, col, _, _) in enumerate(work):
        g = got[2 * p.slices * i : 2 * p.slices * (i + 1)]
        tot = [g[q].bitcast(fx.Float32) for q in range(2)]
        for s in range(1, p.slices):
            tot = [tot[q] + g[2 * s + q].bitcast(fx.Float32) for q in range(2)]
        _lowrank_item(k, p, ok, n, col, tot)


def _extras(k, p, words):
    """The rstd (thread < rows * streams) and the extra words (LDS ``u_x``, per thread)."""
    if not words:
        return
    _store_word(k, k.tid < p.rows * p.streams, p.u_rstd + k.tid, words[0])
    for i, w in enumerate(words[1:]):
        fx.ptr_store(w, k.lds + (p.u_x + k.tid * p.nx + i))


def _lowrank_item(k, p, ok, n, col, tot):
    d = [bf16_round(v) for v in tot]
    gs = [bf16_round(_per_stream(v, p)) for v in d]
    g = [v * sigmoid(v) for v in gs]
    _store_g(k, p, ok, n, col, g)


@traced
def _inject_task0(k, p, t):
    """Task 0: the injection multipliers (only here are the injection tasks' partials polled)."""
    if t == 0:
        _inject_rows(k, p)


def _inject_rows(k, p):
    part = k.mb[f"{p.name}.part"]
    items = p.rows * p.streams
    for c in range(-(-items // THREADS)):
        idx = fx.Int32(c * THREADS) + k.tid
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


def _terms(k, p, valid, st, half, acc, gword, hw):
    """term = sigmoid(bf16(acc)) * normed of the lane's live rows (``acc[e]``: row 4 (lane // 16)
    + e, feature lane % 16) -> LDS [row][stream][32]."""
    f = k.lane % 16
    odd = (f % 2) == 1  # the column's half of its bf16 word (t * 32 + 16 half is even)
    w1 = odd.select(bf2_f32(gword)[1], bf2_f32(gword)[0]) + 1.0
    for e in range(p.er):
        row = (k.lane // 16) * 4 + e
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
        chan.put_bf16(k.mb[p.bf], e, mixed, k.tag)
