# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``GatheredDot``: fp32 dot products of each row with rows of a bf16 table, chosen per row by index (a gathered
rescore: exact scores of candidates an earlier stage screened) or a fixed leading block of the table
(a small dense projection).

Per row ``r`` of the step and column ``c < C``::

    out[r, c] = sum_k x[r, k] * table[row(r, c), k]        (fp32 products and sums, bf16 inputs)

``row(r, c)`` is element ``c`` of row ``r`` of ``idx`` (an ``F32`` channel carrying int32 indices, e.g. a ``TopK``
output; a negative index gives 0) or, without ``idx``, ``c`` itself. ``x``: a BF16 channel of width K; ``table``:
the plain bf16 [rows of the table, K] arg (64-bit row addressing: tables past 2 GiB); ``out``: an F32 channel of
width C (tagged).

Stage: ``rows x C / 16`` tasks; a task polls its x row into LDS once, then wave ``w`` takes columns ``2 w``,
``2 w + 1`` of its 16: lane ``l`` multiplies K / 64 elements (16 B loads of the table row) and the wave sums
them (butterfly). Reduction order (the contract): each lane adds its elements in K order, then the wave butterfly
(32, 16, 8, 4, 2, 1).

**Dense K-split mode** (``kslices`` = S > 0, no ``idx``; a small dense projection such as an MoE router): the table
is read once a step, not once a row. ``C / 16 x S`` tasks: task (tile, s) lands x[rows, K slice s] in
LDS and its 8 waves each run ``K / S / 256`` ``mfma_f32_16x16x32_bf16`` over their 1 / 8 of the slice (16 columns x
16 rows a tile: exact products, fp32 sums; over 16 rows an MFMA per 16-row chunk on the same table operand, a row's
sums those of one chunk), the waves' tiles are summed in wave order, and the slice's partial goes
to a scratch channel (``part``); stage ``<name>.sum`` (a task a row) adds the S partials in slice order, or, with
``sum_stage=False``, the consumer does. Reduction order: the MFMA's, then waves, then slices.

**Norm prologue** (dense mode, ``norm=DotNorm(ssq, gamma, eps)``): ``x`` is the raw row (e.g. a seam's LIN) and the
task normalizes its own K slice, y = bf16(x rstd w) with ``NormQuant(NormSpec(plain=True, plain_once=True))``'s
arithmetic: rstd from ``ssq`` (the per-16-element dot2 partials of ``HCSeam(ssq_out=)``: wave w of the norm's map sums
partials 64 w .. 64 w + 63 by the butterfly, the waves are added in order, rstd = v_rsq(sum / K + eps)). So the same
bits as the norm stage's bf16 copy, one hop fewer (the task does not wait for the norm stage). ``DotNorm(ssq="")``:
no partials channel, the task computes them from x itself (thread i < K / 16: the dot2 sum of squares of x's elements
16 i .. 16 i + 15, word order, as ``NormQuant``'s thread of a one-chunk 8-word map: the same bits), polling the whole
row; ``width``: the norm's width (its mean), when x carries zeros past it; ``centred``: the zero-centred weight,
y = bf16(x rstd (1 + w)) (``NormQuant``'s default arithmetic: its bf16 copy's bits).

**A narrower x** (``k`` > the x channel's width, dense mode): the table's K is ``k``, x's row ``x.width`` (a multiple
of 32), its elements past the width read as zero (e.g. a 2944-wide row against a 3072-wide table).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import (
    bf2_f32,
    bf16_pair,
    hw_rsq,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
    wave_sum,
)
from ...plan.program import Scenario, Stage
from ...target import MI355X
from ..base import Block, Channel, Fmt

COLS = 16  # columns per task: two per wave
WAVES, THREADS = MI355X.waves, MI355X.threads


@dataclass(frozen=True)
class DotNorm:
    """The dense mode's norm prologue (module docstring): ``ssq`` the tagged F32 channel of x's per-16-element partial
    sums of squares (width K / 16), ``gamma`` the bf16 [K] weight arg, ``eps`` the norm's.
    """

    ssq: str  # "": computed in the task from x (module docstring)
    gamma: str
    eps: float
    width: int = 0  # the norm's width (its mean); 0: K
    centred: bool = (
        False  # the zero-centred weight (``NormQuant``'s default): y = bf16(x rstd (1 + w)), (1 + w) in f32
    )


@dataclass(frozen=True)
class _Cfg:
    k: int
    c: int
    x: str
    idx: str  # "" without an index channel
    table: str
    out: str

    @property
    def per_lane(self) -> int:  # table words (bf16 pairs) per lane
        return self.k // 2 // 64


def _land_x(k, c: _Cfg, r):
    """Row r of x into LDS words 0 .. K/2 (every thread polls its share, two pairs a spec)."""
    words = c.k // 2
    per = -(-words // (2 * THREADS))
    specs, dests = [], []
    for i in range(per):
        w = fx.min((k.tid + i * THREADS) * 2, fx.Int32(words - 2))
        specs.append((k.mb[c.x], r * words + w, 2))
        dests.append(w)
    got = mb.poll(specs, k.tag)
    for g, d in zip(got, dests, strict=True):
        fx.ptr_store(
            fx.Vector.from_elements(g, fx.Int32).bitcast(fx.Float32), k.ldsf + d
        )


def _emit(k, t, c: _Cfg):
    r, chunk = t // (c.c // COLS), t % (c.c // COLS)
    _land_x(k, c, r)
    gpu.barrier()
    for h in range(2):
        col = chunk * COLS + k.wave * 2 + h
        if (
            c.idx
        ):  # one column per wave at a time: the index is wave-uniform (the table's buffer base is scalar)
            row = uniform(mb.poll([(k.mb[c.idx], r * c.c + col, 1)], k.tag)[0][0])
        else:
            row = col
        ok = row >= 0
        base = k.args[c.table] + fx.Int64(fx.max(row, fx.Int32(0))) * fx.Int64(c.k * 2)
        r_t = rsrc(base)
        acc = fx.Float32(0.0)
        for j in range(0, c.per_lane, 4):
            w0 = k.lane * c.per_lane + j
            tv = fx.Vector(bo.buffer_load(r_t, w0, vec_width=4, dtype=T.i32))
            xv = fx.Vector(
                fx.ptr_load(k.ldsf + w0, result_type=fx.Vector.make_type(4, fx.Float32))
            ).bitcast(fx.Int32)
            for e in range(4):
                tl, th = bf2_f32(tv[e])
                xl, xh = bf2_f32(xv[e])
                acc = acc + tl * xl
                acc = acc + th * xh
        tot = wave_sum(acc)
        _put(k, c, r * c.c + col, ok.select(tot, fx.Float32(0.0)))
    gpu.barrier()  # the next task reuses the LDS


@traced
def _put(k, c: _Cfg, i, v):
    if k.lane == 0:
        mb.put(k.mb[c.out], i, v, k.tag)


@dataclass(frozen=True)
class _DCfg:  # the dense K-split mode
    k: int
    c: int
    s: int  # K slices
    rows: int
    x: str
    table: str
    part: str  # the partials' channel: width S x C (slice s at s x C)
    out: str
    tiles: int = (
        1  # 16-column tiles a task (T): its 8 waves split T tiles x 8 / T K parts; x is polled C / 16 / T times
    )
    norm: DotNorm | None = None  # the norm prologue (x raw, normalized in the task)
    xw: int = (
        0  # x's row width when narrower than K (its elements past it read as zero); 0: K
    )

    @property
    def ks(self) -> int:  # K per slice
        return self.k // self.s

    @property
    def pitch(
        self,
    ) -> (
        int
    ):  # LDS words per x row (+4: the 16 rows of an operand load on distinct banks)
        return self.ks // 2 + 4

    @property
    def nwaves(
        self,
    ) -> int:  # norm: the waves of the norm's map (K / 16 partials, 64 a wave)
        return self.k // 16 // 64

    @property
    def red(
        self,
    ) -> int:  # norm: LDS word of the waves' sums [row][WAVES], after the x slice
        return self.rows * self.pitch

    @property
    def rc(
        self,
    ) -> int:  # 16-row chunks (the MFMA's N): an MFMA a chunk on each table operand
        return -(-self.rows // 16)

    @property
    def lds_words(
        self,
    ) -> (
        int
    ):  # the x slice (norm: then the waves' sums), then (overlaid) the waves' tiles
        return max(
            self.rows * self.pitch + (self.rows * WAVES if self.norm else 0),
            WAVES * 64 * 4 * self.rc,
        )


def _emit_dense(k, t, c: _DCfg):
    """Task t = tile group x S + slice: columns 16 T g .. + 16 T, every row, K slice s. Wave w: tile w % T of the
    group, K part w // T of the slice (T = 1: every wave a K part of the one tile)."""
    g, s = t // c.s, t % c.s
    words, kw = c.ks // 2, c.k // 2  # x words per row: in the slice, in the row
    wp = WAVES // c.tiles  # K parts a tile
    tile = g * c.tiles + k.wave % c.tiles
    kpart = k.wave // c.tiles
    # the tile's table loads first: they do not depend on x, so they are in flight through x's wait (a dot task
    # starts long before its input lands)
    n = k.lane % 16  # A: the tile's column n; B: x row n (clamped)
    r_t = rsrc(k.args[c.table])
    kbs = [
        kpart * (c.ks // wp) + j * 32 + (k.lane // 16) * 8
        for j in range(c.ks // (wp * 32))
    ]  # K in slice
    avs = [fx.Vector(bo.buffer_load(r_t, (tile * 16 + n) * kw + (s * c.ks + kb) // 2, vec_width=4, dtype=T.i32))
           for kb in kbs]  # fmt: skip
    rocdl.sched_barrier(
        0
    )  # keep them here: LLVM would sink each load to its MFMA, past the poll
    per = -(-(c.rows * words) // (2 * THREADS))
    specs, dests, rws, gws = [], [], [], []
    for i in range(per):
        e = fx.min((k.tid + i * THREADS) * 2, fx.Int32(c.rows * words - 2))
        r, w = e // words, e % words
        specs.append((k.mb[c.x], r * kw + s * words + w, 2))
        dests.append(r * c.pitch + w)
        rws.append(r)
        if c.norm is not None:  # the gamma words of these two x words (before the poll)
            gws.append(
                fx.Vector(
                    bo.buffer_load(
                        rsrc(k.args[c.norm.gamma]),
                        s * words + w,
                        vec_width=2,
                        dtype=T.i32,
                    )
                )
            )
    if c.norm is None:
        got = mb.poll(specs, k.tag)
    elif (
        c.norm.ssq
    ):  # the norm's partials in the same poll: wave w < nwaves, lane l: partial 64 w + l of every row
        nq = c.k // 16
        ip = fx.min(k.wave * 64 + k.lane, fx.Int32(nq - 1))
        got = mb.poll(
            specs + [(k.mb[c.norm.ssq], r * nq + ip, 1) for r in range(c.rows)], k.tag
        )
        got, sq = got[: len(specs)], got[len(specs) :]
        _norm_red(k, c, [g[0].bitcast(fx.Float32) for g in sq])
        gpu.barrier()
        got = [_norm_words(k, c, rws[i], g, gws[i]) for i, g in enumerate(got)]
    for gg, d in zip(got, dests, strict=True):
        fx.ptr_store(
            fx.Vector.from_elements(gg, fx.Int32).bitcast(fx.Float32), k.ldsf + d
        )
    gpu.barrier()
    rows = [
        fx.min(n if ch == 0 else n + 16 * ch, fx.Int32(c.rows - 1))
        for ch in range(c.rc)
    ]
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(c.rc)]
    for kb, a in zip(kbs, avs, strict=True):
        for ch, row in enumerate(rows):
            b = fx.ptr_load(
                k.ldsf + (row * c.pitch + kb // 2),
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
            accs[ch] = mfma_bf16(
                a.bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), accs[ch]
            )
    gpu.barrier()  # every wave's x reads are done: the LDS takes the tiles
    for ch, acc in enumerate(
        accs
    ):  # chunk ch's tiles after chunk ch - 1's (8 waves x 64 lanes x 4 a chunk)
        fx.ptr_store(acc, k.ldsf + ((k.wave * 64 + k.lane) * 4 + ch * WAVES * 256))
    gpu.barrier()
    # output o < 256 T: tile o // 256 of the group, lane l = (o % 256) // 4, element i: column 4 (l // 16) + i,
    # row l % 16 (+ 16 a chunk); its K parts' tiles (waves kp T + tile) summed in K-part order (T = 1: wave order)
    for ch in range(c.rc):
        for ps in range(-(-256 * c.tiles // THREADS)):
            o = k.tid + ps * THREADS
            oc = fx.min(o, fx.Int32(256 * c.tiles - 1))
            ti, rem = oc // 256, oc % 256
            lane, i = rem // 4, rem % 4
            v = fx.Float32(0.0)
            for kp in range(wp):
                v = v + fx.Float32(
                    fx.ptr_load(
                        k.ldsf
                        + (((kp * c.tiles + ti) * 64 + lane) * 4 + i + ch * WAVES * 256)
                    )
                )
            col = (g * c.tiles + ti) * 16 + (lane // 16) * 4 + i
            row = lane % 16 if ch == 0 else lane % 16 + 16 * ch
            _put_part(
                k,
                c,
                row * (c.s * c.c) + s * c.c + col,
                v,
                (o < 256 * c.tiles) & (row < c.rows),
            )
    gpu.barrier()  # the next task reuses the LDS


@traced
def _norm_red(k, c: _DCfg, sq):
    """Wave w < nwaves: each row's partials 64 w .. + 63 summed by the butterfly (``wave_sum``, the norm's) -> LDS
    [row][w]."""
    if k.wave < c.nwaves:
        tot = [wave_sum(v) for v in sq]
        if k.lane == 0:
            for r in range_constexpr(c.rows):
                fx.ptr_store(tot[r], k.ldsf + (c.red + r * WAVES + k.wave))


def _norm_words(k, c: _DCfg, r, words, gwords):
    """Two x words of row ``r`` normalized: rstd from the waves' sums in wave order (``norm.row_rstd``), then y =
    bf16((x rstd) w) per element (``norm.quantize_slot``, plain_once) -> two bf16-pair words.
    """
    tot = fx.ptr_load(k.ldsf + (c.red + r * WAVES))
    for w in range(1, c.nwaves):
        tot = tot + fx.ptr_load(k.ldsf + (c.red + r * WAVES + w))
    rstd = hw_rsq(tot * (1.0 / (c.norm.width or c.k)) + c.norm.eps)
    out = []
    for j in range(2):
        lo, hi = bf2_f32(words[j])
        glo, ghi = bf2_f32(gwords[j])
        out.append(bf16_pair(lo * rstd * glo, hi * rstd * ghi).bitcast(fx.Int32))
    return out


@traced
def _put_part(k, c: _DCfg, i, v, ok):
    if ok:
        mb.put(k.mb[c.part], i, v, k.tag)


def _sum_slices(k, t, c: _DCfg):
    """Row t: each column's S partials added in slice order -> out (a thread a column)."""
    col = fx.min(k.tid, fx.Int32(c.c - 1))
    got = mb.poll(
        [(k.mb[c.part], t * (c.s * c.c) + s * c.c + col, 1) for s in range(c.s)], k.tag
    )
    v = fx.Float32(0.0)
    for g in got:
        v = v + g[0].bitcast(fx.Float32)
    _put_out(k, c, t * c.c + col, v, k.tid < c.c)


@traced
def _put_out(k, c: _DCfg, i, v, ok):
    if ok:
        mb.put(k.mb[c.out], i, v, k.tag)


class GatheredDot(Block):
    """``out = x . table[idx]^T`` per row (module docstring). ``weight``: the table's ``layer.retained`` name
    (``pack``); else the caller provides arg ``table``."""

    def __init__(self, name: str, x: Channel, table: str, out: Channel, idx: Channel | None = None,
                 weight: str | None = None, kslices: int = 0, sum_stage: bool = True, tiles: int = 1,
                 norm: DotNorm | None = None, k: int | None = None):  # fmt: skip
        xw = 0
        self.xw = xw
        if kslices and (
            idx is not None
            or x.width % (kslices * WAVES * 32)
            or (sum_stage and out.width > THREADS)
        ):
            raise ValueError(f"GatheredDot {name}: kslices (dense mode) needs no idx, K a multiple of "
                             f"{WAVES * 32} x kslices and (its sum stage) C <= {THREADS}")  # fmt: skip
        if tiles not in (1, 2, 4, 8) or (
            kslices
            and (
                (out.width // 16) % tiles
                or (x.width // kslices) % (WAVES // tiles * 32)
            )
        ):
            raise ValueError(
                f"GatheredDot {name}: tiles 1 / 2 / 4 / 8 dividing C / 16, K / S a multiple of 256 / tiles"
            )
        if norm is not None and (
            not kslices or x.width % (16 * 64) or x.width // 16 // 64 > WAVES
        ):
            raise ValueError(f"GatheredDot {name}: the norm prologue is the dense mode's, K a multiple of 1024 up to "
                             f"{WAVES * 1024}")  # fmt: skip
        self.kslices, self.sum_stage, self.tiles, self.norm = (
            kslices,
            sum_stage,
            tiles,
            norm,
        )
        # the dense mode's slice partials: [rows][S x C], slice s at s x C (``sum_stage=False``: the consumer adds
        # them in slice order, e.g. MoERouter's route, saving a stage and a round trip)
        self.part = (
            Channel(f"{name}.part", kslices * out.width, Fmt.F32) if kslices else None
        )
        if x.fmt is not Fmt.BF16 or x.flagged or x.width % 512:
            raise ValueError(f"GatheredDot reads a tagged BF16 channel, width a multiple of 512 ({x.name}: "
                             f"{x.fmt.value}, width {x.width}; a padded copy: NormSpec.copy_pitch, "
                             "BLOCK_CATALOG 'K')")  # fmt: skip
        if out.fmt is not Fmt.F32 or out.flagged or out.width % COLS:
            raise ValueError(
                f"out must be a tagged F32 channel, width a multiple of {COLS}"
            )
        if idx is not None and (
            idx.fmt is not Fmt.F32 or idx.flagged or idx.width != out.width
        ):
            raise ValueError(
                "idx must be a tagged F32 channel (int32 bits) as wide as out"
            )
        self.name, self.x, self.table, self.out, self.idx, self.weight = (
            name,
            x,
            table,
            out,
            idx,
            weight,
        )

    def stages(self, sc: Scenario) -> list[Stage]:
        if self.kslices:
            return self._dense_stages(sc)
        c = _Cfg(self.x.width, self.out.width, self.x.name, self.idx.name if self.idx else "", self.table,
                 self.out.name)  # fmt: skip
        return [
            Stage(
                self.name,
                sc.rows * c.c // COLS,
                partial(_emit, c=c),
                lds_words=c.k // 2,
                writes=self.out.regions(sc.rows, (self.name,)),
                args=(self.table,),
                key=("gathered_dot", c),
            )
        ]

    def _dense_stages(self, sc: Scenario) -> list[Stage]:
        part = self.part
        c = _DCfg(self.x.width, self.out.width, self.kslices, sc.rows, self.x.name, self.table, part.name,
                  self.out.name, self.tiles, self.norm, self.xw)  # fmt: skip
        sname = f"{self.name}.sum"
        dot = Stage(
            self.name,
            c.c // 16 // c.tiles * c.s,
            partial(_emit_dense, c=c),
            lds_words=c.lds_words,
            writes=part.regions(sc.rows, (self.name,)),
            args=(self.table,) + ((self.norm.gamma,) if self.norm is not None else ()),
            key=("gathered_dense", c),
        )
        if not self.sum_stage:
            return [dot]
        return [dot, Stage(sname, sc.rows, partial(_sum_slices, c=c), writes=self.out.regions(sc.rows, (sname,)),
                           key=("gathered_sum", c))]  # fmt: skip

    def pack(self, layer) -> dict[str, torch.Tensor]:
        return (
            {self.table: layer.retained[self.weight].contiguous()}
            if self.weight
            else {}
        )
