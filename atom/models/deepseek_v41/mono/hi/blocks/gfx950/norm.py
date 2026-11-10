# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``NormQuant``: zero-centred RMSNorm fused with MXFP8 quantization.

Per row of the step::

    y  = bf16(x * rsqrt(mean(x ** 2) + eps) * (1 + w))     (fp32 math, w = gamma)
    q, s = MXFP8(y)                                         (group 32, E8M0 scales)

exactly a zero-centred RMSNorm (the formula above) followed by
``weights.mx.quantize_mxfp8`` (scale = ceil_pow2(amax / 448) as an E8M0
code clipped to [1, 254], an all-zero group gets 127, elements RNE to E4M3).
The quantization is byte-identical to ``quantize_mxfp8`` whenever the fp32 sum
of squares agrees; the sum's order is this block's numerics contract (below).

Two strategies:

- **stage** (``NormQuant.stages``): ``rows * spec.split`` tasks; task ``t``
  normalizes part ``t % split`` of row ``t // split`` and puts it into an MXFP8
  ``Channel`` (and optionally a BF16 one). With ``split > 1`` every task of a
  row also loads the whole row and recomputes its sum of squares (same code,
  same order, so the same bits): no hand-off between the parts.
- **consumer prologue**: a consumer calls the device functions below inside its
  own task and places the returned fp8 words and scale codes in its own LDS
  layout (``NormQuant.stages`` returns nothing). Uses ``split = 1``.

Device contract (``NormSpec`` fixes the mapping; every function is a free
function). Rows are handled as packed bf16 *words* (2 elements):

- **Thread -> words.** ``WordMap(words, threads, wps)`` maps a range of
  ``words`` words: slot ``s = tid + v * threads`` (chunk ``v <
  map.chunks``) is words ``[s * wps, s * wps + wps)`` of the range, valid iff
  ``tid < map.limit(v)``. A 32-group (16 words) spans ``16 / wps`` adjacent
  lanes. Invalid slots load a clamped duplicate, add 0 to the sum and are not
  stored. ``spec.qmap`` maps the part a task quantizes (the whole row when
  ``split == 1``); default (width 5120): 320 threads x 8 words, i.e. thread
  ``i`` owns elements ``[16 i, 16 i + 16)`` and a group spans 2 lanes.
- **Results per thread, row and chunk:** ``(words, code, y)``: ``wps / 2``
  fp8 words (word ``j`` = elements ``2 (s wps) + 4 j .. + 3``, element ``c`` in
  byte ``c % 4``: the MXFP8 channel's pair ``(base + s wps) / 2 + j``, base =
  the part's first word), the E8M0 code (Int32) of group ``(base + s wps) /
  16`` (the same in its lanes; scale pair ``row * width / 32 + that``), and the
  ``wps`` packed bf16 words of ``y`` (for a BF16 copy).
- **LDS.** ``spec.lds_words(nrows)`` = ``nrows * 8`` f32 words (``red``).
- **Barriers.** ``row_rstd`` (inside ``norm_quant``) executes exactly two CTA
  barriers: one before it writes ``red`` (earlier LDS readers are done) and
  one after (partials visible). There is none at the end: a caller that reuses
  ``red`` before its next barrier must add one. Every thread must call it.
- **Reduction order (numerics contract).** Each thread accumulates its words of
  the sum map (chunk order, then word order) with ``v_dot2_f32_bf16`` (``acc +=
  lo^2 + hi^2``), waves reduce with the butterfly (32, 16, 8, 4, 2, 1)
  (``ops.wave_sum``), then waves 0..n-1 are added left to right; ``mean = sum
  * fp32(1 / width)``; ``rstd = v_rsq_f32(mean + eps)``; ``y = bf16((x * rstd)
  * (1 + w))``. The sum map is ``spec.smap``: ``qmap`` itself when ``split ==
  1``, else 512 threads x 4 words over the whole row.
- **Quantization.** amax is taken over the unrounded ``|x * rstd * (1 + w)|``
  and rounded to bf16 once (RNE is monotonic, so this equals the amax of the
  bf16 ``y``); the elements go bf16 -> E4M3 with ``v_cvt_scalef32_pk_fp8_bf16``
  (exact division by the power-of-two scale, RNE; |y| / scale <= 448 always).

Split-phase use in a consumer (``rows``: a list of row indices; ``split == 1``)::

    gw = load_gamma(spec, k.args["gamma"], k.tid)              # before any wait
    xs = poll_rows_bf16(spec, k.mb["h1"], rows, k.tid, k.tag)  # or load_rows_bf16(...)
    res = norm_quant(spec, xs, gw, red, k.tid, k.lane, k.wave)
    # res[j][v] = (words, code, y) of row rows[j], chunk v: place them (if tid < spec.qmap.limit(v))

Registers: every row's words stay live until ``norm_quant`` returns (``rows x
chunks x wps`` x words, plus the sum map's when it is not shared): with many
rows, call it per row group (two barriers per call for the CTA sum). A per-wave
sum (``sum="wave"``) holds 40 words of each row per lane and polls 20 specs per
row (two poll batches): fine from plain memory, a poor fit for a channel source.
The resource gate refuses builds that spill.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import channels as chan
from ...device.gfx950 import mailbox as mb
from ...device.gfx950 import peer
from ...device.gfx950.ops import (
    WAVES,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    hw_rsq,
    rsrc,
    traced,
    wave_sum,
)
from ...geometry import NormGeometry
from ...plan.program import Scenario, Stage
from ..base import Block, Channel, Fmt

GROUP = 32
GROUP_WORDS = GROUP // 2
WAVE = 64
THREADS = WAVES * WAVE
ZERO_CODE = 127


@dataclass(frozen=True)
class WordMap:
    """Slots of ``wps`` packed bf16 words over ``words`` words, interleaved over ``threads``."""

    words: int
    threads: int
    wps: int
    per_wave: bool = (
        False  # slots indexed by lane (threads == 64): every wave holds the whole range
    )

    def __post_init__(self):
        lanes = GROUP_WORDS // self.wps if self.wps <= GROUP_WORDS else 0
        if (
            self.wps not in (2, 4, 8, 16)
            or self.words % GROUP_WORDS
            or self.threads % lanes
        ):
            raise ValueError(f"bad {self}")
        if not 0 < self.threads <= THREADS or (self.per_wave and self.threads != WAVE):
            raise ValueError(f"{self}: threads must be in 1..{THREADS} (64 per wave)")

    @property
    def slots(self) -> int:
        return self.words // self.wps

    @property
    def chunks(self) -> int:
        return -(-self.slots // self.threads)

    @property
    def lanes(self) -> int:
        """Lanes sharing a 32-group."""
        return GROUP_WORDS // self.wps

    def limit(self, v: int) -> int:
        """Threads ``tid < limit(v)`` hold a valid chunk ``v``."""
        return min(self.threads, self.slots - v * self.threads)

    @property
    def waves(self) -> int:
        """Waves holding data."""
        return -(-self.limit(0) // WAVE)


@dataclass(frozen=True)
class NormSpec:
    """The plan-side constants of the norm (bound into emit by value). ``width`` and ``eps``
    are the norm's geometry (``NormGeometry``); the rest is the device mapping."""

    width: int
    eps: float
    threads: int = 320  # quantizing threads (``qmap``)
    wps: int = 8  # packed bf16 words per quantizing slot
    split: int = 1  # stage tasks per row (each quantizes 1 / split of it)
    sum: str = (
        "cta"  # "cta": sum over the CTA (LDS, 2 barriers); "wave": every quantizing wave sums the row
    )
    plain: bool = (
        False  # plain-weight RMSNorm, two roundings: y = bf16(bf16(x * rstd) * w), not bf16(x * rstd * (1 + w))
    )
    identity: bool = False  # no norm, y = x: a quantize-only stage (no gamma, no sum)
    # with ``plain``: one rounding, y = bf16(x * rstd * w) in fp32 (DeepSeek-V4.1's RMSNorm: the bf16 weight
    # times the fp32 normalized x, then one cast), not bf16(bf16(x * rstd) * w)
    plain_once: bool = False
    # split > 1, the CTA sum: the sum map's threads x words a slot over the whole row (0: 512 x 4). Give the split = 1
    # spec's threads / wps (e.g. 320 x 8) for its sum order, i.e. the same rstd and so the same bits as one task a row
    sum_threads: int = 0
    sum_wps: int = 0
    # the row stride (elements) of the source rows, the MXFP8 channel and the BF16 copy, when wider than the norm
    # (0: ``width``): a model whose width is not a multiple of 128 stores its rows padded (e.g. 2880 in 2944) and
    # the stage writes the tail ``width .. pitch`` as zero (fp8 0, E8M0 127, bf16 0), the K tail a GEMM pads with
    pitch: int = 0
    # the BF16 copy's row stride when it differs from ``pitch`` (0: the same), its tail zero too (e.g. a
    # consumer that wants rows a multiple of 512 wide)
    copy_pitch: int = 0

    def __post_init__(self):
        if self.width % (GROUP * self.split) or self.sum not in ("cta", "wave"):
            raise ValueError(
                f"{self}: width must split into whole groups; sum is 'cta' or 'wave'"
            )
        if self.sum_threads and (
            self.split == 1 or self.sum != "cta" or not self.sum_wps
        ):
            raise ValueError(
                f"{self}: sum_threads / sum_wps: a split CTA-sum spec, both given"
            )
        if self.plain_once and not self.plain:
            raise ValueError(
                f"{self}: plain_once is a rounding of the plain-weight norm (plain=True)"
            )
        if self.pitch and (
            self.pitch < self.width or (self.pitch - self.width) % GROUP
        ):
            raise ValueError(f"{self}: pitch is the width plus whole 32-groups")
        if self.copy_pitch and (
            self.copy_pitch < self.width or (self.copy_pitch - self.width) % 8
        ):
            raise ValueError(f"{self}: copy_pitch is the width plus a multiple of 8")
        self.qmap  # noqa: B018 - validates

    @property
    def row_width(self) -> int:
        """The stored row's elements (``pitch``, else ``width``)."""
        return self.pitch or self.width

    @property
    def copy_width(self) -> int:
        """The BF16 copy's row elements (``copy_pitch``, else ``row_width``)."""
        return self.copy_pitch or self.row_width

    @property
    def qmap(self) -> WordMap:
        return WordMap(self.width // 2 // self.split, self.threads, self.wps)

    @property
    def smap(self) -> WordMap:
        if self.split == 1:
            return self.qmap
        if self.sum_threads:
            return WordMap(self.width // 2, self.sum_threads, self.sum_wps)
        return WordMap(self.width // 2, THREADS, 4)

    @property
    def shared(self) -> bool:
        """Whether the sum reads the quantized words (no second load)."""
        return self.smap == self.qmap

    @property
    def waves(self) -> int:
        """Waves that take part (all of them for the CTA sum: barriers)."""
        return self.qmap.waves if self.sum == "wave" else WAVES

    def lds_words(self, nrows: int = 1) -> int:
        return nrows * WAVES if self.sum == "cta" and not self.identity else 0

    @property
    def tag(self) -> str:
        """Short name: ``<threads>x<wps>x<split>`` + ``w`` for the wave sum."""
        return f"{self.threads}x{self.wps}x{self.split}" + (
            "w" if self.sum == "wave" else ""
        )


# Defaults from the block evals (the norm measurements, width 5120): a
# plain-memory source wants many small tasks and no barrier; a channel source wants one CTA
# per row (a per-wave row poll needs two poll batches, i.e. an extra round trip). Other widths
# keep the same per-thread shapes: 8 words a thread over the row (5120: 320 threads), and
# 64 threads x 4 words per stage task (5120: 10 tasks per row).
CHANNEL_WPS = 8
ARG_THREADS, ARG_WPS = 64, 4


def _row_spec(norm: NormGeometry) -> NormSpec:
    """One CTA per row, ``CHANNEL_WPS`` words a thread (5120: 320 x 8)."""
    return NormSpec(
        norm.width, norm.eps, threads=norm.width // 2 // CHANNEL_WPS, wps=CHANNEL_WPS
    )


def _arg_spec(norm: NormGeometry) -> NormSpec:
    """``ARG_THREADS`` x ``ARG_WPS`` words per task, per-wave sums (5120: 64 x 4 x 10)."""
    part = 2 * ARG_THREADS * ARG_WPS  # elements a task quantizes
    split = norm.width // part if norm.width % part == 0 else None
    if (
        split is None
    ):  # fewer words per task: the smallest split into whole groups that fits
        groups = norm.width // GROUP
        split = next(
            s
            for s in range(1, groups + 1)
            if groups % s == 0 and norm.width // s <= part
        )
    return NormSpec(
        norm.width, norm.eps, threads=ARG_THREADS, wps=ARG_WPS, split=split, sum="wave"
    )


def stage_from_arg(norm: NormGeometry) -> NormSpec:
    return _arg_spec(norm)


def stage_from_channel(norm: NormGeometry) -> NormSpec:
    return _row_spec(norm)


def prologue(norm: NormGeometry) -> NormSpec:
    """The consumer-prologue spec (``linear.NormIn``)."""
    return _row_spec(norm)


# ------------------------------------------------------------------ device: loads


def default_spec(from_channel: bool, strategy: str, norm: NormGeometry) -> NormSpec:
    if strategy == "prologue":
        return prologue(norm)
    return stage_from_channel(norm) if from_channel else stage_from_arg(norm)


def _slot_word(m: WordMap, tid, v):
    """First word (within the map's range) of chunk ``v``'s slot, clamped into the range."""
    i = (tid & (WAVE - 1)) if m.per_wave else tid
    s = i + v * m.threads
    top = (
        (WAVE if m.per_wave else THREADS) - 1 + v * m.threads
    )  # the largest s any thread computes
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


def load_gamma(spec: NormSpec, ptr, tid, part=0):
    """Gamma (bf16 [width] at ``ptr``) of quantizing part ``part``: per chunk ``wps``
    packed words. Independent of upstream results: issue it before waiting on rows."""
    return _map_load(spec.qmap, ptr, part * spec.qmap.words, tid)


def load_rows_bf16(spec: NormSpec, ptr, rows, tid, part=0):
    """Rows ``rows`` (row indices) of a plain bf16 [*, width] tensor at ``ptr``:
    per row ``(q, s)``, the per-chunk words of part ``part`` (``qmap``) and of the
    whole row (``smap``; None when it is ``qmap``)."""
    out = []
    for r in rows:
        base = r * (spec.row_width // 2)
        q = _map_load(spec.qmap, ptr, base + part * spec.qmap.words, tid)
        out.append((q, None if spec.shared else _map_load(spec.smap, ptr, base, tid)))
    return out


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


def poll_rows_bf16(spec: NormSpec, addr, rows, tid, tag, part=0, remap=None):
    """Like ``load_rows_bf16`` from a BF16 channel (``addr`` = ``k.mb[name]``; word
    ``w`` of the row-major tensor is pair ``w``): one poll, batched by ``mb.POLL_MAX``
    16-byte specs. ``remap``: pair ``w`` -> its index from ``addr`` (a gathered region's owners' slots).
    """
    specs = []
    for r in rows:
        base = r * (spec.row_width // 2)
        specs += _map_specs(spec.qmap, addr, base + part * spec.qmap.words, tid)
        if not spec.shared:
            specs += _map_specs(spec.smap, addr, base, tid)
    if remap is not None:
        specs = [(a, remap(w), n) for a, w, n in specs]
    got = mb.poll(specs, tag)
    return _rows_words(spec, rows, got)


def _rows_words(spec: NormSpec, rows, got):
    out = []
    for _ in rows:
        q, got = _map_words(spec.qmap, got)
        s = None
        if not spec.shared:
            s, got = _map_words(spec.smap, got)
        out.append((q, s))
    return out


# ------------------------------------------------------------------ device: math


def _v2bf16(w):
    return as_ir_value(
        fx.Vector.from_elements([fx.Int32(w)], fx.Int32).bitcast(fx.BFloat16)
    )


def dot2_bf16(a, b, acc):
    """v_dot2_f32_bf16: acc + a.lo * b.lo + a.hi * b.hi (packed bf16 words a, b)."""
    op = rocdl.fdot2_f32_bf16(
        T.f32, _v2bf16(a), _v2bf16(b), as_ir_value(fx.Float32(acc))
    )
    return fx.Float32(getattr(op, "result", op))


def fp8x2_scaled(word, scale, old, hi: bool):
    """v_cvt_scalef32_pk_fp8_bf16: the two bf16 of ``word`` divided by the power of
    two ``scale`` (f32), RNE to E4M3, into half ``hi`` of ``old`` (Int32)."""
    o = as_ir_value(
        fx.Vector.from_elements([fx.Int32(old)], fx.Int32).bitcast(fx.Int16)
    )
    r = rocdl.cvt_scalef32_pk_fp8_bf16(
        T.vec(2, T.i16), o, _v2bf16(word), as_ir_value(fx.Float32(scale)), hi
    )
    return fx.Vector(r).bitcast(fx.Int32)[0]


def e8m0_code(amax):
    """E8M0 code of ceil_pow2(amax / 448) for a finite amax >= 0, clipped to [1, 254],
    127 for 0. Exact: 448 = 1.75 * 2^8, so the code follows from amax's exponent
    and whether its mantissa exceeds 1.75 (no division to round across a power of two).
    """
    bits = amax.bitcast(fx.Int32)
    exponent = (bits >> 23) & 255
    code = (
        exponent - 8 + ((bits & 0x7FFFFF) > 0x600000).select(fx.Int32(1), fx.Int32(0))
    )
    code = fx.max(fx.min(code, fx.Int32(254)), fx.Int32(1))
    return fx.Int32((amax == 0.0).select(fx.Int32(ZERO_CODE), code))


ACCS = 4  # independent dot2 accumulators per thread


def _sum_partial(m: WordMap, chunks, tid):
    """This thread's dot2 sum of squares over its valid chunks (contract order:
    chunk ``v`` accumulates into ``acc[v % 4]`` word by word; then ``((acc0 +
    acc1) + (acc2 + acc3))`` over the accumulators used)."""
    zero = fx.Int32(0)
    accs = []
    for v, words in enumerate(chunks):
        # a per-wave map's slots run over the lanes: its last chunk is partial when the slots are not a multiple of 64
        # (2880 wide: 360 slots of 4 words), and its clamped duplicates must add 0
        always = m.limit(v) >= (WAVE if m.per_wave else THREADS)
        ok = (
            None if always else ((tid & (WAVE - 1)) if m.per_wave else tid) < m.limit(v)
        )
        ss = accs[v % ACCS] if v >= ACCS else fx.Float32(0.0)
        for w in words:
            ws = w if always else ok.select(w, zero)
            ss = dot2_bf16(ws, ws, ss)
        if v < ACCS:
            accs.append(ss)
    while len(accs) > 1:
        accs = [
            accs[i] + accs[i + 1] if i + 1 < len(accs) else accs[i]
            for i in range(0, len(accs), 2)
        ]
    return accs[0]


@traced
def _partials(wvs, red, lane, wave):
    """Barrier; lane 0 of each wave writes its per-row partials; barrier."""
    gpu.barrier()
    if lane == 0:
        for r in range_constexpr(len(wvs)):
            fx.ptr_store(wvs[r], red + (r * WAVES + wave))
    gpu.barrier()


def row_rstd(spec: NormSpec, sums, red, lane, wave):
    """Per row: ``rstd`` from this thread's partial ``sums`` (CTA sum: two barriers;
    wave sum: none)."""
    _partials([wave_sum(s) for s in sums], red, lane, wave)
    out = []
    for r in range(len(sums)):
        tot = fx.ptr_load(red + r * WAVES)
        for w in range(1, spec.smap.waves):
            tot = tot + fx.ptr_load(red + (r * WAVES + w))
        out.append(hw_rsq(tot * (1.0 / spec.width) + spec.eps))
    return out


def quantize_slot(m: WordMap, words, g1, rstd, plain=False, identity=False):
    """One slot: ``wps`` packed bf16 x words, ``2 wps`` f32 (1 + w) (``plain``: w) -> (``wps / 2`` fp8
    words, the group's code, ``wps`` packed bf16 y words). ``plain``: the normalized x is rounded to
    bf16 before the gamma product (the two-rounding plain RMSNorm)."""
    v = []
    for w, j in zip(words, range(0, 2 * m.wps, 2)):
        lo, hi = bf2_f32(w)
        if identity:
            v += [lo, hi]
        else:
            v += [lo * rstd * g1[j], hi * rstd * g1[j + 1]]
    y = [bf16_pair(v[2 * j], v[2 * j + 1]).bitcast(fx.Int32) for j in range(m.wps)]
    am = fmath.absf(v[0])
    for x in v[1:]:
        am = fx.max(am, fmath.absf(x))
    am = butterfly(am, tuple(1 << b for b in range(m.lanes.bit_length() - 1)), fx.max)
    code = e8m0_code(bf16_round(am))
    scale = (code << 23).bitcast(fx.Float32)
    q = []
    for j in range(0, m.wps, 2):
        q.append(
            fp8x2_scaled(
                y[j + 1], scale, fp8x2_scaled(y[j], scale, fx.Int32(0), False), True
            )
        )
    return q, code, y


def gamma1(gw, plain=False):
    """(1 + w) in f32 per chunk, from ``load_gamma``'s words (``plain``: w)."""
    out = []
    for words in gw:
        vals = []
        for w in words:
            lo, hi = bf2_f32(w)
            vals += [lo, hi] if plain else [lo + 1.0, hi + 1.0]
        out.append(vals)
    return out


def norm_quant(spec: NormSpec, xs, gw, red, tid, lane, wave):
    """Normalize and quantize rows ``xs`` (``load_rows_bf16`` / ``poll_rows_bf16``)
    with gamma ``gw`` (``load_gamma``); ``red``: an f32 LDS pointer with
    ``spec.lds_words(len(xs))`` words. Returns per row, per chunk ``(words, code,
    y)``. Two CTA barriers; every thread of the CTA calls it (``spec.identity``: none, ``gw`` unused).
    """
    if spec.identity:
        return [
            [quantize_slot(spec.qmap, c, None, None, identity=True) for c in q]
            for q, _ in xs
        ]
    sums = [_sum_partial(spec.smap, q if s is None else s, tid) for q, s in xs]
    g1 = gamma1(gw, spec.plain)
    rstds = row_rstd(spec, sums, red, lane, wave)
    twice = (
        spec.plain and not spec.plain_once
    )  # plain_once: the (1 + w) path's arithmetic with w
    return [
        [quantize_slot(spec.qmap, c, g, rstd, twice) for c, g in zip(q, g1)]
        for (q, _), rstd in zip(xs, rstds)
    ]


# ------------------------------------------------------------------ device: channel stores


@traced
def _put_slot(
    out_q,
    out_s,
    out_bf,
    wbase,
    rgroup,
    w,
    lim,
    lanes,
    tid,
    tag,
    words,
    code,
    y,
    dense=False,
    bbase=None,
):
    """Slot at word ``w`` of the part starting at row-major word ``wbase`` (``dense``: the
    MXFP8 channel is bulk + flag: plain words, 16 bytes per store)."""
    if tid < lim:
        if const_expr(dense and len(words) % 4 == 0):
            for j in range_constexpr(0, len(words), 4):
                chan.store_words(out_q, ((wbase + w) >> 1) + j, words[j : j + 4])
        else:
            for j in range_constexpr(0, len(words), 2):
                chan.put_words(
                    out_q, ((wbase + w) >> 1) + j, words[j : j + 2], tag, dense
                )
        if tid % lanes == 0:
            chan.put_words(out_s, rgroup + (w >> 4), [code], tag, dense)
        if const_expr(out_bf is not None):
            for j in range_constexpr(0, len(y), 2):
                mb.put_words(
                    out_bf,
                    (wbase if bbase is None else bbase) + w + j,
                    y[j : j + 2],
                    tag,
                )


def put_row(
    spec: NormSpec,
    out_q,
    out_s,
    out_bf,
    row,
    tid,
    tag,
    chunks,
    part=0,
    dense=False,
    bcopy=None,
):
    """This thread's ``chunks`` (``norm_quant`` of ``row``, part ``part``) into the
    MXFP8 channel (``out_q`` / ``out_s``: ``k.mb`` addresses; ``dense``: a FLAG channel)
    and, unless ``out_bf`` is None, the (tagged) BF16 channel; ``bcopy`` (width, column): the copy's row width and
    its first column when the copy is a slice of a wider row (``NormQuant(bf16_col=)``).
    """
    m = spec.qmap
    wbase = row * (spec.row_width // 2) + part * m.words
    bbase = (
        None
        if spec.copy_width == spec.row_width
        else row * (spec.copy_width // 2) + part * m.words
    )
    for v, (words, code, y) in enumerate(chunks):
        w = (tid + v * m.threads) * m.wps
        _put_slot(out_q, out_s, out_bf, wbase, wbase >> 4, w, m.limit(v), m.lanes, tid, tag, words, code, y, dense,
                  bbase)  # fmt: skip


# ------------------------------------------------------------------ the stage


def _emit_stage(k, t, spec, src, src_is_channel, gamma, out, bf_out, proto="tagged", gather=None, bcopy=None,
                partials=None):  # fmt: skip
    dense = proto == "flag"
    _stage_body(
        k,
        t,
        spec,
        src,
        src_is_channel,
        gamma,
        out,
        bf_out,
        dense,
        gather,
        bcopy,
        partials,
    )
    if dense:  # bulk + flag: this task's words drained, then its flag (slot = task)
        chan.put_flag(k, f"{out}.f", t, spec.waves)


def _stage_body(k, t, spec, src, src_is_channel, gamma, out, bf_out, flag=False, gather=None, bcopy=None,
                partials=None):  # fmt: skip
    if spec.split == 1:
        row, part = t, 0
    else:
        row, part = t // spec.split, t % spec.split
    gw = None if spec.identity else load_gamma(spec, k.args[gamma], k.tid, part)
    if (
        src_is_channel and gather is not None
    ):  # a gathered source: each pair from its owner's slot (tiled)
        tp, rows, tile, prev = gather
        tag = (
            k.tag - 1 if prev else k.tag
        )  # (the previous launch's: its tag and parity slot)
        xs = poll_rows_bf16(spec, peer.from_rank(k, src, 0, tag if prev else None), [row], k.tid, tag, part,
                            remap=lambda w: peer.tile_index(w, spec.row_width // 2, tp, rows, tile))  # fmt: skip
    elif src_is_channel:
        xs = poll_rows_bf16(spec, k.mb[src], [row], k.tid, k.tag, part)
    chunks = norm_quant(spec, xs, gw, k.ldsf, k.tid, k.lane, k.wave)[0]
    out_bf = k.mb[bf_out] if bf_out is not None else None
    put_row(
        spec,
        k.mb[f"{out}.q"],
        k.mb[f"{out}.s"],
        out_bf,
        row,
        k.tid,
        k.tag,
        chunks,
        part,
        flag,
        bcopy,
    )


class NormQuant(Block):
    """Zero-centred RMSNorm -> MXFP8 channel ``out`` (+ optional BF16 channel ``bf16_out``); with
    ``NormSpec(plain=True)`` plain-weight RMSNorm (``plain_once=True``: rounded once), with
    ``NormSpec(identity=True)`` no norm (``gamma=None``).

    ``src``: a plain bf16 [rows, width] arg name (e.g. the layer input ``h_in``) or
    a BF16 ``Channel`` (e.g. the post-attention residual). ``gamma``: the bf16
    [width] arg name; ``weight``: its ``layer.retained`` name (``pack``).
    ``norm``: the geometry (width, eps), unless ``spec`` gives it.
    ``strategy``: "stage" (own stage, ``rows * spec.split`` tasks) or "prologue"
    (no stage: a consumer calls ``norm_quant`` itself; ``spec`` is its contract).
    ``spec``: the device mapping (default: ``default_spec`` for the source kind).
    """

    def __init__(
        self,
        name: str,
        src: str | Channel,
        out: Channel,
        gamma: str | None,
        weight: str = "input_layernorm",
        bf16_out: Channel | None = None,
        spec: NormSpec | None = None,
        strategy: str = "stage",
        norm: NormGeometry | None = None,
        src_gathered: int = 0,
        src_prev: bool = False,
        bf16_col: int | None = None,
        bf16_writers: tuple[str, ...] = (),
        src_partials: tuple | None = None,
    ):
        if spec is None and norm is None:
            raise ValueError(
                f"NormQuant {name}: give the norm geometry (norm=) or a spec"
            )
        if (
            spec is not None
            and norm is not None
            and (spec.width, spec.eps) != (norm.width, norm.eps)
        ):
            raise ValueError(f"NormQuant {name}: spec {spec} does not match {norm}")
        spec = spec or default_spec(isinstance(src, Channel), strategy, norm)
        if isinstance(src, Channel) and (
            src.fmt is not Fmt.BF16 or src.width != spec.row_width
        ):
            raise ValueError(
                f"NormQuant reads a bf16 arg or a BF16 channel of width {spec.row_width}"
            )
        if (
            isinstance(src, Channel)
            and src.flagged
            or bf16_out is not None
            and bf16_out.flagged
        ):
            raise ValueError(
                "NormQuant's BF16 source / copy are tagged channels (bulk + flag not implemented)"
            )
        if out.fmt is not Fmt.MXFP8 or out.width != spec.row_width:
            raise ValueError(f"out must be an MXFP8 channel of width {spec.row_width}")
        # bf16_col: the copy is columns bf16_col .. + width of a wider BF16 channel's rows (e.g. a concatenation of
        # two norms' rows); None: the copy is the whole row
        if bf16_col is not None and (bf16_out is None or spec.row_width != spec.width or spec.copy_pitch
                                     or bf16_col % 8 or bf16_col + spec.width > bf16_out.width):  # fmt: skip
            raise ValueError(
                f"NormQuant {name}: bf16_col {bf16_col} places the unpadded copy inside bf16_out's rows"
            )
        self.bf16_col = bf16_col
        # (bf16_col) the wide channel's writers, when this block declares its regions (one of the slices' blocks does)
        self.bf16_writers = tuple(bf16_writers)
        wide = bf16_col is not None
        if bf16_out is not None and (
            bf16_out.fmt is not Fmt.BF16
            or (not wide and bf16_out.width != spec.copy_width)
        ):
            raise ValueError(
                f"bf16_out must be a BF16 channel of width {spec.copy_width}"
            )
        if (gamma is None) != spec.identity:
            raise ValueError(
                "NormQuant: a gamma arg exactly when it normalizes (spec.identity: none)"
            )
        if strategy not in ("stage", "prologue") or (
            strategy == "prologue" and spec.split != 1
        ):
            raise ValueError(f"strategy {strategy} with split {spec.split}")
        self.name, self.src, self.out, self.gamma, self.weight = (
            name,
            src,
            out,
            gamma,
            weight,
        )
        self.bf16_out, self.spec, self.strategy = bf16_out, spec, strategy
        # TP, a sharded residual: ``src`` is a gathered PEER region (the owners' column slices, tiles of
        # ``src_gathered`` pairs: ``peer.tile_index``; ``HCSeam(residual="sharded")``'s LIN); 0: this rank's channel
        if src_gathered and (not isinstance(src, Channel) or strategy != "stage"):
            raise ValueError(
                f"NormQuant {name}: src_gathered reads a gathered BF16 channel in a stage"
            )
        if src_prev and not src_gathered:
            raise ValueError(
                f"NormQuant {name}: src_prev reads a gathered source the previous launch put"
            )
        self.src_gathered = src_gathered
        self.src_prev = src_prev  # ... put by the previous launch (``Stage.prev_reads``: its tag and parity slot)
        # TP: the source's values summed from a push's per-rank fp32 partials (region, fmt, tp), e.g. the MoE's with
        # ``MoEExperts(dsum=False)``: the sum stage's hop gone (each row's task sums its own elements); ``src`` names
        # the bf16 channel those sums replace (its width, never polled)
        if src_partials is not None and (not isinstance(src, Channel) or strategy != "stage" or src_gathered
                                         or spec.row_width != spec.width):  # fmt: skip
            raise ValueError(
                f"NormQuant {name}: src_partials replaces an unpadded BF16 channel source in a stage"
            )
        self.src_partials = tuple(src_partials) if src_partials is not None else None

    def stages(self, sc: Scenario) -> list[Stage]:
        is_ch = isinstance(self.src, Channel)
        src = self.src.name if is_ch else self.src
        bf = self.bf16_out.name if self.bf16_out is not None else None
        writes = self.out.regions(
            sc.rows, (self.name,), flags=sc.rows * self.spec.split
        )
        if self.bf16_out is not None and self.bf16_col is None:
            writes += self.bf16_out.regions(sc.rows, (self.name,))
        emit = partial(
            _emit_stage,
            spec=self.spec,
            src=src,
            src_is_channel=is_ch,
            gamma=self.gamma,
            out=self.out.name,
            bf_out=bf,
            proto=self.out.proto.value,
            **(
                {"gather": (sc.tp, sc.rows, self.src_gathered, self.src_prev)}
                if self.src_gathered
                else {}
            ),
            **(
                {"bcopy": (self.bf16_out.width, self.bf16_col)}
                if self.bf16_col is not None
                else {}
            ),
            **(
                {"partials": self.src_partials} if self.src_partials is not None else {}
            ),
        )
        return [
            Stage(
                self.name,
                sc.rows * self.spec.split,
                emit,
                lds_words=self.spec.lds_words(1),
                writes=writes,
                args=tuple(
                    a
                    for a in ((self.gamma,) if is_ch else (src, self.gamma))
                    if a is not None
                ),
                key=(
                    "norm_quant",
                    self.spec,
                    src,
                    is_ch,
                    self.gamma,
                    self.out.name,
                    bf,
                    self.out.proto.value,
                )
                + (
                    (("gathered", sc.tp, self.src_gathered, self.src_prev),)
                    if self.src_gathered
                    else ()
                )
                + (
                    (("bcopy", self.bf16_out.width, self.bf16_col),)
                    if self.bf16_col is not None
                    else ()
                )
                + (
                    (("partials",) + self.src_partials,)
                    if self.src_partials is not None
                    else ()
                ),
                prev_reads=(src,) if self.src_prev else (),
                row_tasks=self.spec.split,
            )
        ]

    def pack(self, layer) -> dict[str, torch.Tensor]:
        return (
            {}
            if self.gamma is None
            else {self.gamma: layer.retained[self.weight].contiguous()}
        )
