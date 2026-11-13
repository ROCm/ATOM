# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Top-k selection on one CTA, and in two levels across CTAs (gfx950).

The ``k`` largest of ``n`` keys, ties to the lower index, output in rank order. Exact: no
sampling, no approximation; the order is total, so the result is unique.

**Keys.** A key is a 32-bit *order key* ``u``, compared unsigned, larger first. Floats
become order keys with :func:`fold_f32`; the policy (torch's ``sort`` / ``topk``):

- NaN is larger than every number, all NaNs equal (canonical ``0xFFFFFFFF``);
- ``-0.0`` equals ``+0.0``;
- ``-inf`` is an ordinary value, below every finite one;
- ``u = 0`` means **no key** (padding): never selected. No float folds to it (``-inf``
  folds to ``0x007FFFFF``), so integer keys must avoid it too.

The order is ``(u desc, index asc)``: equal keys go to the lower index. Internally a key is
the unique 64-bit ``(u, v)``, ``v = 2^B - 1 - index`` (``B = index_bits`` rounded up to a
byte), so the order has no ties at all.

**Layout.** Each thread of the CTA holds ``E`` keys in registers (Python lists of Int32
``us`` and their ``idxs``), in any arrangement (e.g. strided, ``index = base + tid + THREADS
e``, so a wave's loads coalesce).

**One CTA** (``n <= THREADS * E``), inside any stage task:

1. :func:`select` -- the ``min(k, valid)`` best keys go to LDS slots ``0 .. n_out``
   (unordered; returns ``n_out``, uniform). A radix select finds the k-th best key ``T``
   with 8-bit passes over ``(u, v)``, most significant first: four over ``u``, then
   ``B / 8`` over ``v`` (reached only when keys equal to the k-th best's ``u`` are split
   by it). A pass counts the keys matching the prefix so far into a 256-bin LDS histogram
   (``ds_add``; pass 0 into 8 copies, lane ``l`` adding to copy ``l % 8``: its digits are
   the floats' sign and exponent, so few bins take most keys, and lanes adding to one LDS
   word serialize). Every wave then scans the histogram itself (no second barrier): lane
   ``l`` holds bins ``255 - 4l ..`` and a wave prefix sum finds the digit holding the
   ``remain``-th key. A pass whose digit's keys are all taken ends the search early (its
   lower bits zero: every key with the prefix is kept). The kept keys (``>= T``) take
   their slots by one CTA prefix sum.
2. :func:`rank` -- the ``n_out`` slots sorted by ``(u, v)`` descending: thread ``tid``
   ends with the key of rank ``tid`` (a bitonic sort of the 64-bit keys, one per thread,
   over the next power of two ``>= k``; in-wave exchanges by DPP and permlane swaps,
   across waves through LDS with one barrier each).

Thread ``tid < n_out`` can instead read slot ``tid`` unsorted with :func:`slot` (e.g. to
publish a chunk's candidates).

**Two levels** across CTAs (``n`` up to ``chunks * THREADS * E``): chunk task ``c``
selects the best ``k`` of its indices and publishes its slots as candidates ``(u,
index)`` (``u`` 0 past its ``n_out``); a final task polls every chunk's ``k`` candidates
and runs :func:`select` + :func:`rank` on them. The best ``k`` of all keys are among their
chunk's best ``k`` (the order is total), so the result is exact. The Indexer block
(``blocks.indexer``) is the user.

LDS: :func:`lds_words` int32 words from the caller's base (``Lds`` offsets), ``k`` up to ``THREADS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from atom.models.qwen38.mono.kernels.device.dpp_utils import update_dpp_i32
from atom.models.qwen38.mono.kernels.device.ops import (
    fresh,
    fresh_s,
    permlane_swap,
    traced,
    uniform,
    wave_max,
    xshfl,
)
from atom.models.qwen38.mono.kernels.target import MI355X

THREADS = MI355X.threads
WAVES = MI355X.waves
WAVE = MI355X.wave
INT_MIN = -(2**31)
BINS = 256
COPIES = 8  # pass 0's histogram copies (lane l counts into copy l % COPIES)
# copies 4 words apart in bank: distinct banks for the 16-byte scans
H0_STRIDE = BINS + 4


def _digits(index_bits: int) -> int:
    return -(-index_bits // 8)


@dataclass(frozen=True)
class Lds:
    """Word offsets (from the caller's LDS base) of the select / rank scratch for ``k``."""

    k: int
    index_bits: int = 16

    @property
    def passes(self) -> int:
        return 4 + _digits(self.index_bits)

    @property
    def h0(self) -> int:  # pass 0: COPIES histograms
        return 0

    @property
    def hist(self) -> int:  # passes 1 ..: one histogram each
        return COPIES * H0_STRIDE

    @property
    def cnt(self) -> int:  # the waves' prefix-sum totals
        return self.hist + (self.passes - 1) * BINS

    @property
    def slots(self) -> int:  # the kept keys: 64-bit (lo = v, hi = u ^ INT_MIN) each
        return self.cnt + 2 * WAVES

    @property
    def xchg(
        self,
    ) -> (
        int
    ):  # rank's cross-wave exchanges (two buffers of THREADS keys), over the histograms
        return 0

    @property
    def junk(self) -> int:  # one word per lane: the uncounted keys' adds
        return self.slots + 2 * self.k

    @property
    def words(self) -> int:
        return self.junk + WAVE


def lds_words(k: int, index_bits: int = 16) -> int:
    """LDS words :func:`select` and :func:`rank` use for ``k`` (1 <= k <= THREADS) over indices below
    ``2^index_bits`` (1 .. 32)."""
    if not 1 <= k <= THREADS:
        raise ValueError(f"top-k supports 1 <= k <= {THREADS} (ranked), got {k}")
    if not 1 <= index_bits <= 32:
        raise ValueError(f"index_bits {index_bits} outside 1 .. 32")
    lay = Lds(k, index_bits)
    assert 2 * 2 * THREADS <= lay.cnt  # rank's exchange buffers fit over the histograms
    return lay.words


def _i32(m: int):
    m &= 0xFFFFFFFF
    return fx.Int32(m - (1 << 32) if m >= 1 << 31 else m)


def _vmask(index_bits: int) -> int:
    """``2^B - 1``: ``v = mask - index`` (B the index bits rounded up to a byte)."""
    return (1 << (8 * _digits(index_bits))) - 1


# ------------------------------------------------------------------ keys


def fold_f32(x):
    """The order key of an f32 (see the module docstring): NaN largest, ``-0.0 == +0.0``,
    never 0."""
    b = fx.Float32(x).bitcast(fx.Int32)
    b = (b == fx.Int32(INT_MIN)).select(fx.Int32(0), b)
    u = (b < 0).select(b ^ fx.Int32(-1), b | fx.Int32(INT_MIN))
    return (fx.Float32(x) != fx.Float32(x)).select(fx.Int32(-1), u)


def _above(a, b):
    """``a > b`` as unsigned 32-bit keys."""
    return (a ^ fx.Int32(INT_MIN)) > (b ^ fx.Int32(INT_MIN))


# ------------------------------------------------------------------ wave / CTA prefix sums


def wave_prefix(v):
    """Inclusive prefix sum of Int32 ``v`` over the wave's lanes (lane 0 first): DPP row
    shifts within 16-lane rows, then the row broadcasts 15 and 31."""
    v = fx.Int32(v)
    for n in (1, 2, 4, 8):
        v = v + fx.Int32(update_dpp_i32(fx.Int32(0), v, 0x110 + n, 0xF, 0xF, False))
    v = v + fx.Int32(update_dpp_i32(fx.Int32(0), v, 0x142, 0xA, 0xF, False))
    return v + fx.Int32(update_dpp_i32(fx.Int32(0), v, 0x143, 0xC, 0xF, False))


@traced
def cta_prefix(v, lane, wave, cnt):
    """(exclusive prefix of ``v`` in thread order, CTA total), every thread; ``cnt``: WAVES
    words of LDS. Ends with the values readable: a barrier inside, none needed after (the
    caller must not reuse ``cnt`` before another barrier)."""
    incl = wave_prefix(v)
    if lane == WAVE - 1:
        fx.ptr_store(incl, cnt + wave)
    gpu.barrier()
    before = fx.Int32(0)
    total = fx.Int32(0)
    for w in range_constexpr(WAVES):
        c = fx.ptr_load(cnt + w)
        before = before + (wave > w).select(c, fx.Int32(0))
        total = total + c
    return before + incl - fx.Int32(v), total


# ------------------------------------------------------------------ select


@traced
def _clear(lds, tid, lay):
    for i in range_constexpr(-(-lay.cnt // THREADS)):
        if tid + i * THREADS < lay.cnt:
            fx.ptr_store(fx.Int32(0), lds + (tid + i * THREADS))


def _count(lds, hoff, us, vs, lane, prefix, mask, which, shift, copies, junk):
    """Count the keys (u != 0) matching ``prefix`` under ``mask`` (two words) by their digit at
    ``shift`` of word ``which`` into the histogram at word ``hoff`` (``copies`` > 1: lane l's
    copy), then the barrier. Branch-free: a key that does not count adds to its lane's junk
    word instead (no exec-mask branch per key, and no two lanes share a junk word)."""
    if copies > 1:
        hoff = hoff + (lane % copies) * H0_STRIDE
    for e in range(len(us)):
        u = fresh(us[e])  # per pass: the masks are not held across passes
        v = vs[e]
        m = u != 0
        if mask[0] != 0:
            m = m & ((u & _i32(mask[0])) == prefix[0])
        if mask[1] != 0:
            v = fresh(v)
            m = m & ((v & _i32(mask[1])) == prefix[1])
        word = u if which == 0 else v
        at = m.select(hoff + ((word >> shift) & 255), junk + lane)
        fx.atomic_add(lds + at, fx.Int32(1), syncscope="workgroup")
    gpu.barrier()


def _lds4(ptr, off):
    v = fx.Vector(fx.ptr_load(ptr + off, result_type=fx.Vector.make_type(4, fx.Int32)))
    return [v[e] for e in range(4)]


def _digit(h, lane, copies, remain):
    """From histogram ``h`` (the sum of its ``copies``): the digit holding the
    ``remain``-th largest counted key, as one uniform word ``hit << 25 | whole << 24 |
    digit << 16 | remain'`` (``remain'`` the count still to take within the digit,
    ``whole`` 1 if that is all of its keys; 0: fewer than ``remain`` keys were counted).
    """
    v = _lds4(h, 252 - 4 * lane)
    for c in range(1, copies):
        w = _lds4(h + c * H0_STRIDE, 252 - 4 * lane)
        v = [a + b for a, b in zip(v, w)]
    cnt = [v[3], v[2], v[1], v[0]]  # digits 255 - 4l, 254 - 4l, 253 - 4l, 252 - 4l
    tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
    above = wave_prefix(tot) - tot
    pick = fx.Int32(0)
    for j in range(4):
        here = (above < remain) & (above + cnt[j] >= remain)
        whole = (above + cnt[j] == remain).select(fx.Int32(1 << 24), fx.Int32(0))
        word = (
            fx.Int32(1 << 25) | whole | ((255 - 4 * lane - j) << 16) | (remain - above)
        )
        pick = here.select(word, pick)
        above = above + cnt[j]
    return uniform(wave_max(pick))  # one lane at most holds the digit


def _passes(index_bits: int):
    """The radix passes: (word, shift, (mask of u, mask of v) of the bits above the digit)."""
    out, mask = [], [0, 0]
    for which, top, low in ((0, 32, 0), (1, 8 * _digits(index_bits), 0)):
        for shift in range(top - 8, low - 1, -8):
            out.append((which, shift, tuple(mask)))
            mask[which] |= 0xFF << shift
    return out


@traced
def _threshold(lds, us, vs, lane, k, lay):
    """The k-th best key ``(Tu, Tv)`` (lower bits zero after a whole digit: every key with
    its prefix is kept), or (0, 0) when at most ``k`` keys are valid (all kept); and the keys
    still to take with its prefix."""
    p0, p1 = fx.Int32(0), fx.Int32(0)
    remain = fx.Int32(k)
    live = fx.Int32(1)
    passes = _passes(lay.index_bits)
    for p in range_constexpr(len(passes)):
        which, shift, mask = passes[p]
        hoff = lay.h0 if p == 0 else lay.hist + BINS * (p - 1)
        h = lds + hoff
        copies = COPIES if p == 0 else 1
        if live != 0:
            _count(
                lds, hoff, us, vs, lane, [p0, p1], mask, which, shift, copies, lay.junk
            )
            got = _digit(h, lane, copies, remain)
            if const_expr(which == 0):
                p0 = p0 | (((got >> 16) & 255) << shift)
            else:
                p1 = p1 | (((got >> 16) & 255) << shift)
            remain = got & 0xFFFF
            # none hit (pass 0 only: fewer than k valid keys): T = 0 keeps them all
            live = ((got >> 25) & 1) & (((got >> 24) & 1) ^ 1)
    return p0, p1, remain


def _vs(idxs, index_bits):
    vmask = _i32(_vmask(index_bits))
    return [vmask - fx.Int32(i) for i in idxs]


def select(lds, tid, lane, wave, us, idxs, k, index_bits=16):
    """The best ``min(k, keys)`` of the CTA's keys (``us`` / ``idxs``: this thread's; indices
    below ``2^index_bits``) to LDS slots ``0 .. n_out`` (unordered) as 64-bit sort keys.
    Returns ``n_out`` (uniform). Every thread of the CTA calls it; ``lds`` is the caller's
    base (:func:`lds_words`). Ends with a barrier (slots readable)."""
    return _select(lds, tid, lane, wave, us, idxs, k, index_bits)


@traced
def _select(lds, tid, lane, wave, us, idxs, k, index_bits):
    lay = Lds(k, index_bits)
    vs = _vs(idxs, index_bits)
    # opaque copies: lane masks built on tid / lane / wave stay inside the call instead of
    # being hoisted out of the caller's task loop and held in SGPRs (spills)
    tid, lane, wave = fresh(tid), fresh(lane), fresh_s(wave)
    _clear(lds, tid, lay)
    gpu.barrier()
    tu, tv, _ = _threshold(lds, us, vs, lane, k, lay)
    kept = fx.Int32(0)
    for e in range_constexpr(len(us)):
        kept = kept + _kept(us[e], vs[e], tu, tv).select(fx.Int32(1), fx.Int32(0))
    slot, total = cta_prefix(kept, lane, wave, lds + lay.cnt)
    for e in range_constexpr(len(us)):
        # recompute the masks here rather than hold the counting loop's
        u = fresh(us[e])
        v = vs[e]
        keep = _kept(u, v, tu, tv)
        if keep:
            at = lds + (lay.slots + 2 * slot)
            fx.ptr_store(
                fx.Vector.from_elements([v, u ^ fx.Int32(INT_MIN)], fx.Int32), at
            )
        slot = slot + keep.select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    return total


def _kept(u, v, tu, tv):
    """``(u, v) >= (tu, tv)`` as unsigned words, and a key at all."""
    ok = u != 0
    return ok & (
        _above(u, tu)
        | ((u == tu) & ((v ^ fx.Int32(INT_MIN)) >= (tv ^ fx.Int32(INT_MIN))))
    )


def _key(words):
    return fx.Vector.from_elements(list(words), fx.Int32).bitcast(fx.Int64)[0]


def _split(key):
    w = fx.Vector.from_elements([key], fx.Int64).bitcast(fx.Int32)
    return [w[0], w[1]]


def slot(lds, tid, n_out, k, index_bits=16):
    """Slot ``tid`` (any of ``0 .. k - 1``) unsorted: (u, index), (0, -1) past ``n_out``."""
    lay = Lds(k, index_bits)
    v = fx.Vector(
        fx.ptr_load(
            lds + (lay.slots + 2 * fx.min(tid, k - 1)),
            result_type=fx.Vector.make_type(2, fx.Int32),
        )
    )
    have = tid < n_out
    idx = _i32(_vmask(index_bits)) - v[0]
    return have.select(v[1] ^ fx.Int32(INT_MIN), fx.Int32(0)), have.select(
        idx, fx.Int32(-1)
    )


# ------------------------------------------------------------------ rank


def _xchg_wave(key, lane, stride):
    """Lane ``lane ^ stride``'s 64-bit key (DPP below 16, permlane swaps at 16 / 32)."""
    words = _split(key)
    if stride >= 16:
        upper = (lane & stride) != 0
        got = []
        for w in words:
            a, b = permlane_swap(stride, w, w)
            got.append(upper.select(a, b))
        return _key(got)
    return _key([xshfl(w, stride) for w in words])


def wave_sort(key, lane):
    """The wave's 64-bit keys (one a lane) sorted descending, as signed Int64: lane ``l``
    ends with the ``l``-th largest. A bitonic network in registers (21 steps, no LDS).
    """
    size = 2
    while size <= WAVE:
        stride = size // 2
        while stride:
            other = _xchg_wave(key, lane, stride)
            # blocks with (lane & size) == 0 sort descending (all of them at size 64); the lower
            # lane of a pair keeps the larger key there. The lane masks come from an opaque copy
            # of the lane: else LLVM hoists every step's masks out of loops into SGPRs.
            me = fresh(lane)
            keep_big = ((me & size) == 0) == ((me & stride) == 0)
            key = ((other > key) == keep_big).select(other, key)
            stride //= 2
        size *= 2
    return key


def _greater(lds, base, key):
    """How many of the 64 keys sorted descending at LDS word ``base`` exceed ``key``."""
    pos = fx.Int32(0)
    for step in (32, 16, 8, 4, 2, 1, 1):  # binary lifting to 63, then the 64th
        at = pos + step - 1
        v = fx.Vector(
            fx.ptr_load(
                lds + (base + 2 * at), result_type=fx.Vector.make_type(2, fx.Int32)
            )
        )
        pos = pos + (_key([v[0], v[1]]) > key).select(fx.Int32(step), fx.Int32(0))
    return pos


def rank(lds, tid, lane, wave, n_out, k, index_bits=16):
    """Rank the ``n_out`` slots of :func:`select` by (u desc, index asc): returns
    ``(u, index, r)``, a kept key held by this thread and its rank ``r`` (``(0, -1, -1)``
    for a thread holding none). Every thread of the CTA calls it.

    Each wave sorts 64 slots in registers (:func:`wave_sort`); then the sorted waves go to
    LDS and a key's rank is its place in its wave plus, for every other wave, a 6-step binary
    search for the keys there above it. Ends with a barrier, so the caller may reuse the LDS
    (``u = 0`` is no key). Not traced: the network unrolls at trace time."""
    assert k <= THREADS, k
    lay = Lds(k, index_bits)
    tid, lane, wave = fresh(tid), fresh(lane), fresh_s(wave)
    have = tid < n_out
    v = fx.Vector(
        fx.ptr_load(
            lds + (lay.slots + 2 * fx.min(tid, k - 1)),
            result_type=fx.Vector.make_type(2, fx.Int32),
        )
    )
    key = _key([have.select(v[0], fx.Int32(0)), have.select(v[1], fx.Int32(INT_MIN))])
    waves = -(-k // WAVE)
    if waves < WAVES:
        raise ValueError(
            f"topk: {k} slots fill {waves} of {WAVES} waves (every wave holds slots)"
        )
    key = wave_sort(key, lane)
    r = lane
    w = _split(key)
    fx.ptr_store(
        fx.Vector.from_elements(w, fx.Int32),
        lds + (lay.xchg + 2 * (wave * WAVE + lane)),
    )
    gpu.barrier()
    for o in range(waves):
        r = r + (wave != o).select(
            _greater(lds, lay.xchg + 2 * WAVE * o, key), fx.Int32(0)
        )
    gpu.barrier()
    w = _split(key)
    real = w[1] != fx.Int32(INT_MIN)  # u != 0
    idx = _i32(_vmask(index_bits)) - w[0]
    return (
        real.select(w[1] ^ fx.Int32(INT_MIN), fx.Int32(0)),
        real.select(idx, fx.Int32(-1)),
        real.select(r, fx.Int32(-1)),
    )
