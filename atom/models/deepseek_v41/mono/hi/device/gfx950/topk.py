# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Top-k selection on one CTA, and in two levels across CTAs (gfx950).

The ``k`` largest of ``n`` keys, ties to the lower index, output in rank order. Exact: no
sampling, no approximation; the order is total, so the result is unique.

**Keys.** A key is a 32-bit *order key* ``u``, compared unsigned, larger first. Floats
become order keys with :func:`order_key` (aiter's rule, below):

- NaNs by their bits: positive ones above ``+inf``, negative ones below ``-inf``;
- ``-0.0`` below ``+0.0``;
- ``-inf`` is an ordinary value, below every finite one;
- ``u = 0`` means **no key** (padding): never selected, unless :func:`select` takes
  ``valid``. The float bits ``0xFFFFFFFF`` (a NaN) map to it, so float keys take ``valid``
  (every u32 a key); integer keys without ``valid`` must avoid it.

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
(``blocks.gfx950.attention.dsa_indexer``) is the worked example.

**Other uses.**

- MoE routing (top 10 of 512 logits): one CTA, ``E = 1`` (thread ``t`` holds expert
  ``t``), ``k = 10``, ``index_bits = 9``; :func:`rank` then sorts 16 keys in wave 0.
  :func:`key_bits` gives the selected logits' fp32 bits back (bit-exact).
- Sampling (top-k / top-p over a vocabulary of ~150 K): two levels, chunk tasks over
  vocabulary slices, then softmax over the final ``k`` in rank order.

**aiter's rule** (V4.1: ``blocks.gfx950.topk.TopK``, a merge tree of
lists ascending by index; one implementation since the top-k fold):

- keys by :func:`order_key` (``-0.0`` below ``+0.0``, NaNs by their bits): every u32 is a
  key, ``0`` (bits ``0xFFFFFFFF``) included, so :func:`select` takes ``valid`` (a key counts
  where it holds; ``v = 0`` marks the others, so indices stay below ``2^B - 1``);
- ``u_bits = 16`` for bf16 values (keys' low half zero): two passes over ``u``;
- thread ``tid`` holds ``E`` consecutive indices from ``tid E``: the slots then come out in
  index order, read by :func:`slot` (no :func:`rank`), and ``k`` may exceed ``THREADS``
  (the indexer's top 2048);
- :func:`wave_topk_won` / :func:`wave_topk_won_full`: one wave, small ``n`` and ``k`` (the
  MoE router's fused top-k).

LDS: :func:`lds_words` int32 words from the caller's base (``Lds`` offsets). ``k`` up to
``THREADS`` with :func:`rank`; any ``k`` with :func:`select` and :func:`slot` alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32

from ...target import MI355X
from .dpp_utils import update_dpp_i32
from .ops import (
    ballot,
    butterfly,
    fresh,
    fresh_s,
    permlane_swap,
    popcount,
    readlane,
    traced,
    uniform,
    wave_max,
    xshfl,
)

THREADS = MI355X.threads
WAVES = MI355X.waves
WAVE = MI355X.wave
INT_MIN = -(2**31)
BINS = 256
COPIES = 8  # pass 0's histogram copies (lane l counts into copy l % COPIES)
H0_STRIDE = (
    BINS + 4
)  # copies 4 words apart in bank: distinct banks for the 16-byte scans


def _digits(index_bits: int) -> int:
    return -(-index_bits // 8)


@dataclass(frozen=True)
class Lds:
    """Word offsets (from the caller's LDS base) of the select / rank scratch for ``k``."""

    k: int
    index_bits: int = 16
    u_bits: int = 32
    in_order: bool = False  # ties by one prefix sum: no index passes

    @property
    def passes(self) -> int:
        return self.u_bits // 8 + (0 if self.in_order else _digits(self.index_bits))

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


def lds_words(
    k: int,
    index_bits: int = 16,
    u_bits: int = 32,
    ranked: bool = True,
    in_order: bool = False,
) -> int:
    """LDS words :func:`select` and :func:`rank` use for ``k`` (k <= THREADS; any k >= 1 when
    ``ranked`` is False: :func:`select` and :func:`slot` only) over indices below
    ``2^index_bits`` (1 .. 32) and keys of ``u_bits`` (8, 16, 24 or 32) top bits; ``in_order``:
    :func:`select`'s."""
    if not 1 <= k <= (THREADS if ranked else k):
        raise ValueError(f"top-k supports 1 <= k <= {THREADS} (ranked), got {k}")
    if not 1 <= index_bits <= 32:
        raise ValueError(f"index_bits {index_bits} outside 1 .. 32")
    if u_bits not in (8, 16, 24, 32):
        raise ValueError(f"u_bits {u_bits} not 8, 16, 24 or 32")
    lay = Lds(k, index_bits, u_bits, in_order)
    assert 2 * 2 * THREADS <= lay.cnt  # rank's exchange buffers fit over the histograms
    return lay.words


def _i32(m: int):
    m &= 0xFFFFFFFF
    return fx.Int32(m - (1 << 32) if m >= 1 << 31 else m)


def _vmask(index_bits: int) -> int:
    """``2^B - 1``: ``v = mask - index`` (B the index bits rounded up to a byte)."""
    return (1 << (8 * _digits(index_bits))) - 1


# ------------------------------------------------------------------ keys


def order_key(bits):
    """aiter's u32 order key of fp32 bits (``-0.0`` below ``+0.0``, NaNs by their bits: positive
    above ``+inf``, negative below ``-inf``), held in an Int32. Every value is a key (0 included):
    :func:`select` with ``valid``."""
    return (bits < 0).select(~bits, bits | INT_MIN)


def key_bits(key):
    """The fp32 bits :func:`order_key` came from."""
    return (key < 0).select(key ^ INT_MIN, ~key)


def ult(a, b):
    """a < b as u32."""
    return (a ^ INT_MIN) < (b ^ INT_MIN)


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


def _count(
    lds, hoff, us, vs, lane, prefix, mask, which, shift, copies, junk, valid_v=False
):
    """Count the keys (u != 0; ``valid_v``: v != 0) matching ``prefix`` under ``mask`` (two
    words) by their digit at ``shift`` of word ``which`` into the histogram at word ``hoff``
    (``copies`` > 1: lane l's copy), then the barrier. Branch-free: a key that does not count
    adds to its lane's junk word instead (no exec-mask branch per key, and no two lanes share
    a junk word)."""
    if copies > 1:
        hoff = hoff + (lane % copies) * H0_STRIDE
    for e in range(len(us)):
        u = fresh(us[e])  # per pass: the masks are not held across passes
        v = fresh(vs[e]) if valid_v else vs[e]
        m = (v if valid_v else u) != 0
        if mask[0] != 0:
            m = m & ((u & _i32(mask[0])) == prefix[0])
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


def _passes(index_bits: int, u_bits: int = 32, v: bool = True):
    """The radix passes: (word, shift, (mask of u, mask of v) of the bits above the digit)."""
    out, mask = [], [0, 0]
    for which, top, low in (
        (0, 32, 32 - u_bits),
        (1, 8 * _digits(index_bits) if v else 0, 0),
    ):
        for shift in range(top - 8, low - 1, -8):
            out.append((which, shift, tuple(mask)))
            mask[which] |= 0xFF << shift
    return out


@traced
def _threshold(lds, us, vs, lane, k, lay, valid_v=False):
    """The k-th best key ``(Tu, Tv)`` (lower bits zero after a whole digit: every key with
    its prefix is kept), or (0, 0) when at most ``k`` keys are valid (all kept); and the keys
    still to take with its prefix (``in_order``: k when at most k keys are valid)."""
    p0, p1 = fx.Int32(0), fx.Int32(0)
    remain = fx.Int32(k)
    live = fx.Int32(1)
    passes = _passes(lay.index_bits, lay.u_bits, not lay.in_order)
    for p in range_constexpr(len(passes)):
        which, shift, mask = passes[p]
        hoff = lay.h0 if p == 0 else lay.hist + BINS * (p - 1)
        h = lds + hoff
        copies = COPIES if p == 0 else 1
        if live != 0:
            _count(
                lds,
                hoff,
                us,
                vs,
                lane,
                [p0, p1],
                mask,
                which,
                shift,
                copies,
                lay.junk,
                valid_v,
            )
            got = _digit(h, lane, copies, remain)
            if const_expr(which == 0):
                p0 = p0 | (((got >> 16) & 255) << shift)
            if const_expr(lay.in_order):
                remain = (((got >> 25) & 1) != 0).select(got & 0xFFFF, remain)
            # none hit (pass 0 only: fewer than k valid keys): T = 0 keeps them all
            live = ((got >> 25) & 1) & (((got >> 24) & 1) ^ 1)
    return p0, p1, remain


def _vs(idxs, index_bits, valid):
    vmask = _i32(_vmask(index_bits))
    return [
        ok.select(vmask - fx.Int32(i), fx.Int32(0))
        for ok, i in zip(valid, idxs, strict=True)
    ]


def select(
    lds,
    tid,
    lane,
    wave,
    us,
    idxs,
    k,
    index_bits=16,
    u_bits=32,
    valid=None,
    in_order=False,
):
    """The best ``min(k, valid)`` of the CTA's keys (``us`` / ``idxs``: this thread's; indices
    below ``2^index_bits``) to LDS slots ``0 .. n_out`` (unordered) as 64-bit sort keys.
    Returns ``n_out`` (uniform). Every thread of the CTA calls it; ``lds`` is the caller's
    base (:func:`lds_words`). Ends with a barrier (slots readable).

    ``u_bits`` < 32: the keys' low ``32 - u_bits`` bits are zero (bf16 values: 16), so fewer
    passes. ``valid`` (a list of booleans like ``us``): every 32-bit ``u`` is a key (no
    padding value), a key counts where its ``valid`` is true; indices then below
    ``2^index_bits - 1`` (``v = 0`` marks the others). ``in_order`` (with ``valid``): the
    keys are in index order (thread ``t``'s before ``t + 1``'s, each thread's ascending), so
    keys equal to the k-th best's are told apart by the CTA prefix sum that places the slots
    (no passes over the index), and the slots come out in index order."""
    if in_order:
        assert (
            valid is not None and len(us) * THREADS < 1 << 16
        ), "in_order: valid, fewer than 2^16 keys"
        lay = Lds(k, index_bits, u_bits, True)
        return _select_in_order(
            lds, tid, lane, wave, us, _vs(idxs, index_bits, valid), k, lay
        )
    return _select(lds, tid, lane, wave, us, idxs, k, index_bits, u_bits, valid)


@traced
def _select(lds, tid, lane, wave, us, idxs, k, index_bits, u_bits, valid):
    lay = Lds(k, index_bits, u_bits)
    valid_v = valid is not None
    vs = _vs(idxs, index_bits, valid)
    # opaque copies: lane masks built on tid / lane / wave stay inside the call instead of
    # being hoisted out of the caller's task loop and held in SGPRs (spills)
    tid, lane, wave = fresh(tid), fresh(lane), fresh_s(wave)
    _clear(lds, tid, lay)
    gpu.barrier()
    tu, tv, _ = _threshold(lds, us, vs, lane, k, lay, valid_v)
    kept = fx.Int32(0)
    for e in range_constexpr(len(us)):
        kept = kept + _kept(us[e], vs[e], tu, tv, valid_v).select(
            fx.Int32(1), fx.Int32(0)
        )
    slot, total = cta_prefix(kept, lane, wave, lds + lay.cnt)
    for e in range_constexpr(len(us)):
        u = fresh(
            us[e]
        )  # recompute the masks here rather than hold the counting loop's
        v = _fresh_v(vs[e], valid_v)
        keep = _kept(u, v, tu, tv, valid_v)
        if keep:
            at = lds + (lay.slots + 2 * slot)
            fx.ptr_store(
                fx.Vector.from_elements([v, u ^ fx.Int32(INT_MIN)], fx.Int32), at
            )
        slot = slot + keep.select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    return total


def _gt_tie(u, v, tu):
    """1 for a valid key above ``tu``, 1 << 16 for one equal to it, else 0."""
    ok = v != 0
    gt = (ok & _above(u, tu)).select(fx.Int32(1), fx.Int32(0))
    return gt + (ok & (u == tu)).select(fx.Int32(1 << 16), fx.Int32(0))


@traced
def _select_in_order(lds, tid, lane, wave, us, vs, k, lay):
    """:func:`select` ``in_order``: the keys above the k-th best's ``u`` and, of those equal to
    it, the ``need`` lowest-indexed; a key's slot is its place among the kept ones in index
    order: (above before it) + min(equal before it, need), one prefix sum of the two counts
    (16-bit halves: fewer than 2^16 keys a CTA)."""
    tid, lane, wave = fresh(tid), fresh(lane), fresh_s(wave)
    _clear(lds, tid, lay)
    gpu.barrier()
    tu, _, need = _threshold(lds, us, vs, lane, k, lay, True)
    cnt = fx.Int32(0)
    for e in range_constexpr(len(us)):
        cnt = cnt + _gt_tie(us[e], vs[e], tu)
    before, total = cta_prefix(cnt, lane, wave, lds + lay.cnt)
    ab, tb = before & 0xFFFF, (before >> 16) & 0xFFFF
    for e in range_constexpr(len(us)):
        u, v = fresh(us[e]), fresh(
            vs[e]
        )  # the masks again here rather than held across the prefix sum
        ok = v != 0
        gt = ok & _above(u, tu)
        tie = ok & (u == tu)
        if gt | (tie & (tb < need)):
            at = lds + (lay.slots + 2 * (ab + fx.min(tb, need)))
            fx.ptr_store(
                fx.Vector.from_elements([v, u ^ fx.Int32(INT_MIN)], fx.Int32), at
            )
        ab = ab + gt.select(fx.Int32(1), fx.Int32(0))
        tb = tb + tie.select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    return (total & 0xFFFF) + fx.min((total >> 16) & 0xFFFF, need)


def _fresh_v(v, valid_v):
    """``valid_v``: an opaque copy of v (its compares are then not held from the counting loop)."""
    return fresh(v) if valid_v else v


def _kept(u, v, tu, tv, valid_v=False):
    """``(u, v) >= (tu, tv)`` as unsigned words, and a key at all."""
    ok = (v if valid_v else u) != 0
    return ok & (
        _above(u, tu)
        | ((u == tu) & ((v ^ fx.Int32(INT_MIN)) >= (tv ^ fx.Int32(INT_MIN))))
    )


def _key(words):
    return fx.Vector.from_elements(list(words), fx.Int32).bitcast(fx.Int64)[0]


def _split(key):
    w = fx.Vector.from_elements([key], fx.Int64).bitcast(fx.Int32)
    return [w[0], w[1]]


def slot(lds, tid, n_out, k, index_bits=16, u_bits=32, in_order=False):
    """Slot ``tid`` (any of ``0 .. k - 1``) unsorted: (u, index), (0, -1) past ``n_out``."""
    lay = Lds(k, index_bits, u_bits, in_order)
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


@traced
def _sort_some(key, lane, wave, waves):
    out = key
    if wave < waves:
        out = wave_sort(key, lane)
    return out


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


def rank(lds, tid, lane, wave, n_out, k, index_bits=16, u_bits=32):
    """Rank the ``n_out`` slots of :func:`select` by (u desc, index asc): returns
    ``(u, index, r)``, a kept key held by this thread and its rank ``r`` (``(0, -1, -1)``
    for a thread holding none). Every thread of the CTA calls it.

    Each wave sorts 64 slots in registers (:func:`wave_sort`); then (``k > 64``) the sorted
    waves go to LDS and a key's rank is its place in its wave plus, for every other wave, a
    6-step binary search for the keys there above it. Ends with a barrier (``k > 64``), so
    the caller may reuse the LDS. Keys from a ``select`` without ``valid`` (``u = 0`` is no
    key). Not traced: the network unrolls at trace time."""
    assert k <= THREADS, k
    lay = Lds(k, index_bits, u_bits)
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
    if waves < WAVES:  # the waves holding slots sort them (the others hold no key)
        key = _sort_some(key, lane, wave, waves)
    else:
        key = wave_sort(key, lane)
    r = lane
    if waves > 1:
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


# ------------------------------------------------------------------ one wave a row (small n and k; the MoE router)

WAVE_MAX_K = 32


def wave_topk_won(n, k, vals, lane):
    """Top ``k`` of ``n`` <= 64 x len(vals) values in one wave, aiter's rule (:func:`order_key`, ties to
    the lowest index): lane l holds ``vals`` (fp32 bits as Int32) of indices l x per .. + per - 1
    (values at indices >= n are ignored). ``k`` rounds: the wave max of each lane's best remaining
    order key, then the lowest index holding it. Returns the lane's winners as a bitmask (bit d:
    index lane x per + d): one Int32, so ``per`` <= 32."""
    per = len(vals)
    assert per <= 32, f"wave_topk_won: {per} values a lane, the winners' mask holds 32"
    # u32 order keys as signed order (key ^ INT_MIN): signed max / compares
    sk = [order_key(v) ^ INT_MIN for v in vals]
    won = Int32(0)
    big = Int32(2**31 - 1)
    for it in range_constexpr(k):
        bk, bi, bav = Int32(INT_MIN), big, Int32(0)
        for d in range_constexpr(per):
            i = lane * per + d
            av = (i < n) & (((won >> d) & 1) == 0)
            better = av & (
                (bav == 0) | (sk[d] > bk)
            )  # strict: the lowest index among equal keys
            bk = better.select(sk[d], bk)
            bi = better.select(i, bi)
            bav = better.select(1, bav)
        kmax = butterfly(
            (bav != 0).select(bk, Int32(INT_MIN)), (32, 16, 8, 4, 2, 1), fx.max
        )
        # the lowest index holding kmax: lanes hold ascending index ranges and each lane's bi is its lowest, so the
        # first lane in the ballot has it (one ballot + readlane, not a second 6-step butterfly)
        m = ballot((bav != 0) & (bk == kmax))
        first = fx.Int32(popcount((m & (fx.Int64(0) - m)) - fx.Int64(1)))
        win = (m != fx.Int64(0)).select(readlane(bi, fx.min(first, Int32(63))), big)
        mine = (win != big) & (win // per == lane)
        won = mine.select(won | (Int32(1) << ((win - lane * per) & 31)), won)
    return won


def wave_topk_won_full(k, vals, lane):
    """``wave_topk_won`` when all 64 x len(vals) values are valid and ``k`` <= len(vals): the same winners.
    Each lane sorts its keys once (descending; equal keys keep index order: an odd-even transposition sort that
    swaps strictly out-of-order neighbours only), so a round reads the lanes' heads instead of rescanning their
    values, and the winning lane pops its head (only the entries a later round can still reach).
    """
    per = len(vals)
    assert k <= per, (k, per)
    hk = [order_key(v) ^ INT_MIN for v in vals]  # u32 order keys as signed order
    hd = [Int32(d) for d in range(per)]
    for p in range_constexpr(per):
        for j in range_constexpr(p % 2, per - 1, 2):
            sw = hk[j + 1] > hk[j]
            hk[j], hk[j + 1] = sw.select(hk[j + 1], hk[j]), sw.select(hk[j], hk[j + 1])
            hd[j], hd[j + 1] = sw.select(hd[j + 1], hd[j]), sw.select(hd[j], hd[j + 1])
    won = Int32(0)
    for it in range_constexpr(k):
        kmax = butterfly(hk[0], (32, 16, 8, 4, 2, 1), fx.max)
        # lanes hold ascending index ranges and a lane's head is its lowest index holding its key: the first lane
        # in the ballot holds the round's winner
        m = ballot(hk[0] == kmax)
        first = fx.min(
            fx.Int32(popcount((m & (fx.Int64(0) - m)) - fx.Int64(1))), Int32(63)
        )
        mine = lane == first
        won = mine.select(won | (Int32(1) << (hd[0] & 31)), won)
        for j in range_constexpr(k - 1 - it):
            hk[j] = mine.select(hk[j + 1], hk[j])
            hd[j] = mine.select(hd[j + 1], hd[j])
    return won


THRESH_WORDS = (
    4 * WAVE + 4
)  # wave_topk_won_thresh's LDS: maxima, candidate pairs, won words, a junk pair (x4)


def wave_topk_won_thresh(k, vals, lane, lds, at):
    """``wave_topk_won``'s winners (all 64 x len(vals) values valid, k <= 64) without its k serial wave rounds (nor
    ``wave_topk_won_full``'s sort). A bound ``t``: the highest lane maximum that at least k lane maxima reach (k
    lanes hold a key >= t, so every winner is >= t). The candidates (keys >= t) go to LDS in index order, one a lane,
    as the 64-bit ``(key, 2^16 - 1 - index)`` (no ties); each one's rank is the count of candidates above it, the
    winners those of rank < k. More than 64 candidates (many equal keys): ``wave_topk_won_full`` / ``wave_topk_won``.
    ``lds``: an Int32 LDS pointer, ``THRESH_WORDS`` from ``at`` (a multiple of 4: 16-byte reads); one wave (its LDS
    writes land before its reads). A caller with invalid values passes them as bits -1 (order key 0, below every
    value) and masks the winners itself."""
    per = len(vals)
    assert per * WAVE <= 1 << 16 and k <= WAVE, (per, k)
    sk = [order_key(v) ^ INT_MIN for v in vals]  # u32 order keys as signed order
    m = sk[0]
    for d in range_constexpr(1, per):
        m = fx.max(m, sk[d])
    mx, cand, wonw, junk = at, at + WAVE, at + 3 * WAVE, at + 4 * WAVE
    fx.ptr_store(m, lds + (mx + lane))
    # empty slots: the lowest pair (never above a candidate)
    fx.ptr_store(
        fx.Vector.from_elements([Int32(0), Int32(INT_MIN)], fx.Int32),
        lds + (cand + 2 * lane),
    )
    reach = Int32(0)
    for j in range_constexpr(0, WAVE, 4):
        mj = fx.Vector(
            fx.ptr_load(lds + (mx + j), result_type=fx.Vector.make_type(4, fx.Int32))
        )
        for e in range_constexpr(4):
            reach = reach + (Int32(mj[e]) >= m).select(Int32(1), Int32(0))
    t = wave_max((reach >= k).select(m, Int32(INT_MIN)))
    ge = [sk[d] >= t for d in range(per)]
    cnt = Int32(0)
    for d in range_constexpr(per):
        cnt = cnt + ge[d].select(Int32(1), Int32(0))
    pos = wave_prefix(cnt) - cnt
    tot = readlane(pos + cnt, WAVE - 1)
    slot = []
    for d in range_constexpr(per):
        at_d = (ge[d] & (pos < WAVE)).select(cand + 2 * pos, Int32(junk))
        lo = Int32(0xFFFF - d) - lane * per
        fx.ptr_store(fx.Vector.from_elements([lo, sk[d]], fx.Int32), lds + at_d)
        slot.append(pos)
        pos = pos + ge[d].select(Int32(1), Int32(0))
    _won_thresh(k, vals, lane, lds, cand, wonw, tot, ge, slot)
    return Int32(fx.ptr_load(lds + (wonw + lane)))


def _pair64(w):
    return (fx.Int64(Int32(w[1])) << 32) | fx.Int64(fx.Uint32(Int32(w[0])))


def _won_thresh(k, vals, lane, lds, cand, wonw, tot, ge, slot):
    """``wave_topk_won_thresh``'s ranks (at most 64 candidates) or its fallback; the won bits -> LDS ``wonw``. One
    value a lane has at most 64 candidates: no fallback branch."""
    if const_expr(len(vals) == 1):
        won_rank(k, lane, lds, cand, wonw, tot, ge, slot)
    else:
        _won_thresh_if(k, vals, lane, lds, cand, wonw, tot, ge, slot)


@traced
def _won_thresh_if(k, vals, lane, lds, cand, wonw, tot, ge, slot):
    if tot <= WAVE:
        won_rank(k, lane, lds, cand, wonw, tot, ge, slot)
    else:
        per = len(vals)
        won = (
            wave_topk_won_full(k, vals, lane)
            if k <= per
            else wave_topk_won(Int32(per * WAVE), k, vals, lane)
        )
        fx.ptr_store(won, lds + (wonw + lane))


def won_rank(k, lane, lds, cand, wonw, tot, ge, slot):
    """The ``tot`` (<= 64) candidates at LDS ``cand`` (one a lane: the 64-bit (key, 2^16 - 1 - index) pairs; empty
    slots the lowest pair): each one's rank, the count of candidates above it, the winners those of rank < k. Lane
    ``l``'s won bits (bit d: its value d, a candidate ``ge[d]`` at ``slot[d]``) -> LDS ``wonw + l``.
    """
    mine = _pair64(
        fx.Vector(
            fx.ptr_load(
                lds + (cand + 2 * lane), result_type=fx.Vector.make_type(2, fx.Int32)
            )
        )
    )
    above = Int32(0)
    for j in range_constexpr(0, WAVE, 2):
        w = fx.Vector(
            fx.ptr_load(
                lds + (cand + 2 * j), result_type=fx.Vector.make_type(4, fx.Int32)
            )
        )
        for e in range_constexpr(2):
            above = above + (_pair64([w[2 * e], w[2 * e + 1]]) > mine).select(
                Int32(1), Int32(0)
            )
    wins = ballot((lane < tot) & (above < k))
    won = Int32(0)
    for d in range_constexpr(len(ge)):
        hit = ge[d] & (
            ((wins >> fx.Int64(fx.min(slot[d], Int32(WAVE - 1)))) & fx.Int64(1))
            != fx.Int64(0)
        )
        won = won | hit.select(Int32(1 << d), Int32(0))
    fx.ptr_store(won, lds + (wonw + lane))
