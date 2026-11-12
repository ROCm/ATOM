# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A CTA's exact top-k of up to ``LDS_KEYS`` fp32 values, the answer of aiter's
``top_k_per_row_decode(stable=True)``: ranked by the fp32 bits' u32 order key
(-0 below +0, NaN above +inf), ties to the lowest index, emitted ascending by
index, -1 padded. P0.7 pinned the key and the tie rule against aiter;
``lds_topk_timing.py CHECK=1`` (skill mono-indexer) pins this routine against
that rule on ties, +-0, NaN and +-inf at every edge n.

Thread i holds ``per_thread`` consecutive indices from i x per_thread, as
keys in LDS, so thread order is index order and one block scan places the
winners. Three radix passes (11, 11, 10 bits) find the k-th key, stopping
early when the entire boundary bucket is needed. Integer counts make the
answer independent of arrival order.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32

from atom.mono.device.ops import block_excl_scan, traced
from atom.mono.device.select import INT_MIN, u32_lt
from atom.mono.plan.execution import THREADS, WAVES

PER_THREAD = 64  # the largest a CTA's LDS takes alone: 32768 keys
LDS_KEYS = THREADS * PER_THREAD


def order_key(bits):
    """u32 order key of fp32 bits, held in an Int32."""
    return (bits < 0).select(~bits, bits | INT_MIN)


def key_bits(key):
    """The fp32 bits ``order_key`` came from."""
    return (key < 0).select(key ^ INT_MIN, ~key)


# A histogram, then a dummy bin per lane: a key outside the pass's
# prefix counts there, so the adds need no branch and no two lanes share one
RADIX_BITS = (11, 11, 10)
HIST_WORDS = (1 << max(RADIX_BITS)) + 64


def lds_words(per_thread: int = PER_THREAD) -> dict[str, int]:
    """The LDS the routine needs, in 32-bit words."""
    return {"keys": THREADS * per_thread, "hist": 3 * HIST_WORDS, "buf": WAVES + 3}


def _no_extra():
    return None


@traced
def lds_topk(n, k, load, emit, lds, tid, per_thread=PER_THREAD, after_keys=None):
    """Top ``k`` of ``n`` <= THREADS x per_thread values, given a quad at a
    time: ``load(q, i)`` -> the 4 values (fp32 bits as Int32) at indices i ..
    i + 3, for i = per_thread tid + 4 q clamped to the quad holding index n - 1
    (values past n are read but never counted) -> ``emit(pos, d, i, extra)`` by
    the thread holding each winner (index i = per_thread tid + d), pos its
    place ascending, and ``emit(pos, None, -1, extra)`` for pos in [min(n, k),
    k). The keys stay in ``lds["keys"]`` (``key_bits`` of one is its value).
    Barriers included; every thread of the CTA calls it. ``extra`` is what
    ``after_keys()`` returns (None without it), called once the keys are in,
    before the passes: loads it issues overlap the passes without holding up
    the keys' own (the load counter is in order)."""
    lane, wave = tid % 64, tid // 64
    keys, hist, buf = lds["keys"], lds["hist"], lds["buf"]
    base = tid * per_thread
    # quads past the last value's re-read it: in the row, never counted
    last = fx.max(n - 1, 0) & ~3
    kv = []
    for q in range_constexpr(per_thread // 4):
        quad = load(q, fx.min(base + 4 * q, last))
        for e in range_constexpr(4):
            kv.append(order_key(quad[e]))
            fx.ptr_store(kv[-1], keys + (base + 4 * q + e))
    for r in range_constexpr((2 * HIST_WORDS + THREADS - 1) // THREADS):
        if tid + THREADS * r < 2 * HIST_WORDS:  # histograms 0 and 1
            fx.ptr_store(Int32(0), hist + (tid + THREADS * r))
    gpu.barrier()
    extra = (after_keys or _no_extra)()
    prefix = Int32(0)
    mask = Int32(0)
    need = Int32(k)
    done = fx.Boolean(False)
    n = fx.max(n, 0)
    # n <= k takes every key: no k-th to find
    if n > k:
        # Pass p counts into histogram p and clears p + 2 ahead of its use.
        for p in range_constexpr(len(RADIX_BITS)):
            if ~done:
                shift = sum(RADIX_BITS[p + 1 :])
                bins = 1 << RADIX_BITS[p]
                h = hist + p * HIST_WORDS
                dummy = bins + lane
                if const_expr(p == 0):
                    run_bin, run = Int32(dummy), Int32(0)
                    for d in range_constexpr(per_thread):
                        bin_ = (base + d < n).select(
                            (kv[d] >> shift) & (bins - 1), dummy
                        )
                        same = bin_ == run_bin
                        fx.llvm.atomic_add(
                            h + same.select(dummy, run_bin), same.select(0, run),
                            syncscope="workgroup",
                        )  # fmt: skip
                        run = same.select(run + 1, 1)
                        run_bin = bin_
                    fx.llvm.atomic_add(h + run_bin, run, syncscope="workgroup")
                else:
                    for d in range_constexpr(per_thread):
                        key = kv[d]
                        counted = (base + d < n) & ((key & mask) == prefix)
                        bin_ = counted.select((key >> shift) & (bins - 1), dummy)
                        fx.llvm.atomic_add(h + bin_, Int32(1), syncscope="workgroup")
                gpu.barrier()
                if const_expr(p + 2 < len(RADIX_BITS)):
                    for r in range_constexpr((HIST_WORDS + THREADS - 1) // THREADS):
                        if tid + r * THREADS < HIST_WORDS:
                            fx.ptr_store(
                                Int32(0),
                                hist + (p + 2) * HIST_WORDS + tid + r * THREADS,
                            )
                per = bins // THREADS
                cs = [fx.ptr_load(h + (bins - 1 - per * tid - q)) for q in range(per)]
                total = Int32(0)
                for q in range_constexpr(per):
                    total = total + cs[q]
                before, _ = block_excl_scan(total, lane, wave, buf)
                for q in range_constexpr(per):
                    hit = (before < need) & (need <= before + cs[q])
                    if hit:
                        fx.ptr_store(Int32(bins - 1 - per * tid - q), buf + WAVES)
                        fx.ptr_store(need - before, buf + WAVES + 1)
                        fx.ptr_store(cs[q], buf + WAVES + 2)
                    before = before + cs[q]
                gpu.barrier()
                prefix = prefix | (fx.ptr_load(buf + WAVES) << shift)
                mask = mask | Int32(
                    ((bins - 1) << shift) - (1 << 32) if p == 0 else (bins - 1) << shift
                )
                need = fx.ptr_load(buf + WAVES + 1)
                done = fx.ptr_load(buf + WAVES + 2) == need
    take_all = n <= k
    # winners: keys above the k-th, then its ties lowest index first; one scan of
    # (above, ties) packed in 16-bit halves gives both prefixes
    gts = [(base + d < n) & u32_lt(prefix, kv[d]) for d in range(per_thread)]
    eqs = [(base + d < n) & (kv[d] == prefix) for d in range(per_thread)]
    above, ties = Int32(0), Int32(0)
    for d in range_constexpr(per_thread):
        above = above + gts[d].select(1, 0)
        ties = ties + eqs[d].select(1, 0)
    packed, _ = block_excl_scan(above | (ties << 16), lane, wave, buf)
    ab = packed & 0xFFFF
    tb = packed >> 16
    for d in range_constexpr(per_thread):
        i = base + d
        gt, tie = gts[d], eqs[d]
        win = (i < n) & (take_all | gt | (tie & (tb < need)))
        at = take_all.select(i, ab + fx.min(tb, need))
        if win:
            emit(at, d, i, extra)
        ab = ab + gt.select(1, 0)
        tb = tb + tie.select(1, 0)
    n_out = fx.min(n, k)
    for r in range_constexpr((k + THREADS - 1) // THREADS):
        pos = tid + THREADS * r
        if (pos >= n_out) & (pos < k):
            emit(pos, None, Int32(-1), extra)
    gpu.barrier()
