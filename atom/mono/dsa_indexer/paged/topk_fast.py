# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Local three-digit radix selection, keeping stable ascending index output."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32

from atom.mono.device.ops import block_excl_scan, traced, wave_incl_scan
from atom.mono.device.select import u32_lt
from atom.mono.dsa_indexer.paged.topk import key_bits, order_key
from atom.mono.plan.execution import THREADS, WAVES


def key_offset(i):
    """Spread consecutive per-thread key groups across LDS banks."""
    return i ^ ((i >> 5) & 31)


@traced
def lds_topk_fast(
    n, k, load, emit, lds, tid, per_thread, after_keys=None, emit_key=False
):
    lane, wave = tid % 64, tid // 64
    keys, hist, buf = lds["keys"], lds["hist"], lds["buf"]
    base = tid * per_thread
    last_quad = fx.max(n - 1, 0) & ~3
    kv = []
    for q in range_constexpr(per_thread // 4):
        quad = load(q, fx.min(base + 4 * q, last_quad))
        for e in range_constexpr(4):
            kv.append(order_key(quad[e]))
            if const_expr(not emit_key):
                fx.ptr_store(kv[-1], keys + key_offset(base + 4 * q + e))
    if tid == 0:
        fx.ptr_store(Int32(k), buf + WAVES)
    gpu.barrier()
    extra = None if after_keys is None else after_keys()
    prefix, mask, need = Int32(0), Int32(0), Int32(k)
    done = fx.Boolean(False)
    n = fx.max(n, 0)
    if n > k:
        for p in range_constexpr(3):
            bits, shift = ((12, 20), (12, 8), (8, 0))[p]
            bins = 1 << bits
            bin_mask = bins - 1
            if ~done:
                for r in range_constexpr(-(-bins // THREADS)):
                    if tid + r * THREADS < bins:
                        fx.ptr_store(Int32(0), hist + tid + r * THREADS)
                gpu.barrier()
                if const_expr(p == 0):
                    run = Int32(0)
                    for d in range_constexpr(per_thread):
                        digit = (kv[d] >> shift) & bin_mask
                        live = base + d < n
                        run = run + live.select(1, 0)
                        end_run = live
                        if const_expr(d + 1 < per_thread):
                            end_run = end_run & (
                                (base + d + 1 >= n)
                                | (((kv[d + 1] >> shift) & bin_mask) != digit)
                            )
                        if end_run:
                            fx.llvm.atomic_add(hist + digit, run, syncscope="workgroup")
                        run = end_run.select(0, run)
                else:
                    for d in range_constexpr(per_thread):
                        if (base + d < n) & ((kv[d] & mask) == prefix):
                            fx.llvm.atomic_add(
                                hist + ((kv[d] >> shift) & bin_mask),
                                Int32(1),
                                syncscope="workgroup",
                            )
                gpu.barrier()
                per_bin = max(1, bins // THREADS)
                counts = [
                    (per_bin * tid + q < bins).select(
                        fx.ptr_load(hist + fx.max(bins - 1 - (per_bin * tid + q), 0)),
                        Int32(0),
                    )
                    for q in range(per_bin)
                ]
                mine = counts[0]
                for q in range_constexpr(1, per_bin):
                    mine = mine + counts[q]
                incl = wave_incl_scan(mine, lane)
                if lane == 63:
                    fx.ptr_store(incl, buf + wave)
                gpu.barrier()
                above = incl - mine
                for w in range_constexpr(WAVES - 1):
                    above = above + (w < wave).select(fx.ptr_load(buf + w), Int32(0))
                for q in range_constexpr(per_bin):
                    hit = (above < need) & (need <= above + counts[q])
                    if hit:
                        fx.ptr_store(bins - 1 - (per_bin * tid + q), buf + WAVES + 1)
                        fx.ptr_store(need - above, buf + WAVES)
                        fx.ptr_store(counts[q], buf + WAVES + 2)
                    above = above + counts[q]
                gpu.barrier()
                need = fx.ptr_load(buf + WAVES)
                prefix = prefix | (fx.ptr_load(buf + WAVES + 1) << shift)
                mask_here = bin_mask << shift
                mask = mask | Int32(
                    mask_here - (1 << 32) if mask_here >= (1 << 31) else mask_here
                )
                done = fx.ptr_load(buf + WAVES + 2) == need
    if n <= k:
        # Every live value wins in index order; no cross-wave scan is needed.
        for d in range_constexpr(per_thread):
            i = base + d
            if i < n:
                if const_expr(emit_key):
                    emit(i, d, i, extra, key_bits(kv[d]))
                else:
                    emit(i, d, i, extra)
    else:
        above_count, ties = Int32(0), Int32(0)
        for d in range_constexpr(per_thread):
            above_count = above_count + ((base + d < n) & u32_lt(prefix, kv[d])).select(
                1, 0
            )
            ties = ties + ((base + d < n) & (kv[d] == prefix)).select(1, 0)
        packed, _ = block_excl_scan(above_count | (ties << 16), lane, wave, buf)
        ab, tb = packed & 0xFFFF, (packed >> 16) & 0xFFFF
        for d in range_constexpr(per_thread):
            i = base + d
            gt = (i < n) & u32_lt(prefix, kv[d])
            tie = (i < n) & (kv[d] == prefix)
            if (i < n) & (gt | (tie & (tb < need))):
                at = ab + fx.min(tb, need)
                if const_expr(emit_key):
                    emit(at, d, i, extra, key_bits(kv[d]))
                else:
                    emit(at, d, i, extra)
            ab = ab + gt.select(1, 0)
            tb = tb + tie.select(1, 0)
    n_out = fx.min(n, k)
    for r in range_constexpr((k + THREADS - 1) // THREADS):
        pos = tid + THREADS * r
        if (pos >= n_out) & (pos < k):
            emit(pos, None, Int32(-1), extra)
    gpu.barrier()
