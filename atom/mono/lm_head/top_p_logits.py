# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A top-p row's sampling and its logits in the LM head launch, what both of
its draws read (``top_p_candidates``, ``top_p_scan``): which rows draw
within their top-p, the row's ``helpers`` CTAs and their slices of its
logits ``lg`` (bf16 [rows][vocab], the dot's epilogue's), loaded 16 bytes a
thread in batches, and a logit's 16-bit order key."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from atom.mono.device.ops import CM_DEV, fresh, ld_f32
from atom.mono.device.select import INT_MIN, order_key
from atom.mono.plan.execution import THREADS

# 16-byte loads of a row's logits in flight a thread
SCAN_BATCH = 8
# CTAs a row's draw is split over at most (a lead's poll of its helpers'
# records stays a few batches)
HELPERS_MAX = 16


def helpers(rows: int, tasks: int) -> int:
    """CTAs a row's draw is split over: as many as the tasks hold a row, at
    most ``HELPERS_MAX``."""
    return max(1, min(HELPERS_MAX, tasks // rows))


def draws_top_p(t, top_p, greedy_t: float):
    """A row drawn within its top-p: T > ``greedy_t`` and top_p < 1 (the dot
    keeps its greedy keys, its logits go to ``lg``)."""
    return (t > fx.Float32(greedy_t)) & (top_p < fx.Float32(1.0))


def row_sampling(c, row):
    """Row ``row``'s (T, top_p), its request's."""
    p, a = c["key"], c["args"]
    return ld_f32(a["temps"], row // p.rq), ld_f32(a["top_ps"], row // p.rq)


def slice_span(p, j):
    """Helper j's quads [j Q / H, (j + 1) Q / H) of a row's Q, H helpers."""
    quads = p.vocab // 8
    return j * quads // p.helpers, (j + 1) * quads // p.helpers


def key16(w, hi_half):
    """(the bf16 as f32, its 16-bit order key) of the low / high half of word
    ``w`` (-0 as +0)."""
    bits = (w & fx.Int32(-65536)) if hi_half else (w << 16)
    bits = (bits == fx.Int32(INT_MIN)).select(fx.Int32(0), bits)
    return bits.bitcast(fx.Float32), (order_key(bits) >> 16) & 0xFFFF


def _quad(c, r, i, span):
    """This thread's i-th 16-byte quad of its slice ``span`` (first, end
    quad) of row r of ``lg`` (8 logits): its words, its index in the row,
    and whether it is in the slice."""
    p = c["key"]
    first, end = span
    q = first + fx.Int32(i * THREADS) + fresh(c["tid"])
    ok = q < end
    q = fx.min(q, fx.Int32(p.vocab // 8 - 1))
    w = bo.buffer_load(
        c["r_lg"], r * (p.vocab // 2) + q * 4, vec_width=4, dtype=T.i32,
        cache_modifier=CM_DEV,
    )  # fmt: skip
    return fx.Vector(w), q, ok


def batches(p) -> int:
    """A slice's loads in batches of ``SCAN_BATCH`` a thread."""
    return -(-p.scan_loads // SCAN_BATCH)


def slice_batch(c, r, bi, span):
    """Batch ``bi`` of this thread's loads of its slice of row r (each a
    quad's words, its index, whether in the slice), every one issued before
    any use."""
    p = c["key"]
    first = bi * SCAN_BATCH
    return [
        _quad(c, r, i, span)
        for i in range(first, min(first + SCAN_BATCH, p.scan_loads))
    ]
