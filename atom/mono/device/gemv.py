# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/stages/gemv.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""bf16 GEMV pieces mono kernels share (K3's K1 / K2, the DSA indexer): 16
weight rows a task against up to 16 token rows in LDS, one
``mfma_f32_16x16x32_bf16`` per 32 K.

K is walked in 128-wide chunks, wave w taking chunks w, w + 8, ...; a lane reads
64 contiguous bytes of its row a chunk (4 dwordx4 at q 32 + g 8 elements: lane
group g = lane / 16), and the token rows are read at the same K. The K order
inside a chunk is a permutation both operands share, so the dot is unchanged.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.ops import CM_DEV, CM_NT, mfma_bf16, rsrc, traced, xshfl
from atom.mono.plan.execution import THREADS, WAVES

ROWS = 16
KCH = 128
LDS_PAD = 8  # bf16 a padded token row
# A wave takes one chunk from each group of ``WAVES`` chunks, so a K range that
# starts and ends on a group boundary leaves every wave's chunk sequence exactly
# as an unsplit pass would produce it. Splitting anywhere else reorders a wave's
# accumulation and is no longer bit-identical.
KGROUP = WAVES * KCH
# What one stage may hold of the token rows. Chosen so that every row count K3's
# kernels serve up to 8 still stage their whole K in one slab -- those builds are
# then byte-for-byte what they were before slabbing existed.
XL_BUDGET = 132 * 1024


def slabs(k, s):
    """``[(k0, klen)]`` covering ``k`` columns in the fewest slabs that fit
    ``s`` token rows in ``XL_BUDGET``, split only on group boundaries.

    One slab is returned whenever the whole K fits, which is the case for every
    build up to 8 rows.
    """
    if k % KGROUP or s * (k + LDS_PAD) * 2 <= XL_BUDGET:
        return [(0, k)]
    groups = k // KGROUP
    for n in range(2, groups + 1):
        widest = -(-groups // n) * KGROUP
        if s * (widest + LDS_PAD) * 2 <= XL_BUDGET:
            out, start = [], 0
            for j in range(n):
                take = -(-(groups - start // KGROUP) // (n - j)) * KGROUP
                out.append((start, take))
                start += take
            return out
    raise AssertionError(f"{s} rows x {KGROUP} columns exceed {XL_BUDGET} B")


def slab_width(k, s):
    """The widest slab ``slabs`` will hand back, which is what LDS is sized for."""
    return max(n for _, n in slabs(k, s))


def chunks(k):
    """A wave's K chunks for K = ``k``: (count a wave, total)."""
    n = k // KCH
    return (n + WAVES - 1) // WAVES, n


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0, paired=False):
    """This wave's operands of weight rows ``row0`` .. + 16 (row-major, ``ldk``
    bf16 a row at ``w``, K columns ``k0`` .. ``k0 + k``): a list a chunk
    (clamped into K: ``mfmas`` zeroes the extra), ``chunk_ops``' each."""
    per, n = chunks(k)
    ldk = k if ldk is None else ldk
    r_w = rsrc(w)
    cm = CM_NT if nt else 0
    return [
        chunk_ops(
            r_w, row0, ldk, k0 + fx.min(wave + WAVES * i, n - 1) * KCH, lane, paired, cm
        )
        for i in range(per)
    ]


def chunk_ops(r_w, row0, ldk, kcol, lane, paired=False, cm=CM_NT):
    """The four MFMA A operands of one 128-column chunk (columns ``kcol`` ..)
    of weight rows ``row0`` .. + 16 (``ldk`` bf16 a row): slab q's lane holds
    row lane % 16, 8 columns from 32 q + (lane / 16) 8.

    A load takes one slab of the 16 rows (64 B each, rows a row apart), or
    with ``paired`` two slabs of 8 rows -- whole 128 B lines: lane r < 8 the
    even slab of row r, lane r >= 8 the odd slab of row r - 8, and a lane ^ 8
    swap and a select put the layout back, the same values. Coalescing sees a
    load's addresses, not its lane order. It holds the pair's registers a
    while: a build at the VGPR edge spills with it (K3 K1 at S <= 16)."""
    r16 = lane % ROWS
    k = kcol + (lane // ROWS) * 8

    def load(row, col):
        return fx.Vector(
            bo.buffer_load(
                r_w, (row * ldk + col) // 2, vec_width=4, dtype=T.i32, cache_modifier=cm
            )
        )

    if not paired:
        return [load(row0 + r16, k + 32 * q) for q in range(4)]
    # slab pair p of rows 8 h .. 8 h + 7
    got = [
        [load(row0 + 8 * h + r16 % 8, k + 32 * (2 * p + r16 // 8)) for h in range(2)]
        for p in range(2)
    ]
    lo = r16 < 8
    ops = []
    for p in range(2):
        a, b = got[p]
        ops.append(_select4(lo, a, _swap8(b)))  # slab 2 p
        ops.append(_select4(lo, _swap8(a), b))  # slab 2 p + 1
    return ops


def _swap8(v):
    """A 4-dword operand from lane ^ 8 (within a 16-lane row)."""
    return fx.Vector.from_elements([xshfl(v[e], 8) for e in range(4)], fx.Int32)


def _select4(cond, a, b):
    return fx.Vector.from_elements(
        [cond.select(a[e], b[e]) for e in range(4)], fx.Int32
    )


def mfmas(lane, wave, xl, xrow, k, s, ops, acc=None):
    """``loads``' operands against token rows 0 .. s - 1 of the LDS rows ``xl``
    (``xrow`` bf16 a row) -> this wave's 16x16 accumulator.

    ``acc`` carries an earlier slab's partial in, so a K split on group
    boundaries accumulates in the same order one pass over the whole K would.
    """
    per, n = chunks(k)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    if acc is None:
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(per):
        ch = wave + WAVES * i
        live = ch < n
        kk = fx.min(ch, n - 1) * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    xl + (t * xrow + kk + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            if const_live(per, n):
                xb = fx.Vector.from_elements(
                    [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
                )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    return acc


def mfmas_unit(lane, wave, xl, xrow, i, k, s, op, acc):
    """``mfmas``' chunk ``i`` alone: ``op`` (``loads``' entry i) against the
    LDS rows of column group i only (``KGROUP`` columns from local column 0,
    ``xrow`` bf16 a row) -> ``acc``. Groups i = 0, 1, .. in turn accumulate in
    ``mfmas``' order, so a pipelined staging a group at a time is bit-equal."""
    per, n = chunks(k)
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    ch = wave + WAVES * i
    live = ch < n
    kk = wave * KCH + g * 8
    for q in range_constexpr(4):
        xb = fx.Vector(
            fx.ptr_load(
                xl + (t * xrow + kk + 32 * q) // 2,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        if i == per - 1 and const_live(per, n):
            # a wave whose chunk is past K reads stale LDS: select, not x 0
            xb = fx.Vector.from_elements(
                [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
            )
        acc = mfma_bf16(op[q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc)
    return acc


def const_live(per, n):
    """Whether some wave's last chunk is past K (its MFMA must add zero)."""
    return per * WAVES != n


@traced
def rows_to_lds(tid, src, k, s, xl, xrow, cm=0, ldk=None, k0=0):
    """Columns k0 .. k0 + k of token rows 0 .. s - 1 of the bf16 [s, ldk] at
    ``src`` -> LDS (``xrow`` a row), 16 B a lane. Each chunk is stored as it
    arrives: holding every chunk first (``rows_loads``) spilled K3's 16-row
    K2 (1 VGPR, 8 B scratch) and cost it 1.7 us."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    for q in range_constexpr(-(-words // THREADS)):
        v, t, kk = _row_chunk(tid, src, k, ldk, k0, cm, words, q)
        if tid + THREADS * q < words:
            fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


def _row_chunk(tid, src, k, ldk, k0, cm, words, q):
    """Pass q's 16 B chunk of a thread (``words`` chunks over the rows, the
    last re-read past them) and its (row, column): ``rows_to_lds`` /
    ``rows_loads``' one load."""
    e = fx.min(tid + THREADS * q, words - 1)
    t, kk = e // (k // 8), e % (k // 8) * 8
    v = fx.Vector(
        bo.buffer_load(
            rsrc(src),
            (t * ldk + k0 + kk) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cm,
        )
    )
    return v, t, kk


def rows_loads(tid, src, rows, k, ldk, k0=0, cm=CM_DEV):
    """A thread's 16 B chunks of columns k0 .. k0 + k of token rows 0 .. rows
    - 1 of the bf16 [rows, ldk] at ``src``: ``rows_store``'s operands, loaded
    a unit ahead of a pipelined staging (in registers across the unit
    before)."""
    words = rows * k // 8
    return [
        _row_chunk(tid, src, k, ldk, k0, cm, words, q)[0]
        for q in range(-(-words // THREADS))
    ]


@traced
def rows_store(tid, vs, rows, k, xl, xrow):
    """``rows_loads``' chunks into LDS rows of ``xrow`` bf16 at ``xl`` (no
    barrier: the caller's, before the rows are read)."""
    words = rows * k // 8
    for q in range_constexpr(len(vs)):
        e = tid + THREADS * q
        if e < words:
            t, kk = e // (k // 8), e % (k // 8) * 8
            fx.ptr_store(vs[q], xl + (t * xrow + kk) // 2)


def staged_units(n, ahead, bufs, load, store, compute):
    """n units of a GEMV, each a slice of the token rows staged into one of
    the two LDS buffers ``bufs`` (unit u in ``bufs[u % 2]``) and multiplied.
    Unit u's rows are loaded ``ahead`` units before it multiplies (in
    registers meanwhile) and stored behind the barrier that freed its buffer
    (unit u - 1's reads); a barrier between units. ``load(u)`` -> its chunks
    (``rows_loads``), ``store(u, xs, buf)`` (``rows_store``), ``compute(u,
    buf)`` its MFMAs: what it issues before them (the next weights) is in
    flight under them. A whole slice staged at a time, each behind a barrier,
    left a 64-row step's GEMVs waiting on every staging (K3 K1 40 us, the
    latent GEMV 16-24 us past its x)."""
    pending = {u: load(u) for u in range(min(ahead, n))}
    store(0, pending.pop(0), bufs[0])
    gpu.barrier()
    for u in range_constexpr(n):
        if const_expr(u + ahead < n):
            pending[u + ahead] = load(u + ahead)
        compute(u, bufs[u % 2])
        if const_expr(u + 1 < n):
            store(u + 1, pending.pop(u + 1), bufs[(u + 1) % 2])
            gpu.barrier()


def dev_rows_to_lds(tid, src, k, s, xl, xrow):
    """``rows_to_lds`` at device scope: rows another CTA of this launch wrote."""
    rows_to_lds(tid, src, k, s, xl, xrow, CM_DEV)
