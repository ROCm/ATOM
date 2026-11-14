# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/stages/gemv.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""GEMV pieces mono kernels share (K3's K1 / K2, the DSA indexer): 16 bf16
(or ``w8``) weight rows a task against up to 16 token rows in LDS, one
``mfma_f32_16x16x32_bf16`` per 32 K.

K is walked in 128-wide chunks, wave w taking chunks w, w + 8, ...; a lane reads
64 contiguous bytes of its row a chunk (4 dwordx4 at q 32 + g 8 elements: lane
group g = lane / 16), and the token rows are read at the same K. The K order
inside a chunk is a permutation both operands share, so the dot is unchanged.

``w8``: the weight is ATOM's per-token FP8 linear as the production GEMM holds
it -- E4M3 in AITER's (16, 16) preshuffle, ``[N/16][K/32][2][16 rows][16 B]``,
with an fp32 scale a row. A lane's 8 elements of a q step are then 8 contiguous
bytes (``w8_offset``), widened to the same bf16 operand exactly
(``fp8x8_bf16_pk``): the MFMAs and the K order are the bf16 path's, and the
row's scale multiplies the reduced sum (``row_value``), loaded before the MFMAs
so that it lands under them (``pair_scales``).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    fp8x8_bf16_pk,
    ld_f32,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    xshfl,
)
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


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0, paired=False, w8=False):
    """This wave's operands of weight rows ``row0`` .. + 16 (row-major, ``ldk``
    bf16 a row at ``w``, K columns ``k0`` .. ``k0 + k``): a list a chunk
    (clamped into K: ``mfmas`` zeroes the extra), ``chunk_operands``' each."""
    per, n = chunks(k)
    ldk = k if ldk is None else ldk
    r_w = rsrc(w)
    cm = CM_NT if nt else 0

    def kcol(i):
        return k0 + fx.min(wave + WAVES * i, n - 1) * KCH

    return [
        chunk_operands(r_w, row0, ldk, kcol(i), lane, paired, cm, w8)
        for i in range(per)
    ]


def chunk_operands(r_w, row0, ldk, kcol, lane, paired=False, cm=CM_NT, w8=False):
    """A chunk's MFMA A operands: ``chunk_ops``', or (``w8``) those of a
    preshuffled E4M3 plane, ``w8_chunk_ops``'."""
    if w8:
        return w8_chunk_ops(r_w, row0, ldk, kcol, lane, cm)
    return chunk_ops(r_w, row0, ldk, kcol, lane, paired, cm)


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


# dwords from a lane's 8 fp8 of one q step to the next's: 32 K columns, a K
# block (``w8_offset``)
W8_QSTEP = 128


def w8_offset(row0, r, col, ldk):
    """The byte of element (row0 + r, col) of an ``[N, ldk]`` E4M3 plane in
    AITER's (16, 16) preshuffle, ``[N/16][K/32][2][16 rows][16 B]`` (``row0`` a
    multiple of 16, r < 16). The 8 elements from a ``col`` multiple of 8 are
    contiguous: a lane's operand of a q step."""
    return (
        ((row0 // ROWS * (ldk // 32) + col // 32) * 2 + col % 32 // 16) * 256
        + r * 16
        + col % 16
    )


def w8_lane_dwords(row0, lane, kcol, ldk):
    """The dwords of a lane's four q-step operands of chunk ``kcol`` in a
    preshuffled E4M3 plane (``w8_offset``): row row0 + lane % 16, 8 elements
    from kcol + 32 q + 8 (lane / 16), as ``chunk_ops`` takes them."""
    base = w8_offset(row0, lane % ROWS, kcol + 8 * (lane // ROWS), ldk) // 4
    return [base + W8_QSTEP * q for q in range(4)]


def w8_chunk_ops(r_w, row0, ldk, kcol, lane, cm=CM_NT):
    """``chunk_ops`` of a preshuffled E4M3 plane: two dwords a lane a q step
    (``w8_lane_dwords``), which ``operand`` widens. A wave's 64 lanes read a
    step's 512 B in one piece, so there is no ``paired`` form."""
    return [
        fx.Vector(bo.buffer_load(r_w, at, vec_width=2, dtype=T.i32, cache_modifier=cm))
        for at in w8_lane_dwords(row0, lane, kcol, ldk)
    ]


def operand(op, w8=False):
    """A ``chunk_ops`` operand, or (``w8``) a ``w8_chunk_ops`` one widened:
    the bf16 MFMA A operand."""
    return fp8x8_bf16_pk([op[0], op[1]]) if w8 else op.bitcast(fx.BFloat16)


def pair_scales(scale, row0, tid):
    """A ``w8`` GEMV's row scales (fp32 at ``scale``) of thread tid's
    ``row_pair``, rows row0 + 2 (tid % 8) and the next: loaded before the
    MFMAs, so that they land under them rather than after the reduce."""
    r = row0 + 2 * (tid % (ROWS // 2))
    return [ld_f32(scale, r), ld_f32(scale, r + 1)]


def row_value(red, r, t, scale=None):
    """Row r, token t of the waves' accumulators at ``red`` (``row_sum``), times
    ``scale`` (a ``w8`` GEMV's row scale), rounded to bf16."""
    v = row_sum(red, r, t)
    if scale is not None:
        v = v * scale
    return bf16_round(v)


def row_pair(red, rp, t, scales=None):
    """Rows 2 rp and 2 rp + 1, token t (``row_value``; ``scales``:
    ``pair_scales``'): a thread's column pair."""
    scales = [None, None] if scales is None else scales
    return [row_value(red, 2 * rp + i, t, scales[i]) for i in range(2)]


def mfmas(lane, wave, xl, xrow, k, s, ops, acc=None, w8=False):
    """``loads``' operands against token rows 0 .. s - 1 of the LDS rows ``xl``
    (``xrow`` bf16 a row) -> this wave's 16x16 accumulator.

    ``acc`` carries an earlier slab's partial in, so a K split on group
    boundaries accumulates in the same order one pass over the whole K would.
    ``w8``: the operands are ``loads(w8=True)``'s; the sum is unscaled (the
    caller scales the reduced row: ``row_value``).
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
            acc = mfma_bf16(operand(ops[i][q], w8), xb.bitcast(fx.BFloat16), acc)
    return acc


def mfmas_unit(lane, wave, xl, xrow, i, k, s, op, acc, w8=False):
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
        acc = mfma_bf16(operand(op[q], w8), xb.bitcast(fx.BFloat16), acc)
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
