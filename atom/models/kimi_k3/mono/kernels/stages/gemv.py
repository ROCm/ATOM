# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/stages/gemv.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""bf16 GEMV pieces the K3 mono kernels share: 16 weight rows a task against up
to 16 token rows in LDS, one ``mfma_f32_16x16x32_bf16`` per 32 K.

K is walked in 128-wide chunks, wave w taking chunks w, w + 8, ...; a lane reads
64 contiguous bytes of its row a chunk (4 dwordx4 at q 32 + g 8 elements: lane
group g = lane / 16), and the token rows are read at the same K. The K order
inside a chunk is a permutation both operands share, so the dot is unchanged.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.ops import CM_DEV, CM_NT, mfma_bf16, rsrc, traced
from atom.mono.plan.execution import THREADS, WAVES

ROWS = 16
KCH = 128
LDS_PAD = 8  # bf16 a padded token row
# A wave takes one chunk from each group of ``WAVES`` chunks, so a K range that
# starts and ends on a group boundary leaves every wave's chunk sequence exactly
# as an unsplit pass would produce it. Splitting anywhere else reorders a wave's
# accumulation and is no longer bit-identical.
KGROUP = WAVES * KCH
# What one stage may hold of the token rows. Chosen so that every row count this
# kernel serves up to 8 still stages its whole K in one slab -- those builds are
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


def loads(lane, wave, w, row0, k, nt=True, ldk=None, k0=0):
    """This wave's operands of weight rows ``row0`` .. + 16 (row-major, ``ldk``
    bf16 a row at ``w``, K columns ``k0`` .. ``k0 + k``): a list a chunk
    (clamped into K: ``mfmas`` zeroes the extra)."""
    per, n = chunks(k)
    ldk = k if ldk is None else ldk
    r_w = rsrc(w)
    row = row0 + lane % ROWS
    g = lane // ROWS
    out = []
    for i in range(per):
        ch = fx.min(wave + WAVES * i, n - 1)
        base = (row * ldk + k0 + ch * KCH + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT if nt else 0,
                    )
                )
                for q in range(4)
            ]
        )
    return out


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


def const_live(per, n):
    """Whether some wave's last chunk is past K (its MFMA must add zero)."""
    return per * WAVES != n


@traced
def rows_to_lds(tid, src, k, s, xl, xrow, cm=0, ldk=None, k0=0):
    """Columns k0 .. k0 + k of token rows 0 .. s - 1 of the bf16 [s, ldk] at
    ``src`` -> LDS (``xrow`` a row), 16 B a lane."""
    ldk = k if ldk is None else ldk
    words = s * k // 8
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (k // 8)
        kk = e % (k // 8) * 8
        v = fx.Vector(
            bo.buffer_load(
                rsrc(src),
                (t * ldk + k0 + kk) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        if tid + THREADS * i < words:
            fx.ptr_store(v, xl + (t * xrow + kk) // 2)
    gpu.barrier()


def dev_rows_to_lds(tid, src, k, s, xl, xrow):
    """``rows_to_lds`` at device scope: rows another CTA of this launch wrote."""
    rows_to_lds(tid, src, k, s, xl, xrow, CM_DEV)
