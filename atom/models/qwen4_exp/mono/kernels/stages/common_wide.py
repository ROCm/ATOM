"""Device helpers of the wide builds: batched builds of more than ``MAX_ROWS``
request rows (``layout_wide.for_wide``). ``common.py`` stays the source of
the builds of up to ``MAX_ROWS`` rows.

Lane ``l`` of an MFMA result tile holds output column ``l % 16`` of A rows
``4 (l // 16) .. 4 (l // 16) + 3`` in ``acc[0..3]`` (:func:`p2.mfma_p2`), so
request row ``r`` of a 16-row tile is ``acc[r % 4]`` of lanes
``16 (r // 4) .. 16 (r // 4) + 15``. The up-to-4-row builds read rows from
lanes 0-15 only.
"""

from types import SimpleNamespace

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.stages.common import mb_ix, row_ix

PRE_POLL_MAX = 64 * mbx.POLL_MAX  # hints per mbx.pre_poll


def _opaque(x, reg):
    asm = llvm.InlineAsmOp(
        T.i32,
        [fx.Int32(x).ir_value()],
        "; fresh $0",
        f"={reg},0",
        has_side_effects=True,
    )
    return fx.Int32(asm.result)


def fresh_round(c):
    """``c`` with copies of ``tid``, ``lane`` and ``wave`` that LLVM cannot
    merge with another round's (``kernel._fresh``), and its own mailbox
    addresses (``c.round_mailboxes``). A stage of two task rounds runs its
    code twice; on one set LLVM computes the rounds' common index math and
    addresses once and holds them through the second round, which then
    spills (M4 and Q7 from 6 rows)."""
    ids = {
        "tid": _opaque(c.tid, "v"),
        "lane": _opaque(c.lane, "v"),
        "wave": _opaque(c.wave, "s"),
    }
    ck = SimpleNamespace(**{**vars(c), **ids})
    c.round_mailboxes(ck)
    return ck


def fresh_ids(c):
    """Fresh copies of ``c.tid`` and ``c.lane`` (as :func:`fresh_round`'s)
    for each stage of a build of more than ``WAVES`` rows. There one stage's
    thread mask can equal a later stage's (at 16 rows Q7's ``tid < 8 * group``
    and M3's ``tid < 2 * rows``); LLVM computes it once and holds (spills) it
    from one stage to the other."""
    c.tid, c.lane = _opaque(c.tid, "v"), _opaque(c.lane, "v")


def row_lanes(lane, r):
    """Whether ``lane`` holds request row ``r`` (a Python int) of an MFMA
    result tile, as ``acc[r % 4]``."""
    return lane // 16 == r // 4


def pre_poll_wide(hint, n, tag):
    """``mbx.pre_poll`` of ``n`` hints, in batches of ``PRE_POLL_MAX``. Every
    thread must call this."""
    for first in range(0, n, PRE_POLL_MAX):
        part = hint if first == 0 else (lambda m, f=first: hint(m + f))
        mbx.pre_poll(part, min(PRE_POLL_MAX, n - first), tag)


def poll_mixed_wide(c, x8, xe):
    """``common.poll_mixed`` of a wide build: wave 0 waits on row 0's 80
    exponent pairs, then thread ``i < 320`` loads its fp8 words of every row
    and exponents ``i, i + 320, ...`` of rows 1 .. rows - 1."""
    tid, lane, wave = c.tid, c.lane, c.wave
    sx, se = c.L.stride["x8"], c.L.stride["xe"]
    n = C.hidden // 32
    vecs = C.hidden // 8
    ne = (c.rows - 1) * n
    if wave == 0:
        e = mbx.poll(
            [
                (c.mb_xe, mb_ix(lane, 0), 1),
                (c.mb_xe, mb_ix(fx.min(lane + 64, n - 1), 0), 1),
            ],
            c.tag,
        )
        fx.ptr_store(e[0][0], xe + lane)
        if lane < 16:
            fx.ptr_store(e[1][0], xe + (lane + 64))
    gpu.barrier()
    if tid < vecs:
        ks = [fx.min(tid + vecs * m, ne - 1) for m in range(-(-ne // vecs))]
        specs = [(c.mb_x8, mb_ix(tid * 2, r), 2) for r in range(c.rows)]
        specs += [(c.mb_xe, mb_ix(k % n, 1 + k // n), 1) for k in ks]
        v = mbx.poll(specs, c.tag)
        for r in range_constexpr(c.rows):
            fx.ptr_store(v[r][0], x8 + (r * sx + tid * 2))
            fx.ptr_store(v[r][1], x8 + (r * sx + tid * 2 + 1))
        for m in range_constexpr(len(ks)):
            if tid + vecs * m < ne:
                fx.ptr_store(v[c.rows + m][0], xe + ((1 + ks[m] // n) * se + ks[m] % n))
    gpu.barrier()


poll_mixed_wide = ASTRewriter.transform(poll_mixed_wide)


def store_rows(c, acc, red, col, rs):
    """Store an MFMA result tile's rows: ``acc[i]`` of lane ``l`` to column
    ``col`` of request row ``4 (l // 16) + i`` (rows ``rs`` apart), for the
    rows below ``c.rows``. Replaces ``if lane < 16: ... acc[r]``."""
    lane = c.lane
    for i in range_constexpr(4):
        r = (lane // 16) * 4 + i
        if r < c.rows:
            fx.ptr_store(acc[i], red + row_ix(col, r, rs))


store_rows = ASTRewriter.transform(store_rows)
