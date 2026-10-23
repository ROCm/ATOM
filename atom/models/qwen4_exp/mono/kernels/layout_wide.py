"""Layouts of the grouped verify builds (``build_layer(..., group=S)``).

A grouped build of ``rows`` rows runs ``rows / S`` requests of ``S`` rows
each. Up to :data:`layout.MAX_ROWS` rows its layout is the seq build's of as
many rows (``for_rows(rows, seq=True)``): the seq growth of Q5's and Q7's LDS
entries covers a group of up to ``rows`` rows.

Wide builds (more than ``MAX_ROWS`` rows, up to :data:`WIDE_ROWS`) run the
``stages.*_wide`` stages. Their LDS is in phases (:data:`PHASE`): each
phase's entries start at word 0 and overlay every other phase's, and the
kernel runs a CTA barrier ahead of a stage whose phase is not the last one's
(:func:`barrier_before`).

M3 runs each router block's request rows in halves (an even row count) on
two CTAs, ``m3_task_rows`` rows a task. B3 holds every row's mixed row as
bf16 words (``bax``, as ``baw``), so all rows fit one pass. Past ``WAVES``
rows (one request row per wave) M3's tasks run in passes of ``m3_rows`` rows;
its LDS entries hold one pass's rows. A1's norm pack (``n8``, ``ne``) is of one
row. At 16 rows M4's and M5's entries are two phases, both after ``top``
(M4's CTAs route M5's rows into it ahead).
"""

import functools
from types import SimpleNamespace

from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import GRID
from atom.models.qwen4_exp.mono.kernels.layout import (
    _ATTN_QSA,
    _DELTA_TABLE,
    _HC_TABLE,
    _MOE,
    _PER_ROW,
    _PLE,
    _QSA,
    BASE,
    LDS_LIMIT,
    MAX_ROWS,
    ROUNDS,
    RW_ROW,
    SEQ_GROW,
    TASKS,
    WAVES,
    for_rows,
)

WIDE_ROWS = 16
# Two 64-column slices per M4 task: one per task would not fit the grid.
M4_SLICES = 2
# M5's poll blocks (``moe_wide._m5_blocks``): request rows per block.
M5_BLOCK = 4

# The LDS phase of each stage; Q2 holds no LDS. B1, C1 and Q1 share the GEMV
# entries (``red``, ``x8``, ``xe``), B3 has the rest of the DeltaNet table.
PHASE = {
    "a1": "hc", "a2": "hc", "m1": "hc", "m2": "hc",
    "b1": "d1", "c1": "d1", "q1": "d1",
    "b3": "d3",
    "m3": "m3",
    "m4": "m45", "m5": "m45",
    "q3": "q34", "q4": "q34",
    "q5": "q57", "q6": "q57", "q7": "q57",
    "p1": "ple", "p2": "ple",
}  # fmt: skip


def _phases(group):
    """name -> the LDS table of each phase of a wide build in groups of ``group``."""
    return {
        "hc": _HC_TABLE,
        "d1": _DELTA_TABLE[:3],
        # B3's mixed row by part as bf16 words (``delta_group_wide.b3``).
        "d3": tuple((n, RW_ROW if n == "bax" else w) for n, w in _DELTA_TABLE[3:]),
        "m3": _MOE[:3],
        "m45": tuple((n, w * M4_SLICES if n == "gred" else w) for n, w in _MOE[3:]),
        "q34": _QSA,
        # A group's current key heads and value tiles (``qsa_group.q5`` / ``q7``).
        "q57": tuple((n, w + (group - 1) * SEQ_GROW.get(n, 0)) for n, w in _ATTN_QSA),
        "ple": _PLE,
    }


def _place(lds, stride, table, rows, n_of):
    """``layout._place`` with ``n_of.get(name, rows)`` rows of per-row entry
    ``name`` (its row stride that of ``rows`` rows)."""
    w = 0
    for name, words in table:
        assert lds.get(name, w) == w, name
        lds[name] = w
        stride[name] = (words + 3) // 4 * 4
        if rows > 1 and name in _PER_ROW:
            stride[name] += (16 - stride[name]) % 64
        w += stride[name] * (n_of.get(name, rows) if name in _PER_ROW else 1)
    return w


def barrier_before(stages, i, L):
    """Whether wide build ``L`` runs a CTA barrier ahead of ``stages[i]``: its
    phase differs from the last phase of ``stages[:i]``."""
    phase = L.phase
    if phase.get(stages[i]) is None:
        return False
    last = [phase[s] for s in stages[:i] if phase.get(s) is not None]
    return bool(last) and last[-1] != phase[stages[i]]


@functools.cache
def for_wide(rows, group):
    """The layout of a grouped build: ``rows`` rows in groups of ``group``."""
    assert group >= 2 and rows % group == 0 and rows // group >= 2, (rows, group)
    assert rows <= WIDE_ROWS, rows
    if rows <= MAX_ROWS:
        return for_rows(rows, seq=True)
    # The task counts of ``for_rows``. M1's and M2's CTAs wrap past the
    # grid's end.
    tasks, base = dict(TASKS), dict(BASE)
    # A2 from CTA 128: the narrow builds' 176 was measured up to 4 rows only.
    base["a2"] = 128
    for name in ("a1", "m1"):
        tasks[name] = rows * TASKS[name]
    base["m2"] = (base["m1"] + tasks["m1"]) % GRID
    base["m3"] = (base["m2"] + tasks["m2"]) % GRID
    m3_task_rows = rows // 2 if rows % 2 == 0 else rows
    tasks["m3"] = rows // m3_task_rows * TASKS["m3"]
    tasks["m4"] = (rows * C.top_k + 1) * (C.moe_inter // 64 // M4_SLICES)
    # M5 on the CTAs past M4's last partial round when it fits there: those
    # end M4 a round early and poll M5's rows in blocks of ``M5_BLOCK`` as
    # M4's rounds complete them. Else one block of all rows.
    m4_rem = tasks["m4"] % GRID
    m5_block = rows
    if m4_rem and tasks["m5"] <= GRID - m4_rem:
        base["m5"] = (base["m4"] + m4_rem) % GRID
        m5_block = M5_BLOCK
    tasks["q3"] = rows * (GRID // rows)
    for name in ("q4", "q5", "q6", "q7", "p2"):
        tasks[name] = rows * TASKS[name]
    assert base["q4"] + tasks["q4"] <= GRID  # one CTA per row's selection task
    deep = rows > WAVES
    # Q6 in task rounds past WAVES rows (``qsa_wide.q6``).
    rounds = (*ROUNDS, "q6") if deep else ROUNDS
    for name, n in tasks.items():
        assert n <= GRID or name in rounds, f"{name}: one task per CTA at most"
    # M3's passes, A1's one norm-packed row.
    m3_rows = min(WAVES, m3_task_rows)
    n_of = {"n8": 1, "ne": 1} if deep else {}
    n_of |= {"rx": m3_rows}
    tables, phase = _phases(group), dict(PHASE)
    if _place({}, {}, tables["m45"], rows, n_of) > LDS_LIMIT:
        m45 = tables.pop("m45")
        tables["m4"] = tuple(e for e in m45 if e[0] in ("top", "gx8", "gxe", "gred"))
        tables["m5"] = tuple(e for e in m45 if e[0] in ("top", "dn", "ys"))
        assert tables["m4"][0][0] == tables["m5"][0][0] == "top"
        phase |= {"m4": "m4", "m5": "m5"}
    lds, stride, sizes = {}, {}, {}
    for name, table in tables.items():
        sizes[name] = _place(lds, stride, table, rows, n_of)
        assert sizes[name] <= LDS_LIMIT, (name, sizes[name])
    return SimpleNamespace(
        rows=rows, tasks=tasks, base=base, lds=lds, stride=stride,
        lds_words=max(sizes.values()), split=False, m4_slices=M4_SLICES, seq=True,
        group=group, phases=sizes, phase=phase, m3_rows=m3_rows,
        m3_task_rows=m3_task_rows, m5_block=m5_block,
    )  # fmt: skip
