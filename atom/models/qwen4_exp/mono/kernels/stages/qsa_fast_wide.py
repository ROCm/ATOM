"""Q4 of the wide builds (more than ``MAX_ROWS`` request rows):
:func:`qsa_fast.q4` with each branch's mailbox addresses set in the branch.
Set at the start of the stage, LLVM holds one through both branches (and
spills it, from 6 rows); a branch sets its own from a fresh load of
``scratch`` (``c.round_mailboxes``). ``qsa_fast.py`` stays the source of the
builds of up to ``MAX_ROWS`` rows."""

from types import SimpleNamespace

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter

from atom.models.qwen4_exp.mono.kernels.layout import WAVES
from atom.models.qwen4_exp.mono.kernels.stages.common import task_index
from atom.models.qwen4_exp.mono.kernels.stages.qsa import _request, row_len
from atom.models.qwen4_exp.mono.kernels.stages.qsa_fast import (
    CHUNKS,
    FAST_CHUNK,
    Q4_TASKS,
    _chunk,
    _final,
    _mailboxes,
)


def _branch(c):
    """``c`` with a branch's own mailbox addresses (``qsa_fast._mailboxes``)."""
    cn = SimpleNamespace(**vars(c))
    c.round_mailboxes(cn)
    _mailboxes(cn)
    return cn


def q4(c, p):
    t = task_index(c, "q4")
    k0 = c.lds_i + c.L.lds["q4_keys"]
    per = c.L.tasks["q3"] // c.rows  # Q3 tasks of a request row
    rq, j = _request(c, t, Q4_TASKS)
    L = row_len(p, rq)

    if t < c.L.tasks["q4"]:
        cb = (L + 1) // 4
        busy = fx.max(fx.min((cb + WAVES - 1) // WAVES, per), 1)  # Q3 tasks with blocks
        nch = (cb + FAST_CHUNK - 1) // FAST_CHUNK
        if j < CHUNKS:
            if (cb > FAST_CHUNK) & (j < nch):
                _chunk(_branch(c), j, cb, busy, per, k0, rq)
        else:
            _final(_branch(c), cb, busy, per, nch, k0, rq)


q4 = ASTRewriter.transform(q4)
