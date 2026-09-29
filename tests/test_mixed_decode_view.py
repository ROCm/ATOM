# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`_MixedDecodeView` must slice every per-row array its consumers index.

The view presents the decode rows `[n_prefill:]` of a mixed batch as a
standalone decode batch. Consumers then do `batch.<field>[:scheduled_bs]`,
where `scheduled_bs` is the decode row count. If the view did not slice that
field, the read takes the FIRST n_decode rows of the whole batch -- the prefill
rows -- and every decode row gets another row's state.

That has already happened once, and it cost 13 GSM8K points without raising
anything: `per_req_cache_groups` was renamed `state_slots_committed`, the view
followed the rename to the wrong name, and the consumer silently read unsliced
data. Output norms stayed plausible; the damage scaled with the decode row
count, so a one-row mixed batch looked fine.

Two guards, because they fail at different times:
  * the static one below catches a new/renamed per-row read at test time
  * the view's own `__getattr__` length check catches it at the first mixed
    batch, for anything the source scan cannot see
"""

import importlib.util
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

V4_ATTN = (
    Path(__file__).resolve().parent.parent
    / "atom"
    / "model_ops"
    / "attentions"
    / "deepseek_v4_attn.py"
)


def _view_source() -> str:
    src = V4_ATTN.read_text()
    m = re.search(r"class _MixedDecodeView.*?(?=\nclass )", src, re.DOTALL)
    assert m, "_MixedDecodeView not found"
    return m.group(0)


def _sliced_fields() -> set[str]:
    """Attributes `__init__` assigns, i.e. the ones it explicitly slices."""
    return set(re.findall(r"self\.([a-z_][a-z0-9_]*)\s*=", _view_source())) - {
        "_batch",
        "_np",
    }


def _per_row_reads() -> set[str]:
    """Fields ANY attention file indexes by row count, `batch.x[:bs]`.

    Scans the whole package, not just the V4 builder. Scanning one file is how
    `num_scheduled_tokens` was missed: the read lives in `decode_spans` in
    backends.py, a file over from the view, and the decode view served it
    unsliced -- prefill rows' token counts paired with a cu_seqlens built from
    the real batch. The runtime guard caught that one; this makes the static
    check catch the next.
    """
    # TWO shapes, because only matching the first let `state_fork_srcs`
    # through. Its consumer never writes `batch.state_fork_srcs[...]`:
    #
    #     srcs = getattr(batch, "state_fork_srcs", None)   # -> a local
    #     src_np = np.asarray(srcs[:scheduled_bs], ...)    # -> sliced here
    #
    # so a pattern anchored on `batch.` sees nothing. It was found instead by
    # the runtime guard, on a real server, after the guard stopped being
    # swallowable -- two accidents deep for something a text search can answer.
    #
    # The indirect form tracks the LOCAL as well as the name: a field is only
    # this view's problem if something positionally slices it. Reporting every
    # `getattr(batch, ...)` instead was the first attempt, and it demanded a
    # slice for `is_mixed` -- a batch-wide bool, where slicing is meaningless.
    direct = re.compile(
        r"batch\.([a-z_][a-z0-9_]*)\[\s*:\s*"
        r"(?:scheduled_bs|bs|n_d_seqs|running_bs)\s*\]"
    )
    bind = re.compile(
        r"([a-z_][a-z0-9_]*)\s*=\s*getattr\(\s*batch\s*,\s*"
        r"[\"\']([a-z_][a-z0-9_]*)[\"\']"
    )
    found: set[str] = set()
    for f in V4_ATTN.parent.glob("*.py"):
        txt = f.read_text()
        found |= set(direct.findall(txt))
        for local, field in bind.findall(txt):
            sliced = re.search(
                rf"\b{re.escape(local)}\[\s*:\s*"
                r"(?:scheduled_bs|bs|n_d_seqs|running_bs)\s*\]",
                txt,
            )
            if sliced:
                found.add(field)
    return found


def test_every_per_row_read_is_sliced_by_the_view():
    missing = sorted(_per_row_reads() - _sliced_fields())
    assert not missing, (
        f"{missing} are read as `batch.<field>[:scheduled_bs]` but the decode "
        "view does not slice them, so the consumer would get the prefill rows. "
        "Slice them in _MixedDecodeView.__init__."
    )


# `deepseek_v4_attn` imports aiter at module scope, and CI has no aiter -- so
# the two behavioural tests below can only run on a machine that does. They are
# marked rather than `importorskip`ed at module level so the STATIC test above
# keeps running everywhere: that one reads the file as text, needs no import,
# and is the one that catches a renamed or newly-added per-row read.
#
# Worth being explicit about, because a skip that never runs anywhere is worth
# less than no test at all -- see the note in conftest.atom_config_double about
# four tests that stayed red for exactly this reason and nobody was told.
requires_aiter = pytest.mark.skipif(
    importlib.util.find_spec("aiter") is None,
    reason="deepseek_v4_attn imports aiter at module scope; CI has none",
)


class _FakeBatch:
    """Minimal stand-in: one per-row array and one batch-wide scalar."""

    def __init__(self, n_rows: int):
        self.total_seqs_num = n_rows
        self.per_row_thing = np.arange(n_rows)
        self.a_scalar = 7
        self.a_short_list = [1, 2]


@requires_aiter
def test_unsliced_per_row_field_raises_instead_of_falling_through():
    """The failure mode that cost 13 points must now be loud."""
    from atom.model_ops.attentions.deepseek_v4_attn import (
        MixedViewUnslicedField,
        _MixedDecodeView,
    )

    view = _MixedDecodeView.__new__(_MixedDecodeView)
    view._batch = _FakeBatch(8)
    view._np = 3

    with pytest.raises(MixedViewUnslicedField, match="does not slice"):
        _ = view.per_row_thing


@requires_aiter
def test_the_guard_survives_a_getattr_with_a_default():
    """The guard must not be swallowed by the way consumers actually read.

    `_state_slot_in_np` does `getattr(batch, "state_fork_srcs", None)`, and
    `gdn_attn` does the same for two more fields. `getattr` with a default
    swallows AttributeError and ONLY AttributeError -- so while the guard
    raised that, it was silent on exactly these reads: the field came back as
    None, the caller took its early return, and a forked row kept
    `state_slot_in == state_slot_out`. Loud everywhere except where it mattered.
    """
    from atom.model_ops.attentions.deepseek_v4_attn import (
        MixedViewUnslicedField,
        _MixedDecodeView,
    )

    view = _MixedDecodeView.__new__(_MixedDecodeView)
    view._batch = _FakeBatch(8)
    view._np = 3

    # The literal shape of the consumer's read.
    with pytest.raises(MixedViewUnslicedField):
        _ = getattr(view, "per_row_thing", None)

    # And the `or ()` variant gdn_attn uses.
    with pytest.raises(MixedViewUnslicedField):
        _ = getattr(view, "per_row_thing", None) or ()

    # An AttributeError subclass would be swallowed again -- pin it.
    assert not issubclass(MixedViewUnslicedField, AttributeError)


@requires_aiter
def test_private_names_still_raise_attribute_error():
    """`_`-prefixed lookups must stay AttributeError.

    Not the guard's business, and copy/pickle protocols probe for dunders
    expecting the normal miss.
    """
    from atom.model_ops.attentions.deepseek_v4_attn import _MixedDecodeView

    view = _MixedDecodeView.__new__(_MixedDecodeView)
    view._batch = _FakeBatch(8)
    view._np = 3

    with pytest.raises(AttributeError):
        _ = view._not_a_real_internal
    assert getattr(view, "__deepcopy__", None) is None


@requires_aiter
def test_batch_wide_values_still_delegate():
    """Only per-row arrays are refused; everything else passes through."""
    from atom.model_ops.attentions.deepseek_v4_attn import _MixedDecodeView

    view = _MixedDecodeView.__new__(_MixedDecodeView)
    view._batch = _FakeBatch(8)
    view._np = 3

    assert view.a_scalar == 7
    # Shorter than the batch, so not per-row -- delegating is correct.
    assert view.a_short_list == [1, 2]


class _FilterableBatch:
    """A batch whose `state_slots_committed` can be made short on purpose.

    Short is not hypothetical: the scheduler builds that list from the seqs
    passing `has_per_req_cache and state_slot >= 0`, so a row admitted without
    a slot yet simply is not in it.
    """

    def __init__(self, n_rows: int, n_slots: int):
        self.total_seqs_num = n_rows
        self.context_lens = np.arange(n_rows)
        self.block_tables = np.arange(n_rows)
        self.swa_block_tables = np.arange(n_rows)
        self.num_scheduled_tokens = np.ones(n_rows, dtype=np.int32)
        self.num_cached_tokens = list(range(n_rows))
        self.last_block_num_tokens = list(range(n_rows))
        self.state_slots_committed = list(range(n_slots))
        self.total_seqs_num_decode = n_rows - 2
        self.total_tokens_num_decode = n_rows - 2
        self.is_dummy_run = False
        self.num_spec_step = 0


@requires_aiter
def test_filtered_state_slots_is_refused_not_sliced_positionally():
    """A short list means the positional slice is no longer row-aligned.

    Every consumer downstream reads `state_slots_committed[:scheduled_bs]` as
    if it were one entry per row, so the whole chain rests on the filter having
    dropped nothing. When it did drop something, the old code sliced anyway and
    every decode row got some other row's state slot -- silently.
    """
    from atom.model_ops.attentions.deepseek_v4_attn import (
        MixedViewUnslicedField,
        _MixedDecodeView,
    )

    # Filter dropped one row: 8 rows, 7 slots.
    with pytest.raises(MixedViewUnslicedField, match="filter dropped"):
        _MixedDecodeView(_FilterableBatch(8, 7), 2)


@requires_aiter
def test_unfiltered_state_slots_slices_normally():
    """The ordinary case -- filter a no-op -- still works."""
    from atom.model_ops.attentions.deepseek_v4_attn import _MixedDecodeView

    view = _MixedDecodeView(_FilterableBatch(8, 8), 2)
    assert list(view.state_slots_committed) == [2, 3, 4, 5, 6, 7]


@requires_aiter
def test_the_precondition_check_is_not_an_assert():
    """`python -O` strips asserts; what it was guarding is silent corruption."""
    import ast
    import inspect

    from atom.model_ops.attentions.deepseek_v4_attn import _MixedDecodeView

    src = inspect.getsource(_MixedDecodeView.__init__)
    tree = ast.parse(textwrap.dedent(src))
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    assert not asserts, (
        f"{len(asserts)} assert(s) in _MixedDecodeView.__init__; each guards a "
        "silent row-misalignment path and vanishes under -O"
    )
