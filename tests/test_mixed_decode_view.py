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

import re
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
    m = re.search(r"class _MixedDecodeView.*?(?=\nclass )", src, re.S)
    assert m, "_MixedDecodeView not found"
    return m.group(0)


def _sliced_fields() -> set[str]:
    """Attributes `__init__` assigns, i.e. the ones it explicitly slices."""
    return set(re.findall(r"self\.([a-z_][a-z0-9_]*)\s*=", _view_source())) - {
        "_batch",
        "_np",
    }


def _per_row_reads() -> set[str]:
    """Fields the V4 builder indexes by row count, `batch.x[:scheduled_bs]`."""
    return set(
        re.findall(
            r"batch\.([a-z_][a-z0-9_]*)\[\s*:\s*"
            r"(?:scheduled_bs|bs|n_d_seqs|running_bs)\s*\]",
            V4_ATTN.read_text(),
        )
    )


def test_every_per_row_read_is_sliced_by_the_view():
    missing = sorted(_per_row_reads() - _sliced_fields())
    assert not missing, (
        f"{missing} are read as `batch.<field>[:scheduled_bs]` but the decode "
        "view does not slice them, so the consumer would get the prefill rows. "
        "Slice them in _MixedDecodeView.__init__."
    )


class _FakeBatch:
    """Minimal stand-in: one per-row array and one batch-wide scalar."""

    def __init__(self, n_rows: int):
        self.total_seqs_num = n_rows
        self.per_row_thing = np.arange(n_rows)
        self.a_scalar = 7
        self.a_short_list = [1, 2]


def test_unsliced_per_row_field_raises_instead_of_falling_through():
    """The failure mode that cost 13 points must now be loud."""
    from atom.model_ops.attentions.deepseek_v4_attn import _MixedDecodeView

    view = _MixedDecodeView.__new__(_MixedDecodeView)
    view._batch = _FakeBatch(8)
    view._np = 3

    with pytest.raises(AttributeError, match="does not slice"):
        _ = view.per_row_thing


def test_batch_wide_values_still_delegate():
    """Only per-row arrays are refused; everything else passes through."""
    from atom.model_ops.attentions.deepseek_v4_attn import _MixedDecodeView

    view = _MixedDecodeView.__new__(_MixedDecodeView)
    view._batch = _FakeBatch(8)
    view._np = 3

    assert view.a_scalar == 7
    # Shorter than the batch, so not per-row -- delegating is correct.
    assert view.a_short_list == [1, 2]
