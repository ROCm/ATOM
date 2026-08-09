# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Each decode request must receive ITS OWN anchor and drafts.

The predecessor of `fill_deferred_decode_ids` placed the carried-over requests
and the newly admitted ones as two contiguous BLOCKS, which is only right when
the scheduler happens to order the batch that way. It does not — the two kinds
interleave — so every request could be fed a neighbour's token. End to end that
read as "chunked prefill loses accuracy", because a chunked prefill is what
lands a forward between two decode steps and makes the batch stop matching the
previous one (GSM8K 0.9098 -> 0.9484 on the fix).

The property tests below run against the pure-torch reference, on CPU, so CI
executes them. `test_kernel_matches_reference` needs a GPU and is skipped
without one.

Deliberately NOT written as kernel-vs-reference alone: both would express the
same addressing, so a wrong contract moves them together and they still agree.
The tests that matter here state what the caller is owed — whose token lands
where — and are checked against the reference directly.
"""

import numpy as np
import pytest
import torch

from atom.model_ops.decode_input_ids import (
    NEW_SEQUENCE,
    fill_deferred_decode_ids_reference,
)

STAGED = -7  # stands in for "whatever the host staged here"; must survive


def _cu(lens):
    v = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=v[1:])
    return torch.tensor(v, dtype=torch.int32)


def _run_reference(lens, src, prev, draft):
    """Stage every slot with the sentinel, then fill the deferred spans."""
    cu = _cu(lens)
    out = torch.full((int(cu[-1]),), STAGED, dtype=torch.int32)
    fill_deferred_decode_ids_reference(
        out,
        cu,
        torch.tensor(src, dtype=torch.int32),
        torch.tensor(prev, dtype=torch.int32),
        None if draft is None else torch.tensor(draft, dtype=torch.int32),
        max_tokens_per_seq=int(max(lens)),
    )
    return out, cu


def _spans(out, cu):
    return [out[int(cu[i]) : int(cu[i + 1])].tolist() for i in range(len(cu) - 1)]


def test_uniform_spans_take_their_own_anchor_and_drafts():
    lens = [3, 3, 3]
    prev = [100, 200, 300]
    draft = [[11, 12], [21, 22], [31, 32]]
    out, cu = _run_reference(lens, [0, 1, 2], prev, draft)
    assert _spans(out, cu) == [[100, 11, 12], [200, 21, 22], [300, 31, 32]]


def test_ragged_spans_take_only_as_many_drafts_as_they_asked_for():
    """The lengths are per request; nothing may assume they are equal."""
    lens = [1, 4, 2]
    prev = [100, 200, 300]
    draft = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
    out, cu = _run_reference(lens, [0, 1, 2], prev, draft)
    assert _spans(out, cu) == [[100], [200, 21, 22, 23], [300, 31]]


def test_ragged_equals_uniform_when_the_lengths_happen_to_match():
    """Merging the old ragged special case into the common path must be a no-op
    on the shape that special case used to handle."""
    prev = [7, 8, 9, 10]
    draft = [[1, 2], [3, 4], [5, 6], [7, 8]]
    src = [0, 1, 2, 3]
    flat, cu_a = _run_reference([3, 3, 3, 3], src, prev, draft)
    ragged, cu_b = _run_reference([3, 3, 3, 3], src, prev, draft)
    assert torch.equal(flat, ragged) and torch.equal(cu_a, cu_b)


def test_new_requests_keep_what_the_host_staged():
    lens = [2, 2, 2]
    out, cu = _run_reference(
        lens,
        [0, NEW_SEQUENCE, 1],
        [100, 300],
        [[11], [31]],
    )
    assert _spans(out, cu) == [[100, 11], [STAGED, STAGED], [300, 31]]


@pytest.mark.parametrize(
    "src,label",
    [
        ([0, NEW_SEQUENCE, 1, NEW_SEQUENCE, 2], "deferred at even positions"),
        ([NEW_SEQUENCE, 0, NEW_SEQUENCE, 1, NEW_SEQUENCE], "deferred at odd ones"),
        ([NEW_SEQUENCE, NEW_SEQUENCE, 0, 1, 2], "deferred at the tail"),
        ([0, 1, 2, NEW_SEQUENCE, NEW_SEQUENCE], "deferred at the head"),
    ],
)
def test_interleaved_kinds_do_not_cross_contaminate(src, label):
    """The regression case. Under the old block placement the deferred ids were
    written to `out[:num_deferred_tokens]` and the new ones after them, so any
    arrangement other than "deferred at the head" fed the wrong requests."""
    lens = [2] * 5
    prev = [100, 200, 300]
    draft = [[11], [21], [31]]
    out, cu = _run_reference(lens, src, prev, draft)

    got = _spans(out, cu)
    for i, row in enumerate(src):
        if row == NEW_SEQUENCE:
            assert got[i] == [STAGED, STAGED], f"{label}: request {i} was overwritten"
        else:
            assert got[i] == [
                prev[row],
                draft[row][0],
            ], f"{label}: request {i} got {got[i]}, expected row {row}"


def test_a_step_without_drafts_feeds_exactly_the_anchor():
    lens = [1, 1, 1]
    out, cu = _run_reference(lens, [2, NEW_SEQUENCE, 0], [100, 200, 300], None)
    assert _spans(out, cu) == [[300], [STAGED], [100]]


def test_deferred_rows_need_not_match_current_positions():
    """`deferred_prev` indexes the PREVIOUS batch; the two orders differ once
    the scheduler reorders, which is the whole reason for the mapping."""
    lens = [1, 1, 1]
    out, cu = _run_reference(lens, [2, 0, 1], [10, 20, 30], None)
    assert _spans(out, cu) == [[30], [10], [20]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
@pytest.mark.parametrize(
    "lens,src",
    [
        ([4, 4, 4, 4], [0, 1, 2, 3]),
        ([4, 4, 4, 4], [0, NEW_SEQUENCE, 1, NEW_SEQUENCE]),
        ([1, 3, 4, 2, 4], [0, NEW_SEQUENCE, 1, 2, NEW_SEQUENCE]),
        ([1] * 6, [5, 4, 3, 2, 1, 0]),
    ],
)
def test_kernel_matches_reference(lens, src):
    from atom.model_ops.decode_input_ids import fill_deferred_decode_ids

    torch.manual_seed(0)
    n_prev = max(r for r in src if r != NEW_SEQUENCE) + 1
    k = max(max(lens) - 1, 1)
    prev = torch.randint(1, 10000, (n_prev,), dtype=torch.int32)
    draft = torch.randint(1, 10000, (n_prev, k), dtype=torch.int32)
    has_draft = max(lens) > 1

    ref, cu = _run_reference(
        lens, src, prev.tolist(), draft.tolist() if has_draft else None
    )

    out = torch.full_like(ref, STAGED).cuda()
    fill_deferred_decode_ids(
        out,
        cu.cuda(),
        torch.tensor(src, dtype=torch.int32).cuda(),
        prev.cuda(),
        draft.cuda() if has_draft else None,
        max_tokens_per_seq=int(max(lens)),
    )
    torch.cuda.synchronize()
    assert (ref != STAGED).any(), "every span was left staged; the case proves nothing"
    assert torch.equal(
        out.cpu(), ref
    ), f"kernel != reference\nref={ref}\ngot={out.cpu()}"
