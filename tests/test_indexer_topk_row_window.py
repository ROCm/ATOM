# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`top_k_per_row_prefill` must honor each row's window, not just its end.

The FP8 indexer prefill path (`Indexer._score_topk_prefill`) scores a
`[rows, total_committed]` logits matrix whose columns are the batch's
sequences' compressed K *concatenated*. Row `t` of sequence `s` may only
select columns `[seq_base[s], seq_base[s] + visible_end[t])`. Nothing masks
the rest: `fp8_mqa_logits` is called with `clean_logits=False` and the FP4
path hands the kernel a raw `torch.empty`. The per-row window IS the whole
mechanism keeping one request out of another's index.

That contract is invisible in a batch of fresh prompts — they have no
committed K, so `total_committed == 0` and there is nothing to leak. It only
carries weight once a batch mixes a *resumed* chunk (committed K, `seq_base`
pushes later rows to a non-zero offset) with a *fresh* one (empty window over
a non-empty buffer). Those are exactly the shapes below.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises an aiter GPU kernel's row-window contract; needs a real GPU",
        allow_module_level=True,
    )

from aiter.ops.topk import top_k_per_row_prefill

DEV = "cuda"
TOPK = 8


def _run(logits, starts, ends, k=TOPK):
    """Return `[rows, k]` int32 top-k column indices for the given windows."""
    rows = logits.shape[0]
    idx = torch.empty((rows, k), dtype=torch.int32, device=DEV)
    top_k_per_row_prefill(
        logits,
        starts,
        ends,
        idx,
        None,
        rows,
        logits.stride(0),
        logits.stride(1),
        k=k,
    )
    return idx


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEV)


def test_row_never_selects_a_column_before_its_start():
    """The decisive case: a predecessor's columns hold the largest logits.

    A resumed chunk sits first in the batch and owns columns `[0, 128)`; a
    second sequence owns `[128, 200)`. If the kernel scanned from column 0
    instead of from `rowStarts`, the second sequence's rows would index the
    first request's compressed keys — cross-request contamination that no
    single-sequence test can produce.
    """
    width, split = 200, 128
    logits = torch.full((4, width), -1.0, dtype=torch.float32, device=DEV)
    logits[:, :split] = 1000.0  # predecessor's columns: the largest in the buffer
    logits[:, split:] = torch.linspace(0.0, 1.0, width - split, device=DEV)

    starts = _i32([split] * 4)
    ends = _i32([width] * 4)
    idx = _run(logits, starts, ends)

    assert (idx >= 0).all(), "window is wider than k, so no -1 sentinel is due"
    assert int(idx.min()) >= split, (
        f"selected column {int(idx.min())} < rowStart {split} — the kernel "
        f"reached back into the previous sequence's committed K"
    )


def test_the_same_rows_do_select_those_columns_when_they_own_them():
    """Arms the test above: with `rowStart=0` the big columns ARE chosen.

    Without this, `test_row_never_selects_a_column_before_its_start` would
    also pass against a kernel that selects nothing at all.
    """
    width, split = 200, 128
    logits = torch.full((4, width), -1.0, dtype=torch.float32, device=DEV)
    logits[:, :split] = 1000.0
    logits[:, split:] = torch.linspace(0.0, 1.0, width - split, device=DEV)

    idx = _run(logits, _i32([0] * 4), _i32([width] * 4))
    assert int(idx.max()) < split, (
        "with the whole buffer in range the 1000.0 columns must win; they "
        "did not, so the harness cannot detect a back-reach either"
    )


def test_empty_window_over_a_non_empty_buffer_yields_only_sentinels():
    """A fresh prompt batched behind a resumed one: `start == end`.

    A first-chunk prompt has no committed K, so its rows' window is empty
    while `total_committed > 0` because a batch-mate contributed. Every
    output cell must be the -1 sentinel; anything else is another request's
    column.
    """
    width = 192
    logits = torch.full((3, width), 500.0, dtype=torch.float32, device=DEV)

    idx = _run(logits, _i32([width] * 3), _i32([width] * 3))
    assert (idx < 0).all(), (
        f"empty window still returned columns {idx[idx >= 0].tolist()[:8]} — "
        f"a fresh sequence would attend to a batch-mate's compressed keys"
    )


def test_window_shorter_than_k_pads_with_sentinels_not_neighbours():
    """A short window must pad with -1, not spill into adjacent columns."""
    width, start, end = 200, 128, 131  # 3 valid columns, k = 8
    logits = torch.full((2, width), 500.0, dtype=torch.float32, device=DEV)
    logits[:, start:end] = torch.tensor([1.0, 2.0, 3.0], device=DEV)

    idx = _run(logits, _i32([start] * 2), _i32([end] * 2))
    valid = idx[idx >= 0]
    assert valid.numel() == 2 * (end - start), (
        f"expected exactly {end - start} real columns per row, got "
        f"{valid.numel() // 2}"
    )
    assert int(valid.min()) >= start and int(valid.max()) < end


def test_seq_local_topk_is_independent_of_what_precedes_it_in_the_batch():
    """The property the e2e bug would violate: batch-composition invariance.

    Sequence S's own scores are fixed. Run it once alone (`seq_base = 0`) and
    once behind a 128-column batch-mate whose logits are larger. After the
    caller's `- seq_base` remap the two must agree column for column.
    """
    own = torch.tensor(
        [[3.0, 9.0, 1.0, 7.0, 5.0, 2.0, 8.0, 4.0, 6.0, 0.5]],
        dtype=torch.float32,
        device=DEV,
    )
    n_own = own.shape[1]

    alone = _run(own.clone(), _i32([0]), _i32([n_own]), k=4)

    base = 128
    padded = torch.full((1, base + n_own), 1000.0, dtype=torch.float32, device=DEV)
    padded[:, base:] = own
    behind = _run(padded, _i32([base]), _i32([base + n_own]), k=4)

    assert torch.equal(alone, behind - base), (
        f"same sequence, different batch-mate: alone={alone.tolist()} "
        f"behind={(behind - base).tolist()}"
    )
