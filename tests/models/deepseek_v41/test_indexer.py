# SPDX-License-Identifier: MIT
"""The contract the paged index scorer rests on: one id per visible row."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("masked", [False, True])
def test_paged_top_k_returns_one_id_per_visible_row(masked):
    """The selection a captured decode step runs.

    `_indptr_scan` reserves `min(visible, k)` slots per row for this kernel's
    output. Two things could make it emit fewer and leave the difference
    unwritten: a row shorter than `k`, which it documents padding with -1, and
    a row whose surviving scores are `-inf` because the candidate mask removed
    the rest. The second is the one nothing states, and the sparse attention
    kernel is called with `has_invalid=False` -- it dereferences every slot in
    the range the indptr claims.

    Columns past `visible` are left uninitialized on purpose: that is what the
    paged scorer hands over, since its kernel returns before writing them.
    """
    from aiter.ops.topk import top_k_per_row_decode

    rows, width, topk = 6, 512, 64
    visible = torch.tensor([1, 7, 63, 64, 65, width], dtype=torch.int32, device="cuda")
    logits = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    torch.manual_seed(311)
    for row, count in enumerate(visible.tolist()):
        logits[row, :count] = torch.randn(count, device="cuda")
    if masked:
        # What `restrict_to_candidates` leaves behind: every visible row still
        # reachable, but through scores the mask drove to -inf outside a
        # handful of blocks. Keep more than `topk` of them so the count is
        # still bounded by `min(visible, topk)` and not by the mask.
        keep = 128
        for row, count in enumerate(visible.tolist()):
            if count > keep:
                logits[row, keep:count] = -torch.inf
    selected = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    top_k_per_row_decode(
        logits,
        1,
        visible,
        selected,
        rows,
        logits.stride(0),
        logits.stride(1),
        k=topk,
        stable=True,
    )
    expected = visible.clamp(max=topk).to(torch.int64)
    assert torch.equal((selected >= 0).sum(-1), expected), (
        f"visible={visible.tolist()} k={topk} masked={masked} "
        f"got={(selected >= 0).sum(-1).tolist()}"
    )
