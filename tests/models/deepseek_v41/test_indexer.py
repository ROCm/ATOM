# SPDX-License-Identifier: MIT
"""CSA2 index selection, candidate boundaries, and configurable score ties."""

import pytest
import torch

from atom.model_ops.deepseek_v41.indexer import select_indices


@pytest.mark.parametrize("width", [0, 1, 7, 17, 67])
def test_tiled_indexer_candidates_and_reindex_match_dense_oracle(reference, width):
    torch.manual_seed(826)
    batch, queries, heads, dim = 2, 9, 4, 32
    q = torch.rand(batch, queries, heads, dim)
    keys = torch.rand(batch, width, dim)
    weights = torch.rand(batch, queries, heads)
    visible = torch.linspace(0, width, queries).long()
    scores = (torch.einsum("bqhd,bkd->bqhk", q, keys).relu() * weights[..., None]).sum(
        2
    )
    scores.masked_fill_(torch.arange(width) >= visible[:, None], -torch.inf)
    actual, blocks = select_indices(
        q,
        weights,
        keys,
        visible,
        topk=5,
        make_candidates=True,
        block_size=8,
        topk_blocks=2,
        query_tile=3,
        key_tile=16,
    )
    if not width:
        assert actual.shape == (batch, queries, 0)
        assert blocks.shape == (batch, queries, 0)
        return
    expected = scores.topk(min(5, width), -1, sorted=False)
    expected_ids = (
        expected.indices.masked_fill(~torch.isfinite(expected.values), width)
        .sort(-1)
        .values
    )
    expected_ids.masked_fill_(expected_ids == width, -1)
    assert torch.equal(actual, expected_ids)
    dense_mask = reference.select_candidate_blocks(scores, visible[:, None], 2, 8)
    positions = torch.arange(width)
    actual_mask = ((positions[None, None, :, None] // 8) == blocks[..., None, :]).any(
        -1
    )
    assert torch.equal(actual_mask, dense_mask)
    # Reindex scores only gathered candidates, including partial/newest blocks.
    reindex, _ = select_indices(
        q,
        weights,
        keys,
        visible,
        topk=5,
        candidate_blocks=blocks,
        block_size=8,
        query_tile=3,
        key_tile=16,
    )
    expected = scores.masked_fill(~dense_mask, -torch.inf).topk(
        min(5, width), -1, sorted=False
    )
    expected_ids = (
        expected.indices.masked_fill(~torch.isfinite(expected.values), width)
        .sort(-1)
        .values
    )
    expected_ids.masked_fill_(expected_ids == width, -1)
    assert torch.equal(reindex, expected_ids)


@pytest.mark.parametrize("candidates", [False, True])
def test_selection_count_matches_its_closed_form(candidates):
    """How many ids a row returns is `min(visible, columns)`, not a count.

    `_indptr_scan` reserves exactly that many slots ahead of the scorer, and
    the window segment is written at the other end of the same slice. A row
    that returned fewer would leave the difference between them unwritten, and
    the attention kernel reads whatever is in it.

    This is the tiled scorer's half; the paged one is below.
    """
    # A layer with no candidate source takes the fused top-k, which is a GPU
    # kernel; the candidate arm is the one that runs everywhere.
    device = "cpu" if candidates else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    torch.manual_seed(517)
    batch, queries, heads, dim, width, topk = 2, 6, 4, 32, 48, 5
    q = torch.rand(batch, queries, heads, dim, device=device)
    keys = torch.rand(batch, width, dim, device=device)
    weights = torch.rand(batch, queries, heads, device=device)
    # Both sides of the `min`: rows below the top-k and rows far above it.
    visible = torch.tensor([0, 1, 4, 5, 9, width], device=device)
    blocks = None
    if candidates:
        blocks = select_indices(
            q,
            weights,
            keys,
            visible,
            topk=topk,
            make_candidates=True,
            block_size=8,
            topk_blocks=6,
        )[1]
    selected, _ = select_indices(
        q,
        weights,
        keys,
        visible,
        topk=topk,
        candidate_blocks=blocks,
        block_size=8,
        query_tile=3,
        key_tile=16,
    )
    expected = visible.clamp(max=selected.shape[-1]).expand(batch, -1)
    assert torch.equal((selected >= 0).sum(-1), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("masked", [False, True])
def test_paged_top_k_returns_one_id_per_visible_row(masked):
    """The other half, and the one a captured decode step runs.

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("tie_break", ["small_position", "large_position"])
@pytest.mark.parametrize("key_tile", [8, 16, 32])
def test_indexer_zero_score_ties_follow_position_policy(key_tile, tie_break, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    q = torch.zeros(1, 2, 4, 32, device=device)
    keys = torch.ones(1, 33, 32, device=device)
    weights = torch.ones(1, 2, 4, device=device)
    visible = torch.tensor([3, 33], device=device)
    indices, blocks = select_indices(
        q,
        weights,
        keys,
        visible,
        topk=5,
        make_candidates=True,
        block_size=8,
        topk_blocks=2,
        key_tile=key_tile,
        tie_break=tie_break,
    )
    chosen = [28, 29, 30, 31, 32] if tie_break == "large_position" else [0, 1, 2, 3, 4]
    chosen_blocks = [3, 4] if tie_break == "large_position" else [0, 4]
    assert torch.equal(indices.cpu(), torch.tensor([[[0, 1, 2, -1, -1], chosen]]))
    assert torch.equal(blocks.cpu(), torch.tensor([[[0, -1], chosen_blocks]]))
    reindex, _ = select_indices(
        q,
        weights,
        keys,
        visible,
        topk=5,
        candidate_blocks=blocks,
        block_size=8,
        key_tile=key_tile,
        tie_break=tie_break,
    )
    assert torch.equal(reindex, indices)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("tie_break", ["small_position", "large_position"])
def test_indexer_score_precedes_position_for_full_and_reindex(device, tie_break):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    # Integer scores make ties exact. Position must only break equal scores.
    scores = [position % 4 for position in range(33)]
    q = torch.zeros(1, 3, 1, 32, device=device)
    q[..., 0] = 1
    keys = torch.zeros(1, 33, 32, device=device)
    keys[0, :, 0] = torch.tensor(scores, device=device)
    weights = torch.ones(1, 3, 1, device=device)
    lengths = [0, 3, 33]
    visible = torch.tensor(lengths, device=device)
    direction = -1 if tie_break == "large_position" else 1

    def best_positions(positions):
        selected = sorted(positions, key=lambda p: (-scores[p], direction * p))[:5]
        return sorted(selected) + [-1] * (5 - len(selected))

    expected_indices = [best_positions(range(length)) for length in lengths]
    expected_blocks = []
    expected_reindex = []
    for length in lengths:
        blocks = list(range((length + 7) // 8))
        newest = blocks[-1] if blocks else -1
        priority = lambda b, newest=newest, length=length: (
            (
                -float("inf")
                if b == newest
                else -max(scores[b * 8 : min((b + 1) * 8, length)])
            ),
            direction * b,
        )
        selected = sorted(sorted(blocks, key=priority)[:2])
        expected_blocks.append(selected + [-1] * (2 - len(selected)))
        expected_reindex.append(
            best_positions(
                position
                for block in selected
                for position in range(block * 8, min((block + 1) * 8, length))
            )
        )

    for key_tile in (8, 16, 64):
        indices, blocks = select_indices(
            q,
            weights,
            keys,
            visible,
            topk=5,
            make_candidates=True,
            block_size=8,
            topk_blocks=2,
            key_tile=key_tile,
            tie_break=tie_break,
        )
        reindex, _ = select_indices(
            q,
            weights,
            keys,
            visible,
            topk=5,
            candidate_blocks=blocks,
            block_size=8,
            key_tile=key_tile,
            tie_break=tie_break,
        )
        assert indices.cpu().tolist() == [expected_indices]
        assert blocks.cpu().tolist() == [expected_blocks]
        assert reindex.cpu().tolist() == [expected_reindex]
