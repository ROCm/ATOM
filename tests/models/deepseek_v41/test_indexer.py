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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("units", [8, 16])
def test_unit_table_expands_the_page_table_in_place(units):
    """The kernel against the torch body it replaced, padding rows included.

    The batch is ragged, skips a request and is padded past its own end, so a
    token's row depends on the batch id it carries rather than on where it
    sits. Padding rows come back zeroed: the load is masked, which is what
    lets the kernel read no PAGE table for a request that is not there.
    """
    from atom.model_ops.deepseek_v41.unit_table import unit_table

    from .reference_unit_table import unit_table_reference

    torch.manual_seed(1409)
    bs, columns = 5, 7
    block_tables = torch.randint(
        0, 4096, (bs, columns), dtype=torch.int32, device="cuda"
    )
    batch_ids = torch.tensor(
        [0, 0, 0, 1, 2, 2, 4, 4, -1, -1, -1], dtype=torch.int32, device="cuda"
    )
    # Armed: with no padding row the mask goes untested, and with a uniform
    # batch every row would gather the same PAGEs whatever the id said.
    assert (batch_ids < 0).any() and len(set(batch_ids.tolist())) > 3

    actual = unit_table(block_tables, batch_ids, units)
    assert actual.shape == (batch_ids.numel(), columns * units)
    assert actual.dtype == torch.int32
    assert torch.equal(actual, unit_table_reference(block_tables, batch_ids, units))

    # The reference indexes with torch and takes any view, so a kernel that
    # assumed unit stride would agree with it above and disagree only here.
    strided = torch.stack((batch_ids.flip(0), batch_ids), dim=1)[:, 1]
    assert not strided.is_contiguous() and torch.equal(strided, batch_ids)
    assert torch.equal(
        unit_table(block_tables, strided, units),
        unit_table_reference(block_tables, strided, units),
    )

    # A zero-token forward still owes its caller the shape it will index.
    empty = torch.empty(0, dtype=torch.int32, device="cuda")
    assert unit_table(block_tables, empty, units).shape == (0, columns * units)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_banded_score_plane_preserves_full_and_reindex_selection(monkeypatch):
    from atom.model_ops.deepseek_v41 import paged_scoring as scoring
    from atom.model_ops.deepseek_v41.index_write import write_index_rows
    from atom.model_ops.deepseek_v41.unit_table import unit_table

    torch.manual_seed(1921)
    rows, width, heads, dim, per_page = 7, 512, 32, 128, 128
    plane = torch.empty(4, per_page, dim + 4, dtype=torch.uint8, device="cuda")
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda")
    plan = torch.zeros(width, 4, dtype=torch.int32, device="cuda")
    plan[:, 2] = torch.arange(width, device="cuda")
    keys = torch.randn(width, dim, dtype=torch.bfloat16, device="cuda")
    write_index_rows(keys, plane, plan, table, per_page, ratio=1, scale_fmt="fp32")
    tiles = unit_table(table, torch.zeros(rows, dtype=torch.int32, device="cuda"), 8)
    query = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, heads, dtype=torch.float32, device="cuda")
    visible = torch.tensor(
        [0, 1, 33, 64, 129, 511, 512], device="cuda", dtype=torch.int32
    )
    args = (query, weights, plane.view(-1, 16, dim + 4), tiles, visible)
    kwargs = {"topk": 64, "weights_scale": (heads * dim) ** -0.5}
    full, candidates = scoring.score_topk_paged(*args, **kwargs, candidate_count=16)
    reindex, _ = scoring.score_topk_paged(*args, **kwargs, candidates=candidates)
    real_score = scoring.deepgemm_fp8_paged_mqa_logits
    sizes = []

    def observe(*args, **kwargs):
        scores = args[3]
        sizes.append(scores.numel() * scores.element_size())
        return real_score(*args, **kwargs)

    # Three rows per band, including a short last band. Verify through the
    # real quantized paged scorer and selector, not a mocked score function.
    original_plane_rows = scoring.plane_rows
    monkeypatch.setattr(scoring, "plane_rows", lambda width: 3)
    monkeypatch.setattr(scoring, "deepgemm_fp8_paged_mqa_logits", observe)
    actual, chosen = scoring.score_topk_paged(*args, **kwargs, candidate_count=16)
    actual_reindex, _ = scoring.score_topk_paged(*args, **kwargs, candidates=chosen)
    torch.testing.assert_close(actual, full, rtol=0, atol=0)
    torch.testing.assert_close(chosen, candidates, rtol=0, atol=0)
    torch.testing.assert_close(actual_reindex, reindex, rtol=0, atol=0)
    assert len(sizes) == 6 and max(sizes) <= 3 * width * 4
    assert torch.all(actual[0] == -1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_full, replay_candidates = scoring.score_topk_paged(
            *args, **kwargs, candidate_count=16
        )
        replay_reindex, _ = scoring.score_topk_paged(
            *args, **kwargs, candidates=replay_candidates
        )
    visible.copy_(visible.flip(0))
    graph.replay()
    # Compare the captured bands against an unbanded forward after live bounds
    # change. The empty row moves to the end and must not retain stale IDs.
    monkeypatch.setattr(scoring, "plane_rows", original_plane_rows)
    expected, expected_candidates = scoring.score_topk_paged(
        *args, **kwargs, candidate_count=16
    )
    expected_reindex, _ = scoring.score_topk_paged(
        *args, **kwargs, candidates=expected_candidates
    )
    torch.testing.assert_close(replay_full, expected, rtol=0, atol=0)
    torch.testing.assert_close(replay_candidates, expected_candidates, rtol=0, atol=0)
    torch.testing.assert_close(replay_reindex, expected_reindex, rtol=0, atol=0)
