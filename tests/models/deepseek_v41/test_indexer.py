# SPDX-License-Identifier: MIT
"""The contract the paged index scorer rests on: one id per visible row."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tile", [8, 16])
def test_written_plane_scores_as_if_it_were_never_shuffled(tile):
    """`write_index_rows` puts a row where `pa_mqa_logits` looks for it.

    Judged by scoring the plane the writer produced and comparing against the
    same product computed from the unshuffled keys -- so neither side restates
    the other's addressing, which is the only way a placement test can fail
    honestly. `tile` 8 and 16 shuffle in groups of different length, and 8 was
    unreachable before aiter admitted a page below the MFMA tile.
    """
    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    from atom.model_ops.deepseek_v41 import index_write

    torch.manual_seed(20260923)
    rows, heads, dim, per_page = 512, 32, 128, 256
    plane = torch.zeros(2, per_page, dim + 4, dtype=torch.uint8, device="cuda")
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    plan = torch.zeros(rows, 4, dtype=torch.int32, device="cuda")
    plan[:, 2] = torch.arange(rows, device="cuda")
    keys = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda")
    index_write.write_index_rows(
        keys,
        plane,
        plan,
        table,
        per_page,
        ratio=1,
        rows_per_block=tile,
        scale_fmt="fp32",
    )

    query = torch.randn(1, 1, heads, dim, dtype=torch.bfloat16, device="cuda")
    q_fp8 = query.to(torch.float8_e4m3fn)
    weights = torch.rand(1, heads, dtype=torch.float32, device="cuda")
    logits = torch.empty(1, rows, dtype=torch.float32, device="cuda")
    units = plane.view(-1, tile, dim + 4).shape[0]
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        plane.view(-1, tile, 1, dim + 4),
        weights,
        logits,
        torch.tensor([rows], dtype=torch.int32, device="cuda"),
        torch.arange(units, dtype=torch.int32, device="cuda").unsqueeze(0),
        rows,
        KVBlockSize=tile,
        Preshuffle=True,
    )

    # The writer's own quantization, restated on the keys rather than read back
    # out of the plane: one fp32 scale per row, the row being the block.
    scale = (keys.float().abs().amax(-1, keepdim=True).clamp_min(1e-4)) / 448.0
    stored = (keys.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    per_head = torch.einsum(
        "hd,rd->hr", q_fp8.reshape(heads, dim).float(), stored.float() * scale
    )
    expected = (weights.reshape(heads, 1) * per_head.clamp_min(0.0)).sum(0)
    torch.testing.assert_close(logits[0], expected, rtol=2e-2, atol=2e-2)


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
        # A width with -inf inside it: every visible row still reachable, but
        # most of them scored to -inf. The scorer leaves columns that way
        # wherever a chunk runs past a row's bound, so the selector's count has
        # to follow `min(visible, topk)` and not the count of finite scores.
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
    # One length throughout: the plane is paged at what the model picks
    # candidates in, which is what lets the candidate list be a block table.
    block = 8
    write_index_rows(
        keys,
        plane,
        plan,
        table,
        per_page,
        ratio=1,
        rows_per_block=block,
        scale_fmt="fp32",
    )
    tiles = unit_table(
        table, torch.zeros(rows, dtype=torch.int32, device="cuda"), per_page // block
    )
    query = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, heads, dtype=torch.float32, device="cuda")
    visible = torch.tensor(
        [0, 1, 33, 64, 129, 511, 512], device="cuda", dtype=torch.int32
    )
    args = (query, weights, plane.view(-1, block, dim + 4), tiles, visible)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_scoring_inside_the_candidates_picks_what_masking_the_full_width_picked():
    """The compacted block table against the path it replaces.

    `restrict_to_candidates_reference` scores every column and drives the ones
    outside the kept blocks to `-inf`; the compacted table scores only the kept
    blocks. The two are the same selection by construction -- a row the mask
    removed is a row the narrow table never addresses -- so anything but
    bit-identical ids means the compaction lost or reordered something, which a
    score comparison would let through as "close enough".
    """
    from atom.model_ops.deepseek_v41 import paged_scoring as scoring
    from atom.model_ops.deepseek_v41.candidate_table import lift_candidate_selection
    from atom.model_ops.deepseek_v41.index_write import write_index_rows
    from atom.model_ops.deepseek_v41.indexer import restrict_to_candidates_reference
    from atom.model_ops.deepseek_v41.unit_table import unit_table

    torch.manual_seed(5150)
    rows, width, heads, dim, per_page, block = 7, 512, 32, 128, 128, 8
    plane = torch.empty(4, per_page, dim + 4, dtype=torch.uint8, device="cuda")
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda")
    plan = torch.zeros(width, 4, dtype=torch.int32, device="cuda")
    plan[:, 2] = torch.arange(width, device="cuda")
    keys = torch.randn(width, dim, dtype=torch.bfloat16, device="cuda")
    write_index_rows(
        keys,
        plane,
        plan,
        table,
        per_page,
        ratio=1,
        rows_per_block=block,
        scale_fmt="fp32",
    )
    tiles = unit_table(
        table, torch.zeros(rows, dtype=torch.int32, device="cuda"), per_page // block
    )
    query = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, heads, dtype=torch.float32, device="cuda")
    # A row with nothing to see, rows inside one block, rows on and off a block
    # boundary, and a full row -- the partial newest block is what the compacted
    # bound has to get right, and it is only partial off the boundary.
    visible = torch.tensor([0, 1, 33, 64, 129, 511, 512], device="cuda")
    visible = visible.to(torch.int32)
    args = (query, weights, plane.view(-1, block, dim + 4), tiles, visible)
    # The model's candidate length, fixed; `block` is the plane's.
    kwargs = {"topk": 64, "weights_scale": (heads * dim) ** -0.5, "block_size": 8}

    _, candidates = scoring.score_topk_paged(*args, **kwargs, candidate_count=16)
    actual, _ = scoring.score_topk_paged(*args, **kwargs, candidates=candidates)

    # The path this replaces, assembled from the same pieces: full width, the
    # mask, then the same selector.
    from aiter.ops.topk import top_k_per_row_decode

    q_fp8, q_scale = scoring.quantize_query_rows(query)
    scaled = scoring.scale_indexer_weights(
        weights.contiguous(), q_scale.view(rows, heads, 1), kwargs["weights_scale"]
    )
    scores = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    scoring.deepgemm_fp8_paged_mqa_logits(
        q_fp8.view(rows, 1, heads, dim),
        plane.view(-1, block, dim + 4).unsqueeze(-2),
        scaled,
        scores,
        visible,
        tiles,
        width,
        KVBlockSize=block,
        Preshuffle=True,
    )
    restrict_to_candidates_reference(scores, candidates, visible, kwargs["block_size"])
    expected = torch.empty(rows, kwargs["topk"], dtype=torch.int32, device="cuda")
    top_k_per_row_decode(
        scores,
        1,
        visible,
        expected,
        rows,
        scores.stride(0),
        scores.stride(1),
        k=kwargs["topk"],
        stable=True,
    )
    assert torch.equal(actual, expected), (actual, expected)

    # Armed: the ids leaving the selector are columns of the narrow width, so a
    # missing lift would agree with the real rows only where a row kept its very
    # first blocks.
    raw, _ = scoring.score_topk_paged(*args, **kwargs, candidates=candidates)
    twice = raw.clone()
    lift_candidate_selection(twice, candidates, rows_per_block=block)
    assert not torch.equal(twice, raw)

    # A plane paged at anything but the candidate length says so rather than
    # scoring the wrong rows: the ids would be read at one length and the
    # blocks addressed at another, which is a wrong answer and not a fault.
    mismatched = (query, weights, plane.view(-1, 16, dim + 4), tiles, visible)
    with pytest.raises(AssertionError, match="cannot be a block table"):
        scoring.score_topk_paged(*mismatched, **kwargs, candidates=candidates)


def _picked_blocks_reference(logits, visible, block_size, keep):
    """Candidate blocks by sorting, the definition rather than a restatement.

    A block's score is its best visible column, the newest block outranks every
    other, ties go to the smaller id, and the kept ids come back ascending.
    """
    logits, visible = logits.cpu(), visible.cpu().long()
    rows, width = logits.shape
    past = torch.arange(width) >= visible[:, None]
    scores = logits.masked_fill(past, float("-inf"))
    scores = scores.view(rows, -1, block_size).amax(-1)
    ids = torch.arange(scores.shape[1])
    last = (visible[:, None] + block_size - 1) // block_size
    scores = scores.masked_fill(ids == last - 1, float("inf"))
    # A stable descending sort puts equal scores in ascending id order.
    best = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :keep]
    # Blocks past a row's end were never candidates; they sort last as padding.
    best = torch.where(best < last, best, scores.shape[1]).sort(dim=1).values
    return torch.where(best < scores.shape[1], best, -1).to(torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("rows", [48, 4096])
def test_candidate_blocks_come_from_the_visible_prefix_alone(rows):
    """The picked blocks match their definition on rows far narrower than the table.

    Columns past a row's visibility hold `+inf`, and the table is wider than any
    row's context -- the shape serving runs, where the table is `max_model_len`
    wide. `rows` 4096 leaves each lane several tiles to walk, 48 one each.
    """
    from atom.model_ops.deepseek_v41.indexer import pick_candidate_blocks

    torch.manual_seed(11)
    block_size, keep, width = 8, 16, 4096
    edges = [0, 1, 7, 8, 9, keep * block_size, keep * block_size + 1, width]
    visible = torch.randint(0, width + 1, (rows,), dtype=torch.int32)
    visible[: len(edges)] = torch.tensor(edges, dtype=torch.int32)
    visible = visible.cuda()
    # Few distinct values, so ties are common and the tie rule is exercised.
    logits = torch.randint(0, 4, (rows, width), device="cuda").float()
    past = torch.arange(width, device="cuda")[None, :] >= visible[:, None]
    logits = logits.masked_fill(past, float("inf"))
    out = torch.full((rows, keep), 12345, dtype=torch.int32, device="cuda")

    pick_candidate_blocks(logits, visible, block_size, out, tile=16)

    blocks = (visible + block_size - 1) // block_size
    # Armed: rows that keep everything and rows that must choose both occur.
    assert (blocks <= keep).any() and (blocks > keep).any()
    assert torch.equal(
        out.cpu(), _picked_blocks_reference(logits, visible, block_size, keep)
    )


def _candidate_fixture(rows_per_block, seed=7):
    """Ragged kept-lists over a shared plane, including a row that kept none.

    `visible` is chosen so the newest block is partial on some rows and exact
    on others: the compacted bound is a single scalar only because the partial
    block is always the last kept one, and a row whose newest block is full
    would not tell those two apart.
    """
    torch.manual_seed(seed)
    topk_blocks, tokens, units = 16, 6, 160
    tiles = torch.randperm(4096, device="cuda")[:units].to(torch.int32)
    tiles = tiles.unsqueeze(0).repeat(tokens, 1)
    visible = torch.tensor(
        [1, 37, 8, 256, units * rows_per_block, 0], dtype=torch.int32, device="cuda"
    )
    cand = torch.full((tokens, topk_blocks), -1, dtype=torch.int32, device="cuda")
    for token in range(tokens):
        seen = int(visible[token])
        if not seen:
            continue
        newest = (seen - 1) // rows_per_block
        pool = torch.randperm(newest + 1)[: topk_blocks - 1].tolist()
        keep = sorted(set(pool) | {newest})[:topk_blocks]
        cand[token, : len(keep)] = torch.tensor(keep, dtype=torch.int32, device="cuda")
    return cand, tiles, visible


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_candidate_block_table_matches_its_reference_and_covers_what_it_claims():
    """The gather against the loop, and the bound against what it stands for.

    The bound gets a second judge rather than a restatement: it has to equal
    the visible rows the kept blocks actually cover, summed block by block. A
    reference that recomputed `(kept - 1) * rows + ...` would agree with the
    kernel while both were wrong about the partial block.
    """
    from atom.model_ops.deepseek_v41.candidate_table import (
        candidate_block_table,
        candidate_block_table_reference,
    )

    rows_per_block = 8
    cand, tiles, visible = _candidate_fixture(rows_per_block)
    # Armed: a fixture where every row keeps every block, or none is partial,
    # would pass a bound that ignored either term.
    assert (cand < 0).any() and (cand >= 0).any()

    table, context = candidate_block_table(
        cand, tiles, visible, rows_per_block=rows_per_block
    )
    expected_table, expected_context = candidate_block_table_reference(
        cand, tiles, visible, rows_per_block=rows_per_block
    )
    assert torch.equal(table, expected_table)
    assert torch.equal(context, expected_context)
    for token in range(cand.shape[0]):
        kept = [c for c in cand[token].tolist() if c >= 0]
        covered = sum(
            min(rows_per_block, int(visible[token]) - c * rows_per_block) for c in kept
        )
        assert int(context[token]) == covered, (token, int(context[token]), covered)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_lifting_a_selection_restores_real_rows_and_keeps_them_ascending():
    """Compacted columns back to compressed rows, order intact.

    Order is the half that has no other enforcer: `build_indices` reads the
    ids as an ascending prefix, and nothing downstream re-sorts them.
    """
    from atom.model_ops.deepseek_v41.candidate_table import (
        lift_candidate_selection,
        lift_candidate_selection_reference,
    )

    rows_per_block, topk = 8, 24
    cand, _, _ = _candidate_fixture(rows_per_block)
    tokens = cand.shape[0]
    selected = torch.full((tokens, topk), -1, dtype=torch.int32, device="cuda")
    for token in range(tokens):
        kept = [c for c in cand[token].tolist() if c >= 0]
        if not kept:
            continue
        columns = sorted(torch.randperm(len(kept) * rows_per_block)[:topk].tolist())
        selected[token, : len(columns)] = torch.tensor(
            columns, dtype=torch.int32, device="cuda"
        )

    lifted = selected.clone()
    lift_candidate_selection(lifted, cand, rows_per_block=rows_per_block)
    expected = selected.clone()
    lift_candidate_selection_reference(expected, cand, rows_per_block=rows_per_block)
    assert torch.equal(lifted, expected)
    # Armed: the fixture's kept blocks are not contiguous from zero, so a lift
    # that left the columns alone would fail the identity below as well as order.
    assert not torch.equal(lifted, selected)
    for token in range(tokens):
        live = [v for v in lifted[token].tolist() if v >= 0]
        assert live == sorted(live), (token, live)
        assert all(v >= 0 for v in live)
