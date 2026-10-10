# SPDX-License-Identifier: MIT
"""The contract the paged index scorer rests on: one id per visible row."""

import itertools

import pytest
import torch

from atom.model_ops.deepseek_v41.index_plane import IndexUnits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), 1.0])
def test_nonfinite_scores_keep_candidate_context_inside_allocation(score):
    """A short prefill must keep its newest block even when scores are nonfinite.

    With NaN or +inf scores the old selector picked blocks 0..2047, dropping
    block 5312. The consumer then read 42497 columns from a 16384-column
    candidate allocation, causing the paged-logits GPU access fault.
    """
    from atom.model_ops.deepseek_v41.candidate_table import candidate_block_table
    from atom.model_ops.deepseek_v41.indexer import pick_candidate_blocks

    rows, width, block_size, kept = 3, 65536, 8, 2048
    visible = torch.tensor([42497, 42498, 42499], dtype=torch.int32, device="cuda")
    logits = torch.full((rows, width), score, device="cuda")
    candidates = torch.empty(rows, kept, dtype=torch.int32, device="cuda")
    # every row its own request, its PAGEs one block each, in order
    pages = torch.arange(width // block_size, dtype=torch.int32, device="cuda")
    pages = pages.repeat(rows, 1)
    batch_ids = torch.arange(rows, dtype=torch.int32, device="cuda")
    pick_candidate_blocks(logits, visible, block_size, candidates)
    table, bound = candidate_block_table(
        candidates, pages, batch_ids, 1, visible, rows_per_block=block_size
    )

    torch.testing.assert_close(candidates[:, -1], (visible - 1) // block_size)
    assert bool((candidates[:, 1:] > candidates[:, :-1]).all())
    assert bool((bound > (kept - 1) * block_size).all())
    assert bool((bound <= table.shape[1] * block_size).all())


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
    from atom.model_ops.deepseek_v41.candidate_table import bind_candidates
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
    args = (query, weights, IndexUnits(plane.view(-1, block, dim + 4)), tiles, visible)
    kwargs = {"topk": 64, "weights_scale": (heads * dim) ** -0.5}

    def bind(ids):
        batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
        return bind_candidates(
            ids, table, batch, per_page // block, visible, rows_per_block=block
        )

    full, candidates = scoring.score_topk_paged(*args, **kwargs, candidate_count=16)
    reindex, _ = scoring.score_topk_paged(*args, **kwargs, candidates=bind(candidates))
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
    actual_reindex, _ = scoring.score_topk_paged(
        *args, **kwargs, candidates=bind(chosen)
    )
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
            *args, **kwargs, candidates=bind(replay_candidates)
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
        *args, **kwargs, candidates=bind(expected_candidates)
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
    from atom.model_ops.deepseek_v41.candidate_table import (
        bind_candidates,
        lift_candidate_selection,
    )
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
    args = (query, weights, IndexUnits(plane.view(-1, block, dim + 4)), tiles, visible)
    # The model's candidate length, fixed; `block` is the plane's.
    kwargs = {"topk": 64, "weights_scale": (heads * dim) ** -0.5, "block_size": 8}

    _, candidates = scoring.score_topk_paged(*args, **kwargs, candidate_count=16)
    batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    kept = bind_candidates(
        candidates, table, batch, per_page // block, visible, rows_per_block=block
    )
    actual, _ = scoring.score_topk_paged(*args, **kwargs, candidates=kept)

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
    raw, _ = scoring.score_topk_paged(*args, **kwargs, candidates=kept)
    twice = raw.clone()
    lift_candidate_selection(twice, candidates, rows_per_block=block)
    assert not torch.equal(twice, raw)

    # A plane paged at anything but the candidate length says so rather than
    # scoring the wrong rows: the ids would be read at one length and the
    # blocks addressed at another, which is a wrong answer and not a fault.
    mismatched = (
        query,
        weights,
        IndexUnits(plane.view(-1, 16, dim + 4)),
        tiles,
        visible,
    )
    with pytest.raises(AssertionError, match="cannot be a block table"):
        scoring.score_topk_paged(*mismatched, **kwargs, candidates=kept)


def _picked_blocks_reference(logits, visible, block_size, keep):
    """Candidate blocks by sorting, the definition rather than a restatement.

    A block's score is its best visible column, the newest block outranks every
    other, ties go to the smaller id, and the kept ids come back ascending.
    """
    logits, visible = logits.cpu(), visible.cpu().long()
    logits = torch.nan_to_num(
        logits,
        nan=float("-inf"),
        posinf=torch.finfo(logits.dtype).max,
        neginf=float("-inf"),
    )
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
    """Ragged kept-lists over two requests' PAGE tables (16 blocks a PAGE), a
    padding token (batch -1) that kept none.

    `visible` is chosen so the newest block is partial on some rows and exact
    on others: the compacted bound is a single scalar only because the partial
    block is always the last kept one, and a row whose newest block is full
    would not tell those two apart.

    -> (candidates, block_tables, batch_ids, blocks a PAGE, visible)
    """
    torch.manual_seed(seed)
    topk_blocks, tokens, per_page, columns = 16, 6, 16, 10
    blocks = per_page * columns
    pages = torch.randperm(256, device="cuda")[: 2 * columns].to(torch.int32)
    pages = pages.view(2, columns)
    batch_ids = torch.tensor([0, 1, 0, 1, 0, -1], dtype=torch.int32, device="cuda")
    visible = torch.tensor(
        [1, 37, 8, 256, blocks * rows_per_block, 0], dtype=torch.int32, device="cuda"
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
    return cand, pages, batch_ids, per_page, visible


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
    cand, pages, batch_ids, per_page, visible = _candidate_fixture(rows_per_block)
    # Armed: a fixture where every row keeps every block, or none is partial,
    # would pass a bound that ignored either term.
    assert (cand < 0).any() and (cand >= 0).any()

    args = (cand, pages, batch_ids, per_page, visible)
    table, context = candidate_block_table(*args, rows_per_block=rows_per_block)
    expected_table, expected_context = candidate_block_table_reference(
        *args, rows_per_block=rows_per_block
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
@pytest.mark.parametrize("topk_blocks", [16, 2048])
def test_candidate_context_counts_only_kept_rows_when_newest_block_is_absent(
    topk_blocks,
):
    """Candidate compaction bounds itself even if selection loses the newest block."""
    from atom.model_ops.deepseek_v41.candidate_table import (
        candidate_block_table,
        candidate_block_table_reference,
    )

    block = 8
    selections = [
        list(range(topk_blocks)),
        [0, 7, 511],
        [0, 7, 5312],
        [0, 7, 5311],
        [],
        [],
        [5312],
    ]
    visible = torch.tensor(
        [42499, 42499, 42499, 42496, 42499, 0, 42499], dtype=torch.int32, device="cuda"
    )
    candidates = torch.full(
        (len(selections), topk_blocks), -1, dtype=torch.int32, device="cuda"
    )
    for row, kept in enumerate(selections):
        candidates[row, : len(kept)] = torch.tensor(
            kept, dtype=torch.int32, device="cuda"
        )
    # every row its own request, its PAGEs one block each, in reverse
    pages = torch.arange(8192, dtype=torch.int32, device="cuda").flip(0)
    pages = pages.repeat(len(selections), 1)
    batch_ids = torch.arange(len(selections), dtype=torch.int32, device="cuda")
    args = (candidates, pages, batch_ids, 1, visible)
    candidate_block_table(*args, rows_per_block=block)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        table, context = candidate_block_table(*args, rows_per_block=block)
    for advance in (0, 8192):
        visible.add_(advance)
        graph.replay()
        expected_table, expected_context = candidate_block_table_reference(
            *args, rows_per_block=block
        )
        torch.testing.assert_close(table, expected_table)
        torch.testing.assert_close(context, expected_context)
        # Count actual visible token IDs, independently of the bound formula.
        for row, kept in enumerate(selections):
            seen = int(visible[row])
            covered = [
                pos
                for cand in kept
                for pos in range(cand * block, (cand + 1) * block)
                if pos < seen
            ]
            assert int(context[row]) == len(covered)
            assert 0 <= int(context[row]) <= len(kept) * block


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
    cand = _candidate_fixture(rows_per_block)[0]
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


def _workspace_geometry():
    """Two index ratios with different PAGE widths, so a table sized for one
    ratio would be too small or too large for the other."""
    from types import SimpleNamespace

    return SimpleNamespace(
        owners=((0, 1), (1, 2), (2, 2)),
        rows_per_page=lambda ratio: 64 // ratio,
        index_block_rows=8,
        index_blocks_per_page=lambda ratio: 64 // ratio // 8,
        index_fp4=False,
    )


def test_score_workspace_holds_no_tile_tables_for_the_fp4_plane():
    """The FP4 plane's scorers read the PAGE table itself, so the workspace
    reserves no tile table, and asking for one is refused rather than met."""
    import atom.model_ops.deepseek_v41.score_workspace as ws

    fp8 = ws.ScoreWorkspace(_workspace_geometry(), 40, 9, "cpu")
    fp4 = ws.ScoreWorkspace(_pool_geometry(True), 40, 9, "cpu", pages=1)
    assert fp8.unit_table(1, 40, 9 * 8).numel() == 40 * 9 * 8
    assert not fp4._tiles
    with pytest.raises(ValueError, match="FP4"):
        fp4.unit_table(1, 1, 1)


@pytest.mark.parametrize("max_tokens,columns", [(5, 3), (40, 9)])
def test_score_workspace_holds_every_view_the_scorer_asks_for(max_tokens, columns):
    """The largest view of each kind fits and one element more is refused.

    The workspace is sized once, before the memory profile; a request past it
    would otherwise read beyond the allocation or need a fresh one mid-serving.
    """
    from atom.model_ops.deepseek_v41 import score_workspace as ws

    geometry = _workspace_geometry()
    space = ws.ScoreWorkspace(geometry, max_tokens, columns, "cpu")
    for ratio in (1, 2):
        width = columns * geometry.rows_per_page(ratio) // geometry.index_block_rows
        assert space.unit_table(ratio, max_tokens, width).shape == (max_tokens, width)
        with pytest.raises(ValueError):
            space.unit_table(ratio, max_tokens, width + 1)
        plane = columns * geometry.rows_per_page(ratio)
        band = min(max_tokens, ws.plane_rows(plane))
        assert space.logits(band, plane).shape == (band, plane)
    widest = columns * geometry.rows_per_page(1)
    with pytest.raises(ValueError):
        space.logits(min(max_tokens, ws.plane_rows(widest)) + 1, widest)


def test_score_workspace_band_stops_at_the_int32_reach(monkeypatch):
    """A long-context width bounds the band by `plane_rows`, not by the tokens."""
    from atom.model_ops.deepseek_v41 import score_workspace as ws

    monkeypatch.setattr(ws, "plane_rows", lambda width: 2)
    space = ws.ScoreWorkspace(_workspace_geometry(), 40, 3, "cpu")
    assert space.logits(2, 3 * 64).shape == (2, 3 * 64)
    with pytest.raises(ValueError):
        space.logits(3, 3 * 64)


@pytest.mark.parametrize("seed", range(4))
def test_packed_bands_hold_each_row_aligned_and_fill_before_cutting(seed):
    """Bands partition the rows in order; a row starts on `ROW_ALIGN`, at its
    band's running total; no band is past the capacity, and none could have
    taken the next row."""
    import numpy as np

    from atom.model_ops.deepseek_v41 import score_workspace as ws

    rng = np.random.default_rng(seed)
    visible = rng.integers(0, 700, size=50).astype(np.int32)
    visible[::7] = 0
    capacity = 2048
    offsets = np.full(50, -1, dtype=np.int32)
    bands = ws.band_offsets(visible, capacity, offsets)
    spans = ws.aligned_spans(visible.astype(np.int64))
    assert bands[0][0] == 0 and bands[-1][1] == 50
    assert all(a[1] == b[0] for a, b in itertools.pairwise(bands))
    for first, end in bands:
        within = spans[first:end]
        assert np.array_equal(offsets[first:end], np.cumsum(within) - within)
        assert int(within.sum()) <= capacity
        if end < 50:
            assert int(within.sum() + spans[end]) > capacity
    assert not (offsets % ws.ROW_ALIGN).any()
    with pytest.raises(ValueError, match="exceeds"):
        ws.band_offsets(np.array([capacity + 1], np.int32), capacity, offsets)


def _pool_geometry(index_fp4, speculative_tokens=5):
    from types import SimpleNamespace

    return SimpleNamespace(
        **{**vars(_workspace_geometry()), "index_fp4": index_fp4},
        speculative_tokens=speculative_tokens,
    )


def test_packed_capacity_follows_the_pool_up_to_the_plane():
    """Every decode row of a full pool at the widest ratio, its alignment
    for every token row, and never more than the plane it replaces."""
    from atom.model_ops.deepseek_v41 import score_workspace as ws

    geometry = _pool_geometry(True, speculative_tokens=3)
    # widest ratio 1: 64 rows a PAGE, 4 rows a request
    assert ws.packed_capacity(geometry, 10, 7, 1 << 30) == 10 * 64 * 4 + 7 * 64
    assert ws.packed_capacity(geometry, 10, 7, 100) == 100


def test_score_workspace_packs_only_the_fp4_plane_and_holds_its_maxima():
    """Given the pool's PAGEs, the FP4 workspace packs its logits, and holds
    the block maxima beside them, each row's starting at its logits' over the
    block; an FP8 one keeps its plane."""
    import numpy as np

    from atom.model_ops.deepseek_v41 import score_workspace as ws

    fp4 = ws.ScoreWorkspace(_pool_geometry(True), 40, 9, "cpu", pages=10)
    fp8 = ws.ScoreWorkspace(_pool_geometry(False), 40, 9, "cpu", pages=10)
    assert fp4.packed and fp4.capacity == 10 * 64 * 6 + 40 * 64
    assert fp4.packed_logits().numel() == fp4.capacity
    assert fp4.packed_maxima().numel() == fp4.capacity // 8
    visible = np.array([5, 0, 70, 64], np.int32)
    offsets, block_offsets = np.empty(4, np.int32), np.empty(4, np.int32)
    assert fp4.pack(visible, offsets, block_offsets) == ((0, 4),)
    assert offsets.tolist() == [0, 64, 64, 192]
    assert block_offsets.tolist() == [0, 8, 8, 24]
    # 192 elements: rows 0-2 (64 + 0 + 128), then row 3
    small = ws.ScoreWorkspace(_pool_geometry(True, 0), 2, 9, "cpu", pages=1)
    assert small.pack(visible, offsets, block_offsets) == ((0, 3), (3, 4))
    assert not fp8.packed and fp8.capacity == 40 * 9 * 64
    with pytest.raises(ValueError, match="FP4"):
        fp8.packed_logits()


@pytest.mark.parametrize("ratio,longest", [(1, 640), (2, 1409)])
def test_a_step_fits_one_packed_band_exactly_while_its_rows_do(ratio, longest):
    """A step fits (and so runs from a graph, `step_needs_eager`) exactly
    while its rows -- each seeing its request's context in rows of the widest
    ratio, aligned -- fit one band beside the padding rows a graph adds (at
    ratio 1 each sees a row). Ten PAGEs, 40 token rows: ratio 1 leaves 3840
    elements, six rows of 640; ratio 2 leaves 4480, six of 704 (contexts to
    1409 tokens). A plane always fits."""
    from types import SimpleNamespace

    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace

    def fits(geometry, context):
        workspace = ScoreWorkspace(geometry, 40, 9, "cpu", pages=10)
        return workspace.fits((6,), (context,))

    fp4 = SimpleNamespace(**{**vars(_pool_geometry(True)), "owners": ((0, ratio),)})
    assert fits(fp4, longest)
    assert not fits(fp4, longest + 1)
    fp8 = SimpleNamespace(**{**vars(fp4), "index_fp4": False})
    assert fits(fp8, 1 << 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_scorer_runs_in_its_workspace_and_selects_the_same_rows():
    """A workspace changes where the logits band and the tile table live, not
    what is selected; and both are written in the workspace itself."""
    from types import SimpleNamespace

    from atom.model_ops.deepseek_v41 import paged_scoring as scoring
    from atom.model_ops.deepseek_v41.index_write import write_index_rows
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
    from atom.model_ops.deepseek_v41.unit_table import unit_table

    torch.manual_seed(2207)
    rows, heads, dim, per_page, block = 7, 32, 128, 128, 8
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda")
    width = table.shape[1] * per_page
    plane = torch.empty(4, per_page, dim + 4, dtype=torch.uint8, device="cuda")
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
    geometry = SimpleNamespace(
        owners=((0, 1),),
        rows_per_page=lambda ratio: per_page,
        index_block_rows=block,
        index_blocks_per_page=lambda ratio: per_page // block,
        index_fp4=False,
    )
    space = ScoreWorkspace(geometry, rows, table.shape[1], "cuda")
    batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    tiles = unit_table(table, batch, per_page // block)
    in_space = unit_table(table, batch, per_page // block, workspace=space, ratio=1)
    assert torch.equal(in_space, tiles)
    assert in_space.data_ptr() == space.unit_table(1, 1, 1).data_ptr()

    query = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, heads, dtype=torch.float32, device="cuda")
    visible = torch.tensor(
        [0, 1, 33, 64, 129, 511, 512], dtype=torch.int32, device="cuda"
    )
    args = (query, weights, IndexUnits(plane.view(-1, block, dim + 4)), tiles, visible)
    kwargs = {"topk": 64, "weights_scale": (heads * dim) ** -0.5, "candidate_count": 16}
    expected = scoring.score_topk_paged(*args, **kwargs)

    observed = []
    real_score = scoring.deepgemm_fp8_paged_mqa_logits

    def observe(*a, **k):
        observed.append(a[3].data_ptr())
        return real_score(*a, **k)

    scoring.deepgemm_fp8_paged_mqa_logits = observe
    try:
        actual = scoring.score_topk_paged(*args, **kwargs, workspace=space)
    finally:
        scoring.deepgemm_fp8_paged_mqa_logits = real_score
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert observed == [space.logits(1, 1).data_ptr()]
