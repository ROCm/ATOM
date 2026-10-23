# SPDX-License-Identifier: MIT
"""Bounded logits bands must preserve paged top-k and candidate filtering."""

import pytest
import torch

from atom.model_ops.sparse_indexer_chunk import sparse_indexer_row_chunk


def test_million_token_context_does_not_allocate_eight_gib_logits():
    # 8192 queries against a 4x-compressed 1M context used to request ~8 GiB.
    rows, width = 8192, 1048576 // 4
    band = sparse_indexer_row_chunk(rows, width, 2048)
    assert 0 < band < rows
    assert band * width * 4 < 2 * 1024**3


@pytest.mark.parametrize("budget_mb", [0, 8, 64, 2048])
@pytest.mark.parametrize("index_fp4", [False, True])
def test_serving_workspace_respects_logits_budget(monkeypatch, budget_mb, index_fp4):
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", budget_mb)
    geometry = V41PoolGeometry(
        2,
        ((0, 1), (1, 2)),
        256,
        128,
        512,
        128,
        index_block_rows=8,
        index_fp4=index_fp4,
    )
    # Run the actual serving constructor without allocating multi-GiB buffers.
    rows, columns = 8192, 1048576 // geometry.block_size
    workspace = ScoreWorkspace(geometry, rows, columns, "meta", pages=columns)
    capacity = workspace._logits.numel()
    cap_bytes = min(budget_mb or 2048, 2048) * 1024**2
    assert 0 < capacity * workspace._logits.element_size() < cap_bytes
    for ratio in (1, 2):
        full_width = columns * geometry.rows_per_page(ratio)
        for width in (full_width, full_width - 8, full_width // 2, 1024):
            band = workspace.logits_rows(rows, width)
            logits = workspace.logits(band, width)
            assert 0 < band <= rows
            assert logits.numel() * logits.element_size() < cap_bytes
            assert logits.untyped_storage().nbytes() == capacity * 4
            assert workspace._logits.numel() == capacity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("candidate_mode", ["none", "produce", "consume"])
@pytest.mark.parametrize("live_units", [256, 255, 128])
@pytest.mark.parametrize("use_workspace", [False, True])
def test_paged_scoring_bands_match_one_shot(
    monkeypatch, candidate_mode, live_units, use_workspace
):
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.deepseek_v41 import paged_scoring as scoring
    from atom.model_ops.deepseek_v41.candidate_table import bind_candidates
    from atom.model_ops.deepseek_v41.index_plane import IndexUnits
    from atom.model_ops.deepseek_v41.score_workspace import ScoreWorkspace
    from atom.utils import envs

    torch.manual_seed(312)
    rows, heads, dim, tile, units = 257, 32, 128, 8, 256
    width = tile * units
    values = torch.randn(units, tile * dim, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(units, tile, dtype=torch.float32, device="cuda")
    plane = torch.cat((values.view(torch.uint8), scales.view(torch.uint8)), dim=1)
    plane = IndexUnits(plane.view(units, tile, dim + 4))
    query = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(rows, heads, dtype=torch.bfloat16, device="cuda")
    tiles = torch.arange(units, dtype=torch.int32, device="cuda").repeat(rows, 1)
    live_width = tile * live_units
    visible = torch.randint(
        1, live_width + 1, (rows,), dtype=torch.int32, device="cuda"
    )
    visible[::17] = 0
    candidates = None
    if candidate_mode == "consume":
        # Keep half the blocks. Even the compacted 1024-column plane must
        # cross the 1 MiB budget, so this case still exercises row bands.
        # Use the producer's per-row list: valid blocks are ascending,
        # include the newest visible block, and pad short rows with -1.
        _, candidates = scoring.score_topk_paged(
            query,
            weights,
            plane,
            tiles,
            visible,
            topk=64,
            weights_scale=0.01,
            block_size=tile,
            candidate_count=128,
        )
        candidates = bind_candidates(
            candidates,
            tiles,
            torch.arange(rows, dtype=torch.int32, device="cuda"),
            1,
            visible,
            rows_per_block=tile,
        )
    kwargs = {
        "topk": 64,
        "weights_scale": 0.01,
        "candidates": candidates,
        "block_size": tile,
        "candidate_count": 8 if candidate_mode == "produce" else 0,
    }
    monkeypatch.setattr(envs, "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", 2048)
    expected = scoring.score_topk_paged(query, weights, plane, tiles, visible, **kwargs)
    # Both full and compacted widths cross 1 MiB and leave a short tail.
    monkeypatch.setattr(envs, "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", 1)
    bands = []
    real_score = scoring.deepgemm_fp8_paged_mqa_logits

    def observe_band(*args, **kwargs):
        logits = args[3]
        bands.append((logits.shape[0], logits.numel() * logits.element_size()))
        return real_score(*args, **kwargs)

    monkeypatch.setattr(scoring, "deepgemm_fp8_paged_mqa_logits", observe_band)
    # unit_table materializes a contiguous output after slicing page columns.
    live_tiles = tiles[:, :live_units].contiguous()
    workspace = None
    if use_workspace:
        geometry = V41PoolGeometry(
            2, ((0, 1), (1, 2)), 16, 128, 512, 128, index_block_rows=tile
        )
        workspace = ScoreWorkspace(geometry, rows, width // 16, "cuda")
        storage = workspace._logits.data_ptr()
        assert workspace._logits.numel() * 4 < 1024**2
    actual = scoring.score_topk_paged(
        query, weights, plane, live_tiles, visible, workspace=workspace, **kwargs
    )
    assert len(bands) > 1 and sum(count for count, _ in bands) == rows
    assert max(size for _, size in bands) < 1024**2
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    if expected[1] is not None:
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    if workspace is not None:
        assert workspace._logits.data_ptr() == storage


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("index_block_rows", [8, 16, 32])
def test_unit_tiles_bound_prefill_but_keep_decode_capacity(decode, index_block_rows):
    from types import SimpleNamespace

    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.deepseek_v41.unit_table import unit_table

    # A chunk starting after a cached prefix crosses a page boundary. Its
    # bound is the absolute end, not the two query rows in this microbatch.
    tables = torch.arange(2 * 1024, dtype=torch.int32, device="cuda").view(2, 1024)
    batches = torch.tensor([0, 0, 1, -1], dtype=torch.int32, device="cuda")
    step = SimpleNamespace(
        block_tables=tables,
        batch_ids=batches,
        tiles={},
        tile_workspace=None,
        decode=decode,
        requests=(SimpleNamespace(end=1025), SimpleNamespace(end=512)),
    )
    cache = SimpleNamespace(
        workspace=None,
        geometry=V41PoolGeometry(
            1,
            ((0, 2),),
            1024,
            4,
            512,
            128,
            index_block_rows=index_block_rows,
        ),
    )
    actual = PagedAttentionCache.unit_tiles(cache, step, 2)
    units_per_page = 512 // index_block_rows
    full = unit_table(tables, batches, units_per_page)
    columns = 1024 if decode else 2
    assert actual.shape == (4, columns * units_per_page)
    torch.testing.assert_close(actual, full[:, : columns * units_per_page])
    assert PagedAttentionCache.unit_tiles(cache, step, 2) is actual
