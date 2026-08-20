# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU differential tests for the DeepSeek-V4 FP8 paged-prefill indexer.

The production paged scorer is intentionally exercised without constructing a
model or loading weights. A small synthetic cache is packed in the production
FP8 layout, then the new direct-paged scorer is compared with the previous
gather -> dense MQA -> prefill-top-k pipeline.

The one fixture is deliberately awkward: ragged query counts, one sequence
with no visible K, rows shorter than top-k, a 63/64/65-row page boundary, a
non-identity physical page table, and an odd query count for PCP padding.  The
same logical data is also repacked under a second physical-page permutation.
"""

from dataclasses import dataclass

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "compares aiter FP8 indexer kernels on a real GPU",
        allow_module_level=True,
    )

from aiter import cp_gather_indexer_k_quant_cache, dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.topk import top_k_per_row_prefill
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

from atom.models.deepseek_v4 import Indexer

DEV = "cuda"
HEADS = 64
HEAD_DIM = 128
ROWS_PER_PAGE = 64
ROW_BYTES = HEAD_DIM + 4
TOPK = 8
NUM_PHYSICAL_PAGES = 8

# q rows: 1 + 6 + 4 = 11 (odd, so PCP=2 owns one padded query).
COMMITTED = (0, 6, 65)
BATCH_IDS = (0,) + (1,) * 6 + (2,) * 4
# Causal ends exercise empty, short/sentinel, and both sides of a 64-row page.
VISIBLE_ENDS = (0, 1, 2, 3, 4, 5, 6, 63, 64, 65, 65)


@dataclass
class _Case:
    indexer: Indexer
    q_fp8: torch.Tensor
    weights: torch.Tensor
    block_tables: torch.Tensor
    committed: torch.Tensor
    batch_ids: torch.Tensor
    visible_ends: torch.Tensor


@dataclass
class _View:
    indexer: Indexer
    q_fp8: torch.Tensor
    weights: torch.Tensor
    block_tables: torch.Tensor
    meta: dict


def _bare_indexer(cache: torch.Tensor) -> Indexer:
    """Make only the state the two private scoring helpers actually read."""
    indexer = object.__new__(Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.n_heads = HEADS
    indexer.head_dim = HEAD_DIM
    indexer.kv_cache = cache
    return indexer


def _pack_preshuffled_cache(k_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack separate physical-page K/scales into ATOM's FP8 cache layout."""
    num_pages = k_fp8.size(0)
    packed = torch.empty(
        num_pages,
        ROWS_PER_PAGE * ROW_BYTES,
        dtype=torch.uint8,
        device=DEV,
    )
    data_bytes = ROWS_PER_PAGE * HEAD_DIM
    shuffled = shuffle_weight(k_fp8)
    packed[:, :data_bytes] = shuffled.view(torch.uint8).reshape(num_pages, data_bytes)
    packed[:, data_bytes:] = scales.view(torch.uint8).reshape(num_pages, -1)
    return packed.view(dtypes.fp8).reshape(num_pages, ROWS_PER_PAGE, ROW_BYTES)


def _make_case(page_seed: int) -> _Case:
    """Build the same logical K under a deterministic physical-page shuffle."""
    gen = torch.Generator().manual_seed(page_seed)
    block_tables = (
        torch.randperm(NUM_PHYSICAL_PAGES, generator=gen)[: 3 * 2]
        .reshape(3, 2)
        .to(dtype=torch.int32, device=DEV)
    )

    # Keep K exactly representable and encode a wide, deterministic score order
    # in the fp32 row scales. Every unused physical row has a large poison scale,
    # so any out-of-window/page leak wins loudly instead of hiding behind zeros.
    k_fp8 = torch.ones(
        NUM_PHYSICAL_PAGES,
        ROWS_PER_PAGE,
        HEAD_DIM,
        dtype=dtypes.fp8,
        device=DEV,
    )
    scales = torch.full(
        (NUM_PHYSICAL_PAGES, ROWS_PER_PAGE),
        1000.0,
        dtype=torch.float32,
        device=DEV,
    )
    for batch_id, n_rows in enumerate(COMMITTED):
        for row in range(n_rows):
            physical_page = int(block_tables[batch_id, row // ROWS_PER_PAGE])
            scales[physical_page, row % ROWS_PER_PAGE] = float(row + 1)
    cache = _pack_preshuffled_cache(k_fp8, scales)

    total_q = len(BATCH_IDS)
    q_fp8 = torch.ones(total_q, HEADS, HEAD_DIM, dtype=torch.float32, device=DEV).to(
        dtypes.fp8
    )
    weights = torch.ones(total_q, HEADS, dtype=torch.float32, device=DEV)
    return _Case(
        indexer=_bare_indexer(cache),
        q_fp8=q_fp8,
        weights=weights,
        block_tables=block_tables,
        committed=torch.tensor(COMMITTED, dtype=torch.int32, device=DEV),
        batch_ids=torch.tensor(BATCH_IDS, dtype=torch.int32, device=DEV),
        visible_ends=torch.tensor(VISIBLE_ENDS, dtype=torch.int32, device=DEV),
    )


def _build_meta(
    block_tables: torch.Tensor,
    committed: torch.Tensor,
    batch_ids: torch.Tensor,
    visible_ends: torch.Tensor,
) -> dict:
    """The scorer-facing subset of `_build_v4_indexer_meta`'s contract."""
    bs = committed.numel()
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=DEV)
    torch.cumsum(committed, dim=0, out=cu[1:])
    total_committed = int(cu[-1])

    # PCP padding uses batch_id == -1.  Production's packed legacy metadata
    # indexes cu[-1] (an empty range at the packed buffer end), while the paged
    # table maps it to seq 0 as a safe unread placeholder.
    seq_base = cu[batch_ids]
    safe_batch_ids = batch_ids.clamp_min(0)
    max_seq_len = max(int(committed.max()), 1)
    live_pages = (max_seq_len + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    per_token_tables = torch.index_select(
        block_tables[:, :live_pages], 0, safe_batch_ids
    ).contiguous()
    return {
        "total_committed": total_committed,
        "cu_committed_gpu": cu,
        "seq_base_per_token_gpu": seq_base,
        "cu_starts_gpu": seq_base,
        "cu_ends_gpu": seq_base + visible_ends,
        "visible_end_gpu": visible_ends,
        "paged_prefill_block_tables_per_token": per_token_tables,
        "paged_prefill_max_seq_len": max_seq_len,
    }


def _view(
    case: _Case,
    row_ids: list[int],
    seq_ids: list[int],
    *,
    pad_rows: int = 0,
) -> _View:
    """Select query rows and rebase their request ids like a TBO mini-batch."""
    rows = torch.tensor(row_ids, dtype=torch.long, device=DEV)
    q_fp8 = torch.index_select(case.q_fp8, 0, rows)
    weights = torch.index_select(case.weights, 0, rows)
    original_bids = torch.index_select(case.batch_ids, 0, rows)
    visible_ends = torch.index_select(case.visible_ends, 0, rows)

    local_bids = torch.full_like(original_bids, -1)
    for local_id, original_id in enumerate(seq_ids):
        local_bids[original_bids == original_id] = local_id
    assert bool((local_bids >= 0).all()), (row_ids, seq_ids)

    if pad_rows:
        q_pad = torch.zeros(
            pad_rows, HEADS, HEAD_DIM, dtype=torch.float32, device=DEV
        ).to(dtypes.fp8)
        q_fp8 = torch.cat((q_fp8, q_pad), dim=0)
        weights = torch.cat(
            (weights, torch.zeros(pad_rows, HEADS, dtype=torch.float32, device=DEV)),
            dim=0,
        )
        local_bids = torch.cat(
            (local_bids, torch.full((pad_rows,), -1, dtype=torch.int32, device=DEV))
        )
        visible_ends = torch.cat(
            (visible_ends, torch.zeros(pad_rows, dtype=torch.int32, device=DEV))
        )

    seq_idx = torch.tensor(seq_ids, dtype=torch.long, device=DEV)
    block_tables = torch.index_select(case.block_tables, 0, seq_idx).contiguous()
    committed = torch.index_select(case.committed, 0, seq_idx).contiguous()
    return _View(
        indexer=case.indexer,
        q_fp8=q_fp8.contiguous(),
        weights=weights.contiguous(),
        block_tables=block_tables,
        meta=_build_meta(block_tables, committed, local_bids, visible_ends),
    )


def _full_view(case: _Case) -> _View:
    return _view(case, list(range(len(BATCH_IDS))), [0, 1, 2])


def _legacy_score(view: _View) -> torch.Tensor:
    """Test-local copy of the pre-change gather/dense FP8 kernel pipeline."""
    meta = view.meta
    total_committed = meta["total_committed"]
    k_fp8 = torch.empty(total_committed, HEAD_DIM, dtype=dtypes.fp8, device=DEV)
    k_scale = torch.empty(total_committed, 1, dtype=torch.float32, device=DEV)
    cp_gather_indexer_k_quant_cache(
        view.indexer.kv_cache,
        k_fp8,
        k_scale.view(dtypes.fp8),
        view.block_tables,
        meta["cu_committed_gpu"],
        preshuffle=True,
    )
    logits = fp8_mqa_logits(
        Q=view.q_fp8,
        KV=k_fp8,
        kv_scales=k_scale,
        weights=view.weights,
        cu_starts=meta["cu_starts_gpu"],
        cu_ends=meta["cu_ends_gpu"],
        clean_logits=False,
    )
    topk_global = torch.empty(view.q_fp8.size(0), TOPK, dtype=torch.int32, device=DEV)
    top_k_per_row_prefill(
        logits,
        meta["cu_starts_gpu"],
        meta["cu_ends_gpu"],
        topk_global,
        None,
        logits.size(0),
        logits.stride(0),
        logits.stride(1),
        k=TOPK,
    )
    seq_base = meta["seq_base_per_token_gpu"].unsqueeze(1)
    return torch.where(topk_global < 0, topk_global, topk_global - seq_base)


def _paged_score(view: _View) -> torch.Tensor:
    return view.indexer._score_topk_prefill_paged(
        view.q_fp8, view.weights, view.meta, TOPK
    )


def _assert_same(paged: torch.Tensor, legacy: torch.Tensor, label: str) -> None:
    assert torch.equal(_canonical(paged), _canonical(legacy)), (
        f"{label}: paged top-k differs from legacy\n"
        f"paged={paged.cpu().tolist()}\nlegacy={legacy.cpu().tolist()}"
    )


def _canonical(topk: torch.Tensor) -> torch.Tensor:
    return torch.sort(topk, dim=1).values


def _assert_semantic_topk(topk: torch.Tensor, visible_ends: torch.Tensor) -> None:
    """Independent oracle: increasing K scales make the last rows win."""
    for row_id, end in enumerate(visible_ends.cpu().tolist()):
        n_valid = min(end, TOPK)
        row = topk[row_id]
        assert bool((row[:n_valid] >= 0).all()), (row_id, end, row.cpu().tolist())
        assert bool((row[n_valid:] == -1).all()), (row_id, end, row.cpu().tolist())
        expected = set(range(max(0, end - TOPK), end))
        assert set(row[:n_valid].cpu().tolist()) == expected, (
            row_id,
            end,
            row.cpu().tolist(),
            sorted(expected),
        )


def test_paged_matches_legacy_across_ragged_windows_and_page_permutations():
    outputs = []
    for page_seed in (3, 17):
        view = _full_view(_make_case(page_seed))
        legacy = _legacy_score(view)
        paged = _paged_score(view)
        _assert_same(paged, legacy, f"page permutation seed={page_seed}")
        _assert_semantic_topk(paged, view.meta["visible_end_gpu"])
        outputs.append(paged)

    assert torch.equal(
        _canonical(outputs[0]), _canonical(outputs[1])
    ), "the logical result changed when only physical page IDs were permuted"


def test_paged_matches_legacy_when_query_row_chunking_is_forced(monkeypatch):
    view = _full_view(_make_case(5))
    legacy = _legacy_score(view)
    paged_one_shot = _paged_score(view)

    # Boundaries 3/6/9 cut through both a request and the 63/64/65-row case.
    monkeypatch.setattr(
        Indexer,
        "_prefill_chunk_rows",
        staticmethod(lambda total_tokens, row_width: min(total_tokens, 3)),
    )
    paged_chunked = _paged_score(view)

    _assert_same(paged_one_shot, legacy, "one-shot")
    _assert_same(paged_chunked, legacy, "three-row chunks")


def test_paged_matches_legacy_for_pcp_round_robin_query_shards():
    case = _make_case(7)
    full = _full_view(case)
    baseline = _legacy_score(full)
    padded_total = 12  # 11 real rows, padded to PCP=2.

    for rank in range(2):
        owned = list(range(rank, padded_total, 2))
        real_rows = [row for row in owned if row < len(BATCH_IDS)]
        n_pad = len(owned) - len(real_rows)
        local = _view(case, real_rows, [0, 1, 2], pad_rows=n_pad)
        legacy = _legacy_score(local)
        paged = _paged_score(local)
        _assert_same(paged, legacy, f"pcp rank {rank}")
        assert torch.equal(
            _canonical(paged[: len(real_rows)]), _canonical(baseline[real_rows])
        )
        if n_pad:
            assert bool((paged[-n_pad:] == -1).all()), paged[-n_pad:].cpu().tolist()


def test_paged_matches_legacy_for_tbo_split_and_batch_locality():
    case = _make_case(11)
    full = _full_view(case)
    baseline = _legacy_score(full)

    # Token-midpoint TBO split at row 5 cuts seq 1 (its rows are [1, 7)).
    ubatches = (
        _view(case, list(range(5)), [0, 1]),
        _view(case, list(range(5, 11)), [1, 2]),
    )
    pieces = []
    for ubatch_id, ubatch in enumerate(ubatches):
        legacy = _legacy_score(ubatch)
        paged = _paged_score(ubatch)
        _assert_same(paged, legacy, f"tbo ubatch {ubatch_id}")
        pieces.append(paged)
    assert torch.equal(_canonical(torch.cat(pieces)), _canonical(baseline))

    # The 65-row victim is now the only request, changing its packed legacy
    # base from 6 to 0 while keeping its physical pages and query rows identical.
    victim = _view(case, list(range(7, 11)), [2])
    victim_legacy = _legacy_score(victim)
    victim_paged = _paged_score(victim)
    _assert_same(victim_paged, victim_legacy, "victim alone")
    assert torch.equal(_canonical(victim_paged), _canonical(baseline[7:11]))
