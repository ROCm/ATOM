# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Integration coverage for the V4 prefill row-shard orchestration."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm GPU"
)

from atom.model_ops.v4_indexer_utils import cyclic_row_indices
from atom.models import deepseek_v4
from atom.models.deepseek_v4 import Indexer


class _FakeTPGroup:
    world_size = 4

    def __init__(self, rank_in_group, expected_by_rank, gathered):
        self.rank_in_group = rank_in_group
        self._expected_by_rank = expected_by_rank
        self._gathered = gathered

    def all_gather(self, local, dim=0):
        assert dim == 0
        assert torch.equal(local, self._expected_by_rank[self.rank_in_group])
        return self._gathered


def _make_indexer(device: torch.device) -> Indexer:
    indexer = object.__new__(Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.head_dim = 1
    indexer.kv_cache = torch.empty(1, dtype=torch.uint8, device=device)
    return indexer


def test_score_topk_prefill_tp4_row_shard_matches_unsharded(monkeypatch):
    device = torch.device("cuda")
    world_size = 4
    total_rows = 5
    topk = 2
    total_committed = 131072
    seq_base = torch.arange(total_rows, dtype=torch.int32, device=device) * 16
    seq_end = seq_base + 8
    q = torch.arange(total_rows, dtype=torch.float32, device=device).view(-1, 1, 1)
    weights = (q.view(-1, 1) + 100).contiguous()
    indexer_meta = {
        "total_committed": total_committed,
        "cu_committed_gpu": torch.zeros(1, dtype=torch.int32, device=device),
        "seq_base_per_token_gpu": seq_base,
        "cu_ends_gpu": seq_end,
    }
    block_tables = torch.zeros((1, 1), dtype=torch.int32, device=device)
    indexer = _make_indexer(device)

    def fake_cache_gather(*args, **kwargs):
        return None

    def fake_logits(*, Q, KV, kv_scales, weights, cu_starts, cu_ends, **kwargs):
        del KV, kv_scales, kwargs
        row_ids = Q[:, 0, 0].to(torch.int64)
        assert torch.equal(weights[:, 0], Q[:, 0, 0] + 100)
        assert torch.equal(cu_starts, row_ids.to(torch.int32) * 16)
        assert torch.equal(cu_ends, cu_starts + 8)

        logits = torch.full(
            (Q.shape[0], total_committed),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        base_scores = torch.arange(8, dtype=torch.float32, device=device)
        for local_row, row_id in enumerate(row_ids.tolist()):
            start = int(cu_starts[local_row])
            logits[local_row, start : start + 8] = torch.roll(
                base_scores, shifts=row_id
            )
        return logits

    def fake_topk(
        logits,
        row_starts,
        row_ends,
        output_indices,
        output_values,
        rows,
        stride_0,
        stride_1,
        *,
        k,
    ):
        del row_starts, row_ends, output_values, stride_0, stride_1
        assert rows == logits.shape[0]
        output_indices.copy_(torch.topk(logits, k=k, dim=1).indices.to(torch.int32))

    monkeypatch.setattr(
        deepseek_v4, "cp_gather_indexer_k_quant_cache", fake_cache_gather
    )
    monkeypatch.setattr(deepseek_v4, "fp8_mqa_logits", fake_logits)
    monkeypatch.setattr(deepseek_v4, "top_k_per_row_prefill", fake_topk)
    # 1 MiB / (131072 fp32 logits per row) = two rows per chunk.
    monkeypatch.setattr(deepseek_v4, "SPARSE_INDEXER_LOGITS_BUDGET_MB", 1)

    monkeypatch.setenv("ATOM_INDEXER_PREFILL_ROW_SHARD", "0")
    monkeypatch.setattr(
        deepseek_v4,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size, rank_in_group=0),
    )
    expected = indexer._score_topk_prefill(q, weights, block_tables, indexer_meta, topk)

    shard_rows = (total_rows + world_size - 1) // world_size
    padded_shards = []
    expected_by_rank = []
    for rank in range(world_size):
        row_indices = cyclic_row_indices(total_rows, world_size, rank, device=device)
        local = expected[row_indices]
        padded = torch.full((shard_rows, topk), -1, dtype=torch.int32, device=device)
        padded[: local.shape[0]].copy_(local)
        padded_shards.append(padded)
        expected_by_rank.append(padded)
    gathered = torch.cat(padded_shards, dim=0)

    monkeypatch.setenv("ATOM_INDEXER_PREFILL_ROW_SHARD", "1")
    for rank in range(world_size):
        tp_group = _FakeTPGroup(rank, expected_by_rank, gathered)
        monkeypatch.setattr(
            deepseek_v4, "get_tp_group", lambda tp_group=tp_group: tp_group
        )
        actual = indexer._score_topk_prefill(
            q, weights, block_tables, indexer_meta, topk
        )
        assert torch.equal(actual, expected)
