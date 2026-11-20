# SPDX-License-Identifier: MIT
"""MiniMax-M3 index score workspaces, sized from the configuration (no aiter)."""

from types import SimpleNamespace

import pytest

from atom.model_ops.minimax_m3.index_buffers import (
    IndexScoreBuffers,
    allocate_prefill_scores,
    prefill_score_capacity,
)
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

CONFIG = SimpleNamespace(max_num_batched_tokens=256, max_model_len=4096)


def test_the_prefill_workspace_holds_a_full_chunk_at_the_model_length():
    ws = allocate_prefill_scores(CONFIG, "cpu", index_scorers={}, num_idx_heads=4)
    assert ws.logits(4 * 256, 4096 // 128).shape == (1024, 32)
    with pytest.raises(ValueError):
        ws.logits(4 * 256 + 1, 4096 // 128)
    ragged = SimpleNamespace(max_num_batched_tokens=256, max_model_len=4097)
    assert prefill_score_capacity(ragged, num_idx_heads=4) == 4 * 256 * 33


def test_an_fp4_record_scores_into_its_own_workspace():
    record = IndexScoreBuffers(scores=LogitsWorkspace(1, "cpu"))
    scorers = {"": record}
    assert (
        allocate_prefill_scores(CONFIG, "cpu", index_scorers=scorers, num_idx_heads=4)
        is None
    )
