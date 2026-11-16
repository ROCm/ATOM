# SPDX-License-Identifier: MIT
"""MiniMax-M3 index score workspaces, sized from the configuration (no aiter)."""

from types import SimpleNamespace

import pytest

from atom.model_ops.minimax_m3.index_buffers import (
    allocate_prefill_scores,
    prefill_score_capacity,
)


def _config(index_dtype, mnbt=256, max_model_len=4096):
    return SimpleNamespace(
        index_cache_dtype=index_dtype,
        max_num_batched_tokens=mnbt,
        max_model_len=max_model_len,
    )


@pytest.mark.parametrize("index_dtype", ["bf16", "fp8"])
def test_the_prefill_workspace_holds_a_full_chunk_at_the_model_length(index_dtype):
    ws = allocate_prefill_scores(
        _config(index_dtype), "cpu", max_bs=4, max_qlen=1, num_idx_heads=4
    )
    assert ws.logits(4 * 256, 4096 // 128).shape == (1024, 32)
    with pytest.raises(ValueError):
        ws.logits(4 * 256 + 1, 4096 // 128)


def test_verify_rows_past_the_chunk_widen_it():
    config = _config("fp8", mnbt=16)
    capacity = prefill_score_capacity(config, max_bs=8, max_qlen=4, num_idx_heads=2)
    assert capacity == 2 * (8 * 4) * (4096 // 128)


def test_mxfp4_scores_into_its_own_record():
    config = _config("fp4")
    assert (
        allocate_prefill_scores(config, "cpu", max_bs=4, max_qlen=1, num_idx_heads=4)
        is None
    )
