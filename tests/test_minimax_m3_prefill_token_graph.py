# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_M3_PREFILL_TOKEN_GRAPH: only the MiniMax-M3 sparse builder with an FP4 index
cache takes prefill token graphs, and only when the flag is set (default off); the
recorded buckets fit ``max_model_len``."""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter")

from atom.model_ops.attentions.aiter_attention import (
    AiterAttentionMetadataBuilder,
)

_supports = AiterAttentionMetadataBuilder.supports_token_graph.fget
_capture = AiterAttentionMetadataBuilder.build_for_token_graph_capture


def test_default_off(monkeypatch):
    monkeypatch.delenv("ATOM_M3_PREFILL_TOKEN_GRAPH", raising=False)
    assert not _supports(SimpleNamespace(_has_sparse_attention=True, _index_fp4=True))


def test_flag_needs_sparse_builder(monkeypatch):
    monkeypatch.setenv("ATOM_M3_PREFILL_TOKEN_GRAPH", "1")
    assert _supports(SimpleNamespace(_has_sparse_attention=True, _index_fp4=True))
    assert not _supports(SimpleNamespace(_has_sparse_attention=False, _index_fp4=True))


def test_non_fp4_index_keeps_eager_prefill(monkeypatch):
    """A BF16 / FP8 index would run the padded rows' selection on unwritten rows
    (`_metadata_rows` trims FP4 indexes only), so such a cache records no graphs."""
    monkeypatch.setenv("ATOM_M3_PREFILL_TOKEN_GRAPH", "1")
    assert not _supports(SimpleNamespace(_has_sparse_attention=True, _index_fp4=False))


def test_metadata_rows_trims_fp4_pad_rows_only():
    """The padded replay: q has the bucket's rows, the metadata the real ones."""
    import torch

    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl

    q = torch.zeros(512, 4, 8)
    trim = SparseMHAPagedAttentionImpl._metadata_rows
    assert trim(SimpleNamespace(index_cache_dtype="nvfp4"), q, 257) == 257
    assert trim(SimpleNamespace(index_cache_dtype="fp8"), q, 257) == 512


def _recorder(max_model_len, tokens_cap, within):
    recorded = []
    runner = SimpleNamespace(
        attn_metadata_builder=SimpleNamespace(
            token_graph_within_model_len=within,
            build_for_token_graph_capture=lambda tokens: tokens,
        ),
        config=SimpleNamespace(max_model_len=max_model_len),
        warmup_max_tokens=lambda: tokens_cap,
        running_tokens_per_dp_rank=SimpleNamespace(fill=lambda n: None),
        rank=1,
        _build_capture=lambda build, tokens: (tokens,),
        _run_capture_prefill=lambda tokens, mode: recorded.append((tokens, mode)),
    )
    return runner, recorded


def test_buckets_beyond_max_model_len_are_not_recorded(monkeypatch):
    """max_model_len 2048 with the default 3072 bucket: one synthetic request of 3072
    rows would need 24 block-table columns where a request gets 16."""
    import torch

    from atom.model_engine.model_runner import ModelRunner
    from atom.utils import envs

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(
        envs, "ATOM_TOKEN_GRAPH_SIZES", (256, 512, 1024, 2048, 3072), raising=False
    )
    runner, recorded = _recorder(2048, 4096, within=True)
    ModelRunner._record_tr_graphs(runner)
    assert {t for t, _ in recorded} == {256, 512, 1024, 2048}
    # a backend that sizes its own capture block table keeps every bucket
    runner, recorded = _recorder(2048, 4096, within=False)
    ModelRunner._record_tr_graphs(runner)
    assert 3072 in {t for t, _ in recorded}


@pytest.mark.parametrize("tokens", [0, -1, 4097])
def test_capture_bucket_bounds(tokens):
    builder = SimpleNamespace(max_num_batched_tokens=4096)
    with pytest.raises(ValueError, match="token buffer"):
        _capture(builder, tokens)
