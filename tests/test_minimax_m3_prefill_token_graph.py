# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_M3_PREFILL_TOKEN_GRAPH: only the MiniMax-M3 sparse builder takes prefill
token graphs, and only when the flag is set (default off)."""

from types import SimpleNamespace

import pytest

from atom.model_ops.attentions.aiter_attention import AiterAttentionMetadataBuilder

_supports = AiterAttentionMetadataBuilder.supports_token_graph.fget
_capture = AiterAttentionMetadataBuilder.build_for_token_graph_capture


def test_default_off(monkeypatch):
    monkeypatch.delenv("ATOM_M3_PREFILL_TOKEN_GRAPH", raising=False)
    assert not _supports(SimpleNamespace(_has_sparse_attention=True))


def test_flag_needs_sparse_builder(monkeypatch):
    monkeypatch.setenv("ATOM_M3_PREFILL_TOKEN_GRAPH", "1")
    assert _supports(SimpleNamespace(_has_sparse_attention=True))
    assert not _supports(SimpleNamespace(_has_sparse_attention=False))


@pytest.mark.parametrize("tokens", [0, -1, 4097])
def test_capture_bucket_bounds(tokens):
    builder = SimpleNamespace(max_num_batched_tokens=4096)
    with pytest.raises(ValueError, match="token buffer"):
        _capture(builder, tokens)
