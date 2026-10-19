# SPDX-License-Identifier: MIT
"""Verify the C32 adapter preserves native cache and request metadata."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from atom.model_ops.gluon_prefill import run_gluon_prefill


@pytest.mark.parametrize("draft", [False, True])
def test_native_pages_and_padded_metadata_are_passed_without_gather(draft):
    heads = 16 if draft else 1
    q = torch.randn(5, 16, 128, dtype=torch.bfloat16)
    k = torch.empty(8, heads, 8, 128, 16, dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    scales = torch.rand(8, heads, 256)[..., ::2]
    meta = SimpleNamespace(
        has_cached=True,
        block_tables=torch.tensor([[4, 1], [6, 2], [-1, -1]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 5, 999], dtype=torch.int32),
        context_lens=torch.tensor([129, 255, 999], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_k=255,
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_draft=draft, scheduled_bs=2),
        attn_metadata=meta,
    )
    seen = []

    def attention(query, kc, vc, out, **kw):
        assert query is q and kc is k and vc is v
        assert kw["backend"] == "gluon"
        assert kw["k_scale_cache"] is scales and kw["v_scale_cache"] is scales
        assert kw["cu_seqlens_q"].tolist() == [0, 1, 5]
        assert kw["seqused_k"].tolist() == [129, 255]
        assert kw["block_table"].tolist() == [[4, 1], [6, 2]]
        assert kw["block_table"].data_ptr() == meta.block_tables.data_ptr()
        out.copy_(query)
        seen.append(kw)

    with patch("atom.model_ops.gluon_prefill.torch.empty", wraps=torch.empty) as alloc:
        out = run_gluon_prefill(
            attention, q, k, v, scales, scales, ctx, softmax_scale=0.125
        )
    assert len(seen) == 1 and alloc.call_count == 1
    assert alloc.call_args.args[0] == q.shape
    torch.testing.assert_close(out, q)
    assert meta.cu_seqlens_q[-1] == 999


def test_unsupported_layout_is_rejected_before_launch():
    with pytest.raises(ValueError, match="native M3"):
        run_gluon_prefill(
            None,
            torch.empty(1, 16, 128, dtype=torch.bfloat16),
            torch.empty(1, 1, 8, 128, 16, dtype=torch.float8_e4m3fn),
            torch.empty(1, 1, 128, 128, dtype=torch.float8_e4m3fn),
            None,
            None,
            None,
            softmax_scale=0.125,
        )
