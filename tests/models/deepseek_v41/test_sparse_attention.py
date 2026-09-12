# SPDX-License-Identifier: MIT
"""BF16 sparse attention rounding and sink contracts, independent of model wiring."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_v4_bf16_attention_counts_sink_once_for_two_representations():
    from atom.model_ops.sparse_attn_v4 import sparse_attn

    q = torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.full((1, 2, 512), 6.0, dtype=torch.bfloat16, device="cuda")
    # Two representations of the same logical position remain separate entries.
    indices = torch.tensor([[[0, 1, -1]]], dtype=torch.int32, device="cuda")
    output = sparse_attn(q, kv, torch.zeros(8, device="cuda"), indices, 512**-0.5)
    torch.testing.assert_close(output, torch.full_like(output, 4.0), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_sparse_bf16_uses_reference_online_softmax_tiles():
    from atom.model_ops.sparse_attn_v4 import sparse_attn, sparse_attn_triton

    from .oracle_kernels import sparse_attn as oracle

    torch.manual_seed(512)
    for count in (192, 640):
        q = torch.randn(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(1, count, 512, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(16, device="cuda", dtype=torch.float32)
        indices = torch.arange(count, device="cuda", dtype=torch.int32).view(1, 1, -1)
        expected = oracle(q, kv, sink, indices, 512**-0.5)
        actual = sparse_attn_triton(q, kv, sink, indices, 512**-0.5, block_k=64)
        torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=2**-12)
        assert (
            actual.float() - expected.float()
        ).norm() / expected.float().norm() < 1e-4
        # Existing V4 callers retain the previous default tile size.
        legacy = sparse_attn_triton(q, kv, sink, indices, 512**-0.5, block_k=16)
        assert torch.equal(sparse_attn(q, kv, sink, indices, 512**-0.5), legacy)
