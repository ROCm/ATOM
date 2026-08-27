# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kpool compression semantics for GLM-5.3-Flash.

These pin the part of the port that fails *quietly* if it is wrong: the pooling
softmax runs over a pool's slots independently per dimension, not as a scalar
per-slot gate. Both spellings produce finite, plausible-looking keys, so only an
explicit test separates them.

CPU-only; no GPU or aiter required.
"""

import torch

from atom.model_ops.glm5_next.kpool import (
    append_tail_to_topk,
    compress_pools_ref,
    expand_pools_to_tokens,
    hadamard128_ref,
    history_group_budget_for_topk,
    pool_compress_ref,
    quant_fp8_ue8m0_ref,
)

POOL = 4
HEAD_DIM = 128


def test_pool_softmax_is_per_dimension_not_per_slot():
    """The pooling weights must vary along the head dim, not just by slot."""
    torch.manual_seed(0)
    k = torch.randn(3, POOL, HEAD_DIM)
    gate = torch.randn(3, POOL, HEAD_DIM)
    ape = torch.randn(POOL, HEAD_DIM)

    out = pool_compress_ref(k, gate, ape)
    assert out.shape == (3, HEAD_DIM)

    # A scalar-per-slot gate (softmax over slots of the dim-MEAN score) is the
    # plausible wrong implementation. It must NOT match.
    scalar_w = (gate.float().mean(-1) + ape.float().mean(-1)).softmax(dim=1)
    scalar_out = (scalar_w.unsqueeze(-1) * k.float()).sum(dim=1)
    assert not torch.allclose(out, scalar_out, atol=1e-4)


def test_pool_softmax_weights_sum_to_one_per_dimension():
    torch.manual_seed(1)
    gate = torch.randn(5, POOL, HEAD_DIM)
    ape = torch.randn(POOL, HEAD_DIM)
    # Pooling all-ones keys must return exactly ones: the weights are a
    # partition of unity along the slot axis for every dimension.
    ones = torch.ones(5, POOL, HEAD_DIM)
    out = pool_compress_ref(ones, gate, ape)
    torch.testing.assert_close(out, torch.ones(5, HEAD_DIM), atol=1e-5, rtol=1e-5)


def test_ape_is_applied_per_slot():
    """A large ape on one slot must select that slot's key."""
    k = torch.zeros(1, POOL, HEAD_DIM)
    k[0, 2, :] = 7.0
    gate = torch.zeros(1, POOL, HEAD_DIM)
    ape = torch.full((POOL, HEAD_DIM), -30.0)
    ape[2] = 30.0
    out = pool_compress_ref(k, gate, ape)
    torch.testing.assert_close(out, torch.full((1, HEAD_DIM), 7.0), atol=1e-3, rtol=1e-3)


def test_hadamard128_is_an_involution_up_to_scale():
    torch.manual_seed(2)
    x = torch.randn(4, HEAD_DIM)
    torch.testing.assert_close(hadamard128_ref(hadamard128_ref(x)), 128.0 * x,
                               atol=1e-3, rtol=1e-4)


def test_fp8_quant_scale_is_power_of_two_and_bounded():
    torch.manual_seed(3)
    x = torch.randn(6, HEAD_DIM) * 100
    q, scale = quant_fp8_ue8m0_ref(x)
    assert q.abs().max() <= 448.0 + 1e-3
    log2 = torch.log2(scale)
    torch.testing.assert_close(log2, log2.round(), atol=1e-5, rtol=0)
    # Dequantization must stay close to the input.
    assert (q * scale.unsqueeze(-1) - x).abs().max() < 0.05 * x.abs().max()


def test_compress_pools_end_to_end_shapes():
    torch.manual_seed(4)
    k = torch.randn(9, POOL, HEAD_DIM)
    gate = torch.randn(9, POOL, HEAD_DIM)
    ape = torch.randn(POOL, HEAD_DIM)
    q, scale = compress_pools_ref(k, gate, ape)
    assert q.shape == (9, HEAD_DIM)
    assert scale.shape == (9,)
    assert torch.isfinite(q).all() and torch.isfinite(scale).all()


def test_expand_pools_to_tokens():
    topk = 8
    budget = history_group_budget_for_topk(topk, POOL)
    assert budget == 2
    pool_ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32)
    valid = torch.tensor([[True, True], [True, False]])
    out = expand_pools_to_tokens(pool_ids, valid, topk, POOL)
    assert out.shape == (2, topk)
    # Pool 0 -> tokens 0..3; pool 3 -> tokens 12..15.
    assert out[0].tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    # Pool 1 -> 4..7; the invalid slot is masked out.
    assert out[1].tolist() == [4, 5, 6, 7, -1, -1, -1, -1]


def test_append_tail_selects_the_in_progress_pool():
    topk_tokens = torch.full((2, 8), -1, dtype=torch.int32)
    # seq 10: pools cover 0..7, tail is tokens 8,9 (and one padding slot).
    seq_lens = torch.tensor([10, 8], dtype=torch.int32)
    out = append_tail_to_topk(topk_tokens, seq_lens, POOL)
    assert out.shape == (2, 8 + POOL - 1)
    assert out[0, 8:].tolist() == [8, 9, -1]
    # seq 8 is exactly pool-aligned: no tail tokens at all.
    assert out[1, 8:].tolist() == [-1, -1, -1]
