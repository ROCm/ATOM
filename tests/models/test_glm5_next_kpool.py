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
    expand_and_append_tail_ref,
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
    torch.testing.assert_close(
        out, torch.full((1, HEAD_DIM), 7.0), atol=1e-3, rtol=1e-3
    )


def test_hadamard128_is_an_orthonormal_involution():
    """H must be ORTHONORMAL (1/sqrt(128)), not the raw butterfly.

    The rotation is applied to the pooled keys and to the indexer query, so
    only the normalized transform preserves the dot products the logits are.
    And the FP8 scale is ue8m0 -- a power of two -- while 1/sqrt(128) is
    2**-3.5, so the two conventions do NOT quantize to the same bytes: this is
    a correctness constraint, not a choice of units.
    """
    torch.manual_seed(2)
    x = torch.randn(4, HEAD_DIM)
    torch.testing.assert_close(
        hadamard128_ref(hadamard128_ref(x)), x, atol=1e-4, rtol=1e-4
    )


def test_hadamard128_preserves_dot_products():
    torch.manual_seed(9)
    q = torch.randn(16, HEAD_DIM)
    k = torch.randn(16, HEAD_DIM)
    torch.testing.assert_close(
        (hadamard128_ref(q) * hadamard128_ref(k)).sum(-1),
        (q * k).sum(-1),
        atol=1e-3,
        rtol=1e-4,
    )


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


# --------------------------------------------------------------------------
# GPU: every Triton kernel against the reference above.
# --------------------------------------------------------------------------

import pytest

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kpool Triton kernels need a GPU"
)


@requires_gpu
def test_pool_and_rotate_matches_reference():
    from atom.model_ops.glm5_next.kpool import pool_and_rotate

    torch.manual_seed(0)
    for n in (1, 7, 8, 33, 1024):
        k = torch.randn(n, POOL, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(n, POOL, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 2
        ape = torch.randn(POOL, HEAD_DIM, device="cuda", dtype=torch.float32)

        got = pool_and_rotate(k, gate, ape)
        pooled = pool_compress_ref(k, gate, ape).to(torch.bfloat16).float()
        want = hadamard128_ref(pooled).to(torch.bfloat16)
        rel = (got.float() - want.float()).abs().max() / want.float().abs().max()
        assert rel < 2e-2, (n, rel.item())


@requires_gpu
def test_query_rotation_quantizes_identically_to_the_reference():
    from atom.model_ops.glm5_next.kpool import fwht128_quant_fp8

    torch.manual_seed(3)
    for n in (1, 31, 32, 100):
        q = torch.randn(n, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        q_fp8, q_scale = fwht128_quant_fp8(q)
        rot = hadamard128_ref(q.float()).to(torch.bfloat16).float()
        want_q, want_s = quant_fp8_ue8m0_ref(rot)
        # The scale must be bit-identical: it is a power of two, so "close" is
        # a factor of two wrong.
        assert torch.equal(q_scale.squeeze(-1), want_s)
        # quant_fp8_ue8m0_ref returns the UNROUNDED value; round it the way the
        # store does before comparing.
        assert torch.equal(q_fp8.float(), want_q.to(torch.float8_e4m3fn).float())


@requires_gpu
def test_fused_expand_matches_the_torch_composition():
    from atom.model_ops.glm5_next.kpool import expand_pools_and_append_tail

    torch.manual_seed(0)
    for rows, ngroups in ((3, 2), (32, 512), (129, 8)):
        pool_ids = torch.randint(
            -1, 40, (rows, ngroups), dtype=torch.int32, device="cuda"
        )
        seq_lens = torch.randint(1, 200, (rows,), dtype=torch.int32, device="cuda")
        got = expand_pools_and_append_tail(pool_ids, seq_lens, POOL)
        want = expand_and_append_tail_ref(pool_ids, seq_lens, POOL)
        assert torch.equal(got, want), (got[0][:12], want[0][:12])


@requires_gpu
def test_tail_lands_immediately_after_the_valid_history():
    """The consumer reads only `min(pools, groups)*POOL + tail` entries per row.

    So the tail must be compacted against the history, not parked at a fixed
    column: otherwise every sequence whose length is not pool-aligned loses its
    newest tokens -- and those are the ones the model most needs.
    """
    from atom.model_ops.glm5_next.kpool import expand_pools_and_append_tail

    ngroups = 512
    # 4 sequences, one per pool phase, all far below the 2048-token budget.
    seq_lens = torch.tensor([40, 41, 42, 43], dtype=torch.int32, device="cuda")
    # top-k pads past the row's valid pool count with -1, and that padding is
    # exactly what a fixed-column tail would hand to attention. A pool_ids of
    # plain arange(ngroups) hides the bug, because every padded slot then still
    # decodes to a plausible in-range token id.
    pool_ids = torch.arange(ngroups, dtype=torch.int32, device="cuda").repeat(4, 1)
    for r, sl in enumerate(seq_lens.tolist()):
        pool_ids[r, sl // POOL :] = -1
    out = expand_pools_and_append_tail(pool_ids, seq_lens, POOL)
    for r, sl in enumerate(seq_lens.tolist()):
        consumed = min(sl // POOL, ngroups) * POOL + sl % POOL
        assert consumed == sl, (sl, consumed)
        got = out[r, :consumed].tolist()
        assert got == list(range(sl)), (sl, got[:8], got[-8:])


@requires_gpu
@pytest.mark.parametrize("prefill_len", [16, 17, 18, 19, 64, 65, 66, 67])
def test_tail_survives_prefill_to_decode(prefill_len):
    """A pool assembled one token at a time across decode steps must equal the
    same pool compressed in one shot.

    This is the whole tail state machine: the pool that straddles the
    prefill/decode boundary is the one the design can get wrong, and it is
    wrong for only 3 of every 4 sequence lengths -- hence every phase.
    """
    from atom.model_ops.glm5_next.kpool import (
        kpool_decode_stash_and_pool,
        kpool_seed_tail,
        pool_and_rotate,
    )

    torch.manual_seed(prefill_len)
    dev = "cuda"
    n_tok = prefill_len + 12
    k = (torch.randn(n_tok, HEAD_DIM, device=dev) * 2).to(torch.bfloat16)
    gate = (torch.randn(n_tok, HEAD_DIM, device=dev) * 2).to(torch.bfloat16)
    ape = torch.randn(POOL, HEAD_DIM, device=dev, dtype=torch.float32)

    tail = torch.zeros(8, 2, POOL, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    slot_idx = torch.tensor([3], dtype=torch.int32, device=dev)
    kpool_seed_tail(
        tail,
        k[:prefill_len],
        gate[:prefill_len],
        torch.arange(prefill_len, dtype=torch.int32, device=dev),
        torch.tensor([0, prefill_len], dtype=torch.int32, device=dev),
        slot_idx,
        POOL,
    )

    completed = 0
    for pos in range(prefill_len, n_tok):
        got = kpool_decode_stash_and_pool(
            tail,
            k[pos : pos + 1],
            gate[pos : pos + 1],
            torch.tensor([pos], dtype=torch.int32, device=dev),
            slot_idx,
            ape,
            POOL,
        )
        if pos % POOL != POOL - 1:
            continue  # pool incomplete; the caller marks the slot -1
        pid = pos // POOL
        want = pool_and_rotate(
            k[pid * POOL : (pid + 1) * POOL].unsqueeze(0),
            gate[pid * POOL : (pid + 1) * POOL].unsqueeze(0),
            ape,
        )
        assert torch.equal(got, want), (prefill_len, pid)
        completed += 1
    assert completed >= 2, completed
