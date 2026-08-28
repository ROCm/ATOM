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

from atom.config import _MQA_LOGITS_PRESHUFFLE_ROWS, glm5_kpool_block_size
from atom.model_ops.glm5_next.kpool import (
    append_tail_to_topk,
    compress_pools_ref,
    expand_and_append_tail_ref,
    expand_pools_to_tokens,
    hadamard128_ref,
    history_group_budget_for_topk,
    pool_compress_ref,
    pool_slot_mapping,
    pooled_path_enabled,
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


# --------------------------------------------------------------------------
# Pooled cache geometry
#
# One index block per KV block, `pool_rows = kv_cache_block_size // kpool` rows
# in each. Getting this wrong does not raise: pools land in another request's
# blocks, or in rows the gather never reads, and the model simply attends to
# the wrong keys. Nothing else in the suite touches the addressing.
# --------------------------------------------------------------------------

BLOCK_SIZE = glm5_kpool_block_size(POOL)
POOL_ROWS = BLOCK_SIZE // POOL


def test_block_size_makes_the_index_cache_exact():
    """No padding either way: the pooled rows fill the block exactly."""
    for kpool in (2, 4, 8):
        block = glm5_kpool_block_size(kpool)
        assert block % kpool == 0
        assert (block // kpool) % _MQA_LOGITS_PRESHUFFLE_ROWS == 0
        # Smallest such block: one step down and the constraint breaks.
        assert (block - kpool) // kpool % _MQA_LOGITS_PRESHUFFLE_ROWS != 0


def test_pool_rows_stay_in_the_preshuffled_layout():
    """The constraint the block size exists to satisfy.

    `deepgemm_fp8_paged_mqa_logits` is correct only preshuffled, and preshuffle
    needs a multiple of 16 rows per block. At the old 16-token block this gave 4
    and forced one index row per token; 64 gives exactly 16.
    """
    assert POOL_ROWS % _MQA_LOGITS_PRESHUFFLE_ROWS == 0
    assert BLOCK_SIZE // POOL == POOL * POOL_ROWS // POOL  # 16 pools span a block


def test_pool_slot_mapping_addresses_the_requests_own_blocks():
    block_table = torch.tensor([[7, 3], [5, 9]], dtype=torch.int32)
    pool_ids = torch.tensor([0, 1, POOL_ROWS, POOL_ROWS + 2, 0, POOL_ROWS])
    req_idx = torch.tensor([0, 0, 0, 0, 1, 1])
    got = pool_slot_mapping(block_table, pool_ids, req_idx, POOL_ROWS)
    want = torch.tensor(
        [
            7 * POOL_ROWS + 0,  # req 0, first block
            7 * POOL_ROWS + 1,
            3 * POOL_ROWS + 0,  # req 0, second block: no ::kpool striding
            3 * POOL_ROWS + 2,
            5 * POOL_ROWS + 0,  # req 1 uses its own table, not req 0's
            9 * POOL_ROWS + 0,
        ]
    )
    assert torch.equal(got, want)


def test_pool_slot_mapping_passes_negative_ids_through():
    """`-1` is how a token that closes no pool is skipped by the cache write."""
    block_table = torch.tensor([[7, 3]], dtype=torch.int32)
    pool_ids = torch.tensor([-1, 2, -1])
    req_idx = torch.tensor([0, 0, 0])
    got = pool_slot_mapping(block_table, pool_ids, req_idx, POOL_ROWS)
    assert got[0].item() == -1 and got[2].item() == -1
    assert got[1].item() == 7 * POOL_ROWS + 2


def test_every_pool_of_a_full_block_gets_a_distinct_row():
    """No collisions and no gaps: the whole point of reclaiming the space."""
    n_blocks = 4
    block_table = torch.arange(n_blocks, dtype=torch.int32).unsqueeze(0)
    pool_ids = torch.arange(n_blocks * POOL_ROWS)
    req_idx = torch.zeros(n_blocks * POOL_ROWS, dtype=torch.int64)
    slots = pool_slot_mapping(block_table, pool_ids, req_idx, POOL_ROWS)
    assert slots.unique().numel() == slots.numel()
    assert int(slots.max()) == n_blocks * POOL_ROWS - 1


def test_pooled_path_switch_is_read_in_one_place(monkeypatch):
    """Sizing and dispatch must get the same answer, so they share this."""
    monkeypatch.delenv("ATOM_GLM5_KPOOL", raising=False)
    assert pooled_path_enabled(4) is True
    assert pooled_path_enabled(1) is False
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "0")
    assert pooled_path_enabled(4) is False
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "1")
    assert pooled_path_enabled(4) is True
