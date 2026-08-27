# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Semantics of the GLM-5.3-Flash k-pool DSA indexer.

These run on CPU with synthetic weights and assert the invariants the sparse
attention mask depends on: causality, padding, the token budget, and that pool
selection really expands back to whole pools.

Exact numerical parity against ``transformers``' ``Glm5NextTextIndexer`` needs the
real checkpoint and a GPU, so it is not run here -- see the parity procedure in
``recipes/GLM-5.3-Flash.md``.
"""

import torch

from atom.model_ops.kpool_indexer import build_kpools, kpool_topk_indices

HEAD_DIM = 8
N_HEADS = 4
KPOOL = 4


def _inputs(batch, seq, n_left_pad=0, seed=0):
    torch.manual_seed(seed)
    valid = torch.ones(batch, seq, dtype=torch.bool)
    if n_left_pad:
        valid[:, :n_left_pad] = False
    pos = torch.arange(seq)
    visible = (pos[None, None, :] <= pos[None, :, None]) & valid[:, None, :]
    return {
        "q": torch.randn(batch, seq, N_HEADS, HEAD_DIM),
        "keys": torch.randn(batch, seq, HEAD_DIM),
        "gate_scores": torch.randn(batch, seq, HEAD_DIM),
        "head_weights": torch.randn(batch, seq, N_HEADS),
        "valid_keys": valid,
        "visible": visible,
        "ape": torch.randn(KPOOL, HEAD_DIM),
        "softmax_scale": HEAD_DIM**-0.5,
    }


def _run(index_topk, tail=True, **kw):
    return kpool_topk_indices(
        index_topk=index_topk, kpool=KPOOL, always_select_tail=tail, **kw
    )


def test_output_width_is_topk_plus_tail():
    out = _run(16, **_inputs(1, 40))
    assert out.shape == (1, 40, 16 + KPOOL - 1)
    assert out.dtype == torch.int32

    no_tail = _run(16, tail=False, **_inputs(1, 40))
    assert no_tail.shape == (1, 40, 16)


def test_selection_is_causal_and_skips_padding():
    n_pad = 6
    inp = _inputs(2, 48, n_left_pad=n_pad)
    out = _run(16, **inp)
    for b in range(out.shape[0]):
        for t in range(out.shape[1]):
            picked = out[b, t]
            picked = picked[picked >= 0]
            assert torch.all(picked <= t), "indexer selected a future token"
            assert torch.all(picked >= n_pad), "indexer selected a padding token"


def test_pools_are_selected_whole():
    """A selected complete pool contributes all `kpool` of its token indices."""
    inp = _inputs(1, 64)
    out = _run(16, tail=False, **inp)
    # Look at the last query, which can see every pool.
    picked = sorted(x for x in out[0, -1].tolist() if x >= 0)
    assert picked, "expected a non-empty selection"
    by_pool: dict[int, list[int]] = {}
    for idx in picked:
        by_pool.setdefault(idx // KPOOL, []).append(idx)
    for pool_id, members in by_pool.items():
        assert (
            len(members) == KPOOL
        ), f"pool {pool_id} only partially expanded: {members}"


def test_tail_covers_the_incomplete_pool():
    """Tokens in the trailing partial pool are always reachable."""
    seq = 4 * KPOOL + 2  # two tokens past the last complete pool
    inp = _inputs(1, seq)
    out = _run(4096, **inp)  # budget far exceeds the sequence: everything visible
    picked = {x for x in out[0, -1].tolist() if x >= 0}
    assert {seq - 2, seq - 1} <= picked, "trailing partial-pool tokens were dropped"


def test_budget_is_respected_when_pools_exceed_it():
    seq = 512
    inp = _inputs(1, seq)
    topk = 16
    out = _run(topk, tail=False, **inp)
    picked = out[0, -1]
    assert int((picked >= 0).sum()) <= topk
    # topk // kpool pools are chosen, each expanding to kpool tokens.
    assert int((picked >= 0).sum()) == topk


def test_build_kpools_anchors_at_first_real_token():
    """Left padding must not shift pool boundaries."""
    seq, pad = 32, 3
    torch.manual_seed(1)
    keys = torch.randn(1, seq, HEAD_DIM)
    gate = torch.randn(1, seq, HEAD_DIM)
    ape = torch.randn(KPOOL, HEAD_DIM)

    valid = torch.ones(1, seq, dtype=torch.bool)
    valid[:, :pad] = False
    _, idx, pool_valid = build_kpools(keys, gate, valid, ape, KPOOL)

    first_pool = idx[0, 0][idx[0, 0] >= 0].tolist()
    assert first_pool == [pad, pad + 1, pad + 2, pad + 3]
    assert bool(pool_valid[0, 0])


def test_fully_padded_row_selects_nothing():
    inp = _inputs(1, 32, n_left_pad=32)
    out = _run(16, **inp)
    assert int((out >= 0).sum()) == 0
