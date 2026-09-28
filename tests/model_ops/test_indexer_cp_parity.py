# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU parity for the MiniMax-M3 indexer context-parallel decode selection.

The CP path replaces ONE native kernel call with a four-step chain -- score a
shard, reduce it to packed candidate keys, exchange, merge -- so "selection is
bit-identical to the TP path" is a claim about a COMPOSITION, and the pieces
were only ever checked out of tree. The feature ships default-off and is sold on
exactness, which makes an undetected divergence here the worst failure mode it
has: wrong tokens, no error.

WHAT THE REFERENCE IS. Not a 4-head native call. The TP path each rank runs
today is ``minimax_m3_index_topk_decode`` over that rank's OWN index head with
``num_kv_heads=1`` (at TP4 the impl's ``num_kv_heads`` is the per-rank count),
and the fused emit encodes the head into the page id as
``phys16*NUM_KV_HEADS + head`` -- so a 4-head native call would emit page ids
4x apart and compare unequal against the CP merge for reasons that have nothing
to do with selection. Rank h's reference is the single-head call on
``idx_q[:, h:h+1]``, which is literally what that rank computes today.

WHY THIS NEEDS NO 4 GPUs. The exchange is data movement, not math: the
all-to-all delivers ``received_h[r] = keys_r[h]``. Running the P shards
sequentially on one device and stacking that way is the same tensor the
collective would produce, so the two Triton kernels and the packed-key round
trip are covered exactly. The all-gather transport is NOT re-derived here --
``_exchange_via_all_gather`` is called for real against a stub group, so the
int64/int32 view round trip and the (src, head) index order are the shipped
code under test rather than a copy of it.
"""

import pytest
import torch

aiter = pytest.importorskip("aiter", reason="requires the AITER runtime")
pytest.importorskip("triton", reason="requires Triton")

from atom.distributed.indexer_cp import _exchange_via_all_gather
from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
    local_candidate_keys,
    merge_candidate_keys,
)
from atom.model_ops.minimax_m3.indexer_context_parallel import indexer_context_scores

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

# M3's shipped sparse geometry (config.json): 4 index heads == 4 kv heads, which
# is also the CP world size v1 requires, 128-token blocks, top-16.
WORLD = 4
BLOCK = 128
SCALE = 128**-0.5


def _inputs(batch, max_seq_len, max_query_len, cache_dtype, seed):
    """One decode step's worth of index state, laid out as production has it.

    Pages are a random permutation rather than identity so a kernel that indexes
    the block table by position instead of by page id fails here.
    """
    torch.manual_seed(seed)
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    tokens = batch * max_query_len
    idx_q = torch.randn(tokens, WORLD, BLOCK, device="cuda", dtype=torch.bfloat16)
    pages = torch.randperm(batch * blocks, device="cuda")[: batch * blocks]
    block_table = pages.to(torch.int32).view(batch, blocks).contiguous()
    cache = torch.randn(batch * blocks, BLOCK, BLOCK, device="cuda").to(cache_dtype)
    # Every request must hold at least one full query window, and the shortest
    # lengths are the interesting ones: they leave the high shards empty.
    lens = torch.randint(
        max_query_len, max_seq_len + 1, (batch,), device="cuda", dtype=torch.int32
    )
    lens[0] = max_query_len  # one request that occupies a single block
    lens[-1] = max_seq_len  # and one that fills the window
    return idx_q, cache, block_table, lens, blocks


def _tp_reference(idx_q, cache, block_table, lens, max_seq_len, topk, init, local, q):
    """What each TP rank computes today: its own head, ``num_kv_heads=1``."""
    return [
        minimax_m3_index_topk_decode(
            idx_q[:, h : h + 1],
            cache,
            block_table,
            lens,
            max_seq_len,
            topk,
            init,
            local,
            1,
            SCALE,
            emit_sparse_block_table=True,
            max_query_len=q,
        )
        for h in range(WORLD)
    ]


def _cp_candidates(idx_q, cache, block_table, lens, max_seq_len, topk, init, local, q):
    """Every shard's packed candidate keys, [rank][heads, tokens, topk]."""
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    return [
        local_candidate_keys(
            indexer_context_scores(
                idx_q, cache, block_table, lens, max_seq_len, r, WORLD, q, SCALE
            ),
            lens,
            topk,
            r,
            WORLD,
            q,
            blocks,
            init,
            local,
        )
        for r in range(WORLD)
    ]


class _StubGroup:
    """A one-process stand-in for the TP group's all-gather.

    ``custom_all_gather`` concatenates every rank's contribution along dim 0 in
    rank order, which is the whole contract ``_exchange_via_all_gather`` relies
    on. Ignoring the caller's own tensor is faithful: on a real group that
    tensor is already one of the shards being concatenated.
    """

    def __init__(self, per_rank_keys, rank):
        self._shards = per_rank_keys
        self.rank_in_group = rank

    def custom_all_gather(self, _mine):
        return torch.cat([k.view(torch.int32) for k in self._shards], dim=0)


def _exchange(keys, head, transport):
    """Route every shard's candidates for ``head`` to the rank owning it."""
    if transport == "all_to_all":
        # dist.all_to_all_single splits dim 0 and sends chunk j to rank j, so
        # rank `head` receives shard r's head-`head` slice at position r.
        return torch.stack([keys[r][head] for r in range(WORLD)])
    return _exchange_via_all_gather(keys[head], _StubGroup(keys, head))


def _assert_chain_matches_tp(
    batch,
    max_seq_len,
    q,
    topk,
    init,
    local,
    transport,
    seed,
    cache_dtype=torch.bfloat16,
):
    idx_q, cache, block_table, lens, _ = _inputs(
        batch, max_seq_len, q, cache_dtype, seed
    )
    reference = _tp_reference(
        idx_q, cache, block_table, lens, max_seq_len, topk, init, local, q
    )
    keys = _cp_candidates(
        idx_q, cache, block_table, lens, max_seq_len, topk, init, local, q
    )
    for head in range(WORLD):
        got = merge_candidate_keys(
            _exchange(keys, head, transport), block_table, lens, topk, init, local, q
        )
        want = reference[head]
        for name, a, b in zip(("topk_idx", "sparse_bt", "sparse_ctx"), got, want):
            assert torch.equal(a, b), f"head {head} {name} diverged from the TP path"


# ───────────────────────────────────────────────────────────── exactness ──


@needs_gpu
@pytest.mark.parametrize("transport", ["all_to_all", "all_gather"])
@pytest.mark.parametrize("max_query_len", [1, 4])
@pytest.mark.parametrize("max_seq_len", [128, 512, 4096, 16384])
def test_cp_selection_is_identical_to_the_tp_path(
    transport, max_query_len, max_seq_len
):
    """The exactness claim the feature is sold on, over both transports.

    ``max_seq_len=128`` is the corner that matters most: one global block over
    four shards leaves ranks 1-3 with NOTHING to score, so their candidate rows
    are pure padding and the merge has to rank that padding below every real
    key. ``max_query_len=4`` is spec decode (EAGLE3 with 3 draft tokens), where
    each query token carries its own causal cutoff.
    """
    _assert_chain_matches_tp(
        batch=8,
        max_seq_len=max_seq_len,
        q=max_query_len,
        topk=16,
        init=1,
        local=2,
        transport=transport,
        seed=17,
    )


@needs_gpu
@pytest.mark.parametrize("init, local", [(0, 0), (1, 2), (2, 4)])
def test_forced_blocks_survive_the_round_trip(init, local):
    """Sink and sliding-window blocks are pinned twice, and must be.

    A forced block that loses its own shard's top-k never reaches the merge to
    be pinned there, so ``_local_topk`` pins as well -- with counts that have to
    match the merge's. ``(0, 0)`` pins nothing and is the control.
    """
    _assert_chain_matches_tp(
        batch=4,
        max_seq_len=4096,
        q=1,
        topk=16,
        init=init,
        local=local,
        transport="all_to_all",
        seed=23,
    )


@needs_gpu
@pytest.mark.parametrize("topk", [4, 16])
def test_parity_holds_across_top_k(topk):
    """topk sizes the exchange payload and both kernels' selection width."""
    _assert_chain_matches_tp(
        batch=6,
        max_seq_len=8192,
        q=1,
        topk=topk,
        init=1,
        local=2,
        transport="all_to_all",
        seed=29,
    )


@needs_gpu
@pytest.mark.parametrize("max_query_len", [1, 4])
def test_parity_holds_on_an_fp8_index_cache(max_query_len):
    """The one dtype where the two paths do NOT share a dot formulation.

    ``--index-cache-dtype fp8`` is what recipes/MiniMax-M3.md runs, and it is
    the only input that makes the CP scorer and the native selector compute the
    score differently rather than identically: the native kernel has a dedicated
    fp8 branch that casts the QUERY DOWN to fp8 and multiplies in fp8
    (``index_topk.py`` ``if k.dtype.is_fp8()``), while ``_context_score`` casts
    the KEY UP with a plain ``.to(q.dtype)`` and multiplies in bf16. Nothing
    guarantees a priori that two different products rank 512 blocks the same
    way, which is why this is a separate test and not another parametrize case
    on the bf16 one.

    Measured before it was written, in the discriminating regime (65536 ctx =
    512 blocks, top-16, so the selection actually excludes something): zero
    divergence in top-k indices, sparse_bt and sparse_ctx over 25,600 rows
    across 40 seeds, both fp8 containers and both query lengths. That is past
    the ~1-per-25k rate at which ``_pack_score_key`` says fp8 blocks tie on one
    fp32 score. The test pins the result rather than the reasoning: if either
    side's dot changes, this fails instead of silently shifting which blocks
    attention reads.

    ``aiter.dtypes.fp8`` rather than a hardcoded ``float8_e4m3fn`` because the
    two ROCm archs disagree on which one is native, and this must test the
    container the runtime actually allocates.
    """
    _assert_chain_matches_tp(
        batch=8,
        max_seq_len=16384,
        q=max_query_len,
        topk=16,
        init=1,
        local=2,
        transport="all_to_all",
        seed=37,
        cache_dtype=aiter.dtypes.fp8,
    )


@needs_gpu
def test_both_transports_deliver_the_same_tensor():
    """The two transports must be interchangeable, not merely both correct.

    They are picked by payload size at runtime, so a layout bug in the
    all-gather's int64/int32 view round trip would surface only above or only
    below the threshold -- and only in the arm nobody benchmarked.
    """
    idx_q, cache, block_table, lens, _ = _inputs(4, 4096, 1, torch.bfloat16, 31)
    keys = _cp_candidates(idx_q, cache, block_table, lens, 4096, 16, 1, 2, 1)
    for head in range(WORLD):
        a2a = _exchange(keys, head, "all_to_all")
        gathered = _exchange(keys, head, "all_gather")
        assert torch.equal(a2a, gathered), f"transports disagree for head {head}"
