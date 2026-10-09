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
from atom.model_ops.minimax_m3.index_topk import (
    build_index_score_work_map,
    minimax_m3_index_topk_decode,
)
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


def _shard_score(idx_q, cache, block_table, lens, max_seq_len, r, q, pad):
    """One rank's shard scores, with a map built ``pad`` blocks wider than needed.

    That width is not an embellishment: the map's row count IS the scorer's grid
    and a captured decode cannot move its grid, so production bounds the map by
    the MODEL length while the batch is whatever it is. ``pad > 0`` is that case
    -- the shard carries dead trailing blocks, which the selector must rank below
    every real one. ``pad == 0`` is the batch-tight bound.
    """
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    local = (blocks + WORLD - 1) // WORLD
    work_map = build_index_score_work_map(
        lens,
        max_block=local + pad,
        max_query_len=q,
        num_idx_heads=WORLD,
        cp_world=WORLD,
        cp_rank=r,
    )
    scores = indexer_context_scores(
        idx_q, cache, block_table, lens, max_seq_len, r, WORLD, q, SCALE,
        work_map=work_map, max_block=local + pad,
    )  # fmt: skip
    assert scores.shape[2] == local + pad, "the scorer ignored the bound it was given"
    return scores


def _cp_candidates(
    idx_q, cache, block_table, lens, max_seq_len, topk, init, local, q, pad=0
):
    """Every shard's packed candidate keys, [rank][heads, tokens, topk]."""
    blocks = (max_seq_len + BLOCK - 1) // BLOCK
    return [
        local_candidate_keys(
            _shard_score(idx_q, cache, block_table, lens, max_seq_len, r, q, pad),
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
    pad=0,
):
    idx_q, cache, block_table, lens, _ = _inputs(
        batch, max_seq_len, q, cache_dtype, seed
    )
    reference = _tp_reference(
        idx_q, cache, block_table, lens, max_seq_len, topk, init, local, q
    )
    keys = _cp_candidates(
        idx_q, cache, block_table, lens, max_seq_len, topk, init, local, q, pad
    )
    for head in range(WORLD):
        got = merge_candidate_keys(
            _exchange(keys, head, transport), block_table, lens, topk, init, local, q
        )
        want = reference[head]
        for name, a, b in zip(("topk_idx", "sparse_bt", "sparse_ctx"), got, want):
            assert torch.equal(a, b), f"head {head} {name} diverged from the TP path"


# The two bounds a shard can be scored against, tight and model-length-padded.
# Everything downstream of the score is shared, so parametrizing here subjects
# the padded case -- the one production actually captures -- to the same
# exactness claim as the tight one.
#
# Both arms now run the same kernel, and so does `_tp_reference`: there is one
# decode scorer left. That makes this file's parity claim a claim about the
# CHAIN around it (shard, reduce, exchange, merge), not about the score, and
# leaves it blind to a mistake the scorer makes on both sides. That gap is
# covered by `test_the_shard_scores_match_the_definition` below, which is
# anchored on torch rather than on a second kernel.
SCORERS = pytest.mark.parametrize("pad", [0, 3], ids=["tight", "padded"])


# ───────────────────────────────────────────────────────────── exactness ──


@needs_gpu
@SCORERS
@pytest.mark.parametrize("transport", ["all_to_all", "all_gather"])
@pytest.mark.parametrize("max_query_len", [1, 4])
@pytest.mark.parametrize("max_seq_len", [128, 512, 4096, 16384])
def test_cp_selection_is_identical_to_the_tp_path(
    transport, max_query_len, max_seq_len, pad
):
    """The exactness claim the feature is sold on, over both transports.

    ``max_seq_len=128`` is the corner that matters most: one global block over
    four shards leaves ranks 1-3 with NOTHING to score, so their candidate rows
    are pure padding and the merge has to rank that padding below every real
    key. ``max_query_len=4`` is spec decode (EAGLE3 with 3 draft tokens), where
    each query token carries its own causal cutoff.

    That corner is also where the scorer leaves the most behind: it writes only
    the blocks a request has, so an empty shard's row is uninitialized memory
    start to finish. Correct only because `_pack_score_key` sends masked lanes
    to key 0, and this is the test that says so.
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
        pad=pad,
    )


@needs_gpu
@SCORERS
@pytest.mark.parametrize("init, local", [(0, 0), (1, 2), (2, 4)])
def test_forced_blocks_survive_the_round_trip(init, local, pad):
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
        pad=pad,
    )


@needs_gpu
@SCORERS
@pytest.mark.parametrize("topk", [4, 16])
def test_parity_holds_across_top_k(topk, pad):
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
        pad=pad,
    )


@needs_gpu
@SCORERS
@pytest.mark.parametrize("max_query_len", [1, 4])
def test_parity_holds_on_an_fp8_index_cache(max_query_len, pad):
    """``--index-cache-dtype fp8`` is what recipes/MiniMax-M3.md runs.

    It used to be the one dtype where the two sides did not share a dot
    formulation -- the old Triton ``_context_score`` cast the KEY UP to bf16
    while the native selector's fp8 branch cast the QUERY DOWN -- and this test
    existed to show that two different products still ranked 512 blocks the same
    way. Both sides run the same kernel now, so it no longer pins that. What it
    still pins is the shard/exchange/merge chain under fp8, where the scores are
    closer together than in bf16 and ties are ~1 per 25k rows; `_pack_score_key`
    has to break them identically on both sides.

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
        pad=pad,
    )


@needs_gpu
def test_both_transports_deliver_the_same_tensor():
    """The two transports must be interchangeable, not merely both correct.

    They are picked by payload size at runtime, so a layout bug in the
    all-gather's int64/int32 view round trip would surface only above or only
    below the threshold -- and only in the arm nobody benchmarked.
    """
    idx_q, cache, block_table, lens, _ = _inputs(4, 4096, 1, torch.bfloat16, 31)
    keys = _cp_candidates(idx_q, cache, block_table, lens, 4096, 16, 1, 2, 1, pad=0)
    for head in range(WORLD):
        a2a = _exchange(keys, head, "all_to_all")
        gathered = _exchange(keys, head, "all_gather")
        assert torch.equal(a2a, gathered), f"transports disagree for head {head}"


@needs_gpu
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, "fp8"])
@pytest.mark.parametrize("max_query_len", [1, 4])
def test_the_shard_scores_match_the_definition(max_query_len, cache_dtype):
    """The anchor the rest of this file no longer has.

    Every other test here compares the CP chain against `_tp_reference`, and
    both now run the same scorer, so a mistake inside it cancels. This one is
    anchored on torch: score a shard, then check each live column against the
    max over that block's 128 tokens of q.k*scale, causal-masked -- with the
    round-robin remap ``global = local*WORLD + rank`` done here rather than
    assumed, since that remap is the one thing CP adds to the score.

    Compared in the kernel's base-2 units, and only where it is required to
    write: the scorer leaves every block past a request's length untouched.
    """
    dtype = aiter.dtypes.fp8 if cache_dtype == "fp8" else cache_dtype
    max_seq_len, batch = 4096, 4
    idx_q, cache, block_table, lens, blocks = _inputs(
        batch, max_seq_len, max_query_len, dtype, 41
    )
    local = (blocks + WORLD - 1) // WORLD
    scale = SCALE * 1.4426950408889634
    within = torch.arange(BLOCK, device="cuda")
    for rank in range(WORLD):
        got = _shard_score(
            idx_q, cache, block_table, lens, max_seq_len, rank, max_query_len, 0
        )
        for b in range(batch):
            length = int(lens[b])
            rows = slice(b * max_query_len, (b + 1) * max_query_len)
            q = idx_q[rows].reshape(-1, BLOCK).float()  # [S*H, D], col = tok*H+head
            cuts = (
                length
                - max_query_len
                + torch.arange(max_query_len, device="cuda").repeat_interleave(WORLD)
                + 1
            )
            for p in range(local):
                gp = p * WORLD + rank  # this rank owns global block gp
                if gp * BLOCK >= length:
                    break  # past the request: the kernel need not have written
                k = cache[int(block_table[b, gp])].to(idx_q.dtype).float()
                z = (k @ q.T) * scale
                z = z.masked_fill(
                    (gp * BLOCK + within)[:, None] >= cuts[None, :], -torch.inf
                )
                want = z.amax(0).reshape(max_query_len, WORLD).T
                assert torch.allclose(
                    got[:, rows, p], want, rtol=2e-2, atol=2e-2
                ), f"rank {rank} request {b} local block {p} (global {gp})"
