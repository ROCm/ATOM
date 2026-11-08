# SPDX-License-Identifier: MIT
"""Indexer-only CP over MiniMax-M3's MXFP4 index cache (gfx950).

Each CP rank scores every index head over its 1/P of the blocks with aiter's
MXFP4 scorer, keeps its shard's top-k, and the rank owning head h merges every
shard's head-h candidates. The four ranks run in one process here, with the
exchange done by indexing, so what is pinned is the selection itself: the
merged top-k, the page-128 sparse block table and its context must equal the
native TP selector's for the same head, bit for bit.
"""

import pytest
import torch

from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter.ops, "flydsl"),
    reason="needs gfx950 and aiter's FlyDSL",
)

TOPK, WORLD, DIM = 16, 4, 128
SM_SCALE = DIM**-0.5


def make_case(batch, S, seed, max_len, fp8_q=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq = torch.randint(S + 1, max_len, (batch,), device="cuda", generator=g)
    seq[0] = max_len
    blocks = -(-max_len // 128)
    pages = batch * blocks + 3
    cache = torch.randint(
        0, 256, (pages, PAGE_BYTES), device="cuda", generator=g, dtype=torch.uint8
    )
    d = cache.view(pages, 1, PAGE_BYTES)[:, :, 8192:]
    d.copy_(
        torch.randint(125, 130, d.shape, device="cuda", generator=g).to(torch.uint8)
    )
    table = torch.stack(
        [
            torch.randperm(pages, device="cuda", generator=g)[:blocks]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    q = torch.randn(batch * S, WORLD, DIM, device="cuda", generator=g) * 0.7
    q = q.to(torch.float8_e4m3fn) if fp8_q else q.to(torch.bfloat16)
    return cache, table, seq.to(torch.int32), q


def native_head(cache, table, seq, q, S, head, init_blocks, local_blocks):
    from atom.model_ops.minimax_m3.index_mxfp4 import (
        build_index_work_map,
        index_work_map_capacity,
    )
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    max_block = table.shape[1]
    buf = torch.empty(
        (index_work_map_capacity(seq.numel(), max_block, S, 1), 2),
        dtype=torch.int32,
        device="cuda",
    )
    wmap = build_index_work_map(
        seq, max_block=max_block, max_query_len=S, num_idx_heads=1, out=buf
    )
    return minimax_m3_index_topk_decode(
        q[:, head : head + 1].contiguous(), cache, table, seq, int(seq.max()),
        TOPK, init_blocks, local_blocks, 1, SM_SCALE,
        emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
        index_mxfp4=True, pages_per_block=1,
        index_score_work_map=wmap, index_score_max_block=max_block,
    )  # fmt: skip


def cp_shard_keys(cache, table, seq, q, S, rank, init_blocks, local_blocks):
    from atom.model_ops.minimax_m3.index_mxfp4 import (
        build_index_work_map,
        decode_score_mxfp4,
        fresh_index_score,
        index_score_buffer,
        index_work_map_capacity,
    )
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        local_candidate_keys,
    )

    rows = q.shape[0]
    local_bound = -(-table.shape[1] // WORLD)
    buf = torch.empty(
        (index_work_map_capacity(seq.numel(), local_bound, S, WORLD, WORLD, rank), 2),
        dtype=torch.int32,
        device="cuda",
    )
    wmap = build_index_work_map(
        seq,
        max_block=local_bound,
        max_query_len=S,
        num_idx_heads=WORLD,
        out=buf,
        cp_world=WORLD,
        cp_rank=rank,
    )
    out = fresh_index_score(
        index_score_buffer(rows, WORLD, local_bound, "cuda"),
        rows,
        WORLD,
        local_bound,
        S,
        page_contiguous=True,
    )
    scores = decode_score_mxfp4(
        q, cache, table, seq, local_bound, S, WORLD, SM_SCALE, wmap, out,
        cp_world=WORLD, cp_rank=rank,
    )  # fmt: skip
    assert scores.is_contiguous()
    return local_candidate_keys(
        scores, seq, TOPK, rank, WORLD, S, table.shape[1], init_blocks, local_blocks
    )


@pytest.mark.parametrize(
    "batch,S,max_len",
    [(1, 8, 128 * 70), (5, 8, 128 * 70), (17, 8, 128 * 300 + 77), (6, 1, 128 * 41)],
)
@pytest.mark.parametrize("forced", [(0, 1), (1, 1)])
def test_cp_selection_equals_native(batch, S, max_len, forced):
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        merge_candidate_keys,
    )

    init_blocks, local_blocks = forced
    cache, table, seq, q = make_case(batch, S, batch * 100 + S, max_len)
    shard_keys = [
        cp_shard_keys(cache, table, seq, q, S, r, init_blocks, local_blocks)
        for r in range(WORLD)
    ]
    for head in range(WORLD):
        want_idx, want_bt, want_ctx = native_head(
            cache, table, seq, q, S, head, init_blocks, local_blocks
        )
        gathered = torch.stack([k[head] for k in shard_keys])
        idx, sbt, sctx = merge_candidate_keys(
            gathered, table, seq, TOPK, init_blocks, local_blocks, S,
            pages_per_block=1, block_page_stride=1,
        )  # fmt: skip
        assert torch.equal(idx, want_idx), f"head {head} top-k"
        assert torch.equal(sctx, want_ctx), f"head {head} ctx"
        assert torch.equal(sbt, want_bt), f"head {head} sparse block table"
