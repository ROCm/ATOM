# SPDX-License-Identifier: MIT
"""Indexer-only CP decode over the NVFP4 index cache, against the TP selection.

On gfx950 this emulates the four ranks on one GPU: each rank scores every index
head over its blocks ``b % 4 == rank`` with aiter's FlyDSL NVFP4 scorer (its shard
map built by ``make_sparse_decode_metadata``, its scores in a workspace filled with
values no selector may read), reduces them to its own top-k keys, and the merge of
the four ranks' keys for head ``h`` must equal the TP selection of head ``h``
(``minimax_m3_index_topk_decode`` with ``index_nvfp4``) bit for bit. The NVFP4 KV
reads the logical block ids, so the merge emits no page-16 block table.
"""

import pytest
import torch

from atom.config import indexer_cp_unsupported_reason

WORLD = 4
DIM = 128
TOPK = 16
SM_SCALE = DIM**-0.5
PAGE_BYTES = 9216  # 128 tokens x (64 B e2m1 + 8 B e4m3 scales)


def test_the_gate_admits_the_nvfp4_caches_and_refuses_mxfp4():
    args = (["MiniMaxM3ForCausalLM"], WORLD, WORLD, 128, 1, False, False)
    assert indexer_cp_unsupported_reason(*args, mxfp4_caches=False) is None
    assert indexer_cp_unsupported_reason(*args, mxfp4_caches=True) is not None


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() == "gfx950"
    except Exception:
        return False


gfx950 = pytest.mark.skipif(not _gfx950(), reason="the FlyDSL scorer needs gfx950")


def make_case(batch, S, seed, max_len, fp8_q):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq = torch.randint(S + 1, max_len, (batch,), device="cuda", generator=g)
    seq[0] = max_len  # one full-length request
    blocks = -(-max_len // 128)
    pages = batch * blocks + 3
    cache = torch.randint(
        0, 256, (pages, PAGE_BYTES), device="cuda", generator=g, dtype=torch.uint8
    )
    # positive e4m3 scale bytes around unity (0x30 = 0.5 .. 0x40 = 2), never 0x7F
    sc = cache[:, 8192:]
    sc.copy_(torch.randint(0x30, 0x41, sc.shape, device="cuda", generator=g).to(torch.uint8))
    table = torch.stack(
        [
            torch.randperm(pages, device="cuda", generator=g)[:blocks]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    q = torch.randn(batch * S, WORLD, DIM, device="cuda", generator=g) * 0.7
    q = q.to(torch.float8_e4m3fn) if fp8_q else q.to(torch.bfloat16)
    return cache, table, seq.to(torch.int32), q


def native_head(cache, table, seq, q, S, head, forced):
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    return minimax_m3_index_topk_decode(
        q[:, head : head + 1].contiguous(), cache, table, seq, int(seq.max()), TOPK,
        *forced, 1, SM_SCALE, emit_sparse_block_table=False, max_query_len=S,
        index_nvfp4=True,
    )  # fmt: skip


def cp_shard_keys(cache, table, seq, q, S, rank, forced):
    from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity

    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        local_candidate_keys,
    )
    from atom.model_ops.minimax_m3.indexer_context_parallel import (
        indexer_context_scores_nvfp4,
    )
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata
    from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

    batch, blocks = table.shape
    shard = {"cp_world": WORLD, "cp_rank": rank}
    local = -(-blocks // WORLD)  # the builder's bound: the model length's shard
    rows = minimax_m3_index_score_work_map_capacity(batch, local, S, WORLD, **shard)
    map_buf = torch.empty((rows, 2), dtype=torch.int32, device="cuda")
    ws = LogitsWorkspace(WORLD * batch * S * local, "cuda")
    flat = ws.logits(1, WORLD * batch * S * local)
    junk = torch.tensor([float("nan"), float("inf"), -float("inf"), 3e38], device="cuda")
    flat.copy_(junk[torch.randint(0, 4, flat.shape, device="cuda")])
    md = make_sparse_decode_metadata(
        seq_lens=seq,
        block_table=table,
        slot_mapping=torch.zeros(batch * S, dtype=torch.int64, device="cuda"),
        max_seq_len=blocks * 128,
        num_idx_heads=1,
        max_query_len=S,
        index_work_map_out=map_buf,
        index_work_map_max_block=local,
        index_score_workspace=ws,
        index_cp=(WORLD, rank),
    )
    scores = indexer_context_scores_nvfp4(
        q, cache, table, seq, int(seq.max()), rank, WORLD, S, SM_SCALE,
        md.index_work_map, md.index_work_map_max_block, md.index_score_workspace,
    )  # fmt: skip
    return local_candidate_keys(
        scores, seq, TOPK, rank, WORLD, S, table.shape[1], *forced
    )


@gfx950
@pytest.mark.parametrize("forced", [(0, 1), (1, 1)])
@pytest.mark.parametrize(
    "batch,S,max_len,fp8_q",
    [
        (1, 8, 128 * 70, True),
        (5, 8, 128 * 70, True),
        (17, 8, 128 * 300 + 77, True),  # 301 blocks: shards of 76, 75, 75, 75
        (24, 8, 128 * 1200 + 5, True),  # c48-like: ~150K context, 24 requests
        (6, 1, 128 * 41, False),
    ],
)
def test_cp_selection_equals_native(batch, S, max_len, fp8_q, forced):
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        merge_candidate_keys,
    )

    cache, table, seq, q = make_case(batch, S, batch * 100 + S, max_len, fp8_q)
    shard_keys = [
        cp_shard_keys(cache, table, seq, q, S, rank, forced) for rank in range(WORLD)
    ]
    for head in range(WORLD):
        want = native_head(cache, table, seq, q, S, head, forced)
        got, sbt, ctx = merge_candidate_keys(
            torch.stack([keys[head] for keys in shard_keys]), table, seq, TOPK,
            *forced, S, emit_sparse_block_table=False,
        )  # fmt: skip
        assert sbt is None and ctx is None
        assert torch.equal(got, want), f"head {head}"
