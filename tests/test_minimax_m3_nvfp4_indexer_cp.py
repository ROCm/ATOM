# SPDX-License-Identifier: MIT
"""Indexer-only CP over MiniMax-M3's NVFP4 index cache.

The CP decode scores each rank's round-robin shard of the blocks with aiter's NVFP4
scorer, as it does with the MXFP4 one; selection and the candidate merge are the
existing CP path's. The GPU test runs the four ranks in one process and checks each
shard's scores against the TP decode's (`minimax_m3_index_topk_decode` with the step's
NVFP4 scorer record) bit for bit on every live block, and the merged top-k, with no
sparse block table (the NVFP4 KV reads block ids), against the TP decode's.
"""

import types

import pytest
import torch

from atom.model_ops.minimax_m3.nvfp4_pages import (
    INDEX_DATA_PAGE_BYTES,
    INDEX_PAGE_BYTES,
)

aiter = pytest.importorskip("aiter")
gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter.ops, "flydsl"),
    reason="needs gfx950 and aiter's FlyDSL",
)

WORLD, TOPK, DIM = 4, 16, 128
SM_SCALE = DIM**-0.5
INIT, LOCAL = 0, 1  # M3's sparse_init_block / sparse_local_block


def _config(blocks):
    return types.SimpleNamespace(
        max_model_len=blocks * 128,
        max_num_batched_tokens=1,
        index_cache_dtype="nvfp4",
        enable_tbo=False,
        enable_tbo_decode=False,
    )


def make_case(batch, S, blocks, seed, ties):
    """NVFP4 index pages, a block table of shuffled pages, lengths and e4m3 index q.

    ``ties``: "none" (random codes) or "coarse" (four codes, one scale: equal block
    maxima are common).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = batch * blocks + 3
    cache = torch.randint(
        0, 256, (pages, INDEX_PAGE_BYTES), device="cuda", generator=g, dtype=torch.uint8
    )
    scale = cache[:, INDEX_DATA_PAGE_BYTES:]
    # positive e4m3 scales around 1 (0x38), as a quantized activation's are
    scale.copy_(torch.randint(0x30, 0x41, scale.shape, device="cuda", generator=g))
    if ties == "coarse":
        data = cache[:, :INDEX_DATA_PAGE_BYTES]
        codes = torch.tensor([0x00, 0x22, 0x44, 0x26], device="cuda", dtype=torch.uint8)
        data.copy_(codes[torch.randint(0, 4, data.shape, device="cuda", generator=g)])
        scale.fill_(0x38)
    table = torch.stack(
        [
            torch.randperm(pages, device="cuda", generator=g)[:blocks]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    seq = torch.randint(S + 1, blocks * 128 + 1, (batch,), device="cuda", generator=g)
    seq[0] = blocks * 128  # one request at the model length
    seq[-1] = S + 1  # and one in a single block
    q = (torch.randn(batch * S, WORLD, DIM, device="cuda", generator=g) * 0.7).to(
        torch.float8_e4m3fn
    )
    return cache, table, seq.to(torch.int32), q


def _step(table, seq, S, cp):
    """This rank's builder record and the step's metadata, as the builder and
    `make_sparse_decode_metadata` make them (the workspace filled with values no
    score may depend on)."""
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    batch, blocks = table.shape
    record = allocate_index_scorers(
        _config(blocks), "cuda", max_bs=batch, max_qlen=S, num_idx_heads=1, cp=cp
    )[""]
    flat = record.scores.logits(1, record.scores.capacity)
    junk = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 3e38], device="cuda"
    )
    flat.copy_(junk[torch.randint(0, 4, flat.shape, device="cuda")])
    return make_sparse_decode_metadata(
        seq_lens=seq,
        block_table=table,
        slot_mapping=torch.zeros(batch * S, dtype=torch.int64, device="cuda"),
        max_seq_len=blocks * 128,
        num_idx_heads=1,
        max_query_len=S,
        index_scorer=record,
    )


def tp_scores(cache, table, seq, q, S, head):
    """The scores [rows, blocks] TP rank ``head``'s decode selects from."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    md = _step(table, seq, S, (1, 0))
    minimax_m3_index_topk_decode(
        q[:, head : head + 1].contiguous(),
        cache,
        table,
        seq,
        table.shape[1] * 128,
        TOPK,
        INIT,
        LOCAL,
        1,
        SM_SCALE,
        emit_sparse_block_table=False,
        max_query_len=S,
        n_valid_column_per_row=md.n_valid_column_per_row,
        index_scorer=md.index_scorer,
    )
    rows, blocks = q.shape[0], table.shape[1]
    scores = md.index_scorer.scores.logits(rows, md.index_scorer.max_block)
    return scores.view(rows, -1)[:, :blocks].clone()


def cp_scores(cache, table, seq, q, S, rank):
    from atom.model_ops.minimax_m3.indexer_context_parallel import (
        indexer_context_scores_fp4,
    )

    md = _step(table, seq, S, (WORLD, rank))
    return indexer_context_scores_fp4(
        q, cache, table, seq, table.shape[1] * 128, S, SM_SCALE, md.index_scorer
    ).clone()


def test_the_cp_scorer_hands_aiter_the_shard(monkeypatch):
    """The NVFP4 CP decode scores with aiter's public NVFP4 scorer: every index head of the
    step's rows over the shard's local blocks, the cache's two page views, the step's map
    and the record's workspace, and the shard. The call binds to the installed aiter's
    signature, so an aiter whose scorer does not take the shard (ROCm/aiter#8285) fails
    here, with no GPU."""
    import inspect

    flydsl = pytest.importorskip("aiter.ops.flydsl")
    from atom.model_ops.minimax_m3 import index_topk
    from atom.model_ops.minimax_m3 import indexer_context_parallel as icp
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers
    from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

    signature = inspect.signature(flydsl.minimax_m3_index_score_nvfp4_flydsl)
    seen = {}

    def scorer(*args, **kwargs):
        signature.bind(*args, **kwargs)
        seen.update(args=args, kwargs=kwargs)
        return kwargs["out"]

    ops = types.SimpleNamespace(
        minimax_m3_index_score_prefill_nvfp4_flydsl=None,
        minimax_m3_index_score_nvfp4_flydsl=scorer,
    )
    monkeypatch.setattr(index_topk, "flydsl_ops", lambda: ops)
    monkeypatch.setattr(index_topk, "nvfp4_page_views", lambda cache: ("data", "scale"))
    batch, S, local, rank = 3, 2, 5, 1
    rows = batch * S
    wmap = torch.zeros(4, 2, dtype=torch.int32)
    record = IndexScoreBuffers(
        LogitsWorkspace(WORLD * rows * local, "cpu"),
        wmap,
        local,
        nvfp4=True,
        cp_world=WORLD,
        cp_rank=rank,
    )
    q = torch.zeros(rows, WORLD, DIM)
    table = torch.zeros(batch, WORLD * local, dtype=torch.int32)
    seq = torch.ones(batch, dtype=torch.int32)
    out = icp.indexer_context_scores_fp4(
        q, "cache", table, seq, WORLD * local * 128, S, 0.5, record
    )
    assert out.shape == (WORLD, rows, local)
    args, kwargs = seen["args"], seen["kwargs"]
    assert args[0] is q and args[1:3] == ("data", "scale") and args[3] is table
    assert args[4] is seq and args[5:] == (S, WORLD, 0.5, local)
    assert kwargs["work_map"] is wmap and kwargs["out"] is out
    assert (kwargs["cp_world"], kwargs["cp_rank"]) == (WORLD, rank)


@gfx950
@pytest.mark.parametrize("ties", ["none", "coarse"])
@pytest.mark.parametrize(
    "batch,S,blocks",
    [
        (4, 8, 300),
        (64, 8, 1368),  # 175K, B=64 DSpark-7
        (3, 1, 2050),  # plain decode, shards of 513 / 513 / 512 / 512
    ],
)
def test_cp_shard_scores_equal_tp(batch, S, blocks, ties):
    cache, table, seq, q = make_case(batch, S, blocks, batch * 1000 + blocks, ties)
    shards = [cp_scores(cache, table, seq, q, S, r) for r in range(WORLD)]
    causal = seq.repeat_interleave(S) - S + torch.arange(S, device="cuda").repeat(batch)
    for h in range(WORLD):
        tp = tp_scores(cache, table, seq, q, S, h)
        for r in range(WORLD):
            mine = shards[r][h][:, : -(-(blocks - r) // WORLD)]
            ref = tp[:, r::WORLD]
            live = (torch.arange(r, blocks, WORLD, device="cuda")[None, :] * 128) < (
                causal[:, None] + 1
            )
            assert torch.equal(
                torch.where(live, mine.view(torch.int32), 0),
                torch.where(live, ref.view(torch.int32), 0),
            ), f"head {h} shard {r} scores"


@gfx950
@pytest.mark.parametrize("ties", ["none", "coarse"])
@pytest.mark.parametrize("batch,S,blocks", [(4, 8, 300), (3, 1, 2050)])
def test_cp_topk_equals_tp_with_no_sparse_block_table(batch, S, blocks, ties):
    """The four shards' candidates, merged with no sparse block table as the NVFP4 KV
    runs it, give every head's TP top-k: row by row, as sets."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        local_candidate_keys,
        merge_candidate_keys,
    )

    cache, table, seq, q = make_case(batch, S, blocks, batch * 1000 + blocks + 1, ties)
    keys = [
        local_candidate_keys(
            cp_scores(cache, table, seq, q, S, r),
            seq,
            TOPK,
            r,
            WORLD,
            S,
            blocks,
            INIT,
            LOCAL,
        )
        for r in range(WORLD)
    ]
    for h in range(WORLD):
        md = _step(table, seq, S, (1, 0))
        want = minimax_m3_index_topk_decode(
            q[:, h : h + 1].contiguous(),
            cache,
            table,
            seq,
            blocks * 128,
            TOPK,
            INIT,
            LOCAL,
            1,
            SM_SCALE,
            emit_sparse_block_table=False,
            max_query_len=S,
            n_valid_column_per_row=md.n_valid_column_per_row,
            index_scorer=md.index_scorer,
        )
        got = merge_candidate_keys(
            torch.stack([k[h] for k in keys]),
            table,
            seq,
            TOPK,
            INIT,
            LOCAL,
            S,
            emit_sparse_block_table=False,
        )
        assert got.shape == want.shape, f"head {h}"
        assert torch.equal(got.sort(-1).values, want.sort(-1).values), f"head {h}"
