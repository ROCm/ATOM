# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-GPU parity of the indexer-CP decode chain on the FP4 index caches.

Same idea as ``test_indexer_cp_parity.py``: the exchange is data movement, so the P
shards run one after another on one device and are stacked as the all-to-all would
deliver them. Here the scorer is aiter's FlyDSL one (``indexer_context_scores_fp4``,
the production CP scorer with the production work maps), the reference the FlyDSL TP
selector on each rank's own head, and the cache an NVFP4 / MXFP4 one. The real
collective and the impl method are covered by ``run_indexer_cp_fp4_tp4.py`` (4 ranks).

Ties: equal scores resolve to the higher block id (``_pack_score_key``) in both chains,
so the comparison is exact on the ids and their order, not on sets.
"""

import types

import pytest
import torch

aiter = pytest.importorskip("aiter", reason="requires the AITER runtime")
pytest.importorskip("triton", reason="requires Triton")
pytest.importorskip("flydsl", reason="requires FlyDSL")

from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers
from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
    local_candidate_keys,
    merge_candidate_keys,
)
from atom.model_ops.minimax_m3.indexer_context_parallel import (
    indexer_context_scores_fp4,
)
from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

WORLD, BLOCK, TOPK, INIT, LOCAL, SCALE = 4, 128, 16, 1, 2, 128**-0.5
PAGE_BYTES = {"nvfp4": 9216, "fp4": 8704}
# NVFP4 / MXFP4: the layouts' own page-table emit (see `_select_page_kwargs`)
KV_PAGE = {"nvfp4": {}, "fp4": {"pages_per_block": 1, "block_page_stride": 1}}


def _inputs(dtype, batch, ctx, q_len, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks = -(-ctx // BLOCK)
    pages = batch * blocks
    cache = torch.randint(
        0, 256, (pages, PAGE_BYTES[dtype]), device="cuda", generator=g
    )
    cache = cache.to(torch.uint8)
    lo, hi = (0x28, 0x50) if dtype == "nvfp4" else (120, 134)  # e4m3 / e8m0 scale bytes
    cache[:, BLOCK * 64 :] = torch.randint(
        lo, hi, (pages, PAGE_BYTES[dtype] - BLOCK * 64), device="cuda", generator=g
    ).to(torch.uint8)
    q = (torch.randn(batch * q_len, 4, 128, device="cuda", generator=g) * 2).to(
        aiter.dtypes.fp8
    )
    table = torch.randperm(pages, device="cuda", generator=g).to(torch.int32)
    table = table.view(batch, blocks).contiguous()
    lens = torch.randint(q_len, ctx + 1, (batch,), device="cuda", generator=g)
    lens = lens.to(torch.int32)
    lens[0], lens[-1] = q_len, ctx  # one single-block request, one at the window
    return q, cache, table, lens


def _records(dtype, ctx, batch, q_len, rank):
    cfg = types.SimpleNamespace(
        index_cache_dtype=dtype,
        max_model_len=ctx,
        max_num_batched_tokens=max(batch * q_len, 4096),
        enable_tbo=False,
        enable_tbo_decode=False,
    )
    kw = {"max_bs": batch, "max_qlen": q_len, "num_idx_heads": 1}
    tp = allocate_index_scorers(cfg, "cuda", **kw)[""]
    cp = allocate_index_scorers(cfg, "cuda", **kw, cp_world=WORLD, cp_rank=rank)[""]
    return tp, cp


def _step(rec, table, lens, ctx, q_len):
    return make_sparse_decode_metadata(
        seq_lens=lens,
        block_table=table,
        slot_mapping=torch.zeros(
            lens.numel() * q_len, dtype=torch.int64, device="cuda"
        ),
        max_seq_len=ctx,
        num_idx_heads=1,
        max_query_len=q_len,
        index_scorer=rec,
    )


@needs_gpu
@pytest.mark.parametrize("dtype", ["nvfp4", "fp4"])
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("batch, ctx", [(4, 2048), (16, 16384), (3, 131072)])
def test_cp_selection_is_identical_to_the_tp_selector(dtype, q_len, batch, ctx):
    q, cache, table, lens = _inputs(dtype, batch, ctx, q_len, seed=7 + q_len)
    blocks = table.shape[1]
    rows = batch * q_len
    emit = dtype == "fp4"  # MXFP4 KV takes a page table, NVFP4 KV the ids alone
    keys = []
    for r in range(WORLD):
        _, cp = _records(dtype, ctx, batch, q_len, r)
        md = _step(cp, table, lens, ctx, q_len).index_scorer
        assert md.cp_world == WORLD and md.max_block == -(-blocks // WORLD)
        scores = indexer_context_scores_fp4(
            q, cache, table, lens, ctx, r, WORLD, q_len, SCALE, md
        )
        keys.append(
            local_candidate_keys(
                scores, lens, TOPK, r, WORLD, q_len, blocks, INIT, LOCAL
            )
        )
    for head in range(WORLD):
        tp, _ = _records(dtype, ctx, batch, q_len, head)
        md = _step(tp, table, lens, ctx, q_len)
        want = minimax_m3_index_topk_decode(
            q[:, head : head + 1],
            cache,
            table,
            lens,
            ctx,
            TOPK,
            INIT,
            LOCAL,
            1,
            SCALE,
            emit_sparse_block_table=emit,
            max_query_len=q_len,
            n_valid_column_per_row=md.n_valid_column_per_row,
            index_scorer=md.index_scorer,
            **KV_PAGE[dtype],
        )
        want = want if isinstance(want, tuple) else (want,)
        got = merge_candidate_keys(
            torch.stack([keys[r][head] for r in range(WORLD)]),
            table,
            lens,
            TOPK,
            INIT,
            LOCAL,
            q_len,
            emit=emit,
            **KV_PAGE[dtype],
        )
        assert len(got) == len(want)
        for name, a, b in zip(("topk_idx", "sparse_bt", "sparse_ctx"), got, want):
            assert torch.equal(a, b), f"head {head} {name} diverged from the TP path"
        assert got[0].view(rows, TOPK).shape == (rows, TOPK)


@needs_gpu
@pytest.mark.parametrize("dtype", ["nvfp4", "fp4"])
def test_prefill_scores_a_head_slice_of_the_wide_q_as_it_scores_the_copy(dtype):
    """Prefill stays on the TP path under CP: it takes head ``rank`` of the replicated
    full-width projection, a strided view. The FP4 prefill scorers must read it as they
    read the same rows contiguous."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk

    q_lens, prefix, ctx = [37, 200, 5], [0, 128 * 12, 128 * 30 + 3], 4096
    lens = [p + n for p, n in zip(prefix, q_lens)]
    rows = sum(q_lens)
    _, cache, table, _ = _inputs(dtype, len(lens), ctx, 1, seed=5)
    q = (torch.randn(rows, 4, 128, device="cuda") * 2).to(aiter.dtypes.fp8)
    cu = torch.tensor([0, *q_lens], dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    pre = torch.tensor(prefix, dtype=torch.int32, device="cuda")
    cfg = types.SimpleNamespace(
        index_cache_dtype=dtype,
        max_model_len=ctx,
        max_num_batched_tokens=rows,
        enable_tbo=False,
        enable_tbo_decode=False,
    )
    scorer = allocate_index_scorers(
        cfg, "cuda", max_bs=len(lens), max_qlen=1, num_idx_heads=1, cp_world=WORLD
    )[""]

    def select(iq):
        return minimax_m3_index_topk(
            iq, cache, table, cu, seq, pre, max(q_lens), max(lens), TOPK, INIT, LOCAL,
            1, SCALE, emit_sparse_block_table=dtype == "fp4",
            index_scorer=scorer, **KV_PAGE[dtype],
        )  # fmt: skip

    for rank in range(WORLD):
        view = q[:, rank : rank + 1]
        assert not view.is_contiguous()
        got, want = select(view), select(view.contiguous())
        got = got if isinstance(got, tuple) else (got,)
        want = want if isinstance(want, tuple) else (want,)
        assert all(torch.equal(a, b) for a, b in zip(got, want))
