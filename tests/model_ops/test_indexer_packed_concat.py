# SPDX-License-Identifier: MIT
"""Packed FP8 prefill rows, keys gathered in passes: the builder's plan through
the real kernels against dense scoring of every key."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

mla = pytest.importorskip("atom.model_ops.attentions.aiter_mla", exc_type=ImportError)
from aiter import cp_gather_indexer_k_quant_cache, indexer_k_quant_and_cache

from atom.model_ops.dcp_ops import dcp_local_index, dcp_owner_rank
from atom.model_ops.indexer import score
from atom.model_ops.indexer.rows import IndexCache
from atom.model_ops.indexer.workspace import ConcatKeyWorkspace
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU")
BLOCK, HEADS, DIM, FP8 = 64, 32, 128, score.dtypes.fp8
LENGTHS, CU_Q = [513, 769], np.array([0, 4, 7], np.int32)
CU_K = np.array([0, 513, 1282], np.int32)
VISIBLE = np.array([510, 511, 512, 513, 767, 768, 769], np.int32)


def _write(cache, keys, table, request, positions, local):
    """Request `request`'s keys at `positions` into `cache` at `local` places."""
    slots = table[request, local // BLOCK].long() * BLOCK + local % BLOCK
    rows = keys[CU_K[request] + positions]
    indexer_k_quant_and_cache(rows, cache, slots, 128, "ue8m0", preshuffle=True)


def _expected(q, w, keys, table, starts, visible, topk, stable, local):
    """Dense scores and top-k over every key, gathered off a whole cache."""
    cache = torch.zeros(table.numel(), BLOCK, 144, dtype=torch.uint8, device="cuda")
    for r, n in enumerate(LENGTHS):
        span = torch.arange(n, device="cuda")
        _write(cache, keys, table, r, span, span)
    cu = torch.tensor(CU_K, device="cuda")
    k = torch.empty(1282, DIM, dtype=FP8, device="cuda")
    s = torch.empty(1282, 1, dtype=torch.float32, device="cuda")
    cp_gather_indexer_k_quant_cache(cache, k, s.view(FP8), table, cu, preshuffle=True)
    dense = score.fp8_mqa_logits(q, k, s, w, starts, starts + visible)
    out = torch.empty(q.shape[0], topk, dtype=torch.int32, device="cuda")
    bounds = (starts, starts + visible, out, None, q.shape[0], dense.stride(0), 1)
    score.top_k_per_row_prefill(dense, *bounds, k=topk, stable=stable)
    if local:
        out = torch.where(out < 0, out, out - starts[:, None])
    return cache, dense, out


def _builder(rank, world, prefix, plane, keys):
    var = {
        prefix + name: SimpleNamespace(np=np.zeros(16, np.int32), gpu=None)
        for name in ("index_row_ends", "index_logits_offsets")
    }
    var[prefix + "cu_seqlens_q"] = SimpleNamespace(np=CU_Q)
    var[prefix + "cu_seqlens_k"] = SimpleNamespace(np=CU_K)
    builder = mla.AiterMLAMetadataBuilder.__new__(mla.AiterMLAMetadataBuilder)
    builder.model_runner = SimpleNamespace(forward_vars=var)
    builder.is_sparse, builder._indexer_fp4, builder.index_kpool = True, False, 1
    builder.dcp_rank, builder.dcp_world_size = rank, world
    builder.cp_kv_cache_interleave_size, builder.device = 1, torch.device("cuda")
    builder.index_logits_workspace, builder.index_prefill_key_workspace = plane, keys
    return builder, var


def _rows(builder, var, table, starts, pcp, prefix, dcp=False):
    counts = {}
    staged = builder._stage_prefill_index_rows(
        2, 2, VISIBLE, pcp, True, SimpleNamespace(set_counts=counts.update), prefix
    )
    for name in counts:
        var[name].gpu = torch.tensor(var[name].np, device="cuda")
    gathers = None
    if dcp:
        gathers = builder._build_dcp_indexer_prefill_meta(
            SimpleNamespace(has_cached=True), 2, prefix, staged.passes
        )
    meta = SimpleNamespace(
        has_cached=True,
        block_tables=table,
        cu_seqlens_q=torch.tensor(CU_Q, device="cuda"),
        cu_seqlens_k=torch.tensor(CU_K, device="cuda"),
        cu_seqlen_ks=starts,
    )
    builder._finish_prefill_index_rows(meta, 2, staged, 1282, prefix, gathers)
    return meta.index_rows


@needs_gpu
@pytest.mark.parametrize("world,local", [(1, False), (1, True), (2, True), (16, True)])
@pytest.mark.parametrize("mode", ["plain", "pcp", "ubatch"])
@pytest.mark.parametrize("stable", [False, True])
def test_builder_passes_match_dense(monkeypatch, world, local, mode, stable):
    """Two requests past an 800-key workspace: two passes, rows in bands of a
    1536-element plane, every DCP rank its share of each pass."""
    torch.manual_seed(3043)
    topk, prefix, pcp = 64, "ubatch_0_" if mode == "ubatch" else "", mode == "pcp"
    visible, starts = VISIBLE, [0] * 4 + [513] * 3
    if pcp:  # rank 1 of 2 owns rows 1, 3, 5 and a padding row
        monkeypatch.setattr(mla, "get_pcp_world_size", lambda: 2)
        monkeypatch.setattr(mla, "get_pcp_rank", lambda: 1)
        visible, starts = np.array([511, 513, 768, 0], np.int32), [0, 0, 513, 0]
    count = len(visible)
    q = torch.randn(count, HEADS, DIM, device="cuda", dtype=torch.float32).to(FP8)
    w = torch.rand(count, HEADS, device="cuda", dtype=torch.float32)
    keys = torch.randn(1282, DIM, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(2 * 13, device="cuda").to(torch.int32).view(2, 13)
    starts = torch.tensor(starts, device="cuda", dtype=torch.int32)
    vis = torch.tensor(visible, device="cuda", dtype=torch.int32)
    cache, dense, expected = _expected(
        q, w, keys, table, starts, vis, topk, stable, local
    )
    plane = LogitsWorkspace(1536, "cuda")
    storage = torch.full((plane.capacity + 34,), float("nan"), device="cuda")
    plane._logits = storage[17:-17]  # guards past both ends of the plane
    key_ws = ConcatKeyWorkspace(800, DIM, FP8, "cuda")
    monkeypatch.setattr(score, "_all_gather_slot", lambda *a: None)  # one `actual`
    monkeypatch.setattr(score, "get_dcp_group", lambda: None)
    actual = torch.full((count + world, topk), -99, dtype=torch.int32, device="cuda")
    run = {"topk": topk, "stable": stable, "out": actual, "request_local": local}
    for rank in range(world):
        builder, var = _builder(rank, world, prefix, plane, key_ws)
        rows = _rows(builder, var, table, starts, pcp, prefix)
        assert len(rows.passes) == 2 and rows.width == max(visible)
        score.score_topk(q, w, IndexCache(cache, None, BLOCK), rows, **run)
    actual = actual[:count]
    assert torch.isnan(storage[:17]).all() and torch.isnan(storage[-17:]).all()
    if stable:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    base = starts[:, None] if local else 0

    def picked(ids):  # unstable ties may pick other keys of the same score
        values = dense.gather(1, (ids + base).clamp_min(0).long())
        return values.masked_fill(ids < 0, -float("inf")).sort(1).values

    assert torch.equal((actual >= 0).sum(1), vis.clamp_max(topk).long())
    torch.testing.assert_close(picked(actual), picked(expected), rtol=0, atol=0)


@pytest.fixture(scope="module")
def dcp_process_group():
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    yield rank
    dist.destroy_process_group()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="run with torchrun --nproc-per-node=2",
)
@pytest.mark.parametrize("topk", [64, 2048])
def test_dcp_passes_gather_shards_and_selections(monkeypatch, topk, dcp_process_group):
    """Each rank holds its round-robin key shard: passes gather every shard in
    place, and the ranks' selections come back in row order on both."""
    import torch.distributed as dist

    rank = dcp_process_group
    group = SimpleNamespace(rank_in_group=rank, world_size=2, device_group=None)
    monkeypatch.setattr(score, "get_dcp_group", lambda: group)
    torch.manual_seed(3043)  # the same inputs on both ranks
    q = torch.randn(7, HEADS, DIM, device="cuda", dtype=torch.float32).to(FP8)
    w = torch.rand(7, HEADS, device="cuda", dtype=torch.float32)
    keys = torch.randn(1282, DIM, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(2 * 13, device="cuda").to(torch.int32).view(2, 13)
    starts = torch.tensor([0] * 4 + [513] * 3, device="cuda", dtype=torch.int32)
    vis = torch.tensor(VISIBLE, device="cuda", dtype=torch.int32)
    _, _, expected = _expected(q, w, keys, table, starts, vis, topk, True, True)
    shard = torch.zeros(table.numel(), BLOCK, 144, dtype=torch.uint8, device="cuda")
    for r, n in enumerate(LENGTHS):
        pos = np.arange(n)
        mine = torch.from_numpy(pos[dcp_owner_rank(pos, 2, 1) == rank]).cuda()
        local = torch.from_numpy(dcp_local_index(mine.cpu().numpy(), 2, 1)).cuda()
        _write(shard, keys, table, r, mine, local)
    plane = LogitsWorkspace(8192, "cuda")
    key_ws = ConcatKeyWorkspace.for_config(700, 100, 2, DIM, FP8, "cuda", 2, 1)
    builder, var = _builder(rank, 2, "", plane, key_ws)
    rows = _rows(builder, var, table, starts, False, "", dcp=True)
    assert [p.key_base for p in rows.passes] == [0, 513]  # one request each
    out = torch.full((rows.out_rows, topk), -99, dtype=torch.int32, device="cuda")
    score.score_topk(
        q, w, IndexCache(shard, None, BLOCK), rows, topk=topk, stable=True, out=out
    )
    torch.testing.assert_close(out[:7], expected, rtol=0, atol=0)
    dist.barrier()
