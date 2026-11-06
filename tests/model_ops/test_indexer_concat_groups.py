# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fixed-workspace request grouping, including the actual DCP gather on GPUs."""

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops import sparse_indexer_chunk as chunk

rows_module = pytest.importorskip("atom.model_ops.indexer.rows", exc_type=ImportError)
from atom.model_ops.attentions import aiter_mla
from atom.model_ops.indexer import indexer_forward, score


def _builder(monkeypatch, lengths, queries, max_len, tokens, world, budget):
    monkeypatch.setattr(aiter_mla, "FP8_MQA_LOGITS_TAKES_OUT", True)
    monkeypatch.setattr(chunk.envs, "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", budget)
    builder = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    builder.device = "cpu"
    builder.dcp_world_size = world
    builder.cp_kv_cache_interleave_size = 2
    builder.max_bs = len(lengths)
    builder.max_num_batched_tokens = tokens
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=max_len, hf_config=SimpleNamespace(index_head_dim=1)
        ),
        forward_vars={
            name: SimpleNamespace(np=np.cumsum([0, *sizes], dtype=np.int64))
            for name, sizes in [("cu_seqlens_k", lengths), ("cu_seqlens_q", queries)]
        },
    )
    builder._alloc_prefill_concat_keys()
    builder.index_logits_workspace = chunk.LogitsWorkspace(
        builder._prefill_concat_plane_capacity(), "cpu"
    )
    return builder


@pytest.mark.parametrize(
    "lengths, queries, max_len, tokens, world, budget",
    [
        ([30724] * 64, [4] * 64, 32768, 256, 8, 1024),
        ([525312] * 2, [4096] * 2, 1048576, 8192, 8, 8),
        ([300, 300], [1, 1], 512, 2, 1, 1),
    ],
    ids=["many-cached-requests", "one-aligned-row", "non-dcp"],
)
def test_builder_groups_fit_the_actual_plane(
    monkeypatch, lengths, queries, max_len, tokens, world, budget
):
    builder = _builder(monkeypatch, lengths, queries, max_len, tokens, world, budget)
    plane = builder.index_logits_workspace
    key_workspace = builder.index_prefill_key_workspace
    pointers = [t.data_ptr() for t in (*key_workspace._keys, *key_workspace._gathered)]
    groups = builder._concat_groups(SimpleNamespace(has_cached=True), len(lengths))
    assert groups is not None and len(groups) > 1
    assert sum(g.total for g in groups) == sum(lengths)
    assert plane.capacity >= chunk.concat_plane_width(max_len)
    for group in groups:
        assert group.total <= key_workspace.capacity
        assert chunk.concat_plane_width(group.total) <= plane.capacity
        key_workspace.keys(group.total)
        if world > 1:
            key_workspace.gathered(world * group.dcp.local_total)
    assert pointers == [
        t.data_ptr() for t in (*key_workspace._keys, *key_workspace._gathered)
    ]
    keys = rows_module.ConcatKeys(None, None, sum(lengths), None, groups=groups)
    covered = []
    for rank in range(world):
        split = rows_module.RowSplit.of(sum(queries), rank, world)
        rows = rows_module.concat_rows(torch.zeros(sum(queries)), keys, split, plane)
        for first, end in rows.bands:
            group = next(g for g in groups if g.row_first <= first < g.row_end)
            assert end <= group.row_end
            assert (end - first) * chunk.concat_plane_width(
                group.total
            ) <= plane.capacity
            covered.extend(range(first, end))
    assert covered == list(range(sum(queries)))


@pytest.mark.parametrize("rank", [0, 1])
def test_pcp_groups_cover_owned_rows_and_empty_padding(monkeypatch, rank):
    builder = _builder(monkeypatch, [300, 300, 300], [1, 1, 1], 512, 3, 1, 1)
    monkeypatch.setattr(aiter_mla, "get_pcp_world_size", lambda: 2)
    monkeypatch.setattr(aiter_mla, "get_pcp_rank", lambda: rank)
    groups = builder._concat_groups(
        SimpleNamespace(has_cached=True), 3, pcp_split=True, rows=2
    )
    # Rank 1 owns request 1 and a dummy row in the last group.
    starts = torch.tensor([0, 600] if rank == 0 else [300, 0], dtype=torch.int32)
    ends = torch.tensor([1, 1] if rank == 0 else [1, 0], dtype=torch.int32)
    keys = rows_module.ConcatKeys(None, None, 900, starts, groups=groups)
    rows = rows_module.concat_rows(ends, keys, workspace=builder.index_logits_workspace)

    def logits(**kwargs):
        assert (kwargs["cu_starts"] >= 0).all()
        assert (kwargs["cu_ends"] <= kwargs["KV"].shape[0]).all()
        return kwargs["logits"]

    def topk(scores, starts, stops, out, *args, **kwargs):
        out[:, 0] = torch.where(stops > starts, stops - 1, -1)

    monkeypatch.setattr(score, "fp8_mqa_logits", logits)
    monkeypatch.setattr(score, "top_k_per_row_prefill", topk)
    out = torch.full((2, 1), -99, dtype=torch.int32)
    score._score_concat(
        torch.zeros(2, 1, 1),
        torch.ones(2, 1),
        None,
        rows,
        1,
        False,
        out,
        lambda g: (torch.zeros(g.total, 1), torch.ones(g.total, 1)),
        True,
    )
    assert out[:, 0].tolist() == ([0, 0] if rank == 0 else [0, -1])


def _two_rank_gather_worker(rank, rendezvous, interleave):
    """Real cache write, RCCL in-place gather and byte-exact reference comparison."""
    from aiter import dtypes, indexer_k_quant_and_cache

    from atom.distributed.dcp_layout import (
        dcp_concat_gather_plan,
        dcp_local_index,
        dcp_owner_rank,
    )

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:

        class Group:
            world_size = 2
            rank_in_group = rank
            device_group = torch.distributed.group.WORLD

            def all_gather(self, value, dim):
                out = torch.empty(
                    (2 * value.shape[0], *value.shape[1:]),
                    device=value.device,
                    dtype=value.dtype,
                )
                torch.distributed.all_gather_into_tensor(out, value)
                return out

        indexer_forward.get_dcp_group = lambda: Group()
        device = torch.device("cuda", rank)
        lengths = [129, 7, 259]
        cu = np.cumsum([0, *lengths], dtype=np.int64)
        table = torch.tensor(
            [[5, 4], [3, 2], [1, 0], [6, 6]], device=device, dtype=torch.int32
        )
        cache = torch.full((7, 128, 132), 0xA5, device=device, dtype=torch.uint8)
        original = torch.ones(int(cu[-1]), 128, device=device, dtype=torch.bfloat16)
        token = torch.arange(int(cu[-1]), device=device)
        for digit in range(3):
            original[:, 2 * digit] = ((token >> (4 * digit)) & 15) - 8
            original[:, 2 * digit + 1] = -original[:, 2 * digit]
        original *= (2.0 ** (token % 3)).unsqueeze(1)
        for request, length in enumerate(lengths):
            pos = np.arange(length)
            owned = pos[dcp_owner_rank(pos, 2, interleave) == rank]
            local = torch.tensor(dcp_local_index(owned, 2, interleave), device=device)
            slots = table[request, local // 128].long() * 128 + local % 128
            values = original[torch.tensor(cu[request] + owned, device=device)]
            indexer_k_quant_and_cache(
                values, cache, slots, 128, "ue8m0", preshuffle=True
            )

        workspace = chunk.ConcatKeyWorkspace(
            259, 128, dtypes.fp8, device, 2 * (130 + 2 * interleave)
        )
        pointers = [t.data_ptr() for t in (*workspace._keys, *workspace._gathered)]
        keys = rows_module.ConcatKeys(table, None, int(cu[-1]), None, workspace)
        for repeat in range(2):
            for first, end in [(0, 2), (2, 3)]:
                _, _, cu_pad, local_total, src = dcp_concat_gather_plan(
                    cu, first, end, 2, interleave
                )
                plan = rows_module.DcpConcatGather(
                    local_total,
                    torch.tensor(cu_pad, dtype=torch.int32, device=device),
                    torch.tensor(src, dtype=torch.int32, device=device),
                )
                group = rows_module.ConcatGroup(
                    first, end, 0, 0, int(cu[first]), int(cu[end] - cu[first]), plan
                )
                # Deliberately wrong ubatch tables: grouped gather must use the step's.
                metadata = SimpleNamespace(block_tables=table[3:])
                actual = indexer_forward._dcp_gather_indexer_k_prefill(
                    cache, metadata, 128, device, keys, group
                )
                reference = indexer_forward._dcp_gather_indexer_k_prefill(
                    cache, metadata, 128, device, keys._replace(workspace=None), group
                )
                assert torch.equal(
                    actual[0].view(torch.uint8), reference[0].view(torch.uint8)
                )
                torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
                torch.testing.assert_close(
                    actual[0].float() * actual[1],
                    original[cu[first] : cu[end]].float(),
                    rtol=0,
                    atol=0,
                )
        assert pointers == [
            t.data_ptr() for t in (*workspace._keys, *workspace._gathered)
        ]
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs for RCCL")
@pytest.mark.parametrize("interleave", [1, 2])
def test_two_rank_gather_reuses_buffers_and_matches_allocating_path(
    tmp_path, interleave
):
    torch.multiprocessing.spawn(
        _two_rank_gather_worker,
        args=((tmp_path / "rendezvous").as_uri(), interleave),
        nprocs=2,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("request_local", [True, False])
def test_grouped_gpu_topk_matches_the_whole_step(request_local):
    from aiter import dtypes, indexer_k_quant_and_cache

    if not score.FP8_MQA_LOGITS_TAKES_OUT:
        pytest.skip("requires aiter fp8_mqa_logits(logits=), ROCm/aiter#7517")
    torch.manual_seed(3)
    device = "cuda"
    lengths, queries = [300, 127, 400], [3, 2, 2]
    cu = np.cumsum([0, *lengths])
    cu_q = np.cumsum([0, *queries])
    table = torch.tensor(
        [[0, 1, 2, 0], [3, 0, 0, 0], [4, 5, 6, 7]], device=device, dtype=torch.int32
    )
    cache = torch.empty((8, 128, 132), dtype=torch.uint8, device=device)
    slots = torch.cat(
        [
            torch.arange(length, device=device) + page * 128
            for length, page in zip(lengths, [0, 3, 4])
        ]
    )
    key_values = torch.randn(int(cu[-1]), 128, dtype=torch.bfloat16, device=device)
    indexer_k_quant_and_cache(key_values, cache, slots, 128, "ue8m0", preshuffle=True)
    starts = torch.tensor(np.repeat(cu[:-1], queries), dtype=torch.int32, device=device)
    ends = torch.tensor(
        [10, 129, 300, 50, 127, 1, 400], dtype=torch.int32, device=device
    )
    keys = rows_module.ConcatKeys(
        table, torch.tensor(cu, dtype=torch.int32, device=device), int(cu[-1]), starts
    )
    query = torch.randn(7, 32, 128, device=device).to(dtypes.fp8)
    weights = torch.randn(7, 32, device=device)
    index_cache = rows_module.IndexCache(cache, None, 128)
    expected = score.score_topk(
        query,
        weights,
        index_cache,
        rows_module.concat_rows(ends, keys),
        topk=2048,
        stable=True,
        request_local=request_local,
    )
    key_workspace = chunk.ConcatKeyWorkspace(512, 128, dtypes.fp8, device)
    plane = chunk.LogitsWorkspace(1024, device)
    groups = tuple(
        rows_module.ConcatGroup(
            first,
            end,
            int(cu_q[first]),
            int(cu_q[end]),
            int(cu[first]),
            int(cu[end] - cu[first]),
        )
        for first, end in chunk.pack_concat_groups(cu, 512)
    )
    rows = rows_module.concat_rows(
        ends, keys._replace(workspace=key_workspace, groups=groups), workspace=plane
    )
    pointers = [t.data_ptr() for t in (*key_workspace._keys, plane._logits)]
    for _ in range(2):
        actual = score.score_topk(
            query,
            weights,
            index_cache,
            rows,
            topk=2048,
            stable=True,
            request_local=request_local,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pointers == [t.data_ptr() for t in (*key_workspace._keys, plane._logits)]
