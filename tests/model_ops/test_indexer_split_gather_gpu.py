# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The indexer's row split gathered for real (`atom.model_ops.indexer.score`):
each rank writes only its rows into its slot, the in-place all-gather runs on
the TP group's ProcessGroup, and every rank ends up with every rank's slot --
contiguous and striped splits, empty ranks (fewer rows than ranks) included.
A striped FP8 prefill scored across the ranks selects as the whole batch
scored on one. Needs 2+ GPUs."""

import multiprocessing as mp
import time

import pytest
import torch

pytest.importorskip(
    "atom.model_ops.indexer.score",
    reason="the indexer's scorer imports aiter at load",
    exc_type=ImportError,
)

_ROWS = (1, 3, 8, 13)  # 1 over 2 ranks leaves rank 1 with no rows
_TOPK = 4
# Each request's committed keys and new rows, a striped FP8 prefill each: one
# request of a single row; three rows in all, a rank one row or none.
_FP8_BATCHES = (([1000, 37, 2100], [13, 1, 29]), ([1000, 37], [1, 2]))


def _gathers(rank, world):
    """Each split of `_ROWS` rows gathers every rank's rows into its slot."""
    from atom.model_ops.indexer import RowSplit
    from atom.model_ops.indexer.score import _gather_split

    ok = True
    for rows in _ROWS:
        for make in (RowSplit.of, RowSplit.striped):
            splits = [make(rows, r, world, group="tp") for r in range(world)]
            # Rank r's rows: a contiguous run, or every world-th from r.
            dealt = [
                range(r, rows, world) if s.stripe else range(s.first, s.end)
                for r, s in enumerate(splits)
            ]
            split = splits[rank]
            out = torch.full(
                (split.stride * world, _TOPK), -7, dtype=torch.int32, device="cuda"
            )
            mine = torch.tensor(dealt[rank], dtype=torch.int32, device="cuda")
            out[split.slot : split.slot + len(mine)] = mine.unsqueeze(1)
            _gather_split(out, split)
            slots = out.view(world, split.stride, _TOPK).cpu()
            for r, want in enumerate(dealt):
                got = slots[r, : len(want)]
                ok = ok and torch.equal(
                    got, torch.tensor(want, dtype=torch.int32)[:, None].expand_as(got)
                )
    return ok


def _fp8_striped_matches_whole(rank, world, committed, new_rows):
    """A V4 FP8 prefill's rows scored striped across the ranks select as the
    whole batch on one rank, back in row order. At this size both take the
    same logits kernel config, so the selections match exactly."""
    from aiter import QuantType, dtypes, get_hip_quant, indexer_k_quant_and_cache

    from atom.model_ops.indexer import IndexCache, RowSplit, concat_rows, score_topk
    from atom.model_ops.indexer.rows import ConcatKeys

    gen = torch.Generator().manual_seed(2722)
    block, heads, dim, topk = 128, 64, 128, 512
    tokens = sum(new_rows)
    pages = [-(-n // block) for n in committed]
    order = torch.randperm(sum(pages), generator=gen).int()
    table = torch.zeros(len(committed), max(pages), dtype=torch.int32)
    cache = torch.zeros(sum(pages), block, dim + 4, dtype=torch.uint8, device="cuda")
    for b, (n, used) in enumerate(zip(committed, pages)):
        table[b, :used] = order[sum(pages[:b]) : sum(pages[:b]) + used]
        at = torch.arange(n)
        slots = table[b, at // block].long() * block + at % block
        indexer_k_quant_and_cache(
            torch.randn(n, dim, generator=gen).bfloat16().cuda(),
            cache,
            slots.cuda(),
            dim,
            "ue8m0",
            preshuffle=True,
        )
    table = table.cuda()
    batch_ids = torch.repeat_interleave(
        torch.arange(len(new_rows)), torch.tensor(new_rows)
    ).cuda()
    ends = torch.cat(
        [
            torch.randint(1, n + 1, (r,), generator=gen)
            for n, r in zip(committed, new_rows)
        ]
    )
    ends = ends.int().cuda()
    cu = torch.tensor([0, *committed]).cumsum(0).int().cuda()
    q = torch.randn(tokens, heads, dim, generator=gen).bfloat16().cuda()
    w = torch.randn(tokens, heads, generator=gen).bfloat16().cuda()
    q8, qs = get_hip_quant(QuantType.per_1x128)(
        q.reshape(-1, dim), quant_dtype=dtypes.fp8
    )
    q8 = q8.view_as(q)
    weights = w.float() * qs.view(tokens, heads) / 64
    index_cache = IndexCache(cache, None, block)
    total = sum(committed)

    whole = score_topk(
        q8,
        weights,
        index_cache,
        concat_rows(ends, ConcatKeys(table, cu, total, cu[batch_ids])),
        topk=topk,
        stable=True,
    )
    # The builder's form: this rank's rows of the row metadata, the whole
    # batch's query and weights.
    split = RowSplit.striped(tokens, rank, world, group="tp")
    mine = slice(rank, tokens, world)
    striped = score_topk(
        q8,
        weights,
        index_cache,
        concat_rows(
            ends[mine].contiguous(),
            ConcatKeys(table, cu, total, cu[batch_ids[mine]]),
            split,
        ),
        topk=topk,
        stable=True,
    )
    got, want = striped[:tokens].cpu(), whole.cpu()
    # Not vacuous: every row picks as many keys as it sees, up to top-k.
    if not torch.equal((want >= 0).sum(1), ends.cpu().long().clamp(max=topk)):
        return False
    return torch.equal(got.sort(1).values, want.sort(1).values)


def _rank(rank, world, init_method, results):
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(world, 1)
    try:
        results[rank] = (
            _gathers(rank, world),
            *(_fp8_striped_matches_whole(rank, world, *b) for b in _FP8_BATCHES),
        )
    finally:
        torch.cuda.synchronize()
        get_tp_group().barrier()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world", [2, 4])
def test_split_rows_gather_back_in_row_order(world):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        results = manager.dict()
        procs = [
            ctx.Process(target=_rank, args=(r, world, init_method, results))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 300
        for p in procs:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        hung = [p for p in procs if p.is_alive()]
        for p in hung:  # never leave a rank behind for interpreter exit to join
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        assert not hung and all(p.exitcode == 0 for p in procs)
        # (contiguous and striped gathers, each striped FP8 prefill) per rank
        want = (True,) * (1 + len(_FP8_BATCHES))
        assert dict(results) == {r: want for r in range(world)}
