# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DCP sparse prefill's top-k exchanged for real
(`dcp_exchange_owned_index_prefill`): each rank holds only its own rows of the
selection, and every rank ends up with what
`triton_filter_and_convert_dcp_index_prefill` makes of the whole selection on
it -- the same slots, offsets and owned counts, bit for bit. Pad tokens, rows
a rank owns nothing of, slot tails past the last token and ranks with no rows
included. Needs 2+ GPUs."""

import multiprocessing as mp
import time

import pytest
import torch

pytest.importorskip(
    "atom.model_ops.indexer.score",
    reason="the indexer's scorer imports aiter at load",
    exc_type=ImportError,
)

_PAGE, _TOPK = 16, 256
# (request key counts, query rows a request, pad rows, interleave): one row
# over 8 ranks leaves seven with none.
_CASES = (
    ((700, 50, 1300), (37, 5, 61), 3, 1),
    ((900, 64), (11, 2), 0, 4),
    ((300,), (1,), 0, 1),
)


def _selection(gen, lengths, rows_per_request, pads):
    """Each row's request (-1 a pad) and its top-k positions in the request,
    distinct and unsorted, -1 past what the row sees. The first request's
    first two rows see no key and one."""
    ids, picks = [], []
    for b, (keys, rows) in enumerate(zip(lengths, rows_per_request)):
        for r in range(rows):
            seen = (
                r
                if b == 0 and r < 2
                else torch.randint(1, keys + 1, (1,), generator=gen).item()
            )
            row = torch.full((_TOPK,), -1, dtype=torch.int32)
            take = min(seen, _TOPK)
            row[:take] = torch.randperm(seen, generator=gen)[:take]
            ids.append(b)
            picks.append(row)
    for _ in range(pads):  # positions a real row would own
        ids.append(-1)
        picks.append(torch.randint(0, min(lengths), (_TOPK,), generator=gen).int())
    return torch.tensor(ids, dtype=torch.int32), torch.stack(picks)


def _matches_filter(rank, world, lengths, rows_per_request, pads, interleave):
    from atom.model_ops.dcp_ops import (
        dcp_exchange_owned_index_prefill,
        triton_filter_and_convert_dcp_index_prefill,
    )
    from atom.model_ops.indexer import RowSplit

    # One seed: every rank builds the same whole selection and tables.
    gen = torch.Generator().manual_seed(555)
    ids, whole = _selection(gen, lengths, rows_per_request, pads)
    rows = len(ids)
    pages = -(-max(lengths) // (_PAGE * world))
    table = torch.randperm(len(lengths) * pages, generator=gen).int()
    table = table.view(len(lengths), pages).cuda()
    ids, whole = ids.cuda(), whole.cuda()
    indptr_in = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")

    def buffers():
        return (
            torch.full((rows + 1,), -7, dtype=torch.int32, device="cuda"),
            torch.full((rows,), -7, dtype=torch.int32, device="cuda"),
            torch.full((rows * (_TOPK + 1),), -7, dtype=torch.int32, device="cuda"),
        )

    want = buffers()
    triton_filter_and_convert_dcp_index_prefill(
        indptr_in,
        ids,
        whole,
        table,
        rank,
        world,
        _PAGE,
        *want[:2],
        NUM_TOPK_TOKENS=_TOPK,
        out=want[2],
        cp_kv_cache_interleave_size=interleave,
    )

    # This rank's rows in its slot; every other row, its slot's tail
    # included, garbage a real row would own.
    split = RowSplit.of(rows, rank, world)
    mine = torch.randint(0, min(lengths), (split.stride * world, _TOPK)).int().cuda()
    mine[split.first : split.end] = whole[split.first : split.end]
    got = buffers()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        pending = dcp_exchange_owned_index_prefill(
            mine,
            split,
            indptr_in,
            ids,
            table,
            _PAGE,
            *got,
            NUM_TOPK_TOKENS=_TOPK,
            cp_kv_cache_interleave_size=interleave,
        )
    pending.wait()
    # Whole buffers: nothing written that the filter leaves alone either.
    same = all(torch.equal(g, w) for g, w in zip(got, want))
    owned = want[1].cpu()
    return same, bool((owned == 0).any()), bool((owned > 1).any())


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
    ensure_model_parallel_initialized(world, 1, world)
    try:
        results[rank] = tuple(_matches_filter(rank, world, *c) for c in _CASES)
    finally:
        torch.cuda.synchronize()
        get_tp_group().barrier()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world", [2, 8])
def test_each_rank_receives_what_the_filter_keeps_of_the_whole(world):
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
        assert sorted(results.keys()) == list(range(world))
        for rank, cases in results.items():
            same, owns_none, owns_several = zip(*cases)
            assert all(same), (rank, same)
            # Not vacuous: rows a rank owns none of, and several of.
            assert any(owns_none) and any(owns_several), rank
