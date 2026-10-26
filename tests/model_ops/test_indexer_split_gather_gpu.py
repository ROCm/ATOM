# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The indexer's row split gathered for real (`atom.model_ops.indexer.score`):
the ranks agree on pynccl at startup, each rank writes only its rows into its
slot, the in-place all-gather runs on the TP group's pynccl communicator, and
every rank ends up with every row in row order -- empty ranks (fewer rows
than ranks) included. Needs 2+ GPUs."""

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


def _rank(rank, world, init_method, results):
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
    )

    from atom.model_ops.indexer import RowSplit
    from atom.model_ops.indexer.score import (
        _gather_split,
        split_gather_comm,
        split_gather_comm_on_every_rank,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(world, 1)
    try:
        ok = split_gather_comm(get_tp_group()) is not None
        # The startup agreement, under the CUDA default device the runner sets
        # while it builds the attention builders.
        torch.set_default_device("cuda")
        try:
            ok = ok and split_gather_comm_on_every_rank(get_tp_group())
        finally:
            torch.set_default_device(None)
        for rows in _ROWS:
            split = RowSplit.of(rows, rank, world, group="tp")
            out = torch.full(
                (split.stride * world, _TOPK), -7, dtype=torch.int32, device="cuda"
            )
            mine = torch.arange(split.first, split.end, dtype=torch.int32)
            out[split.first : split.end] = mine.unsqueeze(1).expand(-1, _TOPK).cuda()
            _gather_split(out, split)
            torch.cuda.synchronize()
            want = torch.arange(rows, dtype=torch.int32).unsqueeze(1).expand(-1, _TOPK)
            ok = ok and torch.equal(out[:rows].cpu(), want)
        results[rank] = ok
    finally:
        torch.cuda.synchronize()
        get_tp_group().barrier()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world", [2, 4])
def test_split_rows_gather_back_in_row_order_on_pynccl(world):
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
        assert dict(results) == {r: True for r in range(world)}
