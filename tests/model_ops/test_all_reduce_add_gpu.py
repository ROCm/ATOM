# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tensor_model_parallel_all_reduce_add(routed, shared)` against what
`MoE.combine_outputs` did before, `tensor_model_parallel_all_reduce(routed + shared)`:
the same bits on every rank, at V4.1-Flash's MoE combine width (5120 bf16), for
decode row counts the fused kernel takes and prefill ones past its cap, eager and
replayed in a HIP graph. The fused kernel is aiter's HIP Lamport one-shot all-reduce
(ROCm/aiter#7499, 8 MiB at TP4) when aiter has it, else the Gluon all-reduce
(ROCm/aiter#7488, 4 MiB). Needs 4 gfx1250 GPUs and one of them."""

import multiprocessing as mp
import os
import time

import pytest
import torch

pytest.importorskip(
    "aiter.dist.communication_op", reason="needs aiter", exc_type=ImportError
)

_WORLD = 4
_DIM = 5120
_ROW_BYTES = _DIM * 2
_GLUON_CAP = 4 * 1024 * 1024  # 409 rows
# Decode row counts, the Gluon cap and one past it; the Lamport cap and one past it
# are added at run time, and 4096 rows is past both.
_ROWS = (1, 8, 48, 96, 288, 409, 410, 4096)
_GRAPH_ROWS = (48, 8, 288, 1)
_REPLAYS = 10


def _pair(rows, seed, rank):
    gen = torch.Generator(device="cuda").manual_seed(seed * _WORLD + rank)
    routed = torch.randn(rows, _DIM, generator=gen, device="cuda").to(torch.bfloat16)
    shared = torch.randn(rows, _DIM, generator=gen, device="cuda").to(torch.bfloat16)
    return routed, shared


def _differ(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def _hook_fused_kernels(fused_calls):
    """Count the calls that reach a fused all-reduce add kernel; returns the largest
    input in bytes the fused path takes on this aiter."""
    import inspect

    import aiter.ops.triton.comms.all_reduce as gluon
    from aiter.dist.device_communicators import custom_all_reduce
    from aiter.dist.parallel_state import get_tp_group

    plain = gluon.all_reduce

    def counting_gluon(*args, **kwargs):
        fused_calls[0] += kwargs.get("x2") is not None
        return plain(*args, **kwargs)

    gluon.all_reduce = counting_gluon
    # The Gluon kernel folds the add only where it takes the second input (#7488).
    cap = _GLUON_CAP if "x2" in inspect.signature(plain).parameters else 0
    lamport = getattr(custom_all_reduce.CustomAllreduce, "lamport_ar_add", None)
    if lamport is not None:

        def counting_lamport(self, *args, **kwargs):
            fused_calls[0] += 1
            return lamport(self, *args, **kwargs)

        custom_all_reduce.CustomAllreduce.lamport_ar_add = counting_lamport
        ca_comm = get_tp_group().device_communicator.ca_comm
        cap = max(cap, getattr(ca_comm, "_lamport_max_bytes", 0))
    return cap


def _checks(rank):
    """(fused kernel calls, row counts at or under the fused cap, elements differing
    from the add + all-reduce, elements differing across ranks, elements differing
    over graph replays)."""
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group

    import atom.model_ops.communication_op as comm

    comm._tbo_aware_tp_reduce = lambda tp_size: False  # no ATOM config here
    fused_calls = [0]
    cap_rows = _hook_fused_kernels(fused_calls) // _ROW_BYTES
    extra = {cap_rows, cap_rows + 1} if cap_rows else set()
    all_rows = sorted(set(_ROWS) | extra)
    fused_rows = sum(rows <= cap_rows for rows in all_rows)
    group = get_tp_group().device_group
    differ = across = 0
    for i, rows in enumerate(all_rows):
        routed, shared = _pair(rows, i, rank)
        want = comm.tensor_model_parallel_all_reduce(routed + shared)
        got = comm.tensor_model_parallel_all_reduce_add(routed, shared)
        ranks = [torch.empty_like(got) for _ in range(_WORLD)]
        dist.all_gather(ranks, got, group=group)
        differ += _differ(got, want)
        across += sum(_differ(r, got) for r in ranks)
    eager_fused = fused_calls[0]

    ins = [tuple(torch.empty_like(t) for t in _pair(r, 0, rank)) for r in _GRAPH_ROWS]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = [comm.tensor_model_parallel_all_reduce_add(a, b) for a, b in ins]
        want = [comm.tensor_model_parallel_all_reduce(a + b) for a, b in ins]
    replay_differ = 0
    for rep in range(_REPLAYS):
        for j, (a, b) in enumerate(ins):
            new_a, new_b = _pair(_GRAPH_ROWS[j], 100 + rep * 10 + j, rank)
            a.copy_(new_a)
            b.copy_(new_b)
        graph.replay()
        torch.cuda.synchronize()
        replay_differ += sum(_differ(g, w) for g, w in zip(got, want))
    return eager_fused, fused_rows, differ, across, replay_differ


def _rank(rank, init_method, results):
    os.environ["AITER_GLUON_FUSED_AR"] = "1"  # what ATOM_ENABLE_GLUON_FUSED_AR sets
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=_WORLD, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(_WORLD, 1)
    try:
        results[rank] = _checks(rank)
    finally:
        torch.cuda.synchronize()
        get_tp_group().barrier()
        destroy_model_parallel()
        destroy_distributed_environment()


def test_all_reduce_add_matches_add_then_all_reduce():
    if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD:
        pytest.skip(f"needs {_WORLD} GPUs")
    from aiter.dist import communication_op
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250":
        pytest.skip("the fused all-reduce add is gfx1250 only")
    if not hasattr(communication_op, "tensor_model_parallel_all_reduce_add"):
        pytest.skip("aiter predates tensor_model_parallel_all_reduce_add")
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        results = manager.dict()
        procs = [
            ctx.Process(target=_rank, args=(r, init_method, results))
            for r in range(_WORLD)
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
        # Every row count under the cap took the fused kernel, none past it did,
        # and nothing differs anywhere.
        for r in range(_WORLD):
            eager_fused, fused_rows, *differs = results[r]
            assert eager_fused == fused_rows, (r, eager_fused, fused_rows)
            assert differs == [0, 0, 0], (r, differs)
