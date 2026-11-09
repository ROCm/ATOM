# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tensor_model_parallel_all_reduce_add(routed, shared)` against what
`MoE.combine_outputs` did before, `tensor_model_parallel_all_reduce(routed + shared)`:
the same bits on every rank, at V4.1-Flash's MoE combine width (5120 bf16), for
decode row counts the fused kernel takes and prefill ones past its cap, eager and
replayed in a HIP graph. The fused kernel is aiter's HIP Lamport one-shot all-reduce
(ROCm/aiter#7499, 8 MiB at TP4) when aiter has it, else the Gluon all-reduce
(ROCm/aiter#7488, 4 MiB). Requires 4 gfx1250 GPUs and an aiter build with
ROCm/aiter#7488 for the Gluon route or ROCm/aiter#7499 for the Lamport route."""

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
# Decode and prefill row counts; capacity boundaries are added at run time.
_ROWS = (1, 8, 48, 96, 288, 4096)
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
    """Record input bytes for each fused call; return the Gluon all-reduce cap
    and the largest enabled fused-add cap in bytes on this aiter."""
    import inspect

    import aiter.ops.triton.comms.all_reduce as gluon
    from aiter.dist.device_communicators import custom_all_reduce
    from aiter.dist.device_communicators.communicator_cuda import (
        _GLUON_PLAIN_AR_MAX_BYTES,
    )
    from aiter.dist.parallel_state import get_tp_group

    plain = gluon.all_reduce
    signature = inspect.signature(plain)
    device_comm = get_tp_group().device_communicator

    def counting_gluon(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        if bound.get("x2") is not None:
            x = bound["x"]
            fused_calls.append(x.numel() * x.element_size())
        return plain(*args, **kwargs)

    gluon.all_reduce = counting_gluon
    # The Gluon kernel folds the add only where it takes the second input (#7488).
    cap = (
        _GLUON_PLAIN_AR_MAX_BYTES
        if "x2" in signature.parameters and device_comm.gluon_ar_plain is not None
        else 0
    )
    lamport = getattr(custom_all_reduce.CustomAllreduce, "lamport_ar_add", None)
    if lamport is not None:

        def counting_lamport(self, inp, other):
            fused_calls.append(inp.numel() * inp.element_size())
            return lamport(self, inp, other)

        custom_all_reduce.CustomAllreduce.lamport_ar_add = counting_lamport
        ca_comm = device_comm.ca_comm
        if ca_comm is not None and not ca_comm.disabled:
            cap = max(cap, getattr(ca_comm, "_lamport_max_bytes", 0))
    return _GLUON_PLAIN_AR_MAX_BYTES, cap


def _checks(rank):
    """(actual fused input sizes, expected fused input sizes, elements differing
    from the add + all-reduce, elements differing across ranks, elements differing
    over graph replays)."""
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group

    import atom.model_ops.communication_op as comm

    comm._tbo_aware_tp_reduce = lambda tp_size: False  # no ATOM config here
    fused_calls = []
    gluon_cap, fused_cap = _hook_fused_kernels(fused_calls)
    cap_rows = fused_cap // _ROW_BYTES
    assert cap_rows > 0, "no fused all-reduce add kernel is enabled for one row"
    # Include the plain Gluon boundary even on Lamport-only builds: the reference
    # all-reduce changes kernels there. Also test the fused cap and one past it.
    extra = {
        rows
        for cap in (gluon_cap // _ROW_BYTES, cap_rows)
        for rows in (cap, cap + 1)
        if rows > 0
    }
    all_rows = sorted(set(_ROWS) | extra)
    expected_fused = [rows * _ROW_BYTES for rows in all_rows if rows <= cap_rows]
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
    eager_fused = fused_calls.copy()
    fused_calls.clear()

    ins = [tuple(torch.empty_like(t) for t in _pair(r, 0, rank)) for r in _GRAPH_ROWS]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = [comm.tensor_model_parallel_all_reduce_add(a, b) for a, b in ins]
        want = [comm.tensor_model_parallel_all_reduce(a + b) for a, b in ins]
    expected_graph_fused = [
        rows * _ROW_BYTES for rows in _GRAPH_ROWS if rows <= cap_rows
    ]
    assert fused_calls == expected_graph_fused, (
        rank,
        fused_calls,
        expected_graph_fused,
    )
    replay_differ = 0
    for rep in range(_REPLAYS):
        for j, (a, b) in enumerate(ins):
            new_a, new_b = _pair(_GRAPH_ROWS[j], 100 + rep * 10 + j, rank)
            a.copy_(new_a)
            b.copy_(new_b)
        graph.replay()
        torch.cuda.synchronize()
        replay_differ += sum(_differ(g, w) for g, w in zip(got, want))
    return eager_fused, expected_fused, differ, across, replay_differ


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
        # Check each fused input size in order, so missing, duplicate, or
        # out-of-cap calls cannot hide behind the same total count.
        for r in range(_WORLD):
            eager_fused, expected_fused, *differs = results[r]
            assert eager_fused == expected_fused, (r, eager_fused, expected_fused)
            assert differs == [0, 0, 0], (r, differs)
