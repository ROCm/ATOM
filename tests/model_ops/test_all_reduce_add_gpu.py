# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tensor_model_parallel_all_reduce_add(routed, shared)` against what
`MoE.combine_outputs` did before, `tensor_model_parallel_all_reduce(routed + shared)`:
the same bits on every rank, at V4.1-Flash's MoE combine width (5120 bf16).

The fused kernel is aiter's HIP Lamport one-shot all-reduce
(``CustomAllreduce.lamport_ar_add``, ROCm/aiter#7499). Every row count that
kernel accepts must call it, including over a HIP-graph capture; the row past
its cap must not. The cap is aiter's TP4 default, 8 MiB, not whatever
``should_lamport_ar_add`` reports. Requires 4 gfx1250 GPUs and ROCm/aiter#7499.
"""

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
# One prefill size past the TP4 Lamport cap. The search stops here.
_PAST_CAP_ROWS = 4096
# Decode and verify sizes this fused add is meant to take.
_DECODE_ROWS = (1, 8, 48, 96, 288)
_ROWS = _DECODE_ROWS + (_PAST_CAP_ROWS,)
# aiter's unset TP4 limit, 8 MiB, in rows of 5120 bf16. 819.
_DEFAULT_CAP_ROWS = (8 << 20) // (_DIM * 2)
_CAP_ENV = "AITER_CUSTOM_AR_LAMPORT_RMS_MAX_SIZE"
_GRAPH_ROWS = (48, 8, 288, 1)
_REPLAYS = 10


def _pair(rows, seed, rank):
    gen = torch.Generator(device="cuda").manual_seed(seed * _WORLD + rank)
    routed = torch.randn(rows, _DIM, generator=gen, device="cuda").to(torch.bfloat16)
    shared = torch.randn(rows, _DIM, generator=gen, device="cuda").to(torch.bfloat16)
    return routed, shared


def _differ(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).sum())


def _nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def _install_lamport_counter(calls):
    """Count bytes passed to the HIP kernel. Returns the original method."""
    from aiter.dist.device_communicators import custom_all_reduce

    cls = custom_all_reduce.CustomAllreduce
    original = getattr(cls, "lamport_ar_add", None)
    if original is None:
        return None

    def counting(self, inp, other):
        calls.append(_nbytes(inp))
        return original(self, inp, other)

    cls.lamport_ar_add = counting
    return original


def _restore_lamport(original):
    if original is None:
        return
    from aiter.dist.device_communicators import custom_all_reduce

    custom_all_reduce.CustomAllreduce.lamport_ar_add = original


def _cap_rows(accepts):
    """Largest row count ``accepts`` takes, searching only below
    ``_PAST_CAP_ROWS``.

    The cap is a byte limit, so once a row count is rejected every larger one
    is too. ``None`` means ``_PAST_CAP_ROWS`` itself was accepted: that probe
    is the last tensor this search builds.
    """
    if accepts(_PAST_CAP_ROWS):
        return None
    if not accepts(1):
        return 0
    lo, hi = 1, _PAST_CAP_ROWS
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if accepts(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _checks(rank):
    """Per-rank bit and dispatch report. ``error`` is empty on success."""
    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group

    import atom.model_ops.communication_op as comm

    comm._tbo_aware_tp_reduce = lambda tp_size: False  # no ATOM config here
    ca = getattr(get_tp_group().device_communicator, "ca_comm", None)
    if ca is None or not hasattr(ca, "should_lamport_ar_add"):
        return {"error": "aiter has no HIP Lamport all-reduce add"}

    def accepts(rows):
        return ca.should_lamport_ar_add(*_pair(rows, 0, rank))

    cap_rows = _cap_rows(accepts)
    if cap_rows is None:
        return {"error": f"Lamport accepted {_PAST_CAP_ROWS} rows at {_DIM} bf16"}
    if cap_rows < _DECODE_ROWS[-1]:
        return {
            "error": (
                f"Lamport cap is {cap_rows} rows, below decode size "
                f"{_DECODE_ROWS[-1]}"
            )
        }
    # Unset (and blank) is aiter's 8 MiB TP4 default. A predicate that agrees
    # with itself at a smaller cap still fails this.
    if os.environ.get(_CAP_ENV, "").strip() == "" and cap_rows != _DEFAULT_CAP_ROWS:
        return {
            "error": (
                f"Lamport cap is {cap_rows} rows, expected {_DEFAULT_CAP_ROWS} "
                f"at the unset TP4 limit of 8 MiB"
            )
        }
    rows = sorted(set(_ROWS + (cap_rows, cap_rows + 1)))

    calls = []
    original = _install_lamport_counter(calls)
    if original is None:
        return {"error": "aiter has no HIP Lamport all-reduce add"}
    try:
        group = get_tp_group().device_group
        differ = across = 0
        eager = []
        for i, n in enumerate(rows):
            routed, shared = _pair(n, i, rank)
            expect = (
                [_nbytes(routed)] if ca.should_lamport_ar_add(routed, shared) else []
            )
            before = len(calls)
            want = comm.tensor_model_parallel_all_reduce(routed + shared)
            got = comm.tensor_model_parallel_all_reduce_add(routed, shared)
            eager.append((n, calls[before:], expect))
            ranks = [torch.empty_like(got) for _ in range(_WORLD)]
            dist.all_gather(ranks, got, group=group)
            differ += _differ(got, want)
            across += sum(_differ(got, other) for other in ranks)
        ins = [
            tuple(torch.empty_like(t) for t in _pair(n, 0, rank)) for n in _GRAPH_ROWS
        ]
        expected_graph = [
            _nbytes(routed)
            for routed, shared in ins
            if ca.should_lamport_ar_add(routed, shared)
        ]
        calls.clear()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = [comm.tensor_model_parallel_all_reduce_add(a, b) for a, b in ins]
            want = [comm.tensor_model_parallel_all_reduce(a + b) for a, b in ins]
        graph_calls = list(calls)
        replay_differ = 0
        for rep in range(_REPLAYS):
            for j, (routed, shared) in enumerate(ins):
                new_routed, new_shared = _pair(_GRAPH_ROWS[j], 100 + rep * 10 + j, rank)
                routed.copy_(new_routed)
                shared.copy_(new_shared)
            graph.replay()
            torch.cuda.synchronize()
            replay_differ += sum(_differ(g, w) for g, w in zip(got, want))
        return {
            "error": "",
            "eager": eager,
            "graph_calls": graph_calls,
            "expected_graph": expected_graph,
            "differ": differ,
            "across": across,
            "replay": replay_differ,
        }
    finally:
        _restore_lamport(original)


def _rank(rank, init_method, results):
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
    from aiter.dist.communication_op import (
        tensor_model_parallel_all_reduce_add as aiter_all_reduce_add,
    )
    from aiter.dist.device_communicators import custom_all_reduce
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250":
        pytest.skip("the HIP Lamport all-reduce add is gfx1250 only")
    # aiter#7499 is required. A missing symbol fails this test instead of skipping.
    assert callable(aiter_all_reduce_add)

    if not hasattr(custom_all_reduce.CustomAllreduce, "lamport_ar_add"):
        pytest.fail("needs CustomAllreduce.lamport_ar_add from ROCm/aiter#7499")
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        results = manager.dict()
        procs = [
            ctx.Process(target=_rank, args=(r, init_method, results))
            for r in range(_WORLD)
        ]
        for proc in procs:
            proc.start()
        deadline = time.monotonic() + 300
        for proc in procs:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
        hung = [proc for proc in procs if proc.is_alive()]
        for proc in hung:  # never leave a rank behind for interpreter exit to join
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
        assert not hung and all(proc.exitcode == 0 for proc in procs)
        for rank in range(_WORLD):
            result = results[rank]
            assert result["error"] == "", (rank, result["error"])
            for rows, took, expected in result["eager"]:
                assert took == expected, (rank, rows, took, expected)
            assert result["expected_graph"], (
                rank,
                "graph capture took no Lamport call",
            )
            assert result["graph_calls"] == result["expected_graph"], (
                rank,
                result["graph_calls"],
                result["expected_graph"],
            )
            assert (result["differ"], result["across"], result["replay"]) == (
                0,
                0,
                0,
            ), (
                rank,
                result["differ"],
                result["across"],
                result["replay"],
            )
