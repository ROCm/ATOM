# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tbo_mark_used_on_compute`: a tensor allocated on the TBO comm stream and
read on the compute stream must not be handed to the next comm-stream
allocation while the compute stream has yet to read it. Under PCP+TBO the
gathered extend K was overwritten that way by the other ubatch's gather."""

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.utils.tbo import ubatching

N = 1 << 20


def _next_comm_allocation_takes_it(monkeypatch, mark: bool) -> bool:
    """Whether the comm stream's next allocation of the same size gets the
    freed tensor's memory while the compute stream has yet to read it. Asserts
    the allocator's choice rather than the overwrite: whether an overwrite
    lands first also depends on how the two streams share hardware queues."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # no other free block of this size
    comm, compute = torch.cuda.Stream(), torch.cuda.Stream()
    ctx = SimpleNamespace(compute_stream=compute)
    monkeypatch.setattr(ubatching, "_get_current_tbo_context", lambda: ctx)
    with torch.cuda.stream(comm):
        gathered = torch.empty((N,), device="cuda")
    ptr = gathered.data_ptr()
    compute.wait_stream(comm)
    with torch.cuda.stream(compute):
        if mark:
            ubatching.tbo_mark_used_on_compute(gathered)
        torch.cuda._sleep(100_000_000)  # the attention kernel, still queued
        gathered.sum()
    del gathered  # the forward returns; the host frees it
    with torch.cuda.stream(comm):  # the other ubatch's next gather
        taken = torch.empty((N,), device="cuda").data_ptr() == ptr
    torch.cuda.synchronize()
    return taken


def test_marked_tensor_is_held_until_the_compute_stream_reads_it(monkeypatch):
    assert not _next_comm_allocation_takes_it(monkeypatch, mark=True)


@pytest.mark.skipif(
    any(
        "expandable_segments:true" in os.environ.get(var, "").lower().replace(" ", "")
        for var in (
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
        )
    ),
    reason="relies on the default caching allocator handing a freed block of "
    "the same size straight back to the stream that made it",
)
def test_unmarked_tensor_goes_straight_back_to_the_comm_stream(monkeypatch):
    """Pins the hazard itself: if this stops holding, the test above no longer
    shows anything."""
    assert _next_comm_allocation_takes_it(monkeypatch, mark=False)


def test_marking_while_a_graph_records(monkeypatch):
    """Decode TBO's overlapped all-reduce and the side-stream compressors mark
    their tensors while a graph records: the mark must neither break the
    capture nor the replay."""
    comm, compute = torch.cuda.Stream(), torch.cuda.Stream()
    ctx = SimpleNamespace(compute_stream=compute)
    monkeypatch.setattr(ubatching, "_get_current_tbo_context", lambda: ctx)
    x = torch.ones((N,), device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=comm):
        gathered = x * 2
        compute.wait_stream(comm)
        with torch.cuda.stream(compute):
            ubatching.tbo_mark_used_on_compute(gathered)
            out = gathered + 1
        comm.wait_stream(compute)
    del gathered
    x.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(x, 7))
