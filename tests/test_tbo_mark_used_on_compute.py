# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tbo_mark_used_on_compute`: a tensor allocated on the TBO comm stream and
read on the compute stream must not be handed to the next comm-stream
allocation while the compute stream has yet to read it. Under PCP+TBO the
gathered extend K was overwritten that way by the other ubatch's gather."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.utils.tbo.ubatching import tbo_mark_used_on_compute

N = 1 << 20


def _next_comm_allocation_takes_it(mark: bool) -> bool:
    """Whether the comm stream's next allocation of the same size gets the
    freed tensor's memory while the compute stream has yet to read it. Asserts
    the allocator's choice rather than the overwrite: whether an overwrite
    lands first also depends on how the two streams share hardware queues."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # no other free block of this size
    comm, compute = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(comm):
        gathered = torch.empty((N,), device="cuda")
    ptr = gathered.data_ptr()
    compute.wait_stream(comm)
    with torch.cuda.stream(compute):
        if mark:
            tbo_mark_used_on_compute(gathered)
        torch.cuda._sleep(100_000_000)  # the attention kernel, still queued
        gathered.sum()
    del gathered  # the forward returns; the host frees it
    with torch.cuda.stream(comm):  # the other ubatch's next gather
        taken = torch.empty((N,), device="cuda").data_ptr() == ptr
    torch.cuda.synchronize()
    return taken


def test_marked_tensor_is_held_until_the_compute_stream_reads_it():
    assert not _next_comm_allocation_takes_it(mark=True)


def test_unmarked_tensor_goes_straight_back_to_the_comm_stream():
    """Pins the hazard itself: if this stops holding, the test above no longer
    shows anything."""
    assert _next_comm_allocation_takes_it(mark=False)
