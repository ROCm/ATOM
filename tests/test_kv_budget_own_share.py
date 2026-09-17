# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whose memory `gpu_memory_utilization` is a share of.

Device-used minus torch-reserved is the only reading available for memory this
process holds outside the allocator, and on a card shared with anyone else it
over-reports by whatever they hold. Subtracting that from our own utilization
budget charges their memory to us.

It shows up on the sleep/wake path, which re-sizes the KV pool. A colocated RL
trainer allocates after the rollout engine has been sized, so the second reading
is far larger than the first, the pool comes back a fraction of its size, and
the decode graphs captured against the original fault when recaptured against
the replacement.

Imports `kv_budget` rather than `model_runner`, which pulls AITER and therefore
does not load on the non-GPU runner this has to keep working on.
"""

from atom.model_engine.kv_budget import own_non_torch_bytes

GB = 1 << 30

# One card, and what the rollout engine holds on it once it is loaded.
TOTAL = 288 * GB
RESERVED = 60 * GB
# Sized alone: 16GB outside the allocator is ours (RCCL buffers and friends).
FREE_ALONE = 212 * GB
OURS = 16 * GB


def test_first_sizing_reports_the_live_reading():
    # Nothing to compare against, so what the device says is ours by definition.
    assert own_non_torch_bytes(TOTAL, FREE_ALONE, RESERVED, None) == OURS


def test_a_peer_allocating_later_is_not_charged_to_us():
    """The regression: a trainer waking up must not shrink our pool.

    Same engine, same 60GB reserved, but a colocated trainer now holds 38GB, so
    free drops by that much and the live reading comes back at 54GB.
    """
    free_shared = FREE_ALONE - 38 * GB
    assert own_non_torch_bytes(TOTAL, free_shared, RESERVED, None) == 54 * GB
    assert own_non_torch_bytes(TOTAL, free_shared, RESERVED, OURS) == OURS


def test_a_single_tenant_card_is_unaffected():
    """Where nothing else grows, this returns what it always returned."""
    first = own_non_torch_bytes(TOTAL, FREE_ALONE, RESERVED, None)
    assert own_non_torch_bytes(TOTAL, FREE_ALONE, RESERVED, first) == first == OURS


def test_the_baseline_holds_when_a_peer_releases_too():
    """A pool that keeps its size across sleep/wake is the point, so the number
    does not move in either direction once it is established."""
    assert own_non_torch_bytes(TOTAL, FREE_ALONE + 8 * GB, RESERVED, OURS) == OURS


def test_a_full_card_never_reports_negative_overhead():
    assert own_non_torch_bytes(TOTAL, 0, TOTAL, None) == 0
