# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whether and how the runner thread computes on a High-priority stream.

Kept out of the runner so it can be exercised on a machine with no GPU build:
importing the runner pulls in aiter, which resolves the chip architecture
through rocminfo at import time. `torch` imports fine without one, so the
stream construction can live here too; keep anything heavier out.
"""

import logging

import torch

from atom.utils import envs

logger = logging.getLogger("atom")


def high_priority_compute_stream_enabled(config) -> bool:
    """Whether `ATOM_HIGH_PRIORITY_COMPUTE_STREAM` applies to this config.

    TBO overrides the flag. UBatchWrapper computes on the runner's stream, and
    a builder's tbo_comm_stream_priority (High on V4 and V4.1) puts comm ahead
    of the other ubatch's compute. ROCm has a single level above Normal, so a
    High compute stream would erase that ordering and share comm's queue pool.
    """
    if not envs.ATOM_HIGH_PRIORITY_COMPUTE_STREAM:
        return False
    if config.enable_tbo or config.enable_tbo_decode:
        logger.warning(
            "ATOM_HIGH_PRIORITY_COMPUTE_STREAM is ignored with TBO enabled: "
            "the compute stream would match the TBO comm stream's priority."
        )
        return False
    return True


def new_compute_priority_stream(device) -> torch.cuda.Stream:
    """A new stream on `device` at the priority the runner computes on.

    One builder for the full-CU streams the runner computes on, so the
    RapidServe pools cannot drift from `enter_compute_stream` in priority or
    device. The CU-masked streams of a constrained RapidServe rank come from
    `_stream_with_cu_mask` instead and stay at Normal priority; HIP pools
    masked queues separately (`queueWithCUMaskPool_`), so they already cannot
    share a queue with a graph's parallel streams.
    """
    _, high = torch.cuda.Stream.priority_range()
    return torch.cuda.Stream(device, priority=high)


def enter_compute_stream(config, device) -> torch.cuda.Stream | None:
    """Make a High-priority stream the calling thread's current stream.

    HIP pools hardware queues per priority and a graph's parallel streams are
    always Normal, so a Normal launch stream can share their queue and run the
    graph's side branch (the dual-stream shared experts) behind its main branch
    instead of beside it. Only the stream a graph is launched from matters, not
    the one it was captured on, so making this the runner thread's current
    stream covers FULL, PIECEWISE and draft graphs.

    `torch.cuda.set_stream` binds per thread, so this moves the runner thread
    only. Another thread that enqueues work through
    `torch.cuda.current_stream()` still sees the default stream and no longer
    shares one with the forward, so anything that relied on that implicit
    ordering has to wait on an event instead -- see the KV-transfer send
    workers, which wait on the prefill's `kv_cache_ready_event` before
    gathering state the forward writes.
    """
    if not high_priority_compute_stream_enabled(config):
        return None
    stream = new_compute_priority_stream(device)
    # Torch streams do not sync with the default stream; drain it first.
    torch.cuda.synchronize(device)
    torch.cuda.set_stream(stream)
    return stream
