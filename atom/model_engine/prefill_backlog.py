# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A rank's prefill backlog, as `share_cache` DP routing balances on it."""

from __future__ import annotations

import itertools

from atom.model_engine.sequence import SequenceStatus

# Waiting requests walked per call: the router needs the backlog to a few
# percent, so a deep queue is sampled and the rest extrapolated.
BACKLOG_SCAN = 1024


def flush_dp_reuse(scheduler, output_queue) -> None:
    """Queue the scheduler's `share_cache` reuse reports (`Scheduler.
    dp_reuse_reports`) for the CoreManager. Called before a step's outputs are
    queued: a request's report then reaches the router before its first or
    last output, which complete or release its placement."""
    reports = scheduler.dp_reuse_reports
    if reports:
        scheduler.dp_reuse_reports = []
        output_queue.put_nowait(("DP_REUSE", reports))


def prefill_backlog(scheduler, scan: int = BACKLOG_SCAN) -> int:
    """Prefill tokens `scheduler` still has to compute.

    Walks at most `scan` waiting requests and extrapolates the rest. An aborted
    request is no work; a running request that has produced tokens is decoding
    and counts nothing, while a preempted one waits to prefill again and counts
    what it must recompute (`Scheduler.pending_prefill_tokens`), also once it
    runs again: a readmitted one keeps its generated tokens while it prefills in
    chunks (`is_partial_prefill`).
    """
    waiting = scheduler.waiting
    scanned = counted = 0
    for seq in itertools.islice(waiting, scan):
        scanned += 1
        if seq.status != SequenceStatus.ABORTED:
            counted += scheduler.pending_prefill_tokens(seq, estimate=True)
    backlog = counted
    if scanned == scan and len(waiting) > scanned:
        backlog += counted * (len(waiting) - scanned) // scanned
    backlog += sum(
        scheduler.pending_prefill_tokens(seq, estimate=True)
        for seq in scheduler.running
        if seq.num_completion_tokens == 0 or seq.is_partial_prefill
    )
    return backlog
