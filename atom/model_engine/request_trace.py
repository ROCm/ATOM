# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Per-request stage timestamps along the prefill -> decode path.

The engine already reports TTFT as a single number: ``first_token_time -
arrive_time``, logged once per request by ``InputOutputProcessor.postprocess``.
That number says how long the request waited; it says nothing about *where*.
In a 1P1D deployment the wait is a sum over queueing for a block assignment,
the prefill forward itself, the KV transfer, the parked wait on the decode
side, and the first decode step -- and a simulator that reproduces the forward
times can still miss end-to-end TTFT by hundreds of milliseconds because one of
those other terms is not what it modelled.

So this module stamps the transitions that are already there. Every stamp site
is a state change the scheduler was performing anyway -- a queue append, a
status assignment, a block table arriving -- at request rate, never at token
rate and never inside a forward. A finished request is written as one CSV row.

Off unless ``ATOM_REQUEST_TRACE_DIR`` names a directory. When off, `stamp` is
a module-global load and a comparison against None, and nothing is allocated:
the per-request dict of stamps only comes into existence under a live tracer.

Clock: ``time.time()``, not ``time.monotonic()``. The stages span the
front-end, prefill and decode processes, and only wall clock is comparable
across them. They run on one host, so the wall clock is the same clock.
"""

import atexit
import logging
import os
import threading
import time
from collections import deque

logger = logging.getLogger("atom")

# Stage columns, in the order a request passes through them. Not every request
# visits every stage -- a monolithic run has no KV transfer, a decode-side
# request never runs a prefill forward -- and an unvisited stage is written as
# an empty field rather than as a zero, so "never happened" stays
# distinguishable from "happened at the epoch".
STAGES = (
    # Front-end: the request reached the engine and was tokenized.
    "arrive",
    # Scheduler: appended to the waiting queue of this process.
    "queued",
    # Prefill side: the decode process's BlockAssignment landed, so the
    # request can finally be scheduled. The gap from `queued` is the
    # disaggregation handshake, which no single-process model has at all.
    "blocks",
    # First time this request was placed in a scheduled batch here.
    "sched",
    # Decode side: parked on WAITING_FOR_REMOTE_KVS, and released from it.
    "kv_park",
    "kv_ready",
    # Decode side: promoted to RUNNING with the producer's first token in hand.
    "first_decode",
    # Existing fields, restated here so one row holds the whole timeline.
    "first_token",
    "leave",
)

_ROW_COLUMNS = (
    "role",
    "seq_id",
    "request_id",
    "n_prompt",
    "n_cached",
    "n_completion",
    "reason",
) + STAGES

_HEADER = ",".join(_ROW_COLUMNS) + "\n"

# Rows kept in memory between writes. A drain is one buffered write of a few
# tens of kilobytes; 4096 rows is a couple of minutes of completions at the
# rates these benchmarks reach, so a killed process loses seconds, not runs.
_DEFAULT_CAPACITY = 4096
_DRAIN_EVERY = 128


def _fmt(value):
    """Six decimals is a microsecond, which is finer than any stage here."""
    return "" if not value else "%.6f" % value


class RequestTracer:
    """Buffers finished-request rows and appends them to one CSV."""

    def __init__(self, path, capacity=_DEFAULT_CAPACITY):
        self.path = path
        self.rows = deque(maxlen=capacity)
        self.dropped = 0
        self._file = None
        # Scheduler and the connector's receiver thread both retire requests,
        # so the buffer and the write are shared state.
        self._lock = threading.Lock()

    def record(self, seq, role, reason=""):
        stamps = getattr(seq, "_stage_times", None) or {}
        row = (
            role,
            getattr(seq, "id", ""),
            getattr(seq, "external_request_id", "") or "",
            int(getattr(seq, "num_prompt_tokens", 0) or 0),
            int(getattr(seq, "num_cached_tokens", 0) or 0),
            int(getattr(seq, "num_completion_tokens", 0) or 0),
            # Commas would split the row; the reasons in use are single words,
            # but "stop_<token>" is caller-supplied text.
            str(reason or getattr(seq, "leave_reason", "") or "").replace(",", ";"),
        ) + tuple(_fmt(stamps.get(stage)) for stage in STAGES)
        with self._lock:
            if len(self.rows) == self.rows.maxlen:
                self.dropped += 1
            self.rows.append(row)
            drain = len(self.rows) >= _DRAIN_EVERY
        if drain:
            self.drain()

    def drain(self):
        with self._lock:
            if not self.rows:
                return
            rows, self.rows = self.rows, deque(maxlen=self.rows.maxlen)
            if self._file is None:
                fresh = not os.path.exists(self.path)
                self._file = open(self.path, "a", buffering=1 << 16)
                if fresh:
                    self._file.write(_HEADER)
            self._file.write("".join(",".join(str(f) for f in row) + "\n" for row in rows))
            # One write(2) of tens of kilobytes per drain. Without it the rows
            # sit in the buffer, and a run cut short by its scheduler loses
            # exactly the tail that explains why it was cut short.
            self._file.flush()

    def close(self):
        self.drain()
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


_tracer = None


def configure(path, capacity=_DEFAULT_CAPACITY):
    """Turn the trace on, writing to *path*. Idempotent per process."""
    global _tracer
    if _tracer is not None:
        return _tracer
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    _tracer = RequestTracer(path, capacity=capacity)
    atexit.register(close)
    logger.info("Request trace enabled: %s", path)
    return _tracer


def configure_for_role(role):
    """Turn the trace on for *role* if ATOM_REQUEST_TRACE_DIR names a directory.

    One file per process: prefill and decode are separate processes, and under
    data parallelism there is one scheduler per DP rank. Joining them is the
    reader's job -- `request_id` is the same string on both sides.
    """
    from atom.utils import envs

    directory = envs.ATOM_REQUEST_TRACE_DIR
    if not directory:
        return None
    return configure(
        os.path.join(directory, "requests-%s-%d.csv" % (role or "engine", os.getpid()))
    )


def enabled():
    return _tracer is not None


def stamp(seq, stage):
    """Record that *seq* reached *stage*, keeping the first arrival at it.

    First and not last: these stages are entered once per request in the
    common case, but a preempted request is re-queued and re-scheduled, and
    the question being asked is when it *first* got there.
    """
    if _tracer is None:
        return
    stamps = getattr(seq, "_stage_times", None)
    if stamps is None:
        stamps = seq._stage_times = {}
    if stage not in stamps:
        stamps[stage] = time.time()


def adopt(seq, stage, value):
    """Stamp *stage* with an already-taken timestamp, e.g. `arrive_time`."""
    if _tracer is None or not value:
        return
    stamps = getattr(seq, "_stage_times", None)
    if stamps is None:
        stamps = seq._stage_times = {}
    stamps.setdefault(stage, float(value))


def record(seq, role, reason=""):
    """Emit one row for a request that is leaving this process."""
    if _tracer is None:
        return
    adopt(seq, "arrive", getattr(seq, "arrive_time", 0.0))
    adopt(seq, "first_token", getattr(seq, "first_token_time", 0.0))
    stamp(seq, "leave")
    _tracer.record(seq, role, reason)


def close():
    global _tracer
    if _tracer is not None:
        _tracer.close()
        _tracer = None
