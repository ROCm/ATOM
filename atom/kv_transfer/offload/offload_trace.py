# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Per-job timing for the LMCache offload copy pools.

`OFFLOAD_PROFILE=1` already logs what a copy did once it was running --
`pack_ms`, `copy_ms`, `sync_ms`, bytes, chunks. What it does not log is the
part a request actually waits on: a copy runs on a `ThreadPoolExecutor` with
one load worker and `OFFLOAD_COPY_WORKERS` save workers, so between the
submit on the RPC thread and the first instruction of the job there is a queue
whose depth nobody measures. On the TTFT critical path that queue is the
difference between "the copy took 4 ms" and "the request waited 40 ms for a
copy that took 4 ms", and it is exactly the term a model built from the copy
durations alone cannot have.

So this records both ends: the submit, the pickup, the finish, and whatever
the job itself chose to report. One CSV row per job.

Off unless ``ATOM_OFFLOAD_TRACE_DIR`` names a directory. With it off, every
entry point is a module-global load and a comparison against None; the submit
mark is not even taken.

The row's clock is ``time.time()`` for the same reason the request trace uses
it -- these rows are read next to that trace, from a different process -- while
the durations come from ``time.perf_counter()`` deltas, which is the clock
that is actually monotonic.
"""

import atexit
import logging
import os
import threading
import time
from collections import deque

logger = logging.getLogger("atom")

COLUMNS = (
    "role",
    "kind",
    "req_id",
    "operation",
    # Wall clock at submit and at finish, so a row lines up with the request
    # trace and with the engine log.
    "submit",
    "finish",
    # Time the job spent in the executor queue before a worker picked it up.
    "queue_ms",
    # Time from pickup to return, including any event synchronize the job did.
    "run_ms",
    "ok",
    # Whatever the job body reported through `annotate`, when OFFLOAD_PROFILE
    # is not the thing gating it. Empty when the layout does not report.
    "total_bytes",
    "chunks",
    "pack_ms",
    "copy_ms",
    "sync_ms",
    "transfer_ms",
)

_HEADER = ",".join(COLUMNS) + "\n"
_ANNOTATED = ("total_bytes", "chunks", "pack_ms", "copy_ms", "sync_ms", "transfer_ms")

# A copy job is milliseconds, and the pools are a handful of threads, so the
# row rate is bounded by the copies themselves. 8192 rows is minutes of them.
_DEFAULT_CAPACITY = 8192
_DRAIN_EVERY = 256

# Attribute the submit timestamp is parked on. The job metadata is a plain
# dataclass that the connector threads through to the worker, which makes it
# the one object both ends of the queue already hold.
_SUBMIT_ATTR = "_offload_trace_submit"


class OffloadTracer:
    def __init__(self, path, capacity=_DEFAULT_CAPACITY):
        self.path = path
        self.rows = deque(maxlen=capacity)
        self.dropped = 0
        self._file = None
        # Both pools write here, and so does the RPC thread for an inline job.
        self._lock = threading.Lock()

    def add(self, row):
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
            self._file.write("".join(",".join(row) + "\n" for row in rows))
            self._file.flush()

    def close(self):
        self.drain()
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


_tracer = None
_role = "worker"
# The job body reports through `annotate`, which has no handle on the job; the
# thread running it is the handle.
_current = threading.local()


def configure(path, role="worker", capacity=_DEFAULT_CAPACITY):
    global _tracer, _role
    if _tracer is not None:
        return _tracer
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    _tracer = OffloadTracer(path, capacity=capacity)
    _role = role
    atexit.register(close)
    logger.info("Offload trace enabled: %s", path)
    return _tracer


def configure_for_role(role):
    """Turn the trace on if ATOM_OFFLOAD_TRACE_DIR names a directory.

    One file per worker process, named by rank and pid: the copy pools are
    per-worker, and so is the queue being measured.
    """
    from atom.utils import envs

    directory = envs.ATOM_OFFLOAD_TRACE_DIR
    if not directory:
        return None
    return configure(
        os.path.join(directory, "offload-%s-%d.csv" % (role or "worker", os.getpid())),
        role=role or "worker",
    )


def enabled():
    return _tracer is not None


def mark_submit(req):
    """Stamp *req* at the moment it is handed to an executor."""
    if _tracer is None:
        return
    try:
        setattr(req, _SUBMIT_ATTR, (time.time(), time.perf_counter()))
    except AttributeError:
        # A layout whose job metadata does not take attributes still gets a
        # row; it just has no queue time in it.
        pass


class _Job:
    __slots__ = ("kind", "req_id", "operation", "submit", "queue_ms", "t0", "stats")

    def __init__(self, kind, req):
        self.kind = kind
        self.req_id = str(getattr(req, "req_id", req))
        self.operation = str(
            getattr(req, "load_operation" if kind == "load" else "save_operation", "")
            or ""
        )
        submitted = getattr(req, _SUBMIT_ATTR, None)
        self.t0 = time.perf_counter()
        if submitted is None:
            self.submit = ""
            self.queue_ms = ""
        else:
            wall, perf = submitted
            self.submit = "%.6f" % wall
            self.queue_ms = "%.3f" % ((self.t0 - perf) * 1000)
        self.stats = {}


def begin(kind, req):
    """Called by the worker thread as it picks a job up. Returns a token."""
    if _tracer is None:
        return None
    job = _Job(kind, req)
    _current.job = job
    return job


def annotate(stats):
    """Attach a job body's own measurements to the row being built.

    Takes whatever subset of the columns the layout happens to report; a
    layout that reports nothing leaves those fields empty rather than zero,
    because a zero here reads as a copy that moved no bytes.
    """
    if _tracer is None:
        return
    job = getattr(_current, "job", None)
    if job is None or not stats:
        return
    for key in _ANNOTATED:
        if key in stats:
            job.stats[key] = stats[key]


def end(job, ok):
    """Called by the worker thread as the job returns, success or not."""
    if _tracer is None or job is None:
        return
    _current.job = None
    run_ms = (time.perf_counter() - job.t0) * 1000
    row = (
        _role,
        job.kind,
        job.req_id.replace(",", ";"),
        job.operation.replace(",", ";"),
        job.submit,
        "%.6f" % time.time(),
        job.queue_ms,
        "%.3f" % run_ms,
        "1" if ok else "0",
    ) + tuple(_format(job.stats.get(key)) for key in _ANNOTATED)
    _tracer.add(row)


def _format(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.3f" % value
    return str(value)


def close():
    global _tracer
    if _tracer is not None:
        _tracer.close()
        _tracer = None
