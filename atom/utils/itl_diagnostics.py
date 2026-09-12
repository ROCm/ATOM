# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Temporary CPU-only ITL metadata recorder.

Call configure after spawn. Fields must be metadata, never request/response
content or token IDs. Pass immutable values or fresh list snapshots that are
not mutated after emit. Event names must be static, low-cardinality labels.
The writer owns JSON serialization, metrics and all file I/O after configure.
"""

import atexit
import itertools
import json
import os
import re
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

_NORMAL_CAPACITY = 8192
_CRITICAL_CAPACITY = 1024
_DROP_CAPACITY = 1024
_WRITE_PERIOD = 0.05
_CLOSE_TIMEOUT = 2.0
_DURATION_BINS_NS = (10000, 100000, 1000000, 10000000, 100000000, 1000000000)
_CONTENT_KEYS = frozenset(
    (
        "text",
        "content",
        "prompt",
        "response",
        "token_ids",
        "tokens",
        "exception",
        "exception_text",
        "error_message",
        "messages",
        "input_ids",
        "output_ids",
    )
)
_recorder = None
_config_lock = threading.Lock()


def _counter_value(counter):
    # itertools.count increments atomically under CPython's GIL, including drops.
    # count's public repr is count(<next value>); pickle support is deprecated.
    return int(repr(counter)[6:-1])


def _safe_fields(fields, depth=0):
    if depth > 4:
        return None
    if isinstance(fields, dict):
        return {
            key: _safe_fields(value, depth + 1)
            for key, value in fields.items()
            if isinstance(key, str) and key.lower() not in _CONTENT_KEYS
        }
    if fields is None or type(fields) in (bool, int, float, str):
        return fields
    if type(fields) in (list, tuple):
        return [_safe_fields(value, depth + 1) for value in fields]
    # Never call repr/str on tensors, exceptions or arbitrary caller objects.
    return None


class _Recorder:
    def __init__(self, directory, role, rank):
        self.pid = os.getpid()
        self.directory = directory
        self.role = role
        self.rank = rank
        self.path = directory / "itl_events" / f"{role}_{self.pid}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.normal = deque()
        self.critical = deque()
        self.losses = deque(maxlen=_DROP_CAPACITY)
        self.attempts = itertools.count()
        self.dropped = itertools.count()
        self.critical_dropped = itertools.count()
        self.drop_reasons = {
            reason: itertools.count()
            for reason in ("lock_busy", "buffer_full", "closing")
        }
        self.writer_cpu_start_ns = None
        self.accepted = 0
        self.written = 0
        self.high_water = 0
        self.seq = 0
        self.flush_ns = 0
        self.writer_error = None
        self.accepting = True
        self.stop = threading.Event()
        self.done = threading.Event()
        self.deadline = None
        self.phase_counts = {}
        self.duration_bins = {}
        self.metrics_start_ns = time.monotonic_ns()
        self.start_ns = self.metrics_start_ns
        self.thread = threading.Thread(
            target=self._run, name=f"itl-writer-{role}", daemon=True
        )
        self.thread.start()

    def emit(self, event, critical, fields):
        if not self.accepting:
            return
        attempt = next(self.attempts)
        stamp = time.monotonic_ns()
        if not self.lock.acquire(blocking=False):
            self._drop(attempt, stamp, critical, "lock_busy")
            return
        try:
            if not self.accepting:
                self._drop(attempt, stamp, critical, "closing")
                return
            queue = self.critical if critical else self.normal
            capacity = _CRITICAL_CAPACITY if critical else _NORMAL_CAPACITY
            if len(queue) >= capacity:
                self._drop(attempt, stamp, critical, "buffer_full")
                return
            queue.append((attempt, stamp, threading.get_ident(), event, fields))
            self.accepted += 1
            self.high_water = max(
                self.high_water, len(self.normal) + len(self.critical)
            )
        finally:
            self.lock.release()

    def _drop(self, attempt, stamp, critical, reason):
        loss_index = next(self.dropped)
        next(self.drop_reasons[reason])
        if critical:
            next(self.critical_dropped)
        self.losses.append((loss_index, attempt, stamp, critical, reason))

    def _record(self, event, fields, stamp=None, thread=None):
        record = dict(fields)
        record.update(
            event=event,
            monotonic_ns=time.monotonic_ns() if stamp is None else stamp,
            thread=threading.get_ident() if thread is None else thread,
            pid=self.pid,
            rank=self.rank,
            role=self.role,
            seq=self.seq,
        )
        self.seq += 1
        return json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"

    def _stats(self):
        losses = list(self.losses)
        losses.sort()
        return {
            "attempted": _counter_value(self.attempts),
            "accepted": self.accepted,
            "written": self.written,
            "dropped": _counter_value(self.dropped),
            "critical_dropped": _counter_value(self.critical_dropped),
            "drop_reasons": {
                reason: _counter_value(counter)
                for reason, counter in self.drop_reasons.items()
            },
            "writer_cpu_ns": (
                time.thread_time_ns() - self.writer_cpu_start_ns
                if threading.current_thread() is self.thread
                and self.writer_cpu_start_ns is not None
                else None
            ),
            "buffer_depth": len(self.normal) + len(self.critical),
            "buffer_high_water": self.high_water,
            "normal_capacity": _NORMAL_CAPACITY,
            "critical_capacity": _CRITICAL_CAPACITY,
            "flush_duration_ns": self.flush_ns,
            "writer_failure": self.writer_error,
            "unwritten": self.accepted - self.written,
            "loss_detail_overflow": max(0, _counter_value(self.dropped) - len(losses)),
            "loss_window": (
                [min(x[2] for x in losses), max(x[2] for x in losses)]
                if losses
                else None
            ),
            # Evicted loss detail invalidates precise joins over this whole window.
            "loss_detail_unknown_window": (
                [self.start_ns, time.monotonic_ns()]
                if _counter_value(self.dropped) > len(losses)
                else None
            ),
            "loss_ranges": self._loss_ranges(losses),
        }

    @staticmethod
    def _loss_ranges(losses):
        ranges = []
        for _, attempt, stamp, critical, reason in sorted(losses, key=lambda x: x[1]):
            if (
                ranges
                and ranges[-1][1] + 1 == attempt
                and ranges[-1][4:] == [critical, reason]
            ):
                ranges[-1][1] = attempt
                ranges[-1][3] = stamp
            else:
                ranges.append([attempt, attempt, stamp, stamp, critical, reason])
        return ranges

    def _anchor(self):
        before = time.monotonic_ns()
        wall = time.time_ns()
        after = time.monotonic_ns()
        return self._record(
            "recorder.clock_anchor",
            {
                "wall_time_ns": wall,
                "monotonic_before_ns": before,
                "monotonic_after_ns": after,
            },
        )

    def _metrics(self, now):
        record = self._record(
            "recorder.metrics",
            {
                **self._stats(),
                "window_start_ns": self.metrics_start_ns,
                "window_end_ns": now,
                "phase_counts": self.phase_counts,
                "duration_bins": self.duration_bins,
                "duration_bin_upper_ns": _DURATION_BINS_NS,
                "metrics_scope": "accepted_events_only",
            },
        )
        self.phase_counts = {}
        self.duration_bins = {}
        self.metrics_start_ns = now
        return record

    def _write(self, lines):
        if not lines:
            return
        start = time.monotonic_ns()
        self.file.write("".join(lines))
        self.file.flush()
        self.flush_ns = time.monotonic_ns() - start

    def _warn(self, kind):
        # No exception text or metadata; failure makes the data unsuitable for joins.
        print(
            f"ATOM_DIAG_ITL {kind} pid={self.pid} "
            f"attempted={_counter_value(self.attempts)} accepted={self.accepted} "
            f"written={self.written} dropped={_counter_value(self.dropped)} "
            f"critical_dropped={_counter_value(self.critical_dropped)} "
            f"unwritten={self.accepted - self.written}",
            file=sys.stderr,
            flush=True,
        )

    def _run(self):
        self.writer_cpu_start_ns = time.thread_time_ns()
        try:
            self._write([self._record("recorder.start", self._stats()), self._anchor()])
            while True:
                self.stop.wait(_WRITE_PERIOD)
                with self.lock:
                    critical, normal = self.critical, self.normal
                    self.critical, self.normal = deque(), deque()
                # Reserved mappings/terminal records reach storage first. seq is
                # file order; attempt_id and monotonic_ns retain producer order.
                lines = []
                count = 0
                for attempt, stamp, thread, event, fields in itertools.chain(
                    critical, normal
                ):
                    fields = _safe_fields(fields)
                    fields["attempt_id"] = attempt
                    lines.append(self._record(event, fields, stamp, thread))
                    count += 1
                    phase = fields.get("phase", "instant")
                    key = f"{event}:{phase}"
                    if key not in self.phase_counts and len(self.phase_counts) >= 128:
                        key = "other"
                    self.phase_counts[key] = self.phase_counts.get(key, 0) + 1
                    duration = fields.get("duration_ns")
                    if type(duration) in (int, float) and duration >= 0:
                        bins = self.duration_bins.setdefault(key, [0] * 7)
                        index = sum(duration > bound for bound in _DURATION_BINS_NS)
                        bins[index] += 1
                self._write(lines)
                self.written += count
                # Retire metadata outside the producer lock before the next swap.
                del critical, normal
                now = time.monotonic_ns()
                if now - self.metrics_start_ns >= 1_000_000_000:
                    self._write([self._metrics(now), self._anchor()])
                if self.stop.is_set():
                    if not self.normal and not self.critical:
                        break
                    if time.monotonic() >= self.deadline:
                        break
            self._write(
                [
                    self._metrics(time.monotonic_ns()),
                    self._record("recorder.close", self._stats()),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - report failure without payload text
            self.writer_error = type(exc).__name__
            self.accepting = False
            self._warn("writer_failure")
        finally:
            try:
                self.file.close()
            except Exception as exc:  # noqa: BLE001 - never print exception payloads
                self.writer_error = type(exc).__name__
                self._warn("writer_close_failure")
            self.done.set()

    def close(self):
        self.accepting = False
        self.deadline = time.monotonic() + _CLOSE_TIMEOUT
        self.stop.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(_CLOSE_TIMEOUT)
        if not self.done.is_set():
            self.writer_error = "close_timeout"
            self._warn("close_timeout")


def configure(profiler_dir: str | None, role: str, rank: int | None = None) -> None:
    """Enable once per spawned process; disabled configuration has no side effects."""
    global _recorder
    if os.getenv("ATOM_DIAG_ITL", "0") != "1":
        return
    if not profiler_dir:
        raise ValueError("ATOM_DIAG_ITL requires a torch profiler directory")
    if not isinstance(role, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", role):
        raise ValueError("ATOM_DIAG_ITL requires a filename-safe process role")
    directory = Path(profiler_dir).resolve()
    with _config_lock:
        if _recorder is not None and _recorder.pid == os.getpid():
            if _recorder.directory != directory or _recorder.role != role:
                raise ValueError(
                    "ATOM_DIAG_ITL already configured for another role/path"
                )
            if rank is not None and _recorder.rank not in (None, rank):
                raise ValueError("ATOM_DIAG_ITL already configured for another rank")
            if rank is not None:
                _recorder.rank = rank
            return
        _recorder = _Recorder(directory, role, rank)
        atexit.register(close)


def enabled() -> bool:
    """Cheap branch: no environment lookup, allocation or lazy initialization."""
    return _recorder is not None and _recorder.accepting


def emit(event: str, *, critical: bool = False, **fields) -> None:
    recorder = _recorder
    if recorder is not None and recorder.accepting:
        recorder.emit(event, critical, fields)


@contextmanager
def span(event: str, *, critical: bool = False, **fields):
    """Record paired boundaries, preserving exceptions without recording their text."""
    recorder = _recorder
    if recorder is None or not recorder.accepting:
        yield
        return
    start = time.monotonic_ns()
    span_fields = dict(fields, span_id=f"{threading.get_ident()}-{start}")
    recorder.emit(event, critical, dict(span_fields, phase="begin"))
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        recorder.emit(
            event,
            critical,
            dict(
                span_fields,
                phase="end",
                status=status,
                duration_ns=time.monotonic_ns() - start,
            ),
        )


def close() -> None:
    """Bounded best-effort drain; timeout/failure is explicit on stderr."""
    if _recorder is not None and _recorder.pid == os.getpid():
        _recorder.close()
