# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CI-only helpers, copied into atom by prepare_minimax_replay_debug.py.

No GPU synchronization is added. Hooks are outside compiled model functions.
The gate affects only admission of the one armed synthetic target batch.
"""

import faulthandler
import functools
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

_TRACE = None
_TRACE_LOCK = threading.Lock()
_LOCAL = threading.local()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def batch_shape(batch):
    return {
        "request_ids": [int(x) for x in batch.req_ids],
        "done": [int(x) for x in batch.num_cached_tokens],
        "new": [int(x) for x in batch.num_scheduled_tokens],
    }


class ReplayGate:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.plan = None
        self.started = None
        self.released = False
        self.verified = False
        self.observed_count = -1
        self.ordered_ids = None

    def hold(self, scheduler):
        if self.released:
            return False
        if self.plan is None:
            path = self.directory / "armed.json"
            if not path.exists():
                return False
            self.plan = json.loads(path.read_text())
            self.started = time.monotonic()
        lengths = self.plan["lengths"]
        waiting = list(scheduler.waiting)
        if scheduler.running:
            raise RuntimeError("Replay gate requires no running request when armed")
        observed = [int(seq.num_prompt_tokens) for seq in waiting]
        if len(set(observed)) != len(observed) or any(
            x not in lengths for x in observed
        ):
            raise RuntimeError(f"Unexpected requests at replay gate: {observed}")
        if len(waiting) != self.observed_count:
            self.observed_count = len(waiting)
            atomic_json(
                self.directory / "gate-state.json",
                {
                    "state": "waiting",
                    "observed_lengths": observed,
                    "expected_lengths": lengths,
                },
            )
        if len(waiting) < len(lengths):
            if time.monotonic() - self.started > 30:
                raise TimeoutError(f"Replay gate missing requests: {observed}")
            return True
        by_length = {int(seq.num_prompt_tokens): seq for seq in waiting}
        ordered = [by_length[length] for length in lengths]
        scheduler.waiting = deque(ordered)
        self.ordered_ids = [int(seq.id) for seq in ordered]
        self.released = True
        atomic_json(
            self.directory / "gate-state.json",
            {
                "state": "released",
                "ordered_lengths": lengths,
                "request_ids": self.ordered_ids,
                "unix_time": time.time(),
            },
        )
        return False

    def verify(self, batch):
        if not self.released or self.verified:
            return
        observed = batch_shape(batch)
        expected = {
            "request_ids": self.ordered_ids,
            "done": self.plan["done"],
            "new": self.plan["new"],
        }
        matched = observed == expected
        atomic_json(
            self.directory / "gate-batch.json",
            {
                "matched": matched,
                "expected": expected,
                "observed": observed,
                "total_kv": sum(observed["done"]) + sum(observed["new"]),
            },
        )
        if not matched:
            raise RuntimeError(
                f"Replay batch shape mismatch before GPU dispatch: {observed}; wanted {expected}"
            )
        self.verified = True


def hold_for_replay(core):
    if os.environ.get("ATOM_M3_REPLAY_GATE") != "1":
        return False
    gate = getattr(core, "_m3_replay_gate", None)
    if gate is None:
        gate = core._m3_replay_gate = ReplayGate(os.environ["ATOM_M3_REPLAY_DIR"])
    held = gate.hold(core.scheduler)
    if held:
        # Preserve the usual idle connector progress; do not stall the I/O thread.
        core._advance_idle_kv_transfer()
        time.sleep(0.001)
    return held


def verify_batch(core, batch):
    gate = getattr(core, "_m3_replay_gate", None)
    if gate is not None:
        gate.verify(batch)
    mark("core_dispatch", batch=batch)


def active():
    directory = os.environ.get("ATOM_M3_REPLAY_DIR")
    return bool(directory and (Path(directory) / "armed.json").exists())


class Trace:
    def __init__(self):
        self.root = Path(os.environ["ATOM_M3_REPLAY_DIR"])
        self.role = os.environ.get("ATOM_M3_REPLAY_ROLE", "unknown")
        folder = self.root / self.role
        folder.mkdir(parents=True, exist_ok=True)
        self.prefix = folder / f"pid-{os.getpid()}"
        self.fd = os.open(
            str(self.prefix) + ".jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644
        )
        self.stack_file = open(str(self.prefix) + ".stacks.txt", "a")  # noqa: SIM115 - retained for the native watchdog lifetime
        self.lock = threading.Lock()
        self.pending = []
        self.next_event = 0
        self.rank = None
        # Native faulthandler watchdog also works if a GPU query blocks this thread.
        faulthandler.dump_traceback_later(35, repeat=True, file=self.stack_file)
        threading.Thread(
            target=self.watch, name="m3-replay-events", daemon=True
        ).start()

    def emit(self, kind, **fields):
        row = {
            "unix_time": time.time(),
            "pid": os.getpid(),
            "role": self.role,
            "rank": self.rank,
            "thread": threading.get_ident(),
            "kind": kind,
            **fields,
        }
        os.write(self.fd, (json.dumps(row) + "\n").encode())

    def watch(self):
        try:
            while not (self.root / "client-done.json").exists():
                self.poll()
                time.sleep(1)
            self.poll()
        except Exception as exc:  # noqa: BLE001 - report diagnostic failures
            self.emit("watch_error", error=repr(exc))
        finally:
            faulthandler.cancel_dump_traceback_later()
            self.emit("watch_stopped")

    def poll(self):
        with self.lock:
            pending = list(self.pending)
        complete = set()
        for number, stage, event, recorded in pending:
            if event.query():
                complete.add(number)
                self.emit("gpu_complete", event=number, stage=stage)
            elif time.monotonic() - recorded >= 30:
                self.emit(
                    "gpu_pending_30s",
                    event=number,
                    stage=stage,
                    age_seconds=time.monotonic() - recorded,
                )
        with self.lock:
            self.pending = [item for item in self.pending if item[0] not in complete]
        self.emit("heartbeat", pending_events=len(pending) - len(complete))


def mark(stage, *, batch=None, gpu=False, stream=None, rank=None, **fields):
    global _TRACE
    if not active():
        return
    if _TRACE is None:
        with _TRACE_LOCK:
            if _TRACE is None:
                _TRACE = Trace()
    trace = _TRACE
    if rank is not None:
        trace.rank = int(rank)
    if batch is None:
        batch = getattr(_LOCAL, "batch", None)
    if batch is not None:
        fields["batch"] = batch_shape(batch)
    # Persist CPU arrival before entering any HIP/allocator API.
    trace.emit("cpu", stage=stage, **fields)
    if not gpu:
        return
    import torch

    # No event insertion inside CUDA graph capture or compiled operator bodies.
    if torch.cuda.is_current_stream_capturing():
        trace.emit("capture_skipped", stage=stage)
        return
    with trace.lock:
        if len(trace.pending) >= 2048:
            trace.emit("event_budget_exceeded", stage=stage)
            return
        trace.next_event += 1
        number = trace.next_event
    event = torch.cuda.Event(enable_timing=False)
    event.record(stream)
    with trace.lock:
        trace.pending.append((number, stage, event, time.monotonic()))
    trace.emit(
        "gpu_recorded",
        stage=stage,
        event=number,
        device=torch.cuda.current_device(),
        stream=int((stream or torch.cuda.current_stream()).cuda_stream),
    )
    # Read allocator counters only at model stage boundaries, not every copy group.
    if stream is None:
        stats = torch.cuda.memory_stats()
        trace.emit(
            "allocator",
            stage=stage,
            allocated=stats.get("allocated_bytes.all.current"),
            reserved=stats.get("reserved_bytes.all.current"),
            retries=stats.get("num_alloc_retries"),
            oom=stats.get("num_ooms"),
        )


def traced(stage):
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if not active():
                return function(*args, **kwargs)
            previous = getattr(_LOCAL, "batch", None)
            candidates = list(args) + list(kwargs.values())
            batch = next(
                (
                    arg
                    for arg in candidates
                    if hasattr(arg, "num_scheduled_tokens") and hasattr(arg, "req_ids")
                ),
                previous,
            )
            _LOCAL.batch = batch
            rank = getattr(args[0], "rank", None) if args else None
            mark(stage + ":enter", gpu=True, rank=rank)
            try:
                result = function(*args, **kwargs)
                mark(stage + ":return", gpu=True, rank=rank)
                return result
            except Exception as exc:
                mark(stage + ":error", error=repr(exc))
                raise
            finally:
                _LOCAL.batch = previous

        return wrapper

    return decorate
