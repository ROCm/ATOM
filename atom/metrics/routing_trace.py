# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in calibration evidence; never a routing input.

CPU snapshots describe scheduler state at engine boundaries. GPU samples are
whole-batch stream durations, including communication/waits, NOT additive
per-request service times. No prompt text, token IDs or GPU tensors are read.
The first version deliberately supports PP=1 only.
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

from atom.utils import envs

logger = logging.getLogger("atom")


class TraceWriter:
    """Bounded nonblocking producer; JSON serialization and disk I/O off-thread."""

    def __init__(self, directory, *, max_pending=256, max_bytes=256 * 1024**2):
        self.stream_id = uuid.uuid4().hex
        self.path = Path(directory) / f"trace-{os.getpid()}-{self.stream_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", encoding="utf-8")
        self.max_bytes = max_bytes
        self.pending = queue.Queue(maxsize=max_pending)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.attempted = self.written = self.dropped = self.bytes = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def emit(self, kind, **data):
        # Callers transfer ownership of plain CPU data to the writer.
        with self.lock:
            self.attempted += 1
            record = dict(
                schema=1,
                stream_id=self.stream_id,
                event_seq=self.attempted,
                kind=kind,
                wall_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(),
                **data,
            )
            if self.stop.is_set() or self.error is not None:
                self.dropped += 1
                return
            try:
                self.pending.put_nowait(record)
            except queue.Full:
                self.dropped += 1

    def _status(self):
        with self.lock:
            result = {
                "stream_id": self.stream_id,
                "attempted": self.attempted,
                "written": self.written,
                "dropped": self.dropped,
                "pending": self.pending.qsize(),
                "bytes": self.bytes,
                "error": self.error,
                "stopped": self.stop.is_set(),
                "wall_ns": time.time_ns(),
            }
        target = self.path.with_suffix(".status.json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(target)

    def _run(self):
        next_flush = 0.0
        try:
            while not self.stop.is_set() or not self.pending.empty():
                try:
                    record = self.pending.get(timeout=0.2)
                except queue.Empty:
                    record = None
                if record is not None:
                    line = (
                        json.dumps(record, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    )
                    size = len(line.encode("utf-8"))
                    if self.bytes + size <= self.max_bytes:
                        self.file.write(line)
                        with self.lock:
                            self.bytes += size
                            self.written += 1
                    else:
                        with self.lock:
                            self.dropped += 1
                now = time.monotonic()
                if now >= next_flush:
                    self.file.flush()
                    self._status()
                    next_flush = now + 1.0
        except Exception as exc:
            with self.lock:
                self.error = repr(exc)
            logger.exception("Routing calibration trace writer failed")
        finally:
            self.file.close()
            try:
                self._status()
            except OSError:
                logger.exception("Could not write routing trace status")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)


_writer = None
_writer_lock = threading.Lock()
_writer_failed = False


def get_writer():
    global _writer, _writer_failed
    directory = envs.ATOM_ROUTING_TRACE_DIR
    if not directory or _writer_failed:
        return None
    with _writer_lock:
        if _writer is None:
            try:
                _writer = TraceWriter(directory)
            except OSError:
                _writer_failed = True
                logger.exception("Could not initialize routing calibration trace")
        return _writer


def sequence_record(seq):
    return {
        "seq_id": int(seq.id),
        "request_id": seq.external_request_id,
        "parent_request_id": seq.parent_request_id,
        "sibling_index": int(seq.sibling_index),
        "prompt_tokens": int(seq.num_prompt_tokens),
        "num_tokens": int(seq.num_tokens),
        "scheduler_cached_tokens": int(seq.num_cached_tokens),
        "prefix_hit_tokens": int(seq.prefix_cache_hit_tokens),
        "compressed_hit_blocks": int(seq.num_compressed_hit_blocks),
        "wanted_hit_blocks": int(seq.num_wanted_hit_blocks),
        "state_slot": int(seq.state_slot),
        "state_fork_src": int(seq.state_fork_src),
        "checkpoint_demand_pos": int(seq.checkpoint_demand_pos),
        "status": seq.status.name,
        "sequence_type": seq.type.name,
        "partial": bool(seq.is_partial_prefill),
    }


def scheduler_snapshot(scheduler):
    # PP has schedule-time advancement; it needs a separate in-flight ledger.
    if getattr(scheduler, "advance_on_schedule", False):
        raise ValueError("Routing calibration trace requires pipeline_parallel_size=1")
    result = {}
    for name in ("waiting", "running"):
        seqs = getattr(scheduler, name, ())
        result[name] = [sequence_record(seq) for seq in list(seqs)[:4096]]
        result[name + "_count"] = len(seqs)
        result[name + "_truncated"] = len(seqs) > 4096
    return result


def trace_api_request(request_id, headers, data_parallel_rank):
    writer = get_writer()
    if writer is not None:
        # A fresh server UUID identifies each attempt; repeated client IDs must
        # not collapse retries, fan-out siblings or concurrent session turns.
        writer.emit(
            "api_request",
            request_id=request_id,
            client_request_id=headers.get("x-request-id"),
            session_id=headers.get("x-session-id"),
            requested_dp_rank=data_parallel_rank,
        )


def trace_enqueue(scheduler, seq):
    writer = get_writer()
    if writer is not None:
        writer.emit(
            "enqueue",
            dp_rank=scheduler.config.parallel_config.data_parallel_rank,
            request=sequence_record(seq),
        )


class EngineTrace:
    def __init__(self, config, writer):
        if config.pipeline_parallel_size != 1 or config.enable_rapidserve:
            raise ValueError("Routing trace supports PP=1 connector-based serving only")
        self.writer = writer
        self.dp_rank = config.parallel_config.data_parallel_rank
        self.step = 0
        self.last_completed_step = 0
        self.last_completed_monotonic_ns = None

    def begin(self, scheduler):
        self.step += 1
        self.writer.emit(
            "step_start",
            dp_rank=self.dp_rank,
            step=self.step,
            scheduler=scheduler_snapshot(scheduler),
        )

    def batch(self, batch, seqs):
        batch.routing_trace_id = f"{self.writer.stream_id}:{self.step}"
        rows = []
        for i, req_id in enumerate(batch.req_ids):
            row = sequence_record(seqs[req_id])
            q = int(batch.num_scheduled_tokens[i])
            cached = int(batch.num_cached_tokens[i])
            row.update(
                query_tokens=q,
                cached_before=cached,
                kv_end=cached + q,
                prefill=i >= batch.total_seqs_num_decode,
                final_chunk=(
                    bool(batch.is_final_chunk[i])
                    if batch.is_final_chunk is not None
                    else None
                ),
            )
            rows.append(row)
        self.writer.emit(
            "batch_scheduled",
            dp_rank=self.dp_rank,
            step=self.step,
            batch_id=batch.routing_trace_id,
            requests=rows,
        )

    def postprocess(self, batch, finished):
        self.writer.emit(
            "batch_postprocess",
            dp_rank=self.dp_rank,
            step=self.step,
            batch_id=batch.routing_trace_id,
            finished=[sequence_record(seq) for seq in finished],
        )

    def end(self, scheduler, executed):
        # Called only after the engine step returns normally, never from a
        # finally block: a failed forward must not advance observed progress.
        self.last_completed_step = self.step
        self.last_completed_monotonic_ns = time.monotonic_ns()
        self.writer.emit(
            "step_end",
            dp_rank=self.dp_rank,
            step=self.step,
            executed=executed,
            scheduler=scheduler_snapshot(scheduler),
        )

    def snapshot(self, scheduler, metrics):
        self.writer.emit(
            "engine_snapshot",
            dp_rank=self.dp_rank,
            last_completed_step=self.last_completed_step,
            last_completed_monotonic_ns=self.last_completed_monotonic_ns,
            scheduler=scheduler_snapshot(scheduler),
            metrics=metrics,
        )
