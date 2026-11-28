# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""PP IPC sender ownership and receiver consumption protocol.

This file does not import torch or create a process group. TensorOps must bind
the caller and worker CUDA streams, and Channel must own its Gloo work buffers.
Only the outgoing worker uses its outgoing channel; receiver uses another edge.
"""

import logging
import queue
import threading
from dataclasses import dataclass

from .contract import plan_submission, ready_metadata, validate_ready

logger = logging.getLogger(__name__)


@dataclass
class Prepared:
    buffers: dict
    owners: tuple
    producer: object
    plan: dict


def prepare_submission(schema, tensors, owned, ops):
    plan = plan_submission(
        schema, {key: ops.describe(value) for key, value in tensors.items()}, owned
    )
    buffers, extra_owners = {}, []
    for field in plan["fields"]:
        key, action = field["key"], field["action"]
        value = tensors[key]
        if action == "clone-contiguous-on-caller-stream":
            buffers[key] = ops.clone_contiguous(value)
        elif action == "contiguous-copy-and-retain-source-until-producer":
            buffers[key] = ops.contiguous(value)
            extra_owners.append(value)
        else:
            buffers[key] = value
    producer = ops.record_producer()
    return Prepared(
        buffers, tuple(buffers.values()) + tuple(extra_owners), producer, plan
    )


class SendFuture:
    """Duck-types P2PWork.work.wait(); retains all sources through copy fence."""

    def __init__(self, owners=()):
        self.owners = owners
        self.error = None
        self.done = threading.Event()

    def complete(self):
        self.owners = ()
        self.done.set()

    def fail(self, error):
        # Failed GPU completion is uncertain: retain sources for stage teardown.
        self.error = error
        self.done.set()

    def wait(self, timeout=60):
        if not self.done.wait(timeout):
            raise TimeoutError("owned IPC sender has not completed")
        if self.error is not None:
            raise self.error
        return True


@dataclass
class Job:
    sequence: int
    prepared: Prepared
    future: SendFuture


@dataclass
class Drain:
    future: SendFuture


@dataclass
class Abort:
    error: Exception


class AsyncSender:
    def __init__(self, edge, schema, schema_id, channel, ops, timeout=60):
        ready_metadata(edge, 0, 0, schema_id)
        self.edge, self.schema, self.schema_id = edge, schema, schema_id
        self.channel, self.ops, self.timeout = channel, ops, timeout
        self.caller = threading.get_ident()
        self.state_lock = threading.Lock()
        self.jobs = queue.Queue()
        self.sent = self.acked = self.submitted = 0
        self.error = None
        self.failed_jobs = []
        self.closed = False
        self.successfully_closed = False
        self.worker_ready = threading.Event()
        self.ops.attach_sender(self)
        self.worker = threading.Thread(
            target=self._run, name=f"pp-ipc-send-edge{edge}", daemon=True
        )
        self.worker.start()
        if not self.worker_ready.wait(self.timeout):
            raise TimeoutError("sender worker did not initialize")
        if self.error is not None:
            raise self.error

    def _assert_caller(self):
        if threading.get_ident() != self.caller or self.closed:
            raise RuntimeError("sender submissions belong to the live model caller")
        if self.error is not None:
            raise self.error

    def submit(self, tensors, owned=()):
        with self.state_lock:
            self._assert_caller()
            # The native caller commits the previous future before the next send.
            # Refuse unexpected queue growth rather than hiding a main-thread wait.
            if self.submitted != self.sent:
                raise RuntimeError("native one-pending-send contract exceeded")
            try:
                prepared = prepare_submission(self.schema, tensors, owned, self.ops)
            except Exception as exc:
                self.jobs.put(Abort(exc))  # only the worker publishes outgoing ERROR
                raise
            future = SendFuture(prepared.owners)
            self.jobs.put(Job(self.submitted, prepared, future))
            self.submitted += 1
            return future

    def flush(self):
        with self.state_lock:
            self._assert_caller()
            future = SendFuture()
            self.jobs.put(Drain(future))
        future.wait(self.timeout)

    def close(self):
        self._assert_caller()
        self.flush()
        self.closed = True
        self.jobs.put(None)
        self.worker.join(self.timeout)
        if self.worker.is_alive():
            raise TimeoutError("sender worker did not exit")
        if self.error is not None:
            raise self.error
        if self.sent != self.submitted or self.acked != self.sent:
            raise RuntimeError("sender close is missing completion or consumption")
        self.successfully_closed = True
        return self.ops.drained_token(self)

    def _ack(self):
        meta = self.channel.recv(self.timeout)
        if meta.get("kind") == "ERROR":
            raise RuntimeError("peer IPC edge failed: " + str(meta.get("error")))
        expected = {
            "kind": "CONSUMED",
            "edge": self.edge,
            "sequence": self.acked,
            "slot": self.acked % 3,
            "hostCloneComplete": True,
        }
        if (
            meta != expected
            or any(type(meta[k]) is not int for k in ("edge", "sequence", "slot"))
            or meta["hostCloneComplete"] is not True
        ):
            raise ValueError("stale or incomplete IPC CONSUMED")
        self.acked += 1

    def _run(self):
        job = None
        try:
            self.ops.bind_worker()
            self.worker_ready.set()
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                if isinstance(job, Abort):
                    raise job.error
                if isinstance(job, Drain):
                    while self.acked < self.sent:
                        self._ack()
                    self.channel.flush()
                    job.future.complete()
                    job = None
                    continue
                if job.sequence != self.sent:
                    raise ValueError("sender sequence changed")
                if self.sent >= 3 and self.acked <= self.sent - 3:
                    self._ack()
                self.ops.wait_producer(job.prepared.producer)
                if job.prepared.plan["rows"]:
                    self.ops.copy_to_ring(
                        self.sent % 3, job.prepared.buffers, job.prepared.plan["rows"]
                    )
                # wait_producer may only enqueue a local stream dependency.
                # Even zero-byte views can keep upstream storage alive: retain
                # their owners until this worker observes real completion.
                self.ops.copy_fence()
                self.channel.send(
                    ready_metadata(
                        self.edge, self.sent, job.prepared.plan["rows"], self.schema_id
                    )
                )
                with self.state_lock:
                    self.sent += 1
                job.future.complete()
                job = None
        except Exception as exc:
            logger.exception("PP IPC outgoing worker failed")
            with self.state_lock:
                self.error = exc
            self.worker_ready.set()
            if isinstance(job, (Job, Drain)):
                job.future.fail(exc)
                self.failed_jobs.append(job)
            try:
                self.channel.send(
                    {
                        "kind": "ERROR",
                        "edge": self.edge,
                        "sequence": self.sent,
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                )
                self.channel.flush()
            except Exception:
                logger.exception("PP IPC error notification failed")
            while True:
                try:
                    pending = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if isinstance(pending, (Job, Drain)):
                    pending.future.fail(exc)
                    self.failed_jobs.append(pending)
        finally:
            try:
                self.ops.unbind_worker()
            except Exception as exc:
                logger.exception("PP IPC worker cleanup failed")
                with self.state_lock:
                    self.error = self.error or exc


class Receiver:
    def __init__(self, edge, schema, schema_id, channel, ops, timeout=60):
        ready_metadata(edge, 0, 0, schema_id)
        self.edge, self.schema, self.schema_id = edge, schema, schema_id
        self.channel, self.ops, self.timeout = channel, ops, timeout
        self.received = 0
        self.error = None

    def recv(self):
        if self.error is not None:
            raise self.error
        try:
            meta = self.channel.recv(self.timeout)
            if meta.get("kind") == "ERROR":
                raise RuntimeError("peer IPC edge failed: " + str(meta.get("error")))
            rows = validate_ready(meta, self.edge, self.received, self.schema_id)
            if rows:
                tensors = self.ops.clone_from_ring(self.received % 3, rows)
                self.ops.clone_fence()
            else:
                tensors = self.ops.empty_fields(self.schema.fields(0))
            # Validate logical outputs too; no raw metadata can authorize a
            # mismatched tensor return to the real model.
            plan_submission(
                self.schema,
                {key: self.ops.describe(tensor) for key, tensor in tensors.items()},
            )
            self.channel.send(
                {
                    "kind": "CONSUMED",
                    "edge": self.edge,
                    "sequence": self.received,
                    "slot": self.received % 3,
                    "hostCloneComplete": True,
                }
            )
            self.received += 1
            return tensors
        except Exception as exc:
            self.error = exc
            try:
                self.channel.send(
                    {
                        "kind": "ERROR",
                        "edge": self.edge,
                        "sequence": self.received,
                        "error": type(exc).__name__ + ": " + str(exc),
                    }
                )
                self.channel.flush()
            except Exception:
                logger.exception("PP IPC error notification failed")
            raise

    def flush(self):
        self.channel.flush()
