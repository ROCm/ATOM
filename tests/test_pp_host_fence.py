# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Protocol behavior with CPU threads and explicit device-operation substitutes."""

import gc
import queue
import threading
import unittest
import weakref
from copy import deepcopy
from math import prod
from types import SimpleNamespace

from atom.distributed.pp_host_fence.async_core import AsyncSender, Receiver
from atom.distributed.pp_host_fence.contract import (
    plan_submission,
    ready_metadata,
    validate_ready,
)
from atom.distributed.pp_host_fence.k3_edge_schema import EdgeSchema


def reject(fn, classes=(ValueError, RuntimeError, TimeoutError)):
    try:
        fn()
    except classes:
        return
    raise AssertionError("expected rejection")


class Storage:
    def __init__(self, value):
        self.value = value


class Tensor:
    def __init__(self, shape, value=None, contiguous=True, storage=None):
        self.shape = shape
        self.storage = storage or Storage(value)
        self.contiguous = contiguous

    @property
    def value(self):
        return self.storage.value


class Channel:
    def __init__(self, inbox, outbox, records):
        self.inbox, self.outbox, self.records = inbox, outbox, records

    def send(self, value):
        self.records.append((deepcopy(value), threading.get_ident()))
        self.outbox.put(deepcopy(value))

    def recv(self, timeout):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("CPU channel deadline") from exc

    def flush(self):
        pass


class Ops:
    def __init__(self, schema, ring):
        self.schema, self.ring = schema, ring
        self.caller = threading.get_ident()
        self.work_thread = None
        self.pending = []
        self.copy_pending = None
        self.clone_pending = None
        self.copy_submitted = threading.Event()
        self.fence_entered = threading.Event()
        self.clone_submitted = threading.Event()
        self.copy_gate = threading.Event()
        self.clone_gate = threading.Event()
        self.copy_gate.set()
        self.clone_gate.set()
        self.trace = []
        self.copy_failure = False

    def describe(self, tensor):
        return {
            "shape": tensor.shape,
            "dtype": "bfloat16",
            "device": "cuda",
            "bytes": prod(tensor.shape) * 2,
            "contiguous": tensor.contiguous,
        }

    def clone_contiguous(self, value):
        assert threading.get_ident() == self.caller
        result = Tensor(value.shape)
        self.pending.append(lambda: setattr(result.storage, "value", value.value))
        self.trace.append("clone-enqueued")
        return result

    def contiguous(self, value):
        self.trace.append("owned-contiguous-enqueued")
        return self.clone_contiguous(value)

    def record_producer(self):
        assert threading.get_ident() == self.caller
        event = threading.Event()
        self.pending.append(event.set)
        self.trace.append("producer-recorded")
        return event

    def overwrite_after_producer(self, tensor, value):
        self.pending.append(lambda: setattr(tensor.storage, "value", value))

    def progress_caller_stream(self):
        while self.pending:
            self.pending.pop(0)()

    def attach_sender(self, sender):
        self.sender = sender

    def drained_token(self, sender):
        assert sender is self.sender and sender.successfully_closed
        return object()

    def unbind_worker(self):
        pass

    def bind_worker(self):
        self.work_thread = threading.get_ident()
        assert self.work_thread != self.caller

    def wait_producer(self, event):
        assert threading.get_ident() == self.work_thread
        self.producer_dependency = event
        self.trace.append("producer-wait-enqueued")

    def copy_to_ring(self, slot, values, rows):
        assert threading.get_ident() == self.work_thread
        self.copy_pending = (slot, values)
        self.copy_submitted.set()
        self.trace.append("copy-submitted")

    def copy_fence(self):
        assert threading.get_ident() == self.work_thread
        self.fence_entered.set()
        if not self.producer_dependency.wait(2):
            raise TimeoutError("producer did not execute")
        if not self.copy_gate.wait(2):
            raise TimeoutError("copy did not complete")
        if self.copy_failure:
            raise RuntimeError("CPU fixture copy failure")
        if self.copy_pending is not None:
            slot, values = self.copy_pending
            self.ring[slot] = {key: value.value for key, value in values.items()}
        self.copy_pending = None
        self.trace.append("copy-completed")

    def clone_from_ring(self, slot, rows):
        result = {key: Tensor(shape) for key, shape in self.schema.fields(rows).items()}
        self.clone_pending = (slot, result)
        self.clone_submitted.set()
        self.trace.append("clones-submitted")
        return result

    def clone_fence(self):
        if not self.clone_gate.wait(2):
            raise TimeoutError("clones did not complete")
        slot, result = self.clone_pending
        for key, tensor in result.items():
            tensor.storage.value = self.ring[slot][key]
        self.clone_pending = None
        self.trace.append("clones-completed")

    def empty_fields(self, fields):
        self.trace.append("zero-row-metadata")
        return {key: Tensor(shape) for key, shape in fields.items()}


def make_pair(edge=0, schema=None):
    schema = schema or EdgeSchema(2, 2, 1, max_rows=16)
    forward, backward = queue.Queue(), queue.Queue()
    sent_records, ack_records = [], []
    sender_channel = Channel(backward, forward, sent_records)
    receiver_channel = Channel(forward, backward, ack_records)
    ring = {}
    sender_ops, receiver_ops = Ops(schema, ring), Ops(schema, ring)
    sender = AsyncSender(
        edge, schema, "schema-fixture", sender_channel, sender_ops, timeout=2
    )
    receiver = Receiver(
        edge, schema, "schema-fixture", receiver_channel, receiver_ops, timeout=2
    )
    return SimpleNamespace(
        sender=sender,
        receiver=receiver,
        tx=sender_ops,
        rx=receiver_ops,
        forward=forward,
        backward=backward,
        sent=sent_records,
        acks=ack_records,
        ring=ring,
        schema=schema,
    )


def tensors(schema, rows, sequence, noncontiguous_aux=False):
    return {
        key: Tensor(
            shape, sequence * 10 + i, not (noncontiguous_aux and key == "pp_aux")
        )
        for i, (key, shape) in enumerate(schema.fields(rows).items())
    }


def shape_contract():
    for taps in (1, 2, 3, 4):
        schema = EdgeSchema(7168, 8, taps)
        for rows in (0, 1, 17, 129, 16384):
            ts = tensors(schema, rows, 1, noncontiguous_aux=True)
            ops = Ops(schema, {})
            descriptors = {key: ops.describe(value) for key, value in ts.items()}
            result = plan_submission(schema, descriptors, owned=("pp_aux",))
            assert result["rows"] == rows
            assert result["copyBytes"] == rows * 7168 * (9 + taps) * 2
            assert result["fields"][2]["action"] == (
                "metadata-only"
                if rows == 0
                else "contiguous-copy-and-retain-source-until-producer"
            )
    for fault in (
        "missing-aux",
        "unknown-field",
        "dtype",
        "device",
        "bank-shape",
        "bytes",
        "oversized",
    ):
        schema = EdgeSchema(7168, 8, 2)
        ds = {
            key: Ops(schema, {}).describe(value)
            for key, value in tensors(schema, 1, 0).items()
        }
        if fault == "missing-aux":
            del ds["pp_aux"]
        elif fault == "unknown-field":
            ds["residual"] = ds["hidden_states"]
        elif fault in ("dtype", "device", "bytes"):
            ds["pp_aux"][fault] = "wrong"
        elif fault == "bank-shape":
            ds["block_residual"]["shape"] = (1, 7168)
        else:
            ds["hidden_states"]["shape"] = (16385, 7168)
        reject(lambda schema=schema, ds=ds: plan_submission(schema, ds))
    for field in ("edge", "sequence", "slot", "rows", "schemaId", "hostCopyComplete"):
        meta = ready_metadata(1, 1, 1, "a")
        meta[field] = (
            False
            if field == "hostCopyComplete"
            else ("wrong" if field == "schemaId" else True)
        )
        reject(lambda meta=meta: validate_ready(meta, 1, 1, "a"))


def deferred_source_lifetime():
    pair = make_pair()
    ts = tensors(pair.schema, 3, 7)
    expected = {key: value.value for key, value in ts.items()}
    aux = ts["pp_aux"]
    owned_view = Tensor(aux.shape, storage=aux.storage)
    ts["pp_aux"] = owned_view
    weak = weakref.ref(owned_view)
    pair.tx.copy_gate.clear()
    future = pair.sender.submit(ts, owned=("pp_aux",))
    assert not future.done.is_set() and pair.forward.empty()
    for key in ("hidden_states", "block_residual"):
        pair.tx.overwrite_after_producer(ts[key], -999)
    del owned_view, ts, aux
    gc.collect()
    assert weak() is not None, "owned view lost before producer/copy"
    pair.tx.progress_caller_stream()
    assert pair.tx.copy_submitted.wait(1)
    assert pair.forward.empty() and not future.done.is_set()
    # This assertion executes on the still-runnable model caller while the
    # communication worker alone is blocked at its copy fence.
    assert threading.get_ident() == pair.tx.caller != pair.tx.work_thread
    pair.tx.copy_gate.set()
    future.wait(1)
    got = pair.receiver.recv()
    assert {key: value.value for key, value in got.items()} == expected
    pair.sender.close()
    assert not pair.sender.worker.is_alive()
    gc.collect()
    assert weak() is None and not future.owners
    assert all(thread == pair.tx.work_thread for _, thread in pair.sent)


def noncontiguous_owned():
    pair = make_pair()
    ts = tensors(pair.schema, 4, 9, noncontiguous_aux=True)
    future = pair.sender.submit(ts, owned=("pp_aux",))
    assert "owned-contiguous-enqueued" in pair.tx.trace
    pair.tx.progress_caller_stream()
    future.wait(1)
    got = pair.receiver.recv()
    assert got["pp_aux"].value == 92 and got["pp_aux"].contiguous
    pair.sender.close()


def zero_and_flush_continuation():
    pair = make_pair()
    for sequence, rows in enumerate((0, 1, 4, 0, 16, 2, 0)):
        future = pair.sender.submit(
            tensors(pair.schema, rows, sequence), owned=("pp_aux",)
        )
        pair.tx.progress_caller_stream()
        future.wait(1)
        got = pair.receiver.recv()
        assert {key: value.shape for key, value in got.items()} == pair.schema.fields(
            rows
        )
        if rows:
            assert [value.value for value in got.values()] == [
                sequence * 10 + i for i in range(3)
            ]
        if sequence == 3:
            pair.sender.flush()  # later reuse must not wait for an already-drained ACK
    pair.sender.close()
    assert pair.sender.acked == pair.sender.sent == 7
    assert (
        sum(
            message["rows"] == 0
            for message, _ in pair.sent
            if message["kind"] == "READY"
        )
        == 3
    )
    assert (
        pair.tx.trace.count("copy-completed") == 7
        and pair.rx.trace.count("clones-completed") == 4
    )


def zero_producer_boundary():
    pair = make_pair()
    empty = tensors(pair.schema, 0, 0)
    owned = weakref.ref(empty["pp_aux"])
    future = pair.sender.submit(empty, owned=("pp_aux",))
    del empty
    gc.collect()
    assert pair.tx.fence_entered.wait(1)
    assert not future.done.is_set() and pair.forward.empty() and owned() is not None
    assert not pair.tx.copy_submitted.is_set(), "zero rows must not touch raw pointers"
    pair.tx.progress_caller_stream()
    future.wait(1)
    pair.receiver.recv()
    pair.sender.close()
    gc.collect()
    assert owned() is None
    # Effective negative control: removing the actual host completion boundary
    # lets READY/future completion escape while the local producer is pending.
    bad = make_pair()
    bad.tx.copy_fence = lambda: None
    escaped = bad.sender.submit(tensors(bad.schema, 0, 0), owned=("pp_aux",))
    escaped.wait(1)
    assert (
        not bad.forward.empty() and bad.tx.pending
    ), "negative control failed to expose early completion"
    bad.tx.progress_caller_stream()
    bad.receiver.recv()
    bad.sender.close()


def slot_waits_for_clone():
    pair = make_pair()
    for sequence in range(3):
        future = pair.sender.submit(
            tensors(pair.schema, 2, sequence), owned=("pp_aux",)
        )
        pair.tx.progress_caller_stream()
        future.wait(1)
    assert pair.ring[0]["hidden_states"] == 0
    pair.rx.clone_gate.clear()
    values, errors = [], []

    def receive():
        try:
            values.append(pair.receiver.recv())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=receive)
    thread.start()
    assert pair.rx.clone_submitted.wait(1)
    assert pair.backward.empty()
    last = pair.sender.submit(tensors(pair.schema, 2, 3), owned=("pp_aux",))
    pair.tx.progress_caller_stream()
    assert not last.done.wait(0.05), "reused slot before clone completion ACK"
    assert pair.ring[0]["hidden_states"] == 0
    pair.rx.clone_gate.set()
    thread.join(1)
    assert not thread.is_alive() and not errors
    assert values[0]["hidden_states"].value == 0
    last.wait(1)
    assert pair.ring[0]["hidden_states"] == 30
    for expected in (10, 20, 30):
        assert pair.receiver.recv()["hidden_states"].value == expected
    pair.sender.close()


def failure_propagation(kind):
    pair = make_pair()
    if kind == "copy":
        pair.tx.copy_failure = True
        future = pair.sender.submit(tensors(pair.schema, 2, 1), owned=("pp_aux",))
        pair.tx.progress_caller_stream()
        reject(lambda: future.wait(1))
        assert future.owners, "failed completion must preserve source owners"
        reject(pair.receiver.recv)
        reject(lambda: pair.sender.submit(tensors(pair.schema, 1, 2)))
    elif kind == "schema":
        bad = tensors(pair.schema, 2, 0)
        del bad["pp_aux"]
        reject(lambda: pair.sender.submit(bad))
        reject(pair.receiver.recv)
        assert not pair.tx.trace
    elif kind == "ready":
        bad = ready_metadata(0, 99, 1, "schema-fixture")
        pair.forward.put(bad)
        reject(pair.receiver.recv)
        future = pair.sender.submit(tensors(pair.schema, 1, 0))
        pair.tx.progress_caller_stream()
        future.wait(1)
        reject(pair.sender.flush)
    elif kind == "ack":
        future = pair.sender.submit(tensors(pair.schema, 1, 0))
        pair.tx.progress_caller_stream()
        future.wait(1)
        pair.backward.put(
            {
                "kind": "CONSUMED",
                "edge": 0,
                "sequence": 999,
                "slot": 0,
                "hostCloneComplete": True,
            }
        )
        reject(pair.sender.flush)
    pair.sender.worker.join(1)
    assert not pair.sender.worker.is_alive()
    assert pair.sender.error is not None
    assert any(message["kind"] == "ERROR" for message, _ in pair.sent + pair.acks)


class TestHostFenceProtocol(unittest.TestCase):
    def test_field_shape_and_wire_contract(self):
        shape_contract()

    def test_sources_live_until_local_copy_completion(self):
        deferred_source_lifetime()

    def test_noncontiguous_owned_aux(self):
        noncontiguous_owned()

    def test_zero_rows_and_reuse_after_flush(self):
        zero_and_flush_continuation()

    def test_zero_rows_still_wait_for_producer(self):
        zero_producer_boundary()

    def test_slot_reuse_waits_for_clone_acknowledgement(self):
        slot_waits_for_clone()

    def test_copy_failure_retains_sources(self):
        failure_propagation("copy")

    def test_bad_schema_reaches_peer(self):
        failure_propagation("schema")

    def test_stale_ready_is_fatal(self):
        failure_propagation("ready")

    def test_stale_consumed_is_fatal(self):
        failure_propagation("ack")
