# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Actual backend classes with CPU substitutes for HIP, streams and Gloo."""

import gc
import pickle
import queue
import threading
import unittest
import weakref
from collections import deque
from math import prod
from types import SimpleNamespace as NS

from atom.distributed.pp_host_fence.async_core import AsyncSender
from atom.distributed.pp_host_fence.gloo_channel import GlooChannel
from atom.distributed.pp_host_fence.k3_edge_schema import EdgeSchema
from atom.distributed.pp_host_fence.tensor_backend import (
    _PROCESS_RINGS,
    _PROCESS_SEND_OPS,
    RawRing,
    ReceiveOps,
    SendOps,
)


def reject(fn):
    try:
        fn()
    except (ValueError, TypeError, RuntimeError, TimeoutError, AssertionError):
        return
    raise AssertionError("expected rejection")


class Device:
    type = "cuda"
    index = 0

    def __str__(self):
        return "cuda:0"

    def __eq__(self, other):
        return isinstance(other, Device) and self.index == other.index


class Tensor:
    next_pointer = 0x800000

    def __init__(self, shape, dtype="bf16", device=None, pointer=None, trace=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device or Device()
        self.trace = trace if trace is not None else []
        self.pointer = pointer or Tensor.next_pointer
        Tensor.next_pointer += 0x100000
        self.contiguous_flag = True

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return 2 if self.dtype == "bf16" else 1

    def is_contiguous(self):
        return self.contiguous_flag

    def data_ptr(self):
        return self.pointer

    def clone(self, memory_format):
        self.trace.append(("clone", self.pointer, tuple(self.shape), memory_format))
        return Tensor(self.shape, self.dtype, self.device, trace=self.trace)

    def contiguous(self):
        self.trace.append(("contiguous", self.pointer))
        return self.clone("contiguous")

    def record_stream(self, stream):
        self.trace.append(("record-stream", self.pointer, stream.cuda_stream))

    def __getitem__(self, sl):
        assert isinstance(sl, slice) and 0 <= sl.start <= sl.stop <= self.numel()
        return Tensor(
            (sl.stop - sl.start,),
            self.dtype,
            self.device,
            self.pointer + sl.start,
            self.trace,
        )

    def view(self, spec):
        if isinstance(spec, str):
            assert self.numel() % 2 == 0
            return Tensor(
                (self.numel() // 2,), spec, self.device, self.pointer, self.trace
            )
        assert prod(spec) == self.numel()
        return Tensor(spec, self.dtype, self.device, self.pointer, self.trace)


class Stream:
    def __init__(self, cuda, device=None, priority=0):
        self.cuda, self.cuda_stream = cuda, 0x3300 + priority
        self.fail_sync = False

    def wait_event(self, event):
        self.cuda.trace.append(("wait-event", event, threading.get_ident()))

    def synchronize(self):
        self.cuda.trace.append(("stream-sync", self.cuda_stream, threading.get_ident()))
        if self.fail_sync:
            raise RuntimeError("uncertain GPU completion fixture")


class Cuda:
    def __init__(self, trace):
        self.trace, self.local = trace, threading.local()
        self.caller_stream = Stream(self)
        self.fail_sync = False

    def current_device(self):
        return getattr(self.local, "device", 0)

    def set_device(self, device):
        self.local.device = device.index

    def current_stream(self, device):
        return self.caller_stream

    def Stream(self, device, priority):
        self.trace.append(("new-stream", device.index, priority, threading.get_ident()))
        return Stream(self, device, priority)

    def synchronize(self, device):
        self.trace.append(("device-sync", device.index))
        if self.fail_sync:
            raise RuntimeError("ring initialization failed")

    def Event(self, enable_timing):
        assert enable_timing is False
        outer = self

        class Event:
            def record(self, stream):
                outer.trace.append(("producer-record", stream.cuda_stream))

        return Event()


class Torch:
    bfloat16 = "bf16"
    contiguous_format = "contiguous"

    def __init__(self):
        self.trace = []
        self.cuda = Cuda(self.trace)

    def is_tensor(self, value):
        return isinstance(value, Tensor)

    def as_tensor(self, external, device):
        interface = external.__cuda_array_interface__
        self.trace.append(("external-view", interface))
        return Tensor(
            interface["shape"], "uint8", device, interface["data"][0], self.trace
        )

    def empty(self, shape, device, dtype):
        return Tensor(shape, dtype, device, trace=self.trace)

    def inference_mode(self):
        outer = self

        class Inference:
            def __enter__(self):
                outer.trace.append(("inference-enter", threading.get_ident()))

            def __exit__(self, *args):
                outer.trace.append(("inference-exit", threading.get_ident()))

        return Inference()


class HipCopy:
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, dst, src, size, kind, stream):
        self.runtime.copies.append(
            tuple(
                x.value if hasattr(x, "value") else x
                for x in (dst, src, size, kind, stream)
            )
        )
        return 99 if len(self.runtime.copies) == self.runtime.fail_copy else 0


class Runtime:
    def __init__(self):
        self.calls, self.copies = [], []
        self.fail_copy = -1
        self.hip = NS(hipMemcpyAsync=HipCopy(self))

    def allocate_uncached(self, size, flags):
        self.calls.append(("allocate", size, flags))
        return 0x100000

    def get_allocation_base(self, pointer):
        return pointer

    def get_ipc_handle(self, pointer):
        return bytes(64)

    def open_ipc_handle(self, handle):
        self.calls.append(("open", handle))
        return 0x200000

    def library(self):
        return self.hip

    def check(self, error, operation):
        if error:
            raise RuntimeError(operation)

    def close_ipc_handle(self, base):
        self.calls.append(("close", base))


class DeviceBytesFixture:
    def __init__(self, pointer, size):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }


def fixture(aux=1):
    torch, device, runtime, schema = (
        Torch(),
        Device(),
        Runtime(),
        EdgeSchema(2, 2, aux, max_rows=16),
    )
    ring = RawRing(torch, device, schema, runtime, DeviceBytesFixture)
    publication = ring.publish(0, "fixture")
    ops = SendOps(torch, device, schema, runtime, publication, 0, "fixture")
    return NS(
        torch=torch,
        device=device,
        runtime=runtime,
        schema=schema,
        ring=ring,
        ops=ops,
        publication=publication,
    )


def ring_publication_lifetime():
    f = fixture()
    assert f.runtime.calls[0] == ("allocate", f.schema.metadata()["ringBytes"], 0)
    assert [x[0] for x in f.torch.trace][:2] == ["external-view", "device-sync"]
    assert f.ring in _PROCESS_RINGS and f.ops in _PROCESS_SEND_OPS
    ring_ref, ops_ref = weakref.ref(f.ring), weakref.ref(f.ops)
    del f
    gc.collect()
    assert ring_ref() is not None and ops_ref() is not None
    torch, runtime = Torch(), Runtime()
    torch.cuda.fail_sync = True
    before = len(_PROCESS_RINGS)
    reject(
        lambda: RawRing(
            torch,
            Device(),
            EdgeSchema(2, 2, 1, max_rows=16),
            runtime,
            DeviceBytesFixture,
        )
    )
    assert len(_PROCESS_RINGS) == before + 1 and not _PROCESS_RINGS[-1].ready
    reject(lambda: _PROCESS_RINGS[-1].publish(0, "fixture"))
    assert not any(x[0] == "close" for x in runtime.calls)


def publication_rejection():
    f = fixture()
    for key, bad in (
        ("edge", True),
        ("ringBytes", 1),
        ("offset", 2),
        ("offset", -1),
        ("handle", bytes(63)),
        ("handle", "x" * 64),
        ("schemaId", "other"),
    ):
        p = dict(f.publication, **{key: bad})
        before = len(f.runtime.calls)
        reject(
            lambda p=p: SendOps(f.torch, f.device, f.schema, f.runtime, p, 0, "fixture")
        )
        assert len(f.runtime.calls) == before
    reject(lambda: f.ops.close_mapping(None))


def caller_and_copy():
    for aux in (1, 2, 3, 4):
        f = fixture(aux)
        tensors = {
            k: Tensor(shape, trace=f.torch.trace)
            for k, shape in f.schema.fields(7).items()
        }
        assert f.ops.describe(tensors["pp_aux"])["shape"] == (7, aux * 2)
        f.ops.clone_contiguous(tensors["hidden_states"])
        f.ops.contiguous(tensors["pp_aux"])
        event = f.ops.record_producer()
        errors = []

        def worker(f=f, event=event, tensors=tensors, errors=errors):
            try:
                reject(f.ops.record_producer)
                f.ops.bind_worker()
                f.ops.wait_producer(event)
                f.ops.copy_to_ring(2, tensors, 7)
                f.ops.copy_fence()
                f.ops.unbind_worker()
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        t = threading.Thread(target=worker)
        t.start()
        t.join(2)
        assert not t.is_alive() and not errors
        meta = f.schema.metadata()
        assert f.runtime.copies == [
            (
                f.ops.remote + 2 * meta["slotBytes"] + field["offset"],
                tensors[field["key"]].data_ptr(),
                tensors[field["key"]].numel() * 2,
                3,
                0x32FF,
            )
            for field in meta["fields"]
        ]
        names = [x[0] for x in f.torch.trace]
        assert (
            names.index("producer-record")
            < names.index("wait-event")
            < names.index("stream-sync")
            < names.index("inference-exit")
        )
        reject(lambda f=f, tensors=tensors: f.ops.copy_to_ring(0, tensors, 7))
        # Worker exit alone never authorizes unmap, even on a success-shaped fixture.
        reject(lambda f=f: f.ops.close_mapping(object()))


def receiver_offsets():
    f = fixture(4)
    ops = ReceiveOps(f.torch, f.device, f.ring)
    output = ops.clone_from_ring(2, 11)
    assert {k: v.shape for k, v in output.items()} == f.schema.fields(11)
    clones = [x for x in f.torch.trace if x[0] == "clone"]
    meta = f.schema.metadata()
    assert [x[1] for x in clones] == [
        f.ring.pointer + 2 * meta["slotBytes"] + x["offset"] for x in meta["fields"]
    ]
    ops.clone_fence()
    assert f.torch.trace[-1][0] == "stream-sync"
    assert all(v.numel() == 0 for v in ops.empty_fields(f.schema.fields(0)).values())


class Channel:
    def __init__(self, inbox, outbox):
        self.inbox, self.outbox = inbox, outbox

    def send(self, obj):
        self.outbox.put(obj)

    def recv(self, timeout):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("fixture peer timeout") from error

    def flush(self):
        pass


def actual_sender_lifecycle(fail=None):
    f = fixture()
    forward, backward = queue.Queue(), queue.Queue()
    sender = AsyncSender(
        0, f.schema, "fixture", Channel(backward, forward), f.ops, timeout=1
    )
    if fail == "copy":
        f.runtime.fail_copy = 2
    future = sender.submit(
        {k: Tensor(s, trace=f.torch.trace) for k, s in f.schema.fields(4).items()},
        owned=("pp_aux",),
    )
    if fail == "copy":
        reject(lambda: future.wait(1))
        sender.worker.join(2)
        assert len(f.runtime.copies) == 2 and sender.failed_jobs and future.owners
        assert f.ops.inference is None and not sender.worker.is_alive()
        reject(sender.close)
        reject(lambda: f.ops.drained_token(sender))
        reject(lambda f=f: f.ops.close_mapping(object()))
    else:
        future.wait(1)
        ready = forward.get(timeout=1)
        assert ready["hostCopyComplete"] is True
        if fail == "peer":
            backward.put({"kind": "ERROR", "error": "receiver died"})
            reject(sender.close)
            sender.worker.join(2)
            assert f.ops.inference is None
            reject(lambda: f.ops.drained_token(sender))
            reject(lambda f=f: f.ops.close_mapping(object()))
        else:
            backward.put(
                {
                    "kind": "CONSUMED",
                    "edge": 0,
                    "sequence": 0,
                    "slot": 0,
                    "hostCloneComplete": True,
                }
            )
            token = sender.close()
            assert (
                sender.successfully_closed
                and sender.acked == sender.sent == sender.submitted == 1
            )
            reject(lambda f=f: f.ops.close_mapping(object()))
            f.ops.close_mapping(token)
            f.ops.close_mapping(token)
            assert [x[0] for x in f.runtime.calls].count("close") == 1
    if fail:
        assert not any(x[0] == "close" for x in f.runtime.calls)
        ref = weakref.ref(sender)
        del sender
        gc.collect()
        assert ref() is f.ops.sender and f.ring in _PROCESS_RINGS


class CPUValue:
    def __init__(self, value):
        self.value, self.device = value, NS(type="cpu")

    def item(self):
        return self.value[0]

    def numpy(self):
        return self

    def tobytes(self):
        return bytes(self.value)


class Work:
    def __init__(self, action=None):
        self.action, self.failed, self.done = action, False, False

    def wait(self, timeout):
        assert timeout.total_seconds() > 0
        if self.failed:
            return False
        if self.action:
            self.action()
            self.action = None
        self.done = True
        return True

    def is_completed(self):
        return self.done


class CPUTransport:
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing
        self.work, self.count, self.fail_send, self.fail_recv = [], 0, -1, False

    def isend(self, tensor, dst, group):
        self.count += 1
        if self.count == self.fail_send:
            raise RuntimeError("post failure")
        self.outgoing.append(list(tensor.value))
        work = Work()
        self.work.append(work)
        return work

    def irecv(self, tensor, src, group):
        work = Work(lambda: setattr(tensor, "value", self.incoming.popleft()))
        work.failed = self.fail_recv
        return work


class CPUTorch:
    uint8, long = "uint8", "long"

    def __init__(self, incoming, outgoing):
        self.distributed = CPUTransport(incoming, outgoing)

    def frombuffer(self, value, dtype):
        return CPUValue(list(value))

    def tensor(self, value, dtype, device):
        assert device == "cpu"
        return CPUValue(value)

    def empty(self, count, dtype, device):
        assert device == "cpu"
        return CPUValue([0] * count)


def gloo_fixture():
    a, b = deque(), deque()
    left, right = CPUTorch(a, b), CPUTorch(b, a)
    return left, right, GlooChannel(left, "edge0", 1), GlooChannel(right, "edge0", 0)


def gloo_roundtrip_and_pending():
    _lt, _rt, left, right = gloo_fixture()
    value = {"kind": "READY", "handle": bytes(64)}
    left.send(value)
    assert len(left.pending) == 2 and right.recv(1) == value
    refs = [weakref.ref(t) for _, t in left.pending]
    gc.collect()
    assert all(r() is not None for r in refs)
    left.pending[0][0].failed = True
    reject(left.flush)
    assert len(left.pending) == 2 and all(r() is not None for r in refs)
    left.pending[0][0].failed = False
    left.flush()
    gc.collect()
    assert len(left.pending) == 0 and all(r() is None for r in refs)
    right.send({"kind": "CONSUMED"})
    assert left.recv(1) == {"kind": "CONSUMED"}
    right.flush()
    errors = []

    def other():
        try:
            left.send({})
        except RuntimeError:
            errors.append(True)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join(1)
    assert errors == [True]


def gloo_failures_and_bounds():
    lt, rt, left, right = gloo_fixture()
    lt.distributed.fail_send = 2
    reject(lambda: left.send({"x": 1}))
    assert len(left.pending) == 1 and len(left.failed_sends) == 1
    rt.distributed.fail_recv = True
    reject(lambda: right.recv(1))
    assert len(right.failed_receives) == 1
    lt, rt, left, right = gloo_fixture()
    left.max_bytes = 3
    reject(lambda: left.send({"x": 1}))
    assert not lt.distributed.outgoing
    rt.distributed.incoming.append([2**21])
    reject(lambda: right.recv(1))
    payload = pickle.dumps([1, 2])
    rt.distributed.incoming.extend([[len(payload)], list(payload)])
    reject(lambda: right.recv(1))


class TestHostFenceBackends(unittest.TestCase):
    def test_ring_publication_follows_initialization(self):
        ring_publication_lifetime()

    def test_invalid_handles_rejected_before_open(self):
        publication_rejection()

    def test_copy_addresses_and_local_stream_completion(self):
        caller_and_copy()

    def test_receiver_clones_before_slot_consumption(self):
        receiver_offsets()

    def test_close_requires_ack_drain_and_worker_join(self):
        actual_sender_lifecycle()

    def test_failed_copy_never_unmaps(self):
        actual_sender_lifecycle("copy")

    def test_peer_error_never_unmaps(self):
        actual_sender_lifecycle("peer")

    def test_control_buffers_live_through_work_completion(self):
        gloo_roundtrip_and_pending()

    def test_partial_control_failure_retains_buffers(self):
        gloo_failures_and_bounds()
