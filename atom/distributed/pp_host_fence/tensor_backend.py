# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Native HIP memory + local Torch stream operations for the PP transport.

Torch/runtime are injected after exact source/runtime validation. No GPU action
or import occurs merely by reading/importing this module. No IPC event exists.
"""

import ctypes
import threading

# Explicit process-lifetime owners, including partially initialized/failing
# objects. No atexit hook or destructor frees a ring or closes a peer mapping.
_PROCESS_RINGS = []
_PROCESS_SEND_OPS = []


class TensorOps:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.caller = threading.get_ident()

    def _caller_stream(self):
        if (
            threading.get_ident() != self.caller
            or self.torch.cuda.current_device() != self.device.index
        ):
            raise RuntimeError("unexpected model caller thread or active CUDA device")
        return self.torch.cuda.current_stream(self.device)

    def describe(self, tensor):
        torch = self.torch
        if not torch.is_tensor(tensor):
            raise ValueError("K3 IPC fields must be real tensors")
        return {
            "shape": tuple(tensor.shape),
            "dtype": (
                "bfloat16" if tensor.dtype == torch.bfloat16 else str(tensor.dtype)
            ),
            "device": "cuda" if tensor.device == self.device else str(tensor.device),
            "bytes": tensor.numel() * tensor.element_size(),
            "contiguous": tensor.is_contiguous(),
        }

    def clone_contiguous(self, tensor):
        self._caller_stream()
        return tensor.clone(memory_format=self.torch.contiguous_format)

    def contiguous(self, tensor):
        self._caller_stream()
        return tensor.contiguous()

    def record_producer(self):
        stream = self._caller_stream()
        event = self.torch.cuda.Event(enable_timing=False)
        event.record(stream)
        return event


class RawRing:
    """Owner allocation lives until process exit, matching native helper policy."""

    def __init__(self, torch, device, schema, runtime, device_bytes_type):
        self.torch, self.device, self.schema, self.runtime = (
            torch,
            device,
            schema,
            runtime,
        )
        self.metadata = schema.metadata()
        self.ready = False
        if torch.cuda.current_device() != device.index:
            raise RuntimeError("raw ring must allocate on the actual local device")
        _PROCESS_RINGS.append(self)
        self.pointer = runtime.allocate_uncached(self.metadata["ringBytes"], 0)
        self.buffer = torch.as_tensor(
            device_bytes_type(self.pointer, self.metadata["ringBytes"]), device=device
        )
        # Native allocate_uncached uses hipMemset without an explicit stream.
        torch.cuda.synchronize(device)
        self.ready = True

    def publish(self, edge, schema_id):
        if not self.ready:
            raise RuntimeError("ring initialization not complete")
        base = self.runtime.get_allocation_base(self.pointer)
        if base != self.pointer:
            raise RuntimeError("fresh raw ring must be the whole HIP allocation")
        return {
            "edge": edge,
            "schemaId": schema_id,
            "ringBytes": self.metadata["ringBytes"],
            "offset": self.pointer - base,
            "handle": self.runtime.get_ipc_handle(base),
        }


class SendOps(TensorOps):
    def __init__(self, torch, device, schema, runtime, published, edge, schema_id):
        super().__init__(torch, device)
        self.schema, self.metadata, self.runtime = schema, schema.metadata(), runtime
        self._caller_stream()
        if (
            set(published) != {"edge", "schemaId", "ringBytes", "offset", "handle"}
            or published["edge"] != edge
            or published["schemaId"] != schema_id
            or type(published["edge"]) is not int
            or type(published["ringBytes"]) is not int
            or published["ringBytes"] != self.metadata["ringBytes"]
            or type(published["offset"]) is not int
            or published["offset"] != 0
            or not isinstance(published["handle"], bytes)
            or len(published["handle"]) != 64
        ):
            raise ValueError("ring publication does not match this edge")
        self.sender = None
        self._drained_token = None
        _PROCESS_SEND_OPS.append(self)
        self.mapped_base = runtime.open_ipc_handle(published["handle"])
        self.remote = self.mapped_base + published["offset"]
        self.hip = runtime.library()
        self.hip.hipMemcpyAsync.restype = ctypes.c_int
        self.hip.hipMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.stream = None
        self.worker = None
        self.inference = None
        self.closed = False

    def attach_sender(self, sender):
        self._caller_stream()
        if self.sender is not None or sender.ops is not self:
            raise RuntimeError("one sender owns each mapping")
        self.sender = sender

    def bind_worker(self):
        self.worker = threading.get_ident()
        self.torch.cuda.set_device(self.device)
        self.inference = self.torch.inference_mode()
        self.inference.__enter__()
        self.stream = self.torch.cuda.Stream(self.device, priority=-1)

    def unbind_worker(self):
        if self.inference is not None:
            self.inference.__exit__(None, None, None)
            self.inference = None

    def _worker(self):
        if (
            threading.get_ident() != self.worker
            or self.closed
            or self.torch.cuda.current_device() != self.device.index
        ):
            raise RuntimeError("HIP copy belongs to the live outgoing worker")

    def wait_producer(self, event):
        self._worker()
        self.stream.wait_event(event)  # process-local only; this is not completion

    def copy_to_ring(self, slot, buffers, rows):
        self._worker()
        if not 0 < rows <= self.schema.max_rows or slot not in (0, 1, 2):
            raise ValueError("invalid nonempty ring copy")
        for field in self.metadata["fields"]:
            tensor = buffers[field["key"]]
            pointer = self.remote + slot * self.metadata["slotBytes"] + field["offset"]
            error = self.hip.hipMemcpyAsync(
                ctypes.c_void_p(pointer),
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_size_t(tensor.numel() * tensor.element_size()),
                3,
                ctypes.c_void_p(self.stream.cuda_stream),
            )
            self.runtime.check(error, operation="hipMemcpyAsync(" + field["key"] + ")")
            tensor.record_stream(self.stream)

    def copy_fence(self):
        self._worker()
        self.stream.synchronize()

    def drained_token(self, sender):
        """Issue a close capability only after the actual sender cleanly joins."""
        self._caller_stream()
        if (
            sender is not self.sender
            or not sender.successfully_closed
            or sender.error is not None
            or sender.worker.is_alive()
            or sender.acked != sender.sent
            or sender.sent != sender.submitted
            or self.inference is not None
        ):
            raise RuntimeError(
                "mapping cannot close without successful send/ACK drain and join"
            )
        self._drained_token = object()
        return self._drained_token

    def close_mapping(self, drained_token):
        self._caller_stream()
        if drained_token is None or drained_token is not self._drained_token:
            raise RuntimeError(
                "mapping close requires this sender's successful drain capability"
            )
        sender = self.sender
        if (
            sender is None
            or not sender.successfully_closed
            or sender.error is not None
            or sender.worker.is_alive()
            or sender.acked != sender.sent
            or sender.sent != sender.submitted
            or self.inference is not None
        ):
            raise RuntimeError("cannot unmap a failed or live sender")
        if not self.closed:
            self.runtime.close_ipc_handle(self.mapped_base)
            self.closed = True


class ReceiveOps(TensorOps):
    def __init__(self, torch, device, ring):
        super().__init__(torch, device)
        self.ring = ring
        self.clone_stream = None

    def clone_from_ring(self, slot, rows):
        self.clone_stream = self._caller_stream()
        metadata = self.ring.metadata
        output = {}
        for field in metadata["fields"]:
            key = field["key"]
            shape = self.ring.schema.fields(rows)[key]
            count = 2
            for dim in shape:
                count *= dim
            offset = slot * metadata["slotBytes"] + field["offset"]
            view = (
                self.ring.buffer[offset : offset + count]
                .view(self.torch.bfloat16)
                .view(shape)
            )
            output[key] = view.clone(memory_format=self.torch.contiguous_format)
        return output

    def clone_fence(self):
        self._caller_stream()
        self.clone_stream.synchronize()

    def empty_fields(self, fields):
        self._caller_stream()
        return {
            key: self.torch.empty(shape, device=self.device, dtype=self.torch.bfloat16)
            for key, shape in fields.items()
        }
