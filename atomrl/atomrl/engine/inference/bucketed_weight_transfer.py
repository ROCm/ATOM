# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py
# Source: https://github.com/vllm-project/vllm/blob/fad09e8a1f51b31eba1f42ff5d651256c77a734d/examples/offline_inference/rlhf_utils.py

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from verl's bucketed weight transfer and the vLLM IPC example:
# https://github.com/volcengine/verl
# https://github.com/vllm-project/vllm

"""Bucketed weight transfer via ZMQ + CUDA IPC (or shared-memory fallback).

Vendored from verl
(``verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py``) and made
self-contained so AtomRL does not depend on the verl source tree at runtime.
Default behaviour is identical: a training worker (sender) packs weight
tensors into a fixed-size GPU buffer shared via a CUDA IPC handle and streams
them bucket-by-bucket over a ZMQ REQ/REP socket to the colocated vLLM worker
(receiver), which views tensors directly out of the shared buffer and loads
them into the model.

Two sender options differ from verl by default; pass ``gc_collect=True,
double_buffer=False`` for the verl behaviour, which ATOM's hand-rolled
receiver needs:

* ``gc_collect=False`` skips the full ``gc.collect()`` in the sender's cleanup,
  which costs ~0.4 s per sync in a Megatron trainer.
* ``double_buffer=True`` splits the bucket into two slots. The receiver acks a
  slot bucket on receipt (``ack_early`` in the control message) and loads it
  while the sender fills the other slot; its next ack therefore means the
  previous slot is free. The last bucket and direct sends are still acked
  after loading, because nothing follows them to carry that guarantee.

torch.cuda works for both NVIDIA and AMD/ROCm builds, so no device abstraction
layer is required here.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from collections.abc import Callable, Iterable
from multiprocessing import shared_memory
from typing import TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ATOMRL_LOGGING_LEVEL", "INFO"))


def _debug_enabled() -> bool:
    return os.getenv("LUMEN_WEIGHT_TRANSFER_DEBUG", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_timeout_ms() -> int:
    return int(os.getenv("LUMEN_WEIGHT_TRANSFER_ZMQ_TIMEOUT_MS", "0") or 0)


def _device_name() -> str:
    return "cuda"


def _device_id() -> int:
    return torch.cuda.current_device()


def _sync() -> None:
    torch.cuda.synchronize()


def _timed_cleanup(
    stats: dict, *, gc_collect: bool, ipc_collect: bool, empty_cache: bool
) -> None:
    for key, enabled, fn in (
        ("cleanup_gc_s", gc_collect, gc.collect),
        ("cleanup_ipc_collect_s", ipc_collect, torch.cuda.ipc_collect),
        ("cleanup_empty_cache_s", empty_cache, torch.cuda.empty_cache),
    ):
        if enabled:
            t0 = time.perf_counter()
            fn()
            stats[key] = time.perf_counter() - t0


def _debug_log(role: str, message: str, **kwargs) -> None:
    if not _debug_enabled():
        return
    fields = {
        "pid": os.getpid(),
        "device": f"{_device_name()}:{_device_id()}",
        **kwargs,
    }
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info("[weight-ipc][%s] %s %s", role, message, extra)


def _configure_socket(socket, role: str, zmq_handle: str) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    timeout_ms = _debug_timeout_ms()
    if timeout_ms > 0:
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    _debug_log(role, "socket configured", zmq_handle=zmq_handle, timeout_ms=timeout_ms)


async def ensure_async_iterator(iterable):
    """Yield from either an async iterator or a plain (sync) iterable."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def check_bucket_version(metadata: dict, expected_version: int | None) -> None:
    """Reject buckets that do not belong to the sync the receiver was told to expect.

    Equality, not monotonicity: ``_sync_weights_ipc`` runs more than once per
    ``global_step`` (main loop, the pre-rollout refresh when the engine was
    sleeping, and the first sync after resume), so the same version
    legitimately repeats. This mirrors the RDMA path's per-header check.

    ``expected_version=None`` disables the check, which is what keeps senders
    that predate the version field working.
    """
    if expected_version is None:
        return
    version = metadata.get("version")
    if version is None or int(version) != int(expected_version):
        raise RuntimeError(
            f"IPC weight version mismatch: expected {expected_version}, got {version!r}"
        )


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


# Adapted from vllm/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(
    handle: tuple[Callable, tuple], device_id: int | None = None
) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # Patch the device id so it matches the receiver's current device even
        # when sender and receiver have different CUDA_VISIBLE_DEVICES mappings
        # (crucial on ROCm where Ray pins each worker to its own device index).
        list_args[6] = device_id
    return func(*list_args)


def create_shared_memory(size: int, name: str):
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert (
            shm.size >= size
        ), f"Stale shm '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)
    return tensor, shm


class BucketedWeightSender:
    """Send model weights via bucketed IPC transfer over ZMQ (REQ side)."""

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
        version: int | None = None,
        min_bucket_bytes: int = 0,
        gc_collect: bool = False,
        double_buffer: bool = True,
    ):
        # ``min_bucket_bytes`` raises the bucket above the configured size so a
        # receiver that needs one stable buffer per update cycle can be given a
        # bucket that holds the largest single tensor. Left at 0, the bucket is
        # exactly the configured size and oversized tensors go out one at a
        # time through _direct_send_large_weight.
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = max(int(bucket_size_mb) << 20, int(min_bucket_bytes))
        self.use_shm = use_shm
        self.version = version
        self.gc_collect = gc_collect
        self.double_buffer = double_buffer
        self.num_slots = 2 if double_buffer else 1
        # The slots split the configured bucket rather than doubling it: on a
        # colocated GPU every extra byte comes out of vLLM's headroom.
        self.bucket_size //= self.num_slots

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None
        self._slot = 0
        # Wall-clock seconds per phase of the last send, plus counts. ``sync_s``
        # absorbs the async bucket copies; ``ack_s`` is time the receiver held us.
        self.stats: dict[str, float] = {}

    def _control(
        self, bucket_meta: dict, is_last: bool, ack_early: bool = False
    ) -> dict:
        """Build a per-bucket control message.

        ``version`` rides on every bucket rather than on the ``_init_buffer``
        handshake because that handshake is not a dict on the CUDA-IPC path --
        it is the bare tuple ``reduce_tensor()`` returns, which the receiver
        feeds straight to ``rebuild_ipc``. Per-bucket also matches the RDMA
        path, where every header carries the version.
        """
        message = {"bucket_meta": bucket_meta, "is_last": is_last}
        if self.version is not None:
            message["version"] = int(self.version)
        if ack_early:
            message["ack_early"] = True
        return message

    async def async_send_weights(self, weights: Iterable) -> None:
        """Stream ``(name, tensor)`` pairs to the receiver bucket-by-bucket."""
        st = self.stats = {
            "setup_s": 0.0,
            "sync_s": 0.0,
            "ack_s": 0.0,
            "cleanup_s": 0.0,
            "buckets": 0,
            "direct": 0,
            "tensors": 0,
            "bytes": 0,
        }
        self._slot = 0

        def _flush(meta: dict, is_last: bool) -> None:
            t0 = time.perf_counter()
            _sync()
            t1 = time.perf_counter()
            ack_early = self.double_buffer and not is_last
            self.socket.send_pyobj(self._control(meta, is_last, ack_early))
            self.socket.recv()
            st["sync_s"] += t1 - t0
            st["ack_s"] += time.perf_counter() - t1
            st["buckets"] += 1
            # An early ack for this bucket arrives only after the receiver has
            # finished the previous one, so the other slot is free now.
            self._slot = (self._slot + 1) % self.num_slots

        try:
            t0 = time.perf_counter()
            self._init_socket()
            self._init_buffer()
            st["setup_s"] = time.perf_counter() - t0

            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            async for name, weight in ensure_async_iterator(weights):
                weight = weight.contiguous()
                st["tensors"] += 1
                st["bytes"] += weight.nbytes
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    _flush(bucket_meta, False)
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({tuple(weight.shape)}, {weight.dtype}) exceeds the "
                        f"bucket size; increase update_weights_bucket_megabytes "
                        f"({self.bucket_size_mb} MB)."
                    )
                    t0 = time.perf_counter()
                    self._direct_send_large_weight(name, weight)
                    st["ack_s"] += time.perf_counter() - t0
                    st["direct"] += 1
                    continue

                start = self._slot * self.bucket_size + offset
                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": start,
                    "handle": None,
                }
                self.buffer[start : start + weight.nbytes].copy_(
                    weight.view(-1).view(torch.uint8), non_blocking=True
                )
                offset += weight.nbytes

            _flush(bucket_meta, True)
        finally:
            t0 = time.perf_counter()
            self._cleanup()
            st["cleanup_s"] = time.perf_counter() - t0

    def _init_socket(self) -> None:
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        _configure_socket(self.socket, "sender", self.zmq_handle)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self) -> None:
        size = self.bucket_size * self.num_slots
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(
                size, dtype=torch.uint8, device=f"{_device_name()}:{_device_id()}"
            )
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            shm_name = f"lumen_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            self.socket.send_pyobj({"name": shm_name, "size": size})

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        _timed_cleanup(
            self.stats,
            gc_collect=self.gc_collect,
            ipc_collect=True,
            empty_cache=True,
        )

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor) -> None:
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, TensorMetadata] = {
            name: {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": 0,
                "handle": handle,
            }
        }
        self.socket.send_pyobj(self._control(bucket_meta, False))
        self.socket.recv()


class BucketedWeightReceiver:
    """Receive model weights via bucketed IPC transfer over ZMQ (REP side)."""

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
        expected_version: int | None = None,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm
        self.expected_version = expected_version

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None
        # Wall-clock seconds per phase of the last receive, plus counts.
        # ``setup_s`` includes waiting for the sender to start; ``wait_s`` is
        # time blocked on the sender between buckets.
        self.stats: dict[str, float] = {}

    def receive_weights(self, on_bucket_received: Callable) -> None:
        st = self.stats = {
            "setup_s": 0.0,
            "wait_s": 0.0,
            "load_s": 0.0,
            "sync_s": 0.0,
            "cleanup_s": 0.0,
            "buckets": 0,
            "tensors": 0,
        }
        try:
            t0 = time.perf_counter()
            self._init_socket()
            self._init_buffer()
            st["setup_s"] = time.perf_counter() - t0

            while True:
                t0 = time.perf_counter()
                metadata = self.socket.recv_pyobj()
                st["wait_s"] += time.perf_counter() - t0
                check_bucket_version(metadata, self.expected_version)
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = (
                        meta["shape"],
                        meta["dtype"],
                        meta["offset"],
                        meta["handle"],
                    )
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        weights.append((name, tensor))
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = (
                        self.buffer[offset : offset + size]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                # Early ack lets the sender fill the other slot while we load;
                # we must finish this slot before reading the next message.
                ack_early = bool(metadata.get("ack_early"))
                if ack_early:
                    self.socket.send(b"")
                t0 = time.perf_counter()
                on_bucket_received(weights)
                t1 = time.perf_counter()
                _sync()
                st["load_s"] += t1 - t0
                st["sync_s"] += time.perf_counter() - t1
                st["buckets"] += 1
                st["tensors"] += len(weights)
                if not ack_early:
                    self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            t0 = time.perf_counter()
            self._cleanup()
            st["cleanup_s"] = time.perf_counter() - t0

    def _init_socket(self) -> None:
        self.socket = self.zmq_context.socket(zmq.REP)
        _configure_socket(self.socket, "receiver", self.zmq_handle)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self) -> None:
        started_at = time.time()
        comm_metadata = self.socket.recv_pyobj()
        _debug_log(
            "receiver",
            "got initial metadata",
            elapsed_s=f"{time.time() - started_at:.3f}",
        )
        buffer, shm = None, None
        if not self.use_shm:
            buffer = rebuild_ipc(comm_metadata, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        _sync()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        _timed_cleanup(self.stats, gc_collect=True, ipc_collect=True, empty_cache=True)
