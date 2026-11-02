# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mooncake over the GPU scale-up fabric (AMD Helios UALoE, xGMI in one node).

With ``protocol`` ``ualoe`` (or ``hip``) the producer writes the decode rank's
KV pool through Mooncake's HIP transport instead of a NIC:

- Both ranks back their paged KV pool with fabric-exportable VMM memory from
  Mooncake's allocator (``nvlink_allocator.so``), so Mooncake can publish a
  fabric handle for it.
- The engines run in Mooncake's ``EXTERNAL`` metadata mode: there is no
  metadata service and no Mooncake handshake. The decode rank sends its
  exported segment description inside each write request, and the producer
  imports and maps it the first time it sees it (``SegmentImporter``).
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import threading
import time
from typing import Any

import torch

logger = logging.getLogger("atom")

FABRIC_PROTOCOLS = frozenset({"ualoe", "hip"})
FABRIC_METADATA = "EXTERNAL"
# mc_allocator_probe: 1 = hipMemCreate with a fabric handle.
_BACKEND_FABRIC = 1


def is_fabric_protocol(protocol: str | None) -> bool:
    return (protocol or "").strip().lower() in FABRIC_PROTOCOLS


def _allocator_path(kv_transfer_config: dict) -> str:
    configured = kv_transfer_config.get("fabric_allocator")
    if configured:
        return configured
    import mooncake

    path = os.path.join(os.path.dirname(mooncake.__file__), "nvlink_allocator.so")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mooncake's fabric allocator is not at {path}; build mooncake with "
            "USE_HIP or set kv_transfer_config['fabric_allocator']"
        )
    return path


class _Allocator:
    _instances: dict[str, _Allocator] = {}
    _lock = threading.Lock()

    def __init__(self, path: str) -> None:
        lib = ctypes.CDLL(path)
        lib.mc_allocator_probe.argtypes = [ctypes.c_int]
        lib.mc_allocator_probe.restype = ctypes.c_int
        lib.mc_allocator_malloc.argtypes = [
            ctypes.c_ssize_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.mc_allocator_malloc.restype = ctypes.c_void_p
        lib.mc_allocator_free.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ssize_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.mc_allocator_free.restype = None
        self.lib = lib
        self.path = path

    @classmethod
    def get(cls, path: str) -> _Allocator:
        with cls._lock:
            if path not in cls._instances:
                cls._instances[path] = cls(path)
            return cls._instances[path]


class _FabricBuffer:
    """Owns one allocation; torch keeps it alive for as long as the tensor."""

    def __init__(self, allocator: _Allocator, ptr: int, nbytes: int, device: int):
        self._allocator = allocator
        self._ptr = ptr
        self._nbytes = nbytes
        self._device = device
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }

    def __del__(self) -> None:
        try:
            self._allocator.lib.mc_allocator_free(
                self._ptr, self._nbytes, self._device, None
            )
        except Exception:
            pass


# The pool this process allocated, as (base, nbytes); register_kv_caches
# registers it as one buffer (see registration_ranges).
_pool_range: tuple[int, int] | None = None


def allocate_fabric_pool(
    kv_transfer_config: dict, nbytes: int, device: Any
) -> torch.Tensor | None:
    """A zeroed uint8 tensor of ``nbytes`` in fabric memory, or None when the
    device cannot export fabric handles (the caller then allocates with torch,
    which Mooncake can share only within one host)."""
    global _pool_range
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    allocator = _Allocator.get(_allocator_path(kv_transfer_config))
    with torch.cuda.device(index):
        backend = allocator.lib.mc_allocator_probe(index)
        if backend != _BACKEND_FABRIC:
            logger.warning(
                "Mooncake fabric: device %d cannot allocate fabric memory "
                "(probe=%d); the KV pool stays in torch memory and is "
                "reachable only from this host",
                index,
                backend,
            )
            return None
        started = time.perf_counter()
        ptr = allocator.lib.mc_allocator_malloc(nbytes, index, None)
        if not ptr:
            raise RuntimeError(
                f"Mooncake fabric allocator failed for {nbytes} B on device {index}"
            )
        owner = _FabricBuffer(allocator, ptr, nbytes, index)
        tensor = torch.as_tensor(owner, device=device)
        tensor.zero_()
    _pool_range = (ptr, nbytes)
    logger.info(
        "Mooncake fabric: KV pool %.2f GiB at %#x on device %d (%.1f ms)",
        nbytes / (1 << 30),
        ptr,
        index,
        (time.perf_counter() - started) * 1e3,
    )
    return tensor


def registration_ranges(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Mooncake buffers covering ``regions`` ((base, nbytes) each).

    The fabric transport publishes whole physical allocations, so each
    allocation must be registered once: regions inside the fabric pool
    collapse into the pool, the rest merge where they touch or overlap.
    No RDMA MR size limit applies.
    """
    spans = []
    for base, size in regions:
        if size <= 0:
            continue
        end = base + size
        if _pool_range is not None:
            pool_base, pool_size = _pool_range
            if base < pool_base + pool_size and end > pool_base:
                base, end = min(base, pool_base), max(end, pool_base + pool_size)
        spans.append((base, end))
    spans.sort()
    merged: list[list[int]] = []
    for base, end in spans:
        if merged and base <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([base, end])
    return [(base, end - base) for base, end in merged]


def segment_tag(blob: bytes) -> str:
    return hashlib.sha1(blob).hexdigest()


class SegmentImporter:
    """The producer's view of each decode rank's segment.

    A write request carries the decode rank's segment description and a tag
    of it. The description is imported and mapped once per tag; a new tag (the
    decode rank re-registered memory, or restarted on the same address)
    replaces the old mapping.
    """

    def __init__(self, engine: Any, local_device: int) -> None:
        self._engine = engine
        self._local_device = local_device
        self._tags: dict[str, str] = {}
        self._lock = threading.Lock()

    def ensure(self, target: str, request_data: dict) -> None:
        tag = request_data.get("consumer_segment_tag")
        blob = request_data.get("consumer_segment")
        if not tag or blob is None:
            raise RuntimeError(
                f"write request from {target} carries no Mooncake segment; "
                "both sides must use the same fabric protocol"
            )
        if self._tags.get(target) == tag:
            return
        with self._lock:
            if self._tags.get(target) == tag:
                return
            started = time.perf_counter()
            sid = self._engine.import_segment(target, bytes(blob))
            if sid < 0:
                raise RuntimeError(
                    f"Mooncake import_segment({target}) failed: {sid}"
                )
            imported = time.perf_counter()
            ret = self._engine.bind_segment(sid, self._local_device)
            if ret != 0:
                raise RuntimeError(f"Mooncake bind_segment({target}) failed: {ret}")
            self._tags[target] = tag
        logger.info(
            "Mooncake fabric: mapped decode segment %s (import %.2f ms, "
            "bind %.2f ms)",
            target,
            (imported - started) * 1e3,
            (time.perf_counter() - imported) * 1e3,
        )
