# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A device buffer other processes can map: the prefill side's KV pool.

The prefill (producer) rank allocates its paged KV pool here instead of
through the torch allocator, so a decode (consumer) rank can import the pool's
handles, map them, and copy KV straight out of it over the scale-up fabric.

``SharedPool`` is the exporting side. ``descriptor()`` is what the consumer
receives: the handles, plus the pool's base address *in the producer's
process*, which is what the consumer needs to turn the producer's absolute
block addresses into offsets.

Pools exported by this process are kept in a registry so the connector, built
later than the KV cache, finds the one that backs its block regions
(:func:`find_exported`).
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

import torch

from atom.kv_transfer.disaggregation.ualoe import hip_vmm

logger = logging.getLogger("atom")

KIND_POSIX_FD = "posix_fd"
KIND_FABRIC = "fabric"
KINDS = (KIND_POSIX_FD, KIND_FABRIC)

_HANDLE_TYPE = {
    KIND_POSIX_FD: hip_vmm.HANDLE_TYPE_POSIX_FD,
    KIND_FABRIC: hip_vmm.HANDLE_TYPE_FABRIC,
}


@lru_cache(maxsize=1)
def host_id() -> str:
    """This host's boot id: two processes share dma-buf fds only within it."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return os.uname().nodename


class _DeviceBytes:
    """``__cuda_array_interface__`` of ``nbytes`` bytes at a raw device address,
    for a non-owning torch view."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


# The largest physical allocation one handle covers. ROCm 7.2's dma-buf
# export/import maps the wrong memory once a single allocation reaches 4 GiB
# (2**32 bytes, a truncated size), so a pool is built from chunks below that,
# each with its own handle, mapped back to back into one range on both sides.
CHUNK_BYTES = 2 << 30


def chunk_sizes(total: int, granularity: int, chunk: int = CHUNK_BYTES) -> list[int]:
    """``total`` (a multiple of ``granularity``) cut into chunks of at most
    ``chunk`` bytes, each a multiple of ``granularity``."""
    step = max(granularity, chunk // granularity * granularity)
    sizes = [step] * (total // step)
    if total % step:
        sizes.append(total % step)
    return sizes


class SharedPool:
    """Exportable device memory of at least ``nbytes``, mapped on this device.

    ``tensor`` is a uint8 view of the first ``nbytes``; it does not own the
    memory, ``close()`` frees it. The mapping covers the size rounded up to the
    allocation granularity, which is also what an importer maps.
    """

    def __init__(self, nbytes: int, device: torch.device, kind: str, domain: str):
        if kind not in KINDS:
            raise ValueError(f"unknown shared pool kind {kind!r}; one of {KINDS}")
        self.kind = kind
        self.domain = domain
        self.device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        self.base = 0
        self._handles: list[int] = []
        self._fds: list[int] = []
        self._fabric: list[bytes] = []
        handle_type = _HANDLE_TYPE[kind]
        with torch.cuda.device(self.device_index):
            hip_dev = hip_vmm.current_device()
            self.granularity = hip_vmm.granularity(hip_dev, handle_type)
            self.mapped_bytes = -(-nbytes // self.granularity) * self.granularity
            self.chunk_bytes = chunk_sizes(self.mapped_bytes, self.granularity)
            try:
                for size in self.chunk_bytes:
                    self._handles.append(hip_vmm.mem_create(size, hip_dev, handle_type))
                self.base = hip_vmm.map_chunks(
                    list(zip(self._handles, self.chunk_bytes)),
                    self.granularity,
                    hip_dev,
                )
                for handle in self._handles:
                    if kind == KIND_POSIX_FD:
                        # Kept open for the pool's life: importers duplicate them.
                        self._fds.append(hip_vmm.export_posix_fd(handle))
                    else:
                        self._fabric.append(hip_vmm.export_fabric(handle))
            except BaseException:
                self.close()
                raise
        self.nbytes = nbytes
        self.epoch = int.from_bytes(os.urandom(8), "big")
        self.tensor = torch.as_tensor(
            _DeviceBytes(self.base, nbytes), device=f"cuda:{self.device_index}"
        )

    def contains(self, addr: int, length: int = 0) -> bool:
        return self.base <= addr and addr + length <= self.base + self.mapped_bytes

    def descriptor(self) -> dict[str, Any]:
        """What an importer needs; plain types, so it travels in msgpack."""
        return {
            "kind": self.kind,
            "base": self.base,
            "bytes": self.mapped_bytes,
            "chunks": self.chunk_bytes,
            "align": self.granularity,
            "epoch": self.epoch,
            "domain": self.domain,
            "host": host_id(),
            "pid": os.getpid(),
            "fds": self._fds,
            "fabric": self._fabric,
        }

    def close(self) -> None:
        """Free the pool. Importers keep their own reference to the memory."""
        self.tensor = None
        if self.base:
            hip_vmm.unmap_chunks(self.base, self.chunk_bytes)
            self.base = 0
        for handle in self._handles:
            hip_vmm.mem_release(handle)
        self._handles = []
        for fd in self._fds:
            os.close(fd)
        self._fds = []


_exported: list[SharedPool] = []
_exported_lock = threading.Lock()


def allocate(
    nbytes: int, device: torch.device, kind: str, domain: str = ""
) -> SharedPool:
    """A zeroed :class:`SharedPool`, registered for :func:`find_exported`."""
    pool = SharedPool(nbytes, device, kind, domain)
    pool.tensor.zero_()
    with _exported_lock:
        _exported.append(pool)
    logger.info(
        "UALoE: exported a %s KV pool of %.2f GiB at %#x (device %d)",
        kind,
        pool.mapped_bytes / (1 << 30),
        pool.base,
        pool.device_index,
    )
    return pool


def find_exported(spans: list[tuple[int, int]]) -> SharedPool | None:
    """The pool of this process holding every ``(addr, length)`` span, if any."""
    if not spans:
        return None
    with _exported_lock:
        for pool in _exported:
            if all(pool.contains(addr, length) for addr, length in spans):
                return pool
    return None


def release_all() -> None:
    """Free every exported pool (tests and process teardown)."""
    with _exported_lock:
        pools, _exported[:] = list(_exported), []
    for pool in pools:
        pool.close()
