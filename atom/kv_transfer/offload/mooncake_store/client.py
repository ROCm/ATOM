# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A pure Mooncake Store client: no segment of its own, zero-copy calls only.

The owner processes hold the Store's memory; this client mounts nothing
(``global_segment_size=0``) and keeps no staging buffer (``local_buffer_size=0``)
-- every put and get moves bytes straight between a registered caller buffer
and an owner over RDMA (or TCP).

Mooncake facts this wrapper is shaped by (v0.3.14):

* Results are per key: 0 or the object size on success, a negative
  ``ErrorCode`` otherwise, and a batch can partly succeed. A put of a key that
  already exists returns 0.
* Keys inside one call must be unique: the transfer is planned per key, so a
  duplicate silently collapses onto one buffer while both report success.
* There is no per-call timeout. A key's transfer batch waits up to a hard-coded
  60 s after it was posted and then gives up *without cancelling* the RDMA
  work, so a buffer whose key failed that way may still be read or written by
  the NIC. Sooner, a key fails only once every transfer of it has completed or
  failed, or before one was posted (see :func:`buffer_settled`).
* The data, lookup and remove calls release the GIL; ``setup`` and ``close`` do
  not. A blocked call spins its thread's CPU core until the transfer is done.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from typing import Any

logger = logging.getLogger("atom")

# Mooncake Store ErrorCode values (mooncake-store/include/types.h, v0.3.14).
OK = 0
INTERNAL_ERROR = -1
NO_AVAILABLE_HANDLE = -200
INVALID_PARAMS = -600
REPLICA_IS_NOT_READY = -703
OBJECT_NOT_FOUND = -704
OBJECT_ALREADY_EXISTS = -705
OBJECT_HAS_LEASE = -706
LEASE_EXPIRED = -707
TRANSFER_FAIL = -800
RPC_FAIL = -900
RPC_TIMEOUT = -901

_ERROR_NAMES = {
    INTERNAL_ERROR: "INTERNAL_ERROR",
    NO_AVAILABLE_HANDLE: "NO_AVAILABLE_HANDLE",
    INVALID_PARAMS: "INVALID_PARAMS",
    REPLICA_IS_NOT_READY: "REPLICA_IS_NOT_READY",
    OBJECT_NOT_FOUND: "OBJECT_NOT_FOUND",
    OBJECT_ALREADY_EXISTS: "OBJECT_ALREADY_EXISTS",
    OBJECT_HAS_LEASE: "OBJECT_HAS_LEASE",
    LEASE_EXPIRED: "LEASE_EXPIRED",
    TRANSFER_FAIL: "TRANSFER_FAIL",
    RPC_FAIL: "RPC_FAIL",
    RPC_TIMEOUT: "RPC_TIMEOUT",
}

# Seconds after posting a key's transfer that Mooncake stops waiting for it,
# leaving the RDMA work it posted running (transfer_task.cpp,
# wait_for_completion; timed on the wall clock).
BATCH_WAIT_S = 60.0

# Failures after which Mooncake has no transfer outstanding on the key's
# buffer, however long the call took: it never issued one (no space, missing
# or unready object, bad arguments, a failed master RPC before the transfer),
# or the transfer finished and only its result was discarded (LEASE_EXPIRED is
# decided after the bytes arrived; a failed master RPC after it, as a put's
# BatchPutEnd, likewise).
_SETTLED_FAILURES = frozenset(
    {
        NO_AVAILABLE_HANDLE,
        INVALID_PARAMS,
        REPLICA_IS_NOT_READY,
        OBJECT_NOT_FOUND,
        OBJECT_ALREADY_EXISTS,
        OBJECT_HAS_LEASE,
        LEASE_EXPIRED,
        RPC_FAIL,
        RPC_TIMEOUT,
    }
)


def describe(rc: int) -> str:
    """A Mooncake result code by name, e.g. ``OBJECT_NOT_FOUND(-704)``."""
    rc = int(rc)
    name = _ERROR_NAMES.get(rc)
    return f"{name}({rc})" if name is not None else f"rc={rc}"


def buffer_settled(rc: int, call_seconds: float) -> bool:
    """Whether no NIC access can still reach a buffer whose key returned ``rc``.

    ``call_seconds`` is how long the call that returned it took (see
    :class:`CallClock`). Mooncake leaves RDMA work behind a key only when its
    batch wait gives up, :data:`BATCH_WAIT_S` after posting it, so every result
    of a shorter call is settled. From a longer one, a success and the failures
    that leave no transfer behind are; TRANSFER_FAIL and any code this module
    does not know are not.
    """
    rc = int(rc)
    return call_seconds < BATCH_WAIT_S or rc >= 0 or rc in _SETTLED_FAILURES


class CallClock:
    """Times one Store call at least as long as Mooncake's batch wait counts it.

    Mooncake times that wait on the wall clock, which a clock step moves: the
    longer of the monotonic and the wall-clock span never comes out shorter.
    """

    __slots__ = ("_monotonic", "_wall")

    def __init__(self) -> None:
        self._monotonic = time.monotonic()
        self._wall = time.time()

    def seconds(self) -> float:
        """Seconds since the clock was made."""
        return max(time.monotonic() - self._monotonic, time.time() - self._wall)


def _new_distributed_store() -> Any:
    """A fresh ``MooncakeDistributedStore``; imported here, never at load time."""
    from mooncake.store import MooncakeDistributedStore

    return MooncakeDistributedStore()


def _new_replicate_config(group_ids: list[str]) -> Any:
    """A default ``ReplicateConfig`` (one replica) that groups each key."""
    from mooncake.store import ReplicateConfig

    config = ReplicateConfig()
    config.group_ids = list(group_ids)
    return config


class MooncakeStoreClient:
    """One ``MooncakeDistributedStore`` set up as a pure, zero-copy client."""

    def __init__(
        self,
        *,
        local_hostname: str,
        metadata_server: str,
        master_server_addr: str,
        protocol: str,
        rdma_devices: str,
        lookup_batch_keys: int = 8192,
    ) -> None:
        if lookup_batch_keys <= 0:
            raise ValueError("lookup_batch_keys must be positive")
        self.master_server_addr = master_server_addr
        self.metadata_server = metadata_server
        self.protocol = protocol
        self.rdma_devices = rdma_devices
        self._lookup_batch_keys = int(lookup_batch_keys)
        self._stats_lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()
        store = _new_distributed_store()
        # Positional: pybind requires all seven. global_segment_size=0 mounts
        # no segment, local_buffer_size=0 skips the copy-API staging buffer.
        rc = store.setup(
            local_hostname,
            metadata_server,
            0,
            0,
            protocol,
            rdma_devices,
            master_server_addr,
        )
        if rc != 0:
            raise RuntimeError(
                f"Mooncake Store client setup failed ({describe(rc)}): master="
                f"{master_server_addr} metadata={metadata_server} "
                f"protocol={protocol} rdma_devices={rdma_devices or '-'} "
                f"local_hostname={local_hostname}"
            )
        self._store = store

    # -- memory registration ---------------------------------------------
    def register(self, ptr: int, nbytes: int) -> None:
        """Register a buffer the zero-copy calls may move bytes to or from.

        Raises:
            RuntimeError: Mooncake refused it; every later put of it would
                fail with an unregistered address.
        """
        rc = self._store.register_buffer(int(ptr), int(nbytes))
        if rc != 0:
            raise RuntimeError(
                f"Mooncake Store could not register {int(nbytes)} bytes at "
                f"{int(ptr):#x} ({describe(rc)}) on {self.rdma_devices or 'tcp'}. "
                "Host memory must be transparent huge pages -- an ionic NIC "
                "registers only about 3 GiB of 4 KiB pages -- and HBM needs "
                "amdgpu peer memory (MOONCAKE_DISABLE_HIP_DMABUF=1 without dmabuf "
                "support); MC_MAX_MR_SIZE splits large buffers into several MRs."
            )

    def unregister(self, ptr: int) -> None:
        rc = self._store.unregister_buffer(int(ptr))
        if rc != 0:
            logger.warning(
                "Mooncake Store: unregistering the buffer at %#x failed (%s)",
                int(ptr),
                describe(rc),
            )

    # -- data plane ------------------------------------------------------
    def exists(self, keys: list[str]) -> list[int]:
        """Per key: 1 present, 0 absent, negative on error; order kept.

        Split into calls of at most ``lookup_batch_keys`` keys. Every key found
        gets a read lease on the master (10 s by default), which keeps it from
        being evicted before a get that follows promptly.
        """
        results: list[int] = []
        for start in range(0, len(keys), self._lookup_batch_keys):
            batch = keys[start : start + self._lookup_batch_keys]
            codes = list(self._store.batch_is_exist(batch))
            self._check_length("batch_is_exist", codes, batch)
            results.extend(int(code) for code in codes)
            self._count("exists", batch, None, codes, failed=lambda code: code < 0)
        return results

    def put(
        self,
        keys: list[str],
        ptrs: list[int],
        sizes: list[int],
        *,
        group_ids: list[str] | None = None,
    ) -> list[int]:
        """Store each buffer under its key; per key 0 or a negative code.

        ``group_ids`` puts each key in that Mooncake group, which the master
        evicts whole. A key that already exists keeps the group it was put in.
        """
        self._check_unique(keys)
        args: list[Any] = [list(keys), [int(p) for p in ptrs], [int(s) for s in sizes]]
        if group_ids is not None:
            if len(group_ids) != len(keys):
                raise ValueError(
                    f"{len(group_ids)} group ids for {len(keys)} keys; one per key"
                )
            args.append(_new_replicate_config(group_ids))
        codes = [int(code) for code in self._store.batch_put_from(*args)]
        self._check_length("batch_put_from", codes, keys)
        stored = sum(int(s) for s, code in zip(sizes, codes) if code == 0)
        self._count("put", keys, stored, codes, failed=lambda code: code != 0)
        return codes

    def get(self, keys: list[str], ptrs: list[int], sizes: list[int]) -> list[int]:
        """Read each object into its buffer; per key its size or a negative code."""
        self._check_unique(keys)
        codes = [
            int(code)
            for code in self._store.batch_get_into(
                list(keys), [int(p) for p in ptrs], [int(s) for s in sizes]
            )
        ]
        self._check_length("batch_get_into", codes, keys)
        read = sum(code for code in codes if code > 0)
        self._count("get", keys, read, codes, failed=lambda code: code < 0)
        return codes

    def remove(self, key: str, *, force: bool = False) -> int:
        """Remove one object; ``force`` ignores the read leases lookups grant."""
        return int(self._store.remove(key, force))

    def close(self) -> None:
        store = self.__dict__.pop("_store", None)
        if store is None:
            return
        rc = store.close()
        if rc not in (0, None):
            logger.warning("Mooncake Store client close returned %s", describe(rc))

    # -- accounting ------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Cumulative calls, keys, bytes and failures by code, per operation."""
        with self._stats_lock:
            return {"counts": dict(self._counts), "failures": dict(self._failures)}

    def _count(self, op, keys, nbytes, codes, *, failed) -> None:
        """Tally one call; ``nbytes`` is what it moved, None for a lookup."""
        bad = [code for code in codes if failed(int(code))]
        with self._stats_lock:
            self._counts[f"{op}_calls"] += 1
            self._counts[f"{op}_keys"] += len(keys)
            if nbytes is not None:
                self._counts[f"{op}_bytes"] += int(nbytes)
            for code in bad:
                self._failures[f"{op}:{describe(code)}"] += 1

    @staticmethod
    def _check_unique(keys: list[str]) -> None:
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Mooncake Store batch keys must be unique; a duplicate collapses "
                "onto one buffer while both report success"
            )

    @staticmethod
    def _check_length(call: str, codes: list, keys: list[str]) -> None:
        if len(codes) != len(keys):
            raise RuntimeError(
                f"Mooncake Store {call} returned {len(codes)} results for "
                f"{len(keys)} keys"
            )
