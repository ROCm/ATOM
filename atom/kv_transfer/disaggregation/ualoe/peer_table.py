# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The consumer's view of producer KV pools: import, map, translate.

A producer rank is named by ``target`` (its side-channel address). The
descriptor it sends (:meth:`SharedPool.descriptor`) is recorded with
:meth:`PeerTable.attach`; the first read from that target imports and maps
the pool (:meth:`PeerTable.acquire`), later ones reuse the mapping.

A producer that restarts exports a new pool with a new epoch; its old mapping
is dropped once no copy still holds it. A mapping no copy used for ``idle_s``
is dropped too: an imported handle keeps the producer's memory alive, so a
dead producer's pool must not stay pinned by this process forever.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from atom.kv_transfer.disaggregation.ualoe import hip_vmm
from atom.kv_transfer.disaggregation.ualoe.shared_pool import (
    KIND_FABRIC,
    KIND_POSIX_FD,
    host_id,
)

logger = logging.getLogger("atom")


class PeerUnreachable(RuntimeError):
    """This process cannot map the producer's pool (another host, another
    fabric domain, or the import failed)."""


def check_reachable(desc: dict[str, Any], local_domain: str) -> None:
    """Raise :class:`PeerUnreachable` when ``desc`` cannot be mapped from here
    without trying; a failed import says the rest."""
    kind = desc.get("kind")
    if kind == KIND_POSIX_FD:
        if desc.get("host") != host_id():
            raise PeerUnreachable("posix_fd pool on another host")
    elif kind == KIND_FABRIC:
        remote_domain = desc.get("domain") or ""
        if local_domain and remote_domain and local_domain != remote_domain:
            raise PeerUnreachable(
                f"fabric domain {remote_domain!r} is not this one ({local_domain!r})"
            )
    else:
        raise PeerUnreachable(f"unknown pool kind {kind!r}")


@dataclass
class PeerMapping:
    """One producer pool mapped into this process."""

    target: str
    epoch: int
    remote_base: int
    nbytes: int
    local_base: int
    handles: list[int]
    chunks: list[int]
    refs: int = 0
    last_used: float = field(default_factory=time.monotonic)
    retired: bool = False

    def translate(self, addrs: list[int], sizes: list[int]) -> list[int] | None:
        """The local addresses of the producer's ``addrs``; None when any
        ``[addr, addr + size)`` leaves the pool."""
        addr = np.asarray(addrs, dtype=np.int64)
        size = np.asarray(sizes, dtype=np.int64)
        off = addr - self.remote_base
        if addr.size and (int(off.min()) < 0 or int((off + size).max()) > self.nbytes):
            return None
        return (off + self.local_base).tolist()


def _import_and_map(desc: dict[str, Any], device: int) -> tuple[list[int], int]:
    """Import ``desc``'s chunk handles and map them back to back for
    ``device``: (handles, base)."""
    handles: list[int] = []
    try:
        if desc["kind"] == KIND_POSIX_FD:
            for remote_fd in desc["fds"]:
                fd = hip_vmm.dup_fd_from(int(desc["pid"]), int(remote_fd))
                try:
                    handles.append(hip_vmm.import_posix_fd(fd))
                finally:
                    os.close(fd)
        else:
            for blob in desc["fabric"]:
                handles.append(hip_vmm.import_fabric(bytes(blob)))
        base = hip_vmm.map_chunks(
            list(zip(handles, desc["chunks"])), int(desc["align"]), device
        )
    except BaseException:
        for handle in handles:
            hip_vmm.mem_release(handle)
        raise
    return handles, base


def _unmap(mapping: PeerMapping) -> None:
    try:
        hip_vmm.unmap_chunks(mapping.local_base, mapping.chunks)
        for handle in mapping.handles:
            hip_vmm.mem_release(handle)
    except Exception:
        logger.exception("UALoE: unmapping the pool of %s failed", mapping.target)


class PeerTable:
    """Producer pools by target, mapped lazily and dropped when stale or idle."""

    def __init__(
        self,
        local_domain: str = "",
        idle_s: float = 600.0,
        *,
        import_and_map: Callable[[dict[str, Any], int], tuple[list[int], int]] = (
            _import_and_map
        ),
        unmap: Callable[[PeerMapping], None] = _unmap,
    ) -> None:
        self.local_domain = local_domain
        self.idle_s = idle_s
        self._import_and_map = import_and_map
        self._unmap = unmap
        self._lock = threading.Lock()
        # target -> the latest descriptor it sent
        self._descs: dict[str, dict[str, Any]] = {}
        self._maps: dict[str, PeerMapping] = {}
        # superseded mappings still held by a transfer
        self._retired: list[PeerMapping] = []
        # target -> epoch that failed to map; not retried until the epoch moves
        self._unreachable: dict[str, int] = {}

    def attach(self, target: str, desc: dict[str, Any] | None) -> None:
        """Record the latest descriptor ``target`` sent."""
        if desc is None:
            return
        with self._lock:
            self._descs[target] = desc

    def acquire(self, target: str, device: int) -> PeerMapping | None:
        """The mapping to write ``target`` through, held until :meth:`release`.

        None when ``target`` sent no descriptor yet. Raises
        :class:`PeerUnreachable` when it did but this process cannot map it.
        """
        with self._lock:
            desc = self._descs.get(target)
            if desc is None:
                return None
            epoch = int(desc["epoch"])
            mapping = self._maps.get(target)
            if mapping is not None and mapping.epoch == epoch:
                mapping.refs += 1
                mapping.last_used = time.monotonic()
                return mapping
            if self._unreachable.get(target) == epoch:
                raise PeerUnreachable(f"the pool of {target} failed to map before")
            if mapping is not None:
                self._retire_locked(target)
            # Mapped under the lock: a burst of requests to a new producer
            # imports once, and the first transfer pays for it anyway.
            try:
                check_reachable(desc, self.local_domain)
                handles, base = self._import_and_map(desc, device)
            except PeerUnreachable:
                self._unreachable[target] = epoch
                raise
            except Exception as exc:
                self._unreachable[target] = epoch
                raise PeerUnreachable(f"mapping the pool of {target}: {exc}") from exc
            mapping = PeerMapping(
                target=target,
                epoch=epoch,
                remote_base=int(desc["base"]),
                nbytes=int(desc["bytes"]),
                local_base=base,
                handles=handles,
                chunks=list(desc["chunks"]),
                refs=1,
            )
            self._maps[target] = mapping
            logger.info(
                "UALoE: mapped the %s pool of %s (%.2f GiB) at %#x",
                desc["kind"],
                target,
                mapping.nbytes / (1 << 30),
                base,
            )
            return mapping

    def release(self, mapping: PeerMapping) -> None:
        with self._lock:
            mapping.refs -= 1
            mapping.last_used = time.monotonic()
            if mapping.retired and mapping.refs == 0:
                self._retired.remove(mapping)
                self._unmap(mapping)

    def evict_idle(self, now: float | None = None) -> int:
        """Drop mappings no transfer used for ``idle_s``; returns how many."""
        now = time.monotonic() if now is None else now
        dropped = 0
        with self._lock:
            for target, mapping in list(self._maps.items()):
                if mapping.refs == 0 and now - mapping.last_used > self.idle_s:
                    del self._maps[target]
                    self._unmap(mapping)
                    dropped += 1
        return dropped

    def close(self) -> None:
        """Unmap every mapping no transfer holds; a held one is unmapped by
        its last :meth:`release`."""
        with self._lock:
            for target in list(self._maps):
                self._retire_locked(target)
            self._descs.clear()

    def _retire_locked(self, target: str) -> None:
        mapping = self._maps.pop(target)
        if mapping.refs == 0:
            self._unmap(mapping)
        else:
            mapping.retired = True
            self._retired.append(mapping)
