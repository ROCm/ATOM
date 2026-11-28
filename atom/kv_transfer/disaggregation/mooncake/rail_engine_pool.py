# SPDX-License-Identifier: MIT
"""Bounded, single-HCA Mooncake engines for a physically separated RDMA fabric."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger("atom")


class RailEnginePool:
    """Keep each engine on one rail; never mutate a shared engine per request.

    Memory belongs to the connector and must outlive this pool. All engines
    register exactly the same MR ranges, but own separate lkeys and metadata.
    Additional engines are created once, one at a time, on the first request,
    using the RPC address supplied by local_ip_for_device. Registering every
    chunk on a new engine takes about a minute, so a rail already built is
    returned without waiting for another one's build. Multiple rails can
    share a reachable host address while retaining separate HCA filters.
    A failed rail stays failed until restart rather than retrying registration
    on every request or silently falling back to an unreachable rail.
    """

    def __init__(
        self,
        factory,
        primary,
        primary_device,
        devices,
        local_ip_for_device: Callable[[str], str],
    ):
        self._factory = factory
        self._primary_device = primary_device
        self._devices = frozenset(devices)
        if not primary_device or primary_device not in self._devices:
            raise ValueError("The primary HCA must be one of the configured rails")
        self._engines = {primary_device: primary}
        self._errors = {}
        self._local_ip_for_device = local_ip_for_device
        self._regions = None
        # Guards the tables above; held only to read or update them.
        self._lock = threading.Lock()
        self._build_lock = threading.Lock()

    def set_regions(self, ptrs, sizes):
        if len(ptrs) != len(sizes) or not ptrs:
            raise ValueError("Nonempty, aligned registration lists are required")
        with self._lock:
            if self._regions is not None:
                raise RuntimeError(
                    "Replacing live rail memory registrations is unsupported"
                )
            self._regions = tuple(zip(ptrs, sizes))

    def needs_build(self, device) -> bool:
        """Whether ``get(device)`` would first have to build the rail's engine.

        False for a rail that is built or failed, and for one that is not
        configured: ``get`` answers those at once.
        """
        with self._lock:
            return (
                isinstance(device, str)
                and device in self._devices
                and device not in self._engines
                and device not in self._errors
            )

    def get(self, device):
        if not isinstance(device, str) or not device or device not in self._devices:
            raise ValueError(
                f"Consumer HCA {device!r} is not a configured matching rail"
            )
        engine = self._built(device)
        if engine is not None:
            return engine
        with self._build_lock:
            # Another caller may have built it, or failed to, while this one
            # waited for the lock.
            engine = self._built(device)
            if engine is not None:
                return engine
            return self._build(device)

    def _built(self, device):
        """The rail's engine, or None if it is not built yet."""
        with self._lock:
            if self._regions is None:
                raise RuntimeError("Rail memory regions are not ready")
            if device in self._errors:
                raise RuntimeError(
                    f"Rail {device} initialization previously failed: {self._errors[device]}"
                )
            return self._engines.get(device)

    def _build(self, device):
        """Build and register the rail's engine; caller holds ``_build_lock``."""
        engine = None
        registered = []
        started = time.monotonic()
        try:
            # Resolve the control-plane RPC address independently of the
            # HCA filter used for RDMA transfers.
            local_ip = self._local_ip_for_device(device)
            if not isinstance(local_ip, str) or not local_ip:
                raise ValueError(f"No local address resolved for rail {device}")
            engine = self._factory()
            ret = engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", device)
            if ret != 0:
                raise RuntimeError(f"initialize({device}, {local_ip}) returned {ret}")
            # Individual registration makes rollback exact on partial failure.
            for ptr, size in self._regions:
                ret = engine.register_memory(ptr, size)
                if ret != 0:
                    raise RuntimeError(
                        f"register_memory({device}, {ptr:#x}, {size}) returned {ret}"
                    )
                registered.append(ptr)
            logger.info(
                "PD_RAIL_READY device=%s local_ip=%s rpc_port=%d "
                "chunks=%d elapsed_ms=%.3f",
                device,
                local_ip,
                engine.get_rpc_port(),
                len(registered),
                (time.monotonic() - started) * 1000,
            )
            with self._lock:
                self._engines[device] = engine
            return engine
        except Exception as exc:
            with self._lock:
                self._errors[device] = str(exc)
            if engine is not None:
                for ptr in reversed(registered):
                    try:
                        ret = engine.unregister_memory(ptr)
                        if ret != 0:
                            logger.error(
                                "Rail rollback failed: device=%s ptr=%#x ret=%s",
                                device,
                                ptr,
                                ret,
                            )
                    except Exception:
                        logger.exception(
                            "Rail rollback failed: device=%s ptr=%#x", device, ptr
                        )
            raise
