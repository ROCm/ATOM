# SPDX-License-Identifier: MIT
"""Bounded, single-HCA Mooncake engines for a physically separated RDMA fabric."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("atom")


class RailEnginePool:
    """Keep each engine on one rail; never mutate a shared engine per request.

    Memory belongs to the connector and must outlive this pool. All engines
    register exactly the same MR ranges, but own separate lkeys and metadata.
    Additional engines are created once, under a lock, on the first request.
    A failed rail stays failed until restart rather than retrying registration
    on every request or silently falling back to an unreachable rail.
    """

    def __init__(self, factory, primary, primary_device, devices, local_ip):
        self._factory = factory
        self._primary_device = primary_device
        self._devices = frozenset(devices)
        if not primary_device or primary_device not in self._devices:
            raise ValueError("The primary HCA must be one of the configured rails")
        self._engines = {primary_device: primary}
        self._errors = {}
        self._local_ip = local_ip
        self._regions = None
        self._lock = threading.Lock()

    def set_regions(self, ptrs, sizes):
        if len(ptrs) != len(sizes) or not ptrs:
            raise ValueError("Nonempty, aligned registration lists are required")
        with self._lock:
            if self._regions is not None:
                raise RuntimeError(
                    "Replacing live rail memory registrations is unsupported"
                )
            self._regions = tuple(zip(ptrs, sizes))

    def get(self, device):
        if not isinstance(device, str) or not device or device not in self._devices:
            raise ValueError(
                f"Consumer HCA {device!r} is not a configured matching rail"
            )
        with self._lock:
            if self._regions is None:
                raise RuntimeError("Rail memory regions are not ready")
            if device in self._errors:
                raise RuntimeError(
                    f"Rail {device} initialization previously failed: {self._errors[device]}"
                )
            if device in self._engines:
                return self._engines[device]
            engine = None
            registered = []
            started = time.monotonic()
            try:
                engine = self._factory()
                ret = engine.initialize(self._local_ip, "P2PHANDSHAKE", "rdma", device)
                if ret != 0:
                    raise RuntimeError(f"initialize({device}) returned {ret}")
                # Individual registration makes rollback exact on partial failure.
                for ptr, size in self._regions:
                    ret = engine.register_memory(ptr, size)
                    if ret != 0:
                        raise RuntimeError(
                            f"register_memory({device}, {ptr:#x}, {size}) returned {ret}"
                        )
                    registered.append(ptr)
                logger.info(
                    "PD_RAIL_READY device=%s rpc_port=%d chunks=%d elapsed_ms=%.3f",
                    device,
                    engine.get_rpc_port(),
                    len(registered),
                    (time.monotonic() - started) * 1000,
                )
                self._engines[device] = engine
                return engine
            except Exception as exc:
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
