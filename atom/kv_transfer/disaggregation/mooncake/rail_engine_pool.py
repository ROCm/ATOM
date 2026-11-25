# SPDX-License-Identifier: MIT
"""Bounded, single-HCA Mooncake engines for a physically separated RDMA fabric."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("atom")


class RailEnginePool:
    """Keep each engine on one rail; never mutate a shared engine per request.

    Memory belongs to the connector and must outlive this pool. All engines
    register exactly the same MR ranges, but own separate lkeys and metadata.
    The connector builds every additional engine at startup (build_all), in
    parallel, using the RPC address supplied by local_ip_for_device; get()
    still creates a missing engine once, under a lock, on first use. Multiple
    rails can share a reachable host address while retaining separate HCA
    filters. A failed rail stays failed until restart rather than retrying
    registration on every request or silently falling back to an unreachable
    rail.

    Registering one rail costs about as much as the primary's registration
    (with HIP dma-buf, about 1 s of kernel time per 2 GiB chunk, one chunk at
    a time) and holds the process mmap lock while it runs. Built lazily on the
    first transfer, seven extra rails took 61-138 s each on MI355X/ionic,
    serially, while the serving thread blocked in mmap/munmap: the first
    transfers missed their deadline. build_all() moves that cost to startup.
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
            try:
                engine = self._build(device)
            except Exception as exc:
                self._errors[device] = str(exc)
                raise
            self._engines[device] = engine
            return engine

    def build_all(self):
        """Create and register every configured rail now, all rails in parallel.

        Call once at startup, after set_regions and before any transfer. get()
        waits for this to finish. A rail that fails is logged and stays failed
        (get raises for it), exactly as a lazy failure would.
        """
        with self._lock:
            if self._regions is None:
                raise RuntimeError("Rail memory regions are not ready")
            pending = sorted(
                d
                for d in self._devices
                if d not in self._engines and d not in self._errors
            )
            if not pending:
                return
            started = time.monotonic()

            def build(device):
                try:
                    return device, self._build(device), None
                except Exception as exc:
                    logger.exception("PD_RAIL_FAILED device=%s", device)
                    return device, None, exc

            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                results = list(executor.map(build, pending))
            for device, engine, exc in results:
                if exc is None:
                    self._engines[device] = engine
                else:
                    self._errors[device] = str(exc)
            logger.info(
                "PD_RAILS_BUILT ready=%d failed=%d elapsed_ms=%.3f",
                sum(exc is None for _, _, exc in results),
                sum(exc is not None for _, _, exc in results),
                (time.monotonic() - started) * 1000,
            )

    def _build(self, device):
        """Initialize one rail's engine and register every region on it; roll back on failure."""
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
            ptrs = [ptr for ptr, _ in self._regions]
            sizes = [size for _, size in self._regions]
            batch = getattr(engine, "batch_register_memory", None)
            if batch is not None and batch(ptrs, sizes) == 0:
                # Mooncake registers the chunks on parallel threads.
                registered = ptrs
            else:
                if batch is not None:
                    # A failed batch may leave some chunks registered; drop
                    # them (best effort) and redo one by one for exact rollback.
                    logger.warning(
                        "Rail batch registration failed on %s; registering "
                        "chunks individually",
                        device,
                    )
                    for ptr in ptrs:
                        try:
                            engine.unregister_memory(ptr)
                        except Exception:
                            logger.debug(
                                "Rail batch cleanup: device=%s ptr=%#x",
                                device,
                                ptr,
                                exc_info=True,
                            )
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
            return engine
        except Exception:
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
