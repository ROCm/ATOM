# SPDX-License-Identifier: MIT
"""Bounded, single-HCA Mooncake engines for a physically separated RDMA fabric."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("atom")

# Decode peers remembered per pool for restart detection; one per decode
# (host, DP rank, TP rank, DCP rank) and rail, so far above any deployment.
_MAX_TRACKED_PEERS = 4096
# A failed rebuild doubles the wait before the next one, up to this.
_MAX_REBUILD_BACKOFF_S = 3600.0


@dataclass(frozen=True)
class RailRecoveryPolicy:
    """When a non-primary rail engine is replaced (ATOM_PD_ENDPOINT_RECOVERY).

    Mooncake keeps one endpoint per decode peer (``ip:rpc_port@hca``) for the
    life of an engine and exposes no call to drop one. A restarted decode
    comes back on a new RPC port, so its predecessor's endpoints stay behind;
    replacing the engine is the only way to free them from Python.
    """

    rebuild_after_failures: int = 2
    max_stale_peers: int = 8
    min_interval_s: float = 300.0


def _spawn_daemon(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="atom-rail-rebuild", daemon=True).start()


class RailEnginePool:
    """Keep each engine on one rail; never mutate a shared engine per request.

    Memory belongs to the connector and must outlive this pool. All engines
    register exactly the same MR ranges, but own separate lkeys and metadata.
    Additional engines are created once, under a lock, on the first request,
    using the RPC address supplied by local_ip_for_device. Multiple rails can
    share a reachable host address while retaining separate HCA filters.
    A rail whose first initialization failed stays failed until restart
    rather than retrying registration on every request or silently falling
    back to an unreachable rail.

    With a ``recovery`` policy, a non-primary rail engine that keeps failing,
    or whose decode peers restarted often enough, is rebuilt in the
    background: a new engine registers the same memory off the lock while the
    old one keeps serving, then replaces it. Requests already holding the old
    engine finish on it; it is freed, with all of its endpoints, when the last
    of them drops it. The primary engine is the connector's own and is never
    replaced.
    """

    def __init__(
        self,
        factory,
        primary,
        primary_device,
        devices,
        local_ip_for_device: Callable[[str], str],
        *,
        recovery: RailRecoveryPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] = _spawn_daemon,
        retire_wait_s: float = 300.0,
        retire_poll_s: float = 1.0,
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
        self._recovery = recovery
        self._clock = clock
        self._spawn = spawn
        self._retire_wait_s = retire_wait_s
        self._retire_poll_s = retire_poll_s
        # One registration at a time per process: a rebuild re-registers the
        # whole KV pool on its HCA.
        self._rebuild_gate = threading.Lock()
        # Recovery bookkeeping has its own lock: ``_lock`` is held through a
        # rail's first registration (tens of seconds), and every RDMA call
        # reports here. Order: ``_lock`` before ``_state_lock``.
        self._state_lock = threading.Lock()
        self._current = {primary_device: primary}
        self._fail_streak: dict[str, int] = {}
        self._stale_peers: dict[str, int] = {}
        self._peers: OrderedDict[tuple, tuple] = OrderedDict()
        self._rebuilding: set[str] = set()
        self._not_before: dict[str, float] = {}
        self._rebuild_failures: dict[str, int] = {}
        self._primary_warned = False
        self.rebuilds: dict[str, int] = {}

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
                engine = self._build(device, "first use")
            except Exception as exc:
                self._errors[device] = str(exc)
                raise
            self._engines[device] = engine
            with self._state_lock:
                self._current[device] = engine
            return engine

    def _build(self, device, reason):
        """Initialize and register a new engine on ``device``; roll back on failure."""
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
                "chunks=%d elapsed_ms=%.3f reason=%s",
                device,
                local_ip,
                engine.get_rpc_port(),
                len(registered),
                (time.monotonic() - started) * 1000,
                reason,
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

    # ------------------------------------------------------------------
    # Recovery (only with a RailRecoveryPolicy)
    # ------------------------------------------------------------------

    def report(self, engine, ok: bool) -> None:
        """Count one RDMA call's outcome on ``engine``.

        Outcomes of an engine that was already replaced, or of the primary,
        never schedule a rebuild.
        """
        if self._recovery is None:
            return
        job = None
        with self._state_lock:
            device = next(
                (d for d, current in self._current.items() if current is engine),
                None,
            )
            if device is None:
                return
            if ok:
                self._fail_streak[device] = 0
                return
            streak = self._fail_streak.get(device, 0) + 1
            self._fail_streak[device] = streak
            if streak >= self._recovery.rebuild_after_failures:
                job = self._schedule_locked(device, f"{streak} RDMA failures in a row")
        if job is not None:
            self._start(job, device)

    def note_peer(self, device, identity, target, instance=None) -> None:
        """Remember which decode instance ``identity`` is on ``device``.

        ``identity`` names a decode rank independently of its process (host,
        DP, TP, DCP rank); a new ``target`` (RPC port) or ``instance`` for it
        means that decode restarted and this rail's endpoints to the old one
        are stale.
        """
        if self._recovery is None or device not in self._devices:
            return
        key = (device, identity)
        current = (target, instance)
        job = None
        with self._state_lock:
            previous = self._peers.get(key)
            self._peers[key] = current
            self._peers.move_to_end(key)
            while len(self._peers) > _MAX_TRACKED_PEERS:
                self._peers.popitem(last=False)
            if previous is None or previous == current:
                return
            # An instance id is authoritative when both sides carry one;
            # otherwise a new RPC port is the restart.
            if previous[1] is not None and instance is not None:
                if previous[1] == instance:
                    return
            elif previous[0] == target:
                return
            stale = self._stale_peers.get(device, 0) + 1
            self._stale_peers[device] = stale
            logger.info(
                "PD_RAIL_PEER_RESTART device=%s peer=%s old=%s new=%s stale=%d",
                device,
                identity,
                previous[0],
                target,
                stale,
            )
            if stale >= self._recovery.max_stale_peers:
                job = self._schedule_locked(device, f"{stale} decode peers restarted")
        if job is not None:
            self._start(job, device)

    def _schedule_locked(self, device, reason):
        """Claim a rebuild of ``device`` under ``_state_lock``.

        Returns the job for the caller to start once the lock is released, or
        None.
        """
        if device == self._primary_device:
            if not self._primary_warned:
                self._primary_warned = True
                logger.warning(
                    "PD_RAIL_REBUILD_SKIPPED device=%s reason=%s: the primary "
                    "engine is not rebuilt; only a prefill restart frees its "
                    "stale endpoints",
                    device,
                    reason,
                )
            return None
        if device not in self._current or device in self._rebuilding:
            return None
        if self._clock() < self._not_before.get(device, float("-inf")):
            return None
        self._rebuilding.add(device)
        logger.warning("PD_RAIL_REBUILD_SCHEDULED device=%s reason=%s", device, reason)
        return lambda: self._rebuild(device, reason)

    def _start(self, job, device) -> None:
        try:
            self._spawn(job)
        except RuntimeError:  # e.g. can't start new thread
            logger.exception("PD_RAIL_REBUILD_FAILED device=%s: cannot start", device)
            with self._state_lock:
                self._rebuilding.discard(device)

    def _retire(self, old, device) -> None:
        """Release a replaced engine's memory registrations off the GIL.

        Requests that took ``old`` before the swap still hold it; wait for
        them (up to ``retire_wait_s``) by its reference count: the caller's
        local, this frame's argument and getrefcount's own. Then unregister
        its regions (Mooncake releases the GIL there), so that the destructor,
        which runs with the GIL held wherever the last reference drops, only
        tears down QPs and endpoints. If a request still holds it after the
        wait, its registrations go with the destructor instead.
        """
        deadline = time.monotonic() + self._retire_wait_s
        while sys.getrefcount(old) > 3:
            if time.monotonic() >= deadline:
                logger.warning(
                    "PD_RAIL_RETIRE_DEFERRED device=%s: the replaced engine is "
                    "still in use after %.0fs; it is freed by its last user",
                    device,
                    self._retire_wait_s,
                )
                return
            time.sleep(self._retire_poll_s)
        started = time.monotonic()
        for ptr, _ in reversed(self._regions):
            try:
                ret = old.unregister_memory(ptr)
                if ret != 0:
                    logger.error(
                        "Rail retire unregister failed: device=%s ptr=%#x ret=%s",
                        device,
                        ptr,
                        ret,
                    )
            except Exception:
                logger.exception(
                    "Rail retire unregister failed: device=%s ptr=%#x", device, ptr
                )
        logger.info(
            "PD_RAIL_RETIRED device=%s chunks=%d elapsed_ms=%.3f",
            device,
            len(self._regions),
            (time.monotonic() - started) * 1000,
        )

    def _rebuild(self, device, reason) -> None:
        try:
            with self._rebuild_gate:
                started = time.monotonic()
                try:
                    engine = self._build(device, f"rebuild: {reason}")
                except Exception:
                    with self._state_lock:
                        failures = self._rebuild_failures.get(device, 0) + 1
                        self._rebuild_failures[device] = failures
                        wait = min(
                            _MAX_REBUILD_BACKOFF_S,
                            self._recovery.min_interval_s * 2**failures,
                        )
                        self._not_before[device] = self._clock() + wait
                    logger.exception(
                        "PD_RAIL_REBUILD_FAILED device=%s reason=%s; keeping the "
                        "current engine, next try in %.0fs",
                        device,
                        reason,
                        wait,
                    )
                    return
                with self._lock, self._state_lock:
                    old = self._engines.get(device)
                    self._engines[device] = engine
                    self._current[device] = engine
                    self._fail_streak[device] = 0
                    self._stale_peers[device] = 0
                    self._rebuild_failures[device] = 0
                    self._not_before[device] = (
                        self._clock() + self._recovery.min_interval_s
                    )
                    self.rebuilds[device] = self.rebuilds.get(device, 0) + 1
                logger.warning(
                    "PD_RAIL_REBUILT device=%s reason=%s elapsed_ms=%.3f; retiring "
                    "the old engine once its in-flight transfers end",
                    device,
                    reason,
                    (time.monotonic() - started) * 1000,
                )
                del engine
                self._retire(old, device)
                del old
        finally:
            with self._state_lock:
                self._rebuilding.discard(device)
