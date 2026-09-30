# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA landing for Mooncake P/D with a DCP decode (opt-in: ``ATOM_PD_MLA_LANDING``).

A DCP decode rank owns every ``dcp_size``-th token of a prefill block, so its
MLA rows cannot be written as whole blocks. The staged path
(``_execute_staged_mla_regions``) sends one RDMA descriptor per destination
page, which caps a NIC at about 10 GB/s with fragmented decode block tables.
Landing sends a rank's rows packed in rank order instead: one descriptor per
slot of a decode-side GPU landing pool, which the decode rank then scatters
into its paged KV cache.

Protocol, per decode rank (every rank has its own pool and connector):

1. The rank splits its pool into one partition per prefill stage endpoint and
   advertises the stage's partition in every ``write_request``
   (``mla_landing = {epoch, base, slot_bytes, slots}``).
2. The stage keeps the partition as credits (``LandingCredits``), shared by
   all its requests to that rank. A send worker takes a credit, gathers rows
   into its staging slot, writes the slot to ``base + slot * slot_bytes`` and
   sends ``MSG_LANDING_READY`` (request, nonce, stage, seq, slot, items). With
   no credit within ``ATOM_PD_MLA_LANDING_CREDIT_WAIT_MS`` it sends the rest
   of the transfer through the staged path. Write-done then lists the slot of
   every READY it sent (``landed_slots``), so a lost READY is detected and its
   slot still returned.
3. The rank (``LandingReceiver``) scatters READY slots on its own stream and
   returns their credits (``MSG_LANDING_CREDIT``) once the scatter finished.
4. A request completes when every stage's write-done arrived, every landed
   slot it announced was received, and every scatter finished. A failed
   request is reported only after all its stages ended (or after
   ``ATOM_PD_MLA_LANDING_FAIL_WAIT_S``); READY slots that arrive for it later
   are dropped and their credits returned, so a late write can only ever land
   in its own stage's partition and is never scattered.

Credits never move between stages or ranks, so no slot is reused while a
stage may still write it, and no side waits on another stream's progress.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import msgpack
import numpy as np
import torch

from atom.kv_transfer.disaggregation.landing_scatter import (
    DevicePointer,
    landing_segments,
    scatter_segments,
    segment_table,
    segment_table_capacity,
)
from atom.kv_transfer.disaggregation.types import KVTransferRegion

logger = logging.getLogger("atom")

MSG_LANDING_READY = b"landing_ready"
MSG_LANDING_CREDIT = b"landing_credit"

_STATS_INTERVAL_S = 30.0
# A finished request's stage addresses are kept this long so READY slots that
# arrive after it finished can still be dropped with their credit returned.
_TOMBSTONE_TTL_S = 600.0


# ---------------------------------------------------------------------------
# Producer side
# ---------------------------------------------------------------------------


@dataclass
class _TargetCredits:
    epoch: int
    free: list[int]
    in_flight: set[int] = field(default_factory=set)


class LandingCredits:
    """The landing slots this prefill stage may write, per decode rank.

    ``target`` is the decode rank's Mooncake session (``host:rpc_port``). A
    decode restart comes with a new ``epoch``, which resets the credits.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._targets: dict[str, _TargetCredits] = {}
        self.stats = {"landed": 0, "fallbacks": 0, "wait_s": 0.0}

    def sync(self, target: str, landing: dict) -> None:
        """Adopt the partition a write request advertises for ``target``."""
        epoch = int(landing["epoch"])
        with self._cv:
            current = self._targets.get(target)
            if current is not None and current.epoch == epoch:
                return
            self._targets[target] = _TargetCredits(
                epoch=epoch, free=[int(s) for s in landing["slots"]]
            )
            self._cv.notify_all()

    def acquire(self, target: str, epoch: int, timeout_s: float) -> int | None:
        """A free slot for ``target``, or None after ``timeout_s``."""
        start = time.monotonic()
        deadline = start + timeout_s
        with self._cv:
            try:
                while True:
                    credits = self._targets.get(target)
                    if credits is None or credits.epoch != epoch:
                        return None
                    if credits.free:
                        slot = credits.free.pop()
                        credits.in_flight.add(slot)
                        return slot
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cv.wait(remaining)
            finally:
                self.stats["wait_s"] += time.monotonic() - start

    def release(self, target: str, epoch: int, slots: list[int]) -> None:
        """Return slots the decode rank finished with (or that never landed)."""
        with self._cv:
            credits = self._targets.get(target)
            if credits is None or credits.epoch != epoch:
                return
            for slot in slots:
                if slot not in credits.in_flight:
                    logger.warning(
                        "[PD-LANDING] %s returned slot %d that was not in flight",
                        target,
                        slot,
                    )
                    continue
                credits.in_flight.discard(slot)
                credits.free.append(slot)
            self._cv.notify_all()


# ---------------------------------------------------------------------------
# Consumer side
# ---------------------------------------------------------------------------


@dataclass
class _Stream:
    """One prefill stage's transfer of one request to this rank."""

    stage_addr: str
    slots: frozenset[int]
    seen: set[int] = field(default_factory=set)
    done: bool = False


@dataclass
class _Request:
    nonce: int
    dst_block_ids: np.ndarray
    streams: dict[int, _Stream]
    expected: int
    compute_event: torch.cuda.Event | None
    event_waited: bool = False
    pending: int = 0
    failed: bool = False
    failed_at: float = 0.0


@dataclass
class _Task:
    req_id: str
    nonce: int
    pp_rank: int
    slot: int
    items: list[tuple[int, int, int, int]]


class LandingReceiver:
    """A decode rank's landing pool, request bookkeeping and scatter thread.

    ``send(addr, parts)`` sends a ZMQ message to a prefill stage endpoint;
    ``finish(req_id, failed)`` reports a request to the connector. Both are
    called without this object's lock held.
    """

    def __init__(
        self,
        *,
        device: int | None,
        pool_slots: int,
        slot_bytes: int,
        block_size: int,
        consumer_key: str,
        region_bases: list[int],
        region_block_bytes: list[int],
        mla_regions: list[int],
        send: Callable[[str, list], None],
        finish: Callable[[str, bool], None],
        fail_wait_s: float,
    ) -> None:
        if not mla_regions:
            raise ValueError("MLA landing needs at least one MLA region")
        # None keeps everything on the CPU (unit tests replace _copy_segments).
        self.device = (
            torch.device("cpu") if device is None else torch.device("cuda", device)
        )
        self.slot_bytes = slot_bytes
        self.block_size = block_size
        self.consumer_key = consumer_key
        self.epoch = int.from_bytes(os.urandom(7), "big")
        self._send = send
        self._finish = finish
        self._fail_wait_s = fail_wait_s
        self._region_bases = np.asarray(region_bases, dtype=np.int64)
        self._region_block_bytes = np.asarray(region_block_bytes, dtype=np.int64)
        self._is_mla = np.zeros(len(region_bases), dtype=bool)
        self._is_mla[mla_regions] = True
        for c in mla_regions:
            if region_block_bytes[c] % (4 * block_size) or region_bases[c] % 4:
                raise ValueError(
                    f"MLA landing needs 4-byte aligned rows; region {c} has "
                    f"{region_block_bytes[c]} bytes per block at {region_bases[c]:#x}"
                )
        self._dst_origin = int(self._region_bases[mla_regions].min())
        self.pool = torch.empty(
            (pool_slots, slot_bytes), dtype=torch.uint8, device=self.device
        )
        self._pool_slots = pool_slots
        # Worst case per slot: every page of every row plus one split per item.
        rows_per_slot = slot_bytes // int(self._region_block_bytes[mla_regions].min())
        self._max_segments = segment_table_capacity(pool_slots * (rows_per_slot + 64))
        self._table_host = self._table_dev = self._stream = None
        if self.device.type == "cuda":
            self._table_host = torch.empty(
                (3, self._max_segments), dtype=torch.int64
            ).pin_memory()
            self._table_dev = torch.empty(
                (3, self._max_segments), dtype=torch.int64, device=self.device
            )
            self._stream = torch.cuda.Stream(device=self.device)

        self._lock = threading.Lock()
        self._partitions: dict[str, list[int]] = {}
        self._slot_owner: dict[int, str] = {}
        self._unassigned = list(range(pool_slots))
        self._warned_spent = False
        self._requests: dict[str, _Request] = {}
        self._tombstones: dict[tuple[str, int], tuple[float, dict[int, str]]] = {}
        self._queue: queue.SimpleQueue[_Task] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._disabled = False
        self._stats = {"slots": 0, "bytes": 0, "scatter_s": 0.0, "batches": 0}
        self._stats_at = time.monotonic()

    # -- setup ---------------------------------------------------------------

    def region(self) -> KVTransferRegion:
        return KVTransferRegion(
            base_addr=self.pool.data_ptr(),
            total_bytes=self.pool.numel(),
            unit_bytes=self.slot_bytes,
            semantic_role="mla.landing",
        )

    def start(self) -> None:
        """Compile the scatter kernel, then start the scatter thread.

        Runs before CUDA graph capture, so the thread never JIT-compiles or
        allocates while graphs are being captured.
        """
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            # A copy of slot 0 onto itself, through the same table buffer and
            # a 16-byte aligned origin like the real one, so serving never
            # meets a new Triton specialization.
            offset = (self.pool.data_ptr() - self._dst_origin) // 4
            self._table_dev.zero_()
            self._table_dev[1, :2] = offset
            self._table_dev[2, :2] = 1
            scatter_segments(
                self.pool,
                DevicePointer(self._dst_origin, self.device),
                self._table_dev,
                2,
            )
        self._stream.synchronize()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mooncake-mla-landing"
        )
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return not self._disabled

    # -- dispatch (main thread) ------------------------------------------------

    def advertise(self, stage_addr: str, num_stages: int) -> dict | None:
        """The ``mla_landing`` field for a write request to ``stage_addr``.

        The first request to a stage endpoint assigns it a partition of
        ``pool_slots // num_stages`` slots; None once the pool is spent.
        """
        with self._lock:
            if self._disabled:
                return None
            slots = self._partitions.get(stage_addr)
            if slots is None:
                share = max(1, self._pool_slots // max(1, num_stages))
                slots = self._unassigned[:share]
                if not slots:
                    if not self._warned_spent:
                        self._warned_spent = True
                        logger.warning(
                            "[PD-LANDING] pool spent: stage %s and later new "
                            "stage endpoints use staged per-page writes",
                            stage_addr,
                        )
                    return None
                del self._unassigned[:share]
                self._partitions[stage_addr] = slots
                for slot in slots:
                    self._slot_owner[slot] = stage_addr
                logger.info(
                    "[PD-LANDING] stage %s gets %d landing slots (%d unassigned)",
                    stage_addr,
                    len(slots),
                    len(self._unassigned),
                )
        return {
            "epoch": self.epoch,
            "base": self.pool.data_ptr(),
            "slot_bytes": self.slot_bytes,
            "slots": slots,
        }

    def begin(
        self,
        req_id: str,
        nonce: int,
        dst_block_ids: list[int],
        stage_addrs: dict[int, str],
        expected: int,
    ) -> None:
        """Track a request before any write request for it goes out.

        Records an event on the compute stream: the request's fresh pages may
        still be read by their previous owner's queued forward, and the
        scatter must not overwrite them before that finished.
        """
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        streams = {
            pp: _Stream(addr, frozenset(self._partitions.get(addr, ())))
            for pp, addr in stage_addrs.items()
        }
        with self._lock:
            self._requests[req_id] = _Request(
                nonce=nonce,
                dst_block_ids=np.asarray(dst_block_ids, dtype=np.int64),
                streams=streams,
                expected=expected,
                compute_event=event,
            )

    def tracks(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self._requests

    # -- notifications (listener thread) ---------------------------------------

    def on_ready(self, data: dict) -> None:
        req_id = data["request_id"]
        nonce = data.get("write_nonce", 0)
        pp_rank = data.get("pp_rank", 0)
        slot = int(data["slot"])
        seq = int(data["seq"])
        credit_addr = None
        with self._lock:
            request = self._requests.get(req_id)
            if request is None or request.nonce != nonce:
                # Retired or unknown: never scatter, but the slot's owner
                # still gets it back.
                credit_addr = self._slot_owner.get(slot)
                if (req_id, nonce) not in self._tombstones:
                    logger.warning(
                        "[PD-LANDING] READY for unknown req %s stage %d slot %d; "
                        "dropped",
                        req_id,
                        pp_rank,
                        slot,
                    )
            else:
                stream = request.streams.get(pp_rank)
                if stream is None or slot not in stream.slots:
                    logger.error(
                        "[PD-LANDING] req %s stage %d sent slot %d outside its "
                        "partition; failing the request",
                        req_id,
                        pp_rank,
                        slot,
                    )
                    self._fail_locked(request)
                elif seq in stream.seen:
                    pass  # duplicate; the first copy owns the credit
                elif request.failed or stream.done:
                    stream.seen.add(seq)
                    credit_addr = stream.stage_addr
                else:
                    stream.seen.add(seq)
                    request.pending += 1
                    self._queue.put(
                        _Task(
                            req_id,
                            nonce,
                            pp_rank,
                            slot,
                            [tuple(item) for item in data["items"]],
                        )
                    )
        if credit_addr is not None:
            self._send_credits(credit_addr, [slot])

    def stream_done(
        self,
        req_id: str,
        pp_rank: int,
        nonce: int,
        success: bool,
        landed_slots: list[int] | None,
    ) -> bool:
        """Record a stage's write-done; False if this request is not tracked.

        ``landed_slots`` lists the slot of every READY the stage sent, by
        ``seq`` (None from a producer that does not land). A READY that never
        arrived fails the request, and its slot goes back to the stage: the
        RDMA write finished before the write-done was sent.
        """
        outcome = None
        lost: list[int] = []
        credit_addr = None
        with self._lock:
            request = self._requests.get(req_id)
            if request is None:
                return (req_id, nonce) in self._tombstones
            if request.nonce != nonce:
                logger.error(
                    "[PD-LANDING] write-done nonce mismatch for req %s", req_id
                )
                return True
            stream = request.streams.get(pp_rank)
            if stream is None or stream.done:
                return True
            stream.done = True
            if landed_slots:
                lost = [
                    slot
                    for seq, slot in enumerate(landed_slots)
                    if seq not in stream.seen and slot in stream.slots
                ]
                credit_addr = stream.stage_addr
            if not success:
                self._fail_locked(request)
            elif lost:
                logger.error(
                    "[PD-LANDING] req %s stage %d landed %d slots but %d READY "
                    "never arrived; failing the request",
                    req_id,
                    pp_rank,
                    len(landed_slots),
                    len(lost),
                )
                self._fail_locked(request)
            outcome = self._settle_locked(req_id, request)
        if lost:
            self._send_credits(credit_addr, lost)
        if outcome is not None:
            self._finish(req_id, outcome)
        return True

    # -- sweep (main thread) ----------------------------------------------------

    def sweep(self) -> None:
        """Report failed requests whose other stages never ended; log stats."""
        now = time.monotonic()
        expired: list[str] = []
        with self._lock:
            for req_id, request in list(self._requests.items()):
                if (
                    request.failed
                    and not request.pending
                    and now - request.failed_at > self._fail_wait_s
                ):
                    self._retire_locked(req_id, request)
                    expired.append(req_id)
            for key, (at, _) in list(self._tombstones.items()):
                if now - at > _TOMBSTONE_TTL_S:
                    del self._tombstones[key]
            stats = None
            if now - self._stats_at >= _STATS_INTERVAL_S:
                stats, self._stats = self._stats, {
                    "slots": 0,
                    "bytes": 0,
                    "scatter_s": 0.0,
                    "batches": 0,
                }
                elapsed, self._stats_at = now - self._stats_at, now
        for req_id in expired:
            logger.error(
                "[PD-LANDING] req %s: stages still running %ds after a failure; "
                "reporting it failed",
                req_id,
                self._fail_wait_s,
            )
            self._finish(req_id, True)
        if stats is not None and stats["slots"]:
            logger.info(
                "[PD-LANDING] %.1fs: %d slots, %.2f GB landed (%.2f GB/s), "
                "%d scatter batches, scatter busy %.1f%%",
                elapsed,
                stats["slots"],
                stats["bytes"] / 1e9,
                stats["bytes"] / 1e9 / elapsed,
                stats["batches"],
                100 * stats["scatter_s"] / elapsed,
            )

    # -- scatter thread ---------------------------------------------------------

    def _run(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            try:
                self._process(self._drain(block=True))
            except Exception:
                # Never let the thread die: queued READY would wait forever.
                logger.exception("[PD-LANDING] scatter loop error")

    def _drain(self, block: bool) -> list[_Task]:
        """Every queued task, waiting for the first one if ``block``."""
        tasks: list[_Task] = []
        try:
            tasks.append(self._queue.get(block=block))
            while True:
                tasks.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return tasks

    def _process(self, tasks: list[_Task]) -> None:
        if not tasks:
            return
        try:
            bad = self._scatter(tasks)
        except Exception:
            logger.exception(
                "[PD-LANDING] scatter failed; failing %d slot(s) and "
                "disabling landing on this rank",
                len(tasks),
            )
            with self._lock:
                self._disabled = True
            bad = {t.req_id for t in tasks}
        # Exactly once per task, whatever the scatter did.
        self._complete(tasks, failed_ids=bad)

    def _scatter(self, tasks: list[_Task]) -> set[str]:
        """Scatter the tasks' slots; returns requests whose items were bad."""
        live: list[tuple[_Task, _Request]] = []
        events: list[torch.cuda.Event] = []
        with self._lock:
            for task in tasks:
                request = self._requests.get(task.req_id)
                if request is None or request.nonce != task.nonce or request.failed:
                    continue
                live.append((task, request))
                if not request.event_waited:
                    request.event_waited = True
                    if request.compute_event is not None:
                        events.append(request.compute_event)
        bad: set[str] = set()
        parts = []
        landed_bytes = 0
        for task, request in live:
            items = np.asarray(task.items, dtype=np.int64).reshape(-1, 4)
            region, row_start, row_count, offset = items.T
            rows = request.dst_block_ids.size * self.block_size
            if (
                (region < 0).any()
                or (region >= self._is_mla.size).any()
                or not self._is_mla[region].all()
                or (row_start < 0).any()
                or (row_start + row_count > rows).any()
                or (
                    offset
                    + row_count * self._region_block_bytes[region] // self.block_size
                    > self.slot_bytes
                ).any()
            ):
                logger.error(
                    "[PD-LANDING] req %s stage %d slot %d: items out of range",
                    task.req_id,
                    task.pp_rank,
                    task.slot,
                )
                bad.add(task.req_id)
                continue
            parts.append(
                landing_segments(
                    row_start,
                    row_count,
                    task.slot * self.slot_bytes + offset,
                    self._region_bases[region],
                    self._region_block_bytes[region],
                    request.dst_block_ids,
                    self.block_size,
                    self._dst_origin,
                )
            )
            landed_bytes += int(
                (row_count * self._region_block_bytes[region] // self.block_size).sum()
            )
        start = time.monotonic()
        if parts:
            src, dst, nbytes = (np.concatenate(p) for p in zip(*parts))
            self._copy_segments(src, dst, nbytes, events)
        with self._lock:
            self._stats["scatter_s"] += time.monotonic() - start
            self._stats["slots"] += len(live)
            self._stats["bytes"] += landed_bytes
            self._stats["batches"] += 1
        return bad

    def _copy_segments(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        nbytes: np.ndarray,
        events: list[torch.cuda.Event],
    ) -> None:
        """Copy pool bytes ``src`` to ``dst_origin + dst``; returns when done."""
        with torch.cuda.stream(self._stream):
            for event in events:
                self._stream.wait_event(event)
            origin = DevicePointer(self._dst_origin, self.device)
            table = self._table_host.numpy()
            for lo in range(0, src.size, self._max_segments):
                hi = min(src.size, lo + self._max_segments)
                n = segment_table(src[lo:hi], dst[lo:hi], nbytes[lo:hi], table)
                self._table_dev[:, :n].copy_(self._table_host[:, :n], non_blocking=True)
                scatter_segments(self.pool, origin, self._table_dev, n)
                if hi < src.size:
                    # The next chunk reuses the pinned table.
                    self._stream.synchronize()
        self._stream.synchronize()

    def _complete(self, tasks: list[_Task], failed_ids: set[str]) -> None:
        """Release the tasks' pending counts and credits; settle requests."""
        credits: dict[str, list[int]] = {}
        outcomes: list[tuple[str, bool]] = []
        with self._lock:
            touched: dict[str, _Request] = {}
            for task in tasks:
                request = self._requests.get(task.req_id)
                if request is None or request.nonce != task.nonce:
                    continue
                request.pending -= 1
                stream = request.streams[task.pp_rank]
                credits.setdefault(stream.stage_addr, []).append(task.slot)
                if task.req_id in failed_ids:
                    self._fail_locked(request)
                touched[task.req_id] = request
            for req_id, request in touched.items():
                outcome = self._settle_locked(req_id, request)
                if outcome is not None:
                    outcomes.append((req_id, outcome))
        for addr, slots in credits.items():
            try:
                self._send_credits(addr, slots)
            except Exception:
                logger.exception("[PD-LANDING] returning slots to %s failed", addr)
        for req_id, failed in outcomes:
            try:
                self._finish(req_id, failed)
            except Exception:
                logger.exception("[PD-LANDING] reporting req %s failed", req_id)

    # -- helpers (lock held) -------------------------------------------------

    def _fail_locked(self, request: _Request) -> None:
        if not request.failed:
            request.failed = True
            request.failed_at = time.monotonic()

    def _settle_locked(self, req_id: str, request: _Request) -> bool | None:
        """Retire the request once nothing is pending and every stage ended.

        Returns whether it failed, or None if it is still open.
        """
        if request.pending:
            return None
        if sum(s.done for s in request.streams.values()) < request.expected:
            return None
        self._retire_locked(req_id, request)
        return request.failed

    def _retire_locked(self, req_id: str, request: _Request) -> None:
        del self._requests[req_id]
        self._tombstones[(req_id, request.nonce)] = (
            time.monotonic(),
            {pp: s.stage_addr for pp, s in request.streams.items()},
        )

    def _send_credits(self, addr: str, slots: list[int]) -> None:
        payload = msgpack.dumps(
            {"consumer": self.consumer_key, "epoch": self.epoch, "slots": slots}
        )
        self._send(addr, [MSG_LANDING_CREDIT, payload])
