# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA landing for Mooncake P/D with a DCP decode (``ATOM_PD_MLA_LANDING``, default on).

A DCP decode rank owns every ``dcp_size``-th token of a prefill block, so its
MLA rows cannot be written as whole blocks. The staged path
(``_execute_staged_mla_regions``) sends one RDMA descriptor per destination
page, which caps a NIC at about 10 GB/s with fragmented decode block tables.
Landing sends a rank's rows packed in rank order instead: one descriptor per
slot of a decode-side GPU landing pool, which the decode rank then scatters
into its paged KV cache.

Protocol, per decode rank (every rank has its own pool and connector):

1. On first contact the rank gives a prefill stage endpoint a partition of
   ``pool_slots // pp_size`` slots (``pp_size`` is the prefill's), first
   come, first served, and takes it back only from a stage that reports it
   cannot land (no MLA staging pool: a DCP prefill, ``ATOM_PD_MLA_STAGING=0``
   or a build without landing); such an endpoint is not offered slots
   again. With several landing prefill instances per decode rank, a later
   one may get fewer slots, or none and keep the staged path. The rank
   advertises the stage's partition in every ``write_request``
   (``mla_landing = {epoch, base, slot_bytes, slots}``).
2. The stage keeps the partition as credits (``LandingCredits``), shared by
   all its requests to that rank. A send worker takes a credit, gathers rows
   into its staging slot, writes the slot to ``base + slot * slot_bytes`` and
   sends ``MSG_LANDING_READY`` (request, nonce, stage, seq, slot, items). With
   no credit within ``MLA_LANDING_CREDIT_WAIT_S`` it sends the rest
   of the transfer through the staged path. Write-done then lists the slot of
   every READY it sent (``landed_slots``), so a lost READY is detected and its
   slot still returned; a stage that cannot land sends ``None`` instead.
3. The rank (``LandingReceiver``) scatters READY slots on its own stream and
   returns their credits (``MSG_LANDING_CREDIT``) once the scatter finished.
4. A request completes when every stage's write-done arrived, every landed
   slot it announced was received, and every scatter finished. A failed
   request is reported only after all its stages ended: a stage still
   running may yet write the final KV pages directly (staged path, index
   regions), and reporting the failure lets the scheduler reuse them. A stage
   that acknowledged the request (``MSG_WRITE_ACK``) and stays silent past
   the write window the ack named has stopped writing (its transfer deadline
   started before the ack), so once every silent stage is past its window
   the request fails anyway. A stage that never acknowledged may still get
   the request, so it keeps the request open. READY slots that arrive after
   the request failed or their stage ended are dropped, so they never reach
   the KV cache. Each slot goes back
   once: after its scatter, with its READY if that is dropped, or with the
   write-done if its READY had not arrived by then. ZMQ keeps order only
   within one connection, so a READY can still trail its write-done after a
   reconnect; it then returns nothing. This holds while the write-done lists
   every READY's slot at its seq, as the producer sends it. A malformed list
   counts no READY lost, so a READY that arrives after its request retired
   leaves its slot in flight on the stage.

Credits never move between stages or ranks (a partition taken back comes
from a stage that never takes a credit), so no slot is reused while a stage
may still write it, and no side waits on another stage's or rank's progress.
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
from atom.kv_transfer.disaggregation.pd_producer import plain_mla_model
from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.utils import envs, get_hf_text_config

logger = logging.getLogger("atom")

MSG_LANDING_READY = b"landing_ready"
MSG_LANDING_CREDIT = b"landing_credit"

# Prefill: a transfer needing fewer landing slots than this keeps the staged
# per-page path; the scatter round trip would not pay off.
MLA_LANDING_MIN_SLOTS = 2
# Prefill: how long a send worker waits for a free landing slot before it
# sends the rest of the transfer through the staged per-page path.
MLA_LANDING_CREDIT_WAIT_S = 0.010
# Decode: one rank's landing pool, held back from the KV budget and handed
# out to prefill stages first come, first served (``advertise``).
MLA_LANDING_SLOT_BYTES = 8 << 20
MLA_LANDING_POOL_BYTES = 256 << 20
MLA_LANDING_POOL_SLOTS = MLA_LANDING_POOL_BYTES // MLA_LANDING_SLOT_BYTES

_STATS_INTERVAL_S = 30.0
# A request still waiting on a stage this long is logged once.
_STALL_LOG_S = 30.0
# A finished request is remembered this long so READY slots that arrive after
# it finished are dropped quietly; on_ready says why they return no credit.
_TOMBSTONE_TTL_S = 600.0


# ---------------------------------------------------------------------------
# Producer side
# ---------------------------------------------------------------------------


@dataclass
class _TargetCredits:
    epoch: int
    free: list[int]
    in_flight: set[int] = field(default_factory=set)
    # Slot -> time.monotonic() from which it is free again: a failed write
    # may still land in it until then (ATOM_PD_FAILED_WRITE_HOLD_S).
    quarantined: dict[int, float] = field(default_factory=dict)


class LandingCredits:
    """The landing slots this prefill stage may write, per decode rank.

    ``target`` is the decode rank's Mooncake session (``host:rpc_port``). A
    decode restart comes with a new ``epoch``, which resets the credits.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._targets: dict[str, _TargetCredits] = {}

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
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                credits = self._targets.get(target)
                if credits is None or credits.epoch != epoch:
                    return None
                if credits.quarantined:
                    self._unquarantine_locked(target, credits)
                if credits.free:
                    slot = credits.free.pop()
                    credits.in_flight.add(slot)
                    return slot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)

    def quarantine(self, target: str, epoch: int, slot: int) -> None:
        """Keep a slot out of use for a while after a write to it failed.

        Mooncake gives up on a sync write without cancelling the work it
        posted, so that write may still land in the slot. Another request
        writing and announcing the slot meanwhile would then be scattered
        with the wrong rows. The partition shrinks for
        ``ATOM_PD_FAILED_WRITE_HOLD_S``; with no slot left meanwhile, every
        transfer to ``target`` takes the staged path.
        """
        hold_s = envs.ATOM_PD_FAILED_WRITE_HOLD_S
        with self._cv:
            credits = self._targets.get(target)
            if credits is None or credits.epoch != epoch:
                return
            credits.in_flight.discard(slot)
            credits.quarantined[slot] = time.monotonic() + hold_s
            left = len(credits.free) + len(credits.in_flight)
        logger.warning(
            "[PD-LANDING] %s slot %d is out of use for %.0fs after a failed write; "
            "%d slots left meanwhile",
            target,
            slot,
            hold_s,
            left,
        )

    @staticmethod
    def _unquarantine_locked(target: str, credits: _TargetCredits) -> None:
        """Free the quarantined slots whose hold has ended."""
        now = time.monotonic()
        back = [slot for slot, until in credits.quarantined.items() if until <= now]
        for slot in back:
            del credits.quarantined[slot]
            credits.free.append(slot)
        if back:
            logger.info(
                "[PD-LANDING] %s slots %s are back in use after a failed write",
                target,
                back,
            )

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


def mla_landing_configured(config) -> bool:
    """Whether a Mooncake connector with ``config`` gets an MLA landing pool.

    ``config`` holds one Mooncake connector's ``kv_transfer_config`` (see
    ``KVConnectorFactory.kv_budget_reserve_bytes``). Only a ``kv_consumer``
    running DCP on a plain MLA model (``plain_mla_model``) lands. The reserve
    and the allocation in ``register_kv_caches`` both ask this, so no pool
    goes unreserved.
    """

    return bool(
        envs.ATOM_PD_MLA_LANDING
        and getattr(config, "decode_context_parallel_size", 1) > 1
        and config.kv_transfer_config.get("kv_role") == "kv_consumer"
        and plain_mla_model(config)
    )


def mla_landing_table_segments(config) -> int:
    """Segments a landing rank's scatter table holds, from ``config`` alone.

    One per destination page of a full pool, plus 64 item splits per slot.
    An MLA row takes at least ``kv_lora_rank`` bytes, so a page takes at
    least ``block_size * kv_lora_rank``. A batch needing more segments is
    copied in several chunks.
    """

    kv_lora_rank = get_hf_text_config(config.hf_config).kv_lora_rank
    page_bytes = config.kv_cache_block_size * kv_lora_rank
    pages_per_slot = MLA_LANDING_SLOT_BYTES // page_bytes
    return segment_table_capacity(MLA_LANDING_POOL_SLOTS * (pages_per_slot + 64))


def mla_landing_reserve_bytes(config) -> int:
    """HBM to hold back from the KV budget for this rank's MLA landing.

    The pool plus the scatter's int64 ``[3, segments]`` device table, both
    allocated in ``register_kv_caches`` after the KV cache is sized.
    """

    if not mla_landing_configured(config):
        return 0
    table_bytes = 3 * 8 * mla_landing_table_segments(config)
    return MLA_LANDING_POOL_SLOTS * MLA_LANDING_SLOT_BYTES + table_bytes


@dataclass
class _StageTransfer:
    """One prefill stage's transfer of one request to this rank."""

    stage_addr: str
    slots: frozenset[int]
    # READY seq -> the slot that READY named, or that the write-done listed
    # when it counted the READY lost.
    seen: dict[int, int] = field(default_factory=dict)
    done: bool = False
    # When the stage's write window ends (time.monotonic()), from its ack;
    # None until the stage acknowledged the request.
    window_end: float | None = None


@dataclass
class _Request:
    nonce: int
    dst_block_ids: np.ndarray
    stages: dict[int, _StageTransfer]
    # None once a scatter has waited on it.
    compute_event: torch.cuda.Event | None
    # When its write requests went out (time.monotonic()).
    started_at: float
    pending: int = 0
    failed: bool = False
    # Failed once its silent stages overran their windows (at ``expired_at``):
    # reported without waiting for them, once nothing is pending.
    expired: bool = False
    expired_at: float = 0.0
    # The connector has heard the outcome; still tracked while a stage that
    # may yet report holds slots, so its late messages return them.
    reported: bool = False
    stall_logged: bool = False


@dataclass
class _Task:
    req_id: str
    nonce: int
    pp_rank: int
    slot: int
    items: np.ndarray


def _ready_index(value) -> int:
    """A READY's ``slot`` or ``seq``: a non-negative int, else ``ValueError``.

    The listener drops a READY that raises here as unreadable, and its stage's
    write-done then counts it lost. A negative ``seq`` must never get further:
    ``stage_done`` reads ``landed_slots[seq]``, which would index from the end
    and match another READY's slot.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"landing READY index {value!r} is not a non-negative int")
    return value


def _ready_items(data: dict) -> np.ndarray | None:
    """A READY's items as an int64 ``(n, 4)`` array, n >= 1; None if malformed."""
    try:
        items = np.asarray(data["items"], dtype=np.int64)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    # msgpack never yields a (0, 4) array ([] is 1-D); rejected all the same.
    if items.ndim != 2 or items.shape[1] != 4 or not items.shape[0]:
        return None
    return items


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
        table_segments: int,
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
        # Reserved with the pool (``mla_landing_table_segments``).
        self._max_segments = table_segments
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
        self._unassigned = list(range(pool_slots))
        # Stage endpoints that reported they cannot land.
        self._cannot_land: set[str] = set()
        self._warned_spent = False
        self._requests: dict[str, _Request] = {}
        self._tombstones: dict[tuple[str, int], float] = {}
        self._queue: queue.SimpleQueue[_Task] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._disabled = False
        self._stats = {"slots": 0, "bytes": 0, "scatter_s": 0.0, "batches": 0}
        self._stats_at = time.monotonic()
        # Credits whose send raised, by stage endpoint; sweep sends them again.
        self._unsent_credits: dict[str, list[int]] = {}

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

    # -- dispatch (main thread) ------------------------------------------------

    def advertise(self, stage_addr: str, num_stages: int) -> dict | None:
        """The ``mla_landing`` field for a write request to ``stage_addr``.

        The first request to a stage endpoint assigns it a partition of
        ``pool_slots // num_stages`` slots; None once the pool is spent, for
        an endpoint that cannot land, or when landing is disabled (a scatter
        error).
        """
        with self._lock:
            if self._disabled or stage_addr in self._cannot_land:
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
    ) -> None:
        """Track a request before any write request for it goes out.

        ``stage_addrs`` maps every prefill stage that writes the request to
        its endpoint; the request settles only after each one's write-done.
        Records an event on the compute stream: the request's fresh pages may
        still be read by their previous owner's queued forward, and the
        scatter must not overwrite them before that finished.
        """
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        stages = {
            pp: _StageTransfer(addr, frozenset(self._partitions.get(addr, ())))
            for pp, addr in stage_addrs.items()
        }
        with self._lock:
            # Insertion order is start order: sweep walks the oldest first.
            self._requests[req_id] = _Request(
                nonce=nonce,
                dst_block_ids=np.asarray(dst_block_ids, dtype=np.int64),
                stages=stages,
                compute_event=event,
                started_at=time.monotonic(),
            )

    # -- notifications (listener thread) ---------------------------------------

    def stage_acked(
        self, req_id: str, pp_rank: int, nonce: int, window_end: float
    ) -> bool:
        """Record a stage's ack and when its write window ends.

        False if this request is not tracked here.
        """
        with self._lock:
            request = self._requests.get(req_id)
            if request is None:
                return (req_id, nonce) in self._tombstones
            stage = request.stages.get(pp_rank)
            if request.nonce == nonce and stage is not None:
                stage.window_end = window_end
            return True

    def on_ready(self, data: dict) -> None:
        req_id = data["request_id"]
        nonce = data.get("write_nonce", 0)
        pp_rank = data.get("pp_rank", 0)
        slot = _ready_index(data["slot"])
        seq = _ready_index(data["seq"])
        # Malformed items fail only their request (below), never the listener
        # thread or the scatter batch.
        items = _ready_items(data)
        credit_addr = None
        with self._lock:
            request = self._requests.get(req_id)
            if request is None or request.nonce != nonce:
                # Retired or unknown: never scatter, and no credit goes back.
                # A READY can trail its write-done across a reconnect, but
                # each READY of a retired request was seen before its stage's
                # write-done or counted lost by it, so its slot went back
                # already, unless that write-done did not list it (a malformed
                # list counts none lost): the slot then stays in flight on the
                # stage. A READY of a request this rank never knew belongs to
                # no transfer it tracks.
                if (req_id, nonce) not in self._tombstones:
                    logger.warning(
                        "[PD-LANDING] READY for unknown req %s stage %d slot %d; "
                        "dropped",
                        req_id,
                        pp_rank,
                        slot,
                    )
            else:
                stage = request.stages.get(pp_rank)
                if stage is None or slot not in stage.slots:
                    logger.error(
                        "[PD-LANDING] req %s stage %d sent slot %d outside its "
                        "partition; failing the request",
                        req_id,
                        pp_rank,
                        slot,
                    )
                    self._fail_locked(request)
                elif seq in stage.seen:
                    # A duplicate: the first copy, or the write-done that
                    # counted it lost, owns the credit.
                    if stage.seen[seq] != slot:
                        logger.error(
                            "[PD-LANDING] req %s stage %d seq %d names slot %d, "
                            "not %d; failing the request",
                            req_id,
                            pp_rank,
                            seq,
                            slot,
                            stage.seen[seq],
                        )
                        self._fail_locked(request)
                elif request.failed or stage.done:
                    if not request.failed:
                        # Its stage's write-done ended the stage without it.
                        logger.error(
                            "[PD-LANDING] req %s stage %d seq %d is missing from "
                            "its write-done; failing the request",
                            req_id,
                            pp_rank,
                            seq,
                        )
                        self._fail_locked(request)
                    stage.seen[seq] = slot
                    credit_addr = stage.stage_addr
                elif items is None:
                    logger.error(
                        "[PD-LANDING] req %s stage %d slot %d: malformed items; "
                        "failing the request",
                        req_id,
                        pp_rank,
                        slot,
                    )
                    stage.seen[seq] = slot
                    self._fail_locked(request)
                    # The slot's RDMA write finished before READY: return it now.
                    credit_addr = stage.stage_addr
                else:
                    stage.seen[seq] = slot
                    request.pending += 1
                    self._queue.put(
                        _Task(
                            req_id,
                            nonce,
                            pp_rank,
                            slot,
                            items,
                        )
                    )
        if credit_addr is not None:
            self._send_credits(credit_addr, [slot])

    def stage_done(
        self,
        req_id: str,
        pp_rank: int,
        nonce: int,
        success: bool,
        landed_slots: list[int] | None,
    ) -> bool:
        """Record a stage's write-done; False if this request is not tracked.

        ``landed_slots`` lists the slot of every READY the stage sent, by
        ``seq``. A READY that never arrived fails the request, and its slot
        goes back to the stage: the RDMA write finished before the write-done
        was sent. A list that names a slot outside the stage's partition, or
        that disagrees with the READYs that arrived, fails the request and
        returns nothing. ``None`` comes from a stage that cannot land, which
        never takes a credit, so its partition goes back to the pool.
        """
        outcome = None
        lost: dict[int, int] = {}
        reclaimed = 0
        cannot_land = landed_slots is None
        with self._lock:
            request = self._requests.get(req_id)
            if request is None:
                return (req_id, nonce) in self._tombstones
            if request.nonce != nonce:
                logger.error(
                    "[PD-LANDING] write-done nonce mismatch for req %s", req_id
                )
                return True
            stage = request.stages.get(pp_rank)
            if stage is None or stage.done:
                return True
            # Every check runs before the stage ends, so none can leave it
            # ended with the request unsettled.
            if landed_slots is not None and not (
                isinstance(landed_slots, list)
                and all(
                    isinstance(slot, int)
                    and not isinstance(slot, bool)
                    and slot in stage.slots
                    for slot in landed_slots
                )
            ):
                # Which READY never arrived is unknown: count a failed write-done.
                logger.error(
                    "[PD-LANDING] req %s stage %d sent malformed landed slots; "
                    "failing the request",
                    req_id,
                    pp_rank,
                )
                landed_slots, success = None, False
            elif any(
                seq >= len(landed_slots or ()) or landed_slots[seq] != slot
                for seq, slot in stage.seen.items()
            ):
                # A READY the list does not name: trust neither, return nothing.
                logger.error(
                    "[PD-LANDING] req %s stage %d: its write-done and its READYs "
                    "disagree on the landed slots; failing the request",
                    req_id,
                    pp_rank,
                )
                success = False
            elif landed_slots:
                lost = {
                    seq: slot
                    for seq, slot in enumerate(landed_slots)
                    if seq not in stage.seen
                }
            stage.done = True
            # Returned here: a lost READY that a reconnect reordered behind
            # this write-done is then a duplicate.
            stage.seen.update(lost)
            if cannot_land and stage.slots and not stage.seen:
                reclaimed = self._reclaim_locked(stage.stage_addr)
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
        if reclaimed:
            logger.warning(
                "[PD-LANDING] stage %s cannot land (a DCP prefill, "
                "ATOM_PD_MLA_STAGING=0 or a build without landing); its %d "
                "slots go back to the pool and it keeps staged per-page writes",
                stage.stage_addr,
                reclaimed,
            )
        if lost:
            self._send_credits(stage.stage_addr, list(lost.values()))
        if outcome is not None:
            self._finish(req_id, outcome)
        return True

    # -- sweep (main thread) ----------------------------------------------------

    def sweep(self) -> None:
        """Fail requests whose silent stages overran their windows; log stats.

        A failed request is otherwise never reported early: only a stage's
        write-done proves its writes into the request's KV pages have ended,
        and an acknowledged stage's write window proves the same once it
        ended. A request waiting on a stage longer than ``_STALL_LOG_S`` is
        logged once. Tombstones are dropped oldest first.
        """
        now = time.monotonic()
        stalled: list[tuple[str, float, list[int], list[int]]] = []
        expired: list[tuple[str, list[int]]] = []
        abandoned: list[tuple[str, list[int]]] = []
        outcomes: list[tuple[str, bool]] = []
        with self._lock:
            for req_id, request in list(self._requests.items()):
                if request.expired:
                    # Reported, or waiting for a scatter (_complete settles it).
                    # A stage still silent this long after has died or lost its
                    # messages: whatever slots it holds stay in flight.
                    if request.reported and now - request.expired_at > (
                        _TOMBSTONE_TTL_S
                    ):
                        self._retire_locked(req_id, request)
                        abandoned.append(
                            (
                                req_id,
                                [pp for pp, s in request.stages.items() if not s.done],
                            )
                        )
                    continue
                silent = {pp: s for pp, s in request.stages.items() if not s.done}
                if silent and all(
                    s.window_end is not None and s.window_end < now
                    for s in silent.values()
                ):
                    request.expired, request.expired_at = True, now
                    self._fail_locked(request)
                    expired.append((req_id, list(silent)))
                    outcome = self._settle_locked(req_id, request)
                    if outcome is not None:
                        outcomes.append((req_id, outcome))
                elif (
                    not request.stall_logged and now - request.started_at > _STALL_LOG_S
                ):
                    request.stall_logged = True
                    unacked = [pp for pp, s in silent.items() if s.window_end is None]
                    stalled.append(
                        (req_id, now - request.started_at, list(silent), unacked)
                    )
            stale = []
            for key, at in self._tombstones.items():
                if now - at <= _TOMBSTONE_TTL_S:
                    break
                stale.append(key)
            for key in stale:
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
        for req_id, waited, running, unacked in stalled:
            (logger.error if unacked else logger.warning)(
                "[PD-LANDING] req %s has waited %.0fs for stage(s) %s%s",
                req_id,
                waited,
                running,
                (
                    f"; stage(s) {unacked} never acknowledged it, so it waits for "
                    "them without a deadline and its pages stay held"
                    if unacked
                    else ""
                ),
            )
        for req_id, silent in expired:
            logger.error(
                "[PD-LANDING] req %s: stage(s) %s acknowledged it but sent no "
                "write-done within their write window; failing it",
                req_id,
                silent,
            )
        for req_id, silent in abandoned:
            logger.error(
                "[PD-LANDING] req %s: stage(s) %s never reported after it "
                "failed; the landing slots they hold stay out of use",
                req_id,
                silent,
            )
        for req_id, failed in outcomes:
            try:
                self._finish(req_id, failed)
            except Exception:
                # The other requests still have to be reported.
                logger.exception("[PD-LANDING] reporting req %s failed", req_id)
        with self._lock:
            unsent, self._unsent_credits = self._unsent_credits, {}
        for addr, slots in unsent.items():
            self._send_credits(addr, slots)  # queued again if it raises
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
                if request.compute_event is not None:
                    events.append(request.compute_event)
                    request.compute_event = None
        bad: set[str] = set()
        parts = []
        landed_bytes = 0
        for task, request in live:
            items = np.asarray(task.items, dtype=np.int64).reshape(-1, 4)
            region, row_start, row_count, offset = items.T
            rows = request.dst_block_ids.size * self.block_size
            # Upper bounds subtract from local sizes rather than add READY
            # values: such a sum can wrap past int64 and pass.
            if (
                (region < 0).any()
                or (region >= self._is_mla.size).any()
                or not self._is_mla[region].all()
                or (row_start < 0).any()
                or (row_count <= 0).any()
                or (row_count > rows - row_start).any()
                # The scatter copies 4-byte words of the slot.
                or (offset < 0).any()
                or (offset % 4).any()
                or (
                    row_count * self._region_block_bytes[region] // self.block_size
                    > self.slot_bytes - offset
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
                # Row by row: each row slice is contiguous, so the copy reads
                # the pinned table directly. A [:, :n] slice is not, and copy_
                # would stage it through a pageable host buffer, synchronously.
                for row in range(3):
                    self._table_dev[row, :n].copy_(
                        self._table_host[row, :n], non_blocking=True
                    )
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
                stage = request.stages[task.pp_rank]
                credits.setdefault(stage.stage_addr, []).append(task.slot)
                if task.req_id in failed_ids:
                    self._fail_locked(request)
                touched[task.req_id] = request
            for req_id, request in touched.items():
                outcome = self._settle_locked(req_id, request)
                if outcome is not None:
                    outcomes.append((req_id, outcome))
        for addr, slots in credits.items():
            self._send_credits(addr, slots)
        for req_id, failed in outcomes:
            try:
                self._finish(req_id, failed)
            except Exception:
                logger.exception("[PD-LANDING] reporting req %s failed", req_id)

    # -- helpers (lock held) -------------------------------------------------

    def _reclaim_locked(self, stage_addr: str) -> int:
        """Return a stage endpoint's partition to the pool; slots moved.

        Only for an endpoint that cannot land: it never acquires a credit,
        so none of its slots can be in flight.
        """
        if stage_addr in self._cannot_land:
            return 0
        self._cannot_land.add(stage_addr)
        slots = self._partitions.pop(stage_addr, [])
        self._unassigned.extend(slots)
        return len(slots)

    def _fail_locked(self, request: _Request) -> None:
        request.failed = True

    def _settle_locked(self, req_id: str, request: _Request) -> bool | None:
        """Report and retire the request when it is ready; the outcome to report.

        A request is reported once nothing is pending and every stage ended,
        or, expired, once nothing is pending. An expired request stays
        tracked until its silent stages end too, so their late READYs and
        write-done still return the slots they name, the way they would for
        a request still open. Returns whether it failed when the connector
        is to hear of it now, else None.
        """
        if request.pending:
            return None
        ended = all(s.done for s in request.stages.values())
        if request.reported:
            if ended:
                self._retire_locked(req_id, request)
            return None
        if not (ended or request.expired):
            return None
        request.reported = True
        if ended:
            self._retire_locked(req_id, request)
        return request.failed

    def _retire_locked(self, req_id: str, request: _Request) -> None:
        del self._requests[req_id]
        # Insertion order is expiry order: sweep drops the oldest first.
        self._tombstones[(req_id, request.nonce)] = time.monotonic()

    def _send_credits(self, addr: str, slots: list[int]) -> None:
        payload = msgpack.dumps(
            {"consumer": self.consumer_key, "epoch": self.epoch, "slots": slots}
        )
        try:
            self._send(addr, [MSG_LANDING_CREDIT, payload])
        except Exception:
            # Not raised: callers send after their bookkeeping is done. A send
            # that raised queued nothing, so sweep sends these slots again;
            # until then the stage holds them in flight.
            with self._lock:
                first = addr not in self._unsent_credits
                self._unsent_credits.setdefault(addr, []).extend(slots)
            if first:
                logger.exception(
                    "[PD-LANDING] returning slots to %s failed; retrying", addr
                )
