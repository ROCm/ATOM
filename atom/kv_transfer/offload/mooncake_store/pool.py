# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The registered memory every Store transfer of one worker goes through.

One allocation, registered with the Store client once, cut into chunk-sized
slots in two regions: ``save`` (GPU pack -> slot -> put) and ``load`` (get ->
slot -> GPU unpack). A slot holds bytes only while a transfer is in flight;
nothing is cached here. Each slot exposes ``.tensor``, a contiguous uint8 view
of exactly one chunk, so the block GPU connector copies to and from it as it
would any staged object.

On the GPU (the default) the NIC reads and writes HBM directly and the copies
on either side of it are device-to-device. ``pool_device: cpu`` puts the pool
in pinned host memory instead -- the same code, with D2H/H2D copies -- which
only suits a small pool: an ionic NIC registers about 3 GiB of 4 KiB pages,
shared by every process on it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import torch

logger = logging.getLogger("atom")

REGIONS = ("save", "load")
# A slot starts on a page; the pool's base on a huge page, so splitting the
# registration into MC_MAX_MR_SIZE pieces never cuts one.
_SLOT_ALIGN = 4096
_BASE_ALIGN = 2 << 20
# A region's usable slots make at least this many windows, however few threads
# share it: when one transfer never settles and its slots are held back, the
# rest of the region keeps working.
_MIN_WINDOWS_PER_REGION = 4


class SlotPoolExhausted(RuntimeError):
    """A region has fewer usable slots than a request needs.

    Not worth waiting for: slots held back by ``quarantine`` return minutes
    later at the earliest, and retired ones never do.
    """


class Slot:
    """One chunk-sized piece of the registered pool."""

    __slots__ = ("held_until", "index", "leased", "ptr", "region", "tensor")

    def __init__(self, region: str, index: int, tensor: torch.Tensor) -> None:
        self.region = region
        self.index = index
        # Exactly one chunk's bytes; `memory_object_as_uint8` accepts it as is.
        self.tensor = tensor
        self.ptr = int(tensor.data_ptr())
        self.leased = False
        # time.monotonic() at which a held slot returns to its region.
        self.held_until = 0.0

    def __repr__(self) -> str:
        return f"Slot({self.region}#{self.index}@{self.ptr:#x})"


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // multiple) * multiple


def _allocate_pool_tensor(nbytes: int, device: torch.device) -> torch.Tensor:
    """The pool's backing memory: HBM on a GPU device, else pinned host memory."""
    if device.type == "cpu":
        return torch.empty((nbytes,), dtype=torch.uint8, pin_memory=True)
    return torch.empty((nbytes,), dtype=torch.uint8, device=device)


class TransferSlotPool:
    """Chunk slots of one registered allocation, handed out per transfer window.

    ``acquire`` blocks until a region has the slots free; a caller asking for
    more than the region can hand out (``quarantine`` shrinks it) gets
    :class:`SlotPoolExhausted` instead of waiting. Callers that keep their
    requests within ``window(region, threads)`` and release one window before
    asking for the next never wait on each other for long, and never deadlock.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        chunk_bytes: int,
        save_bytes: int,
        load_bytes: int,
        client: Any,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_bytes = int(chunk_bytes)
        if self.chunk_bytes <= 0:
            raise ValueError("transfer pool chunk size must be positive")
        self.slot_stride = _round_up(self.chunk_bytes, _SLOT_ALIGN)
        capacity = {
            "save": int(save_bytes) // self.slot_stride,
            "load": int(load_bytes) // self.slot_stride,
        }
        for region, knob in (
            ("save", "mooncake_store.save_pool_mib"),
            ("load", "mooncake_store.load_pool_mib"),
        ):
            if capacity[region] < 1:
                raise ValueError(
                    f"{knob} is smaller than one {self.chunk_bytes}-byte chunk "
                    f"(slot stride {self.slot_stride})"
                )
        self._capacity = capacity
        self.nbytes = (capacity["save"] + capacity["load"]) * self.slot_stride
        backing = _allocate_pool_tensor(self.nbytes + _BASE_ALIGN, self.device)
        offset = (-int(backing.data_ptr())) % _BASE_ALIGN
        self._backing = backing
        self._buffer = backing[offset : offset + self.nbytes]
        self.base_ptr = int(self._buffer.data_ptr())
        # Register before any slot is handed out: a put from an unregistered
        # address fails every time.
        client.register(self.base_ptr, self.nbytes)
        self._client = client
        self._slots: dict[str, list[Slot]] = {}
        position = 0
        for region in REGIONS:
            slots = []
            for index in range(capacity[region]):
                view = self._buffer[position : position + self.chunk_bytes]
                slots.append(Slot(region, index, view))
                position += self.slot_stride
            self._slots[region] = slots
        self._free = {region: deque(self._slots[region]) for region in REGIONS}
        # Out of use: held back until `Slot.held_until`, or retired for good.
        self._held: dict[str, list[Slot]] = {region: [] for region in REGIONS}
        self._retired = dict.fromkeys(REGIONS, 0)
        # Regions whose every slot is out of use, reported once until one returns.
        self._exhausted: set[str] = set()
        self._cond = threading.Condition()
        self._closed = False

    # -- geometry ----------------------------------------------------------
    def capacity(self, region: str) -> int:
        """Slots the region was built with."""
        return self._capacity[self._region(region)]

    def usable(self, region: str) -> int:
        """Slots the region can hand out now: capacity minus quarantined."""
        region = self._region(region)
        with self._cond:
            self._readmit_locked()
            return self._capacity[region] - self._quarantined_locked(region)

    def quarantined(self, region: str) -> int:
        """Slots of the region out of use now, held back or retired."""
        region = self._region(region)
        with self._cond:
            self._readmit_locked()
            return self._quarantined_locked(region)

    def window(self, region: str, n_threads: int) -> int:
        """Slots one of ``n_threads`` threads may hold at once in ``region``.

        At most a quarter of the usable slots, however few threads there are
        (see ``_MIN_WINDOWS_PER_REGION``).
        """
        shares = max(_MIN_WINDOWS_PER_REGION, int(n_threads))
        return max(1, self.usable(region) // shares)

    # -- lifecycle of a slot --------------------------------------------
    def acquire(self, region: str, n: int) -> list[Slot]:
        """Take ``n`` slots of ``region``, waiting until they are free.

        Raises:
            ValueError: ``n`` is not positive or exceeds the region.
            SlotPoolExhausted: Quarantine leaves the region fewer than ``n``.
            RuntimeError: The pool is closed.
        """
        region = self._region(region)
        n = int(n)
        if n <= 0 or n > self._capacity[region]:
            raise ValueError(
                f"cannot acquire {n} slots of the {self._capacity[region]}-slot "
                f"{region} region"
            )
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("the transfer pool is closed")
                self._readmit_locked()
                quarantined = self._quarantined_locked(region)
                usable = self._capacity[region] - quarantined
                if n > usable:
                    raise SlotPoolExhausted(
                        f"the {region} region has {usable} usable slots of "
                        f"{self._capacity[region]} ({quarantined} quarantined); "
                        f"{n} requested"
                    )
                free = self._free[region]
                if len(free) >= n:
                    slots = [free.popleft() for _ in range(n)]
                    for slot in slots:
                        slot.leased = True
                    return slots
                # A release wakes this; so must a held slot coming back.
                self._cond.wait(self._seconds_to_readmit_locked(region))

    def release(self, slots: list[Slot]) -> None:
        """Return slots no GPU or NIC access can still touch."""
        if not slots:
            return
        with self._cond:
            for slot in slots:
                self._require_leased(slot)
                slot.leased = False
                self._free[slot.region].append(slot)
            self._cond.notify_all()

    def quarantine(
        self,
        slots: list[Slot],
        *,
        reason: str = "last access unconfirmed",
        hold_s: float | None = None,
    ) -> None:
        """Take slots out of use: their last GPU or NIC access is unconfirmed.

        Reusing one could let a late copy or RDMA transfer of the old chunk
        land in, or read from, the next chunk's bytes. With ``hold_s`` the
        slots return to their region that many seconds from now, which must
        bound how late such an access can come; without it they are retired
        for good. A healthy device and fabric never get here; an unhealthy one
        shrinks the pool instead of corrupting what passes through it.
        """
        if not slots:
            return
        now = time.monotonic()
        with self._cond:
            for slot in slots:
                self._require_leased(slot)
                slot.leased = False
                if hold_s is None:
                    self._retired[slot.region] += 1
                else:
                    slot.held_until = now + float(hold_s)
                    self._held[slot.region].append(slot)
            counts = {region: self._quarantined_locked(region) for region in REGIONS}
            # Region -> whether a held slot will come back to it.
            exhausted = {
                region: bool(self._held[region])
                for region in REGIONS
                if counts[region] == self._capacity[region]
                and region not in self._exhausted
            }
            self._exhausted.update(exhausted)
            self._cond.notify_all()
        logger.warning(
            "Mooncake Store offload: quarantined %d transfer slot(s) %s (%s); "
            "out of use now: save %d/%d, load %d/%d",
            len(slots),
            "for good" if hold_s is None else f"for {float(hold_s):.0f}s",
            reason,
            counts["save"],
            self._capacity["save"],
            counts["load"],
            self._capacity["load"],
        )
        for region, returning in exhausted.items():
            logger.error(
                "Mooncake Store offload: every %s slot of this worker is out of "
                "use; its %ss fail at once %s",
                region,
                region,
                "until a held slot returns" if returning else "for good",
            )

    def close(self) -> None:
        """Unregister the pool from the Store client and drop its memory."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            leased = sum(
                1 for slots in self._slots.values() for slot in slots if slot.leased
            )
            self._cond.notify_all()
        if leased:
            logger.warning(
                "Mooncake Store offload: closing the transfer pool with %d slot(s) "
                "still in use",
                leased,
            )
        try:
            self._client.unregister(self.base_ptr)
        finally:
            self._slots = {region: [] for region in REGIONS}
            self._free = {region: deque() for region in REGIONS}
            self._buffer = None
            self._backing = None

    def _region(self, region: str) -> str:
        if region not in REGIONS:
            raise ValueError(f"unknown transfer pool region {region!r}")
        return region

    def _quarantined_locked(self, region: str) -> int:
        return self._retired[region] + len(self._held[region])

    def _readmit_locked(self) -> None:
        """Return every held slot whose hold is over to its region's free list."""
        now = time.monotonic()
        for region in REGIONS:
            held = self._held[region]
            if not held or min(slot.held_until for slot in held) > now:
                continue
            back = [slot for slot in held if slot.held_until <= now]
            self._held[region] = [slot for slot in held if slot.held_until > now]
            self._free[region].extend(back)
            self._exhausted.discard(region)
            self._cond.notify_all()
            logger.info(
                "Mooncake Store offload: %d held %s slot(s) back in use; out of "
                "use now: %d/%d",
                len(back),
                region,
                self._quarantined_locked(region),
                self._capacity[region],
            )

    def _seconds_to_readmit_locked(self, region: str) -> float | None:
        """Until the region's next held slot returns; None if none is held."""
        held = self._held[region]
        if not held:
            return None
        return max(0.0, min(slot.held_until for slot in held) - time.monotonic())

    @staticmethod
    def _require_leased(slot: Slot) -> None:
        if not slot.leased:
            raise RuntimeError(f"{slot!r} is not held by a transfer")
