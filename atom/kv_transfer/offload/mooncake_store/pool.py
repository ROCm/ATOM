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
from collections import deque
from itertools import pairwise
from typing import Any

import torch

logger = logging.getLogger("atom")

REGIONS = ("save", "load")
# A slot starts on a page; the pool's base on a huge page, so splitting the
# registration into MC_MAX_MR_SIZE pieces never cuts one.
_SLOT_ALIGN = 4096
_BASE_ALIGN = 2 << 20
# A region's usable slots make at least this many windows, however few threads
# share it: when one transfer never settles and its slots are quarantined, the
# rest of the region keeps working.
_MIN_WINDOWS_PER_REGION = 4

# Backing memory of closed pools with a quarantined or still-leased slot. Kept
# registered and referenced for the life of the process: a late RDMA transfer
# or GPU copy then lands in memory nothing else is given. Each closed pool adds
# at most one entry; a healthy device and fabric add none.
_UNSETTLED_ALLOCATIONS: list[torch.Tensor] = []


class SlotPoolExhausted(RuntimeError):
    """A region has fewer usable slots than a request needs.

    Not worth waiting for: a quarantined slot never returns.
    """


class Slot:
    """One chunk-sized piece of the registered pool."""

    __slots__ = ("index", "leased", "ptr", "region", "tensor")

    def __init__(self, region: str, index: int, tensor: torch.Tensor) -> None:
        self.region = region
        self.index = index
        # Exactly one chunk's bytes; `memory_object_as_uint8` accepts it as is.
        self.tensor = tensor
        self.ptr = int(tensor.data_ptr())
        self.leased = False

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
        # Slots out of use for good, per region; see `quarantine`.
        self._quarantined = dict.fromkeys(REGIONS, 0)
        # Regions whose every slot is quarantined, reported once.
        self._exhausted: set[str] = set()
        self._cond = threading.Condition()
        self._closed = False

    # -- geometry ----------------------------------------------------------
    def capacity(self, region: str) -> int:
        """Slots the region was built with."""
        return self._capacity[self._region(region)]

    def usable(self, region: str) -> int:
        """Slots the region can hand out: capacity minus quarantined."""
        region = self._region(region)
        with self._cond:
            return self._capacity[region] - self._quarantined[region]

    def quarantined(self, region: str) -> int:
        """Slots of the region out of use for good."""
        region = self._region(region)
        with self._cond:
            return self._quarantined[region]

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
                quarantined = self._quarantined[region]
                usable = self._capacity[region] - quarantined
                if n > usable:
                    raise SlotPoolExhausted(
                        f"the {region} region has {usable} usable slots of "
                        f"{self._capacity[region]} ({quarantined} quarantined); "
                        f"{n} requested"
                    )
                if len(self._free[region]) >= n:
                    slots = self._take_locked(region, n)
                    for slot in slots:
                        slot.leased = True
                    return slots
                # A release wakes this, and so does a quarantine, which can
                # leave the region too small for `n`.
                self._cond.wait()

    def contiguous_view(self, slots: list[Slot]) -> torch.Tensor | None:
        """``slots`` as one chunk-major buffer, or None when they are not one.

        Slots in index order whose bytes follow one another with no padding
        between them -- a chunk size that is a page multiple, as every
        GLM-5.2 stage's is -- are a buffer the codec packs or unpacks in one
        call, with no staging copy.
        """
        if not slots or self.slot_stride != self.chunk_bytes:
            return None
        first = slots[0]
        for previous, slot in pairwise(slots):
            if slot.region != first.region or slot.index != previous.index + 1:
                return None
        offset = first.ptr - self.base_ptr
        return self._buffer[offset : offset + len(slots) * self.chunk_bytes]

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
        self, slots: list[Slot], *, reason: str = "last access unconfirmed"
    ) -> None:
        """Take slots out of use for good: their last GPU or NIC access is unconfirmed.

        Reusing one could let a late copy or RDMA transfer of the old chunk
        land in, or read from, the next chunk's bytes. Nothing bounds how late
        that can come: Mooncake gives up on a transfer without cancelling its
        RDMA work, and a NIC whose completion queue stalls holds that work
        indefinitely. A healthy device and fabric never get here; an unhealthy
        one shrinks the pool instead of corrupting what passes through it, and
        `close` keeps the memory of a pool with quarantined slots.
        """
        if not slots:
            return
        with self._cond:
            for slot in slots:
                self._require_leased(slot)
                slot.leased = False
                self._quarantined[slot.region] += 1
            counts = dict(self._quarantined)
            exhausted = [
                region
                for region in REGIONS
                if counts[region] == self._capacity[region]
                and region not in self._exhausted
            ]
            self._exhausted.update(exhausted)
            self._cond.notify_all()
        logger.warning(
            "Mooncake Store offload: quarantined %d transfer slot(s) for good "
            "(%s); out of use now: save %d/%d, load %d/%d",
            len(slots),
            reason,
            counts["save"],
            self._capacity["save"],
            counts["load"],
            self._capacity["load"],
        )
        for region in exhausted:
            logger.error(
                "Mooncake Store offload: every %s slot of this worker is out of "
                "use; its %ss fail at once for good",
                region,
                region,
            )

    def close(self) -> None:
        """Unregister the pool and drop its memory, if no access can still reach it.

        A quarantined slot, or one still leased, may yet be read or written by
        the NIC or by a GPU copy. Unregistering or freeing the memory under it
        would let that access reach memory the process hands out again, so
        such a pool stays registered and allocated for the life of the
        process (``_UNSETTLED_ALLOCATIONS``); the Store client's own teardown
        then ends its RDMA work.
        """
        with self._cond:
            if self._closed:
                return
            self._closed = True
            leased = sum(
                1 for slots in self._slots.values() for slot in slots if slot.leased
            )
            quarantined = sum(self._quarantined.values())
            self._cond.notify_all()
        try:
            if leased or quarantined:
                _UNSETTLED_ALLOCATIONS.append(self._backing)
                logger.warning(
                    "Mooncake Store offload: closing the transfer pool with %d "
                    "quarantined and %d leased slot(s); its %d bytes stay "
                    "registered and allocated for the life of the process",
                    quarantined,
                    leased,
                    self.nbytes,
                )
            else:
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

    def _take_locked(self, region: str, n: int) -> list[Slot]:
        """``n`` free slots of ``region`` in index order, one run if there is one.

        The lowest run of ``n`` consecutive slots, so a window is one buffer
        (``contiguous_view``); only a region that quarantine or other windows
        have broken up hands out scattered slots, and those go through the
        staging copy instead.
        """
        free = self._free[region]
        ordered = sorted(free, key=lambda slot: slot.index)
        chosen = ordered[:n]
        run_start = 0
        for position in range(1, len(ordered) + 1):
            if (
                position == len(ordered)
                or ordered[position].index != ordered[position - 1].index + 1
            ):
                if position - run_start >= n:
                    chosen = ordered[run_start : run_start + n]
                    break
                run_start = position
        taken = {slot.index for slot in chosen}
        self._free[region] = deque(slot for slot in free if slot.index not in taken)
        return chosen

    @staticmethod
    def _require_leased(slot: Slot) -> None:
        if not slot.leased:
            raise RuntimeError(f"{slot!r} is not held by a transfer")
