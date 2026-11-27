# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The Mooncake image embedding transfer: P pulls E's rows over RDMA.

E registers one region with a Mooncake TransferEngine: a slot table, then a
pool of rows. Each entry's rows sit in the pool, and its slot holds
``(generation, address, length, ack)``; rows never overlap a slot. E answers
with a handle naming the rows and the slot, and P reads the rows, then the
slot, with ``transfer_sync_read``. A slot whose three values differ from the
handle's means the rows may have been given up or overwritten, and P encodes
the images itself.

E reuses pool space and slots oldest-first, and only once an entry's lease
has expired. Before writing over an entry it zeroes the entry's slot, and it
writes a new entry's slot after its rows. Generations only grow, so a slot
never matches a handle given out for another entry. ``ack`` is written 0 and
reserved for P to acknowledge a read, so that E can reclaim the entry before
its lease ends.

Neither ``MooncakeEcWriter`` nor ``MooncakeEcReader`` locks: each is used by
one thread at a time (E's encoder thread puts, P's model runner reads).
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from atom.ec_transfer.base import EcStoreFull, dtype_name, handle_mismatch
from atom.utils import envs, mooncake_hca
from atom.utils.network import get_ip
from atom.utils.pci_topology import gpu_pci_bdf

logger = logging.getLogger("atom")

try:
    from mooncake.engine import TransferEngine

    MOONCAKE_AVAILABLE = True
except ImportError:
    MOONCAKE_AVAILABLE = False

TRANSPORT = "mooncake"
DEFAULT_PROTOCOL = "rdma"
_SLOT_FIELDS = 4  # generation, address, length, ack: int64 each
_SLOT = _SLOT_FIELDS * 8  # bytes of a slot
_ALIGN = 64  # entry start alignment in the pool
# One slot per this many pool bytes: an entry is at least one image's rows
# (hundreds of KiB), so space runs out before slots do; the table is 1/2048
# of the region.
BYTES_PER_SLOT = 64 << 10


def _open_engine(
    device: torch.device,
    protocol: str,
    ib_device: str = "",
    *,
    enable_alternate_hca: bool = False,
    hca_count: int = 8,
) -> tuple[Any, str]:
    """A TransferEngine on the HCAs ``mooncake_hca`` picks, and its session.
    ``ib_device`` names them; else ``ATOM_MOONCAKE_IB_DEVICE``; else the
    HCA on the GPU's PCIe switch."""
    if not MOONCAKE_AVAILABLE:
        raise RuntimeError(
            "Mooncake is not installed; image embedding transport needs it"
        )
    if envs.ATOM_MOONCAKE_MATCHED_RAILS:
        logger.warning(
            "ATOM_MOONCAKE_MATCHED_RAILS is set, but image embedding transport "
            "does not pick a matched rail; on a rail-isolated fabric its reads "
            "may fail and P encodes the images itself"
        )
    configured = ib_device or envs.ATOM_MOONCAKE_IB_DEVICE
    gpu_bdf = None
    if protocol.strip().lower() != "tcp" and not configured and device.type == "cuda":
        # The GPU this engine's memory is on, whatever *_VISIBLE_DEVICES masks it.
        gpu_bdf = gpu_pci_bdf(device.index or 0)
    devices = mooncake_hca.ib_device_filter(
        protocol,
        configured,
        gpu_bdf,
        enable_alternate_hca=enable_alternate_hca,
        hca_count=hca_count,
    )
    mooncake_hca.configure_transport(protocol)
    ip = get_ip()
    engine = TransferEngine()
    ret = engine.initialize(ip, "P2PHANDSHAKE", protocol, devices)
    if ret != 0:
        raise RuntimeError(
            f"Mooncake TransferEngine.initialize() failed (ret={ret}) on ip={ip}, "
            f"protocol={protocol}, ib_device={devices}"
        )
    return engine, f"{ip}:{engine.get_rpc_port()}"


def _register(engine: Any, buf: torch.Tensor) -> None:
    ret = engine.register_memory(buf.data_ptr(), buf.numel())
    if ret != 0:
        raise RuntimeError(f"Mooncake register_memory failed (ret={ret})")


def _sync(buf: torch.Tensor) -> None:
    # A remote read sees GPU memory only once the stream that wrote it is done.
    if buf.is_cuda:
        torch.cuda.current_stream(buf.device).synchronize()


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


@dataclass
class _Entry:
    start: int
    end: int
    slot: int
    created: float


def default_slots(size_bytes: int) -> int:
    """Slots for a region of ``size_bytes``: one per ``BYTES_PER_SLOT``."""
    return max(1, size_bytes // BYTES_PER_SLOT)


def slot_table_bytes(slots: int) -> int:
    """Bytes the table of ``slots`` takes at the front of E's region."""
    return -(-slots * _SLOT // _ALIGN) * _ALIGN


class MooncakeEcWriter:
    """E's registered region of ``size_bytes``: a slot table, then a pool that
    stores rows under a key for P to read. ``slots`` defaults to one per
    ``BYTES_PER_SLOT`` of the region."""

    def __init__(
        self,
        size_bytes: int,
        device: torch.device | str,
        *,
        protocol: str = DEFAULT_PROTOCOL,
        slots: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if slots is None:
            slots = default_slots(size_bytes)
        table = slot_table_bytes(slots)
        if table >= size_bytes:
            raise ValueError(
                f"a {size_bytes} byte region leaves no pool after {slots} slots"
            )
        # One registration of exactly size_bytes (E sizes it in whole GiB):
        # the slot table, then the pool.
        self._region = torch.zeros(size_bytes, dtype=torch.uint8, device=device)
        self._slots = self._region[: slots * _SLOT].view(torch.int64).view(slots, -1)
        self._buf = self._region[table:]
        self._engine, self.session = _open_engine(self._region.device, protocol)
        _register(self._engine, self._region)
        self._lease_s = envs.ATOM_EC_LEASE_S
        self._clock = clock
        self._entries: deque[_Entry] = deque()
        self._free_slots: deque[int] = deque(range(slots))
        # A random start: a restarted E never reissues a past generation.
        self._generation = int.from_bytes(os.urandom(6), "little") + 1

    def _slot_address(self, slot: int) -> int:
        return self._region.data_ptr() + slot * _SLOT

    def _place(self, need: int) -> tuple[int, int]:
        """Start of ``need`` free bytes and a free slot, evicting expired
        entries oldest-first."""
        if need > self._buf.numel():
            raise EcStoreFull(
                f"{need} bytes of embeddings exceed the {self._buf.numel()} byte pool"
            )
        start = self._entries[-1].end if self._entries else 0
        if start + need > self._buf.numel():
            start = 0
        end = start + need
        now = self._clock()
        victims = []
        while not self._free_slots or any(
            e.start < end and start < e.end for e in self._entries
        ):
            oldest = self._entries[0]
            if now - oldest.created < self._lease_s:
                self._entries.extendleft(reversed(victims))
                for e in victims:
                    self._free_slots.remove(e.slot)
                raise EcStoreFull(
                    f"pool of {self._buf.numel()} bytes and "
                    f"{len(self._slots)} slots is full: every entry in the way "
                    f"is inside its {self._lease_s} s lease"
                )
            victims.append(self._entries.popleft())
            self._free_slots.append(victims[-1].slot)
        if victims:
            for e in victims:
                self._slots[e.slot].zero_()
            _sync(self._region)
        return start, self._free_slots.popleft()

    def put(self, key: int, rows: torch.Tensor) -> dict[str, Any]:
        """Store ``rows`` and return the handle P reads them by.
        Raises ``EcStoreFull`` when no space or slot is past its lease."""
        payload = rows.contiguous().view(-1).view(torch.uint8)
        need = -(-payload.numel() // _ALIGN) * _ALIGN
        start, slot = self._place(need)
        self._generation += 1
        address = self._buf.data_ptr() + start
        self._buf[start : start + payload.numel()].copy_(payload)
        # The rows, then their slot: a slot P reads as matching names rows
        # already written. Both land before the handle leaves E.
        self._slots[slot].copy_(
            torch.tensor([self._generation, address, payload.numel(), 0])
        )
        _sync(self._region)
        self._entries.append(_Entry(start, start + need, slot, self._clock()))
        return {
            "transport": TRANSPORT,
            "key": key,
            "rows": rows.shape[0],
            "dtype": dtype_name(rows.dtype),
            TRANSPORT: {
                "session": self.session,
                "address": address,
                "length": payload.numel(),
                "generation": self._generation,
                "slot": self._slot_address(slot),
            },
        }


class MooncakeEcReader:
    """P's side: reads rows from E's pool into a registered buffer, which
    grows, at least doubling, up to ``max_bytes`` of rows.

    ``fallbacks`` counts the reads that returned None, by reason; each is
    logged with its reason and the running counts."""

    def __init__(
        self,
        size_bytes: int,
        device: torch.device | str,
        *,
        max_bytes: int | None = None,
        protocol: str = DEFAULT_PROTOCOL,
        kv_transfer_config: dict[str, Any] | None = None,
    ) -> None:
        """``kv_transfer_config``: this server's, whose ``ib_device``,
        ``ib_enable_alternate_hca`` and ``ib_hca_count`` the reader's engine
        takes as the KV connector's does."""
        kv = kv_transfer_config or {}
        self._max_bytes = size_bytes if max_bytes is None else max_bytes
        # The reread slot, then the rows.
        self._buf = torch.zeros(size_bytes + _SLOT, dtype=torch.uint8, device=device)
        self._engine, _ = _open_engine(
            self._buf.device,
            protocol,
            kv.get("ib_device", ""),
            enable_alternate_hca=bool(kv.get("ib_enable_alternate_hca", False)),
            hca_count=int(kv.get("ib_hca_count", 8)),
        )
        _register(self._engine, self._buf)
        self.fallbacks: Counter[str] = Counter()

    def _fallback(self, reason: str, session: Any = None) -> None:
        self.fallbacks[reason] += 1
        logger.warning(
            "Image embeddings from %s unusable: %s; running the vision tower "
            "(fallbacks so far: %s)",
            session,
            reason,
            dict(self.fallbacks),
        )

    def _fit(self, length: int) -> str | None:
        """Grow the buffer to hold ``length`` bytes of rows if ``max_bytes``
        allows; why it cannot, or None when it holds them."""
        capacity = self._buf.numel() - _SLOT
        if length <= capacity:
            return None
        if length > self._max_bytes:
            return f"{length} bytes exceed the {self._max_bytes} byte read limit"
        old = self._buf
        size = min(max(length, 2 * capacity), self._max_bytes)
        try:
            buf = torch.zeros(size + _SLOT, dtype=torch.uint8, device=old.device)
            _register(self._engine, buf)
        except RuntimeError as e:
            return f"cannot grow the read buffer: {e}"
        self._buf = buf
        ret = self._engine.unregister_memory(old.data_ptr())
        if ret != 0:
            logger.warning("Mooncake unregister_memory failed (ret=%d)", ret)
        return None

    def get(
        self, handle: dict[str, Any], key: int, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """The rows E stored for ``key``, or None if P must encode itself."""
        reason = handle_mismatch(handle, TRANSPORT, key, dtype)
        if reason is not None:
            self._fallback(reason)
            return None
        where = handle[TRANSPORT]
        session = where.get("session")
        address, length = where.get("address"), where.get("length")
        generation, slot = where.get("generation"), where.get("slot")
        # A generation of 0 would match a given-up or never-written slot.
        if not (
            isinstance(session, str)
            and _positive_int(address)
            and _positive_int(generation)
            and _positive_int(length)
            and _positive_int(slot)
            and length % (handle["rows"] * dtype.itemsize) == 0
        ):
            self._fallback("malformed handle", session)
            return None
        too_large = self._fit(length)
        if too_large is not None:
            self._fallback(too_large, session)
            return None
        # Work queued on this buffer (its zero fill, the previous read's
        # clone) must finish before the NIC writes into it.
        _sync(self._buf)
        ptr = self._buf.data_ptr()
        ret = self._engine.transfer_sync_read(session, ptr + _SLOT, address, length)
        if ret == 0:
            ret = self._engine.transfer_sync_read(session, ptr, slot, _SLOT)
        if ret != 0:
            self._fallback(f"read failed (ret={ret})", session)
            return None
        seen = self._buf[:_SLOT].view(torch.int64).tolist()
        if seen[:3] != [generation, address, length]:
            self._fallback("entry given up or reused", session)
            return None
        rows = self._buf[_SLOT : _SLOT + length].view(dtype)
        return rows.view(handle["rows"], -1).clone()
