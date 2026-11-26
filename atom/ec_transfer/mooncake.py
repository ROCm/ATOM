# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The Mooncake image embedding transfer: P pulls E's rows over RDMA.

E keeps each request's embedding rows in a pool of GPU memory registered with
a Mooncake TransferEngine and answers with a handle; P reads the rows with
``transfer_sync_read``. An entry is ``[generation word | rows]``. E reuses
space oldest-first, and only once an entry's lease has expired; it changes an
entry's word before writing over the entry. P reads the rows, then rereads the
word: a word other than the one in the handle means the rows may be another
request's, and P encodes the images itself.

Neither ``MooncakeEcWriter`` nor ``MooncakeEcReader`` locks: each is used by
one thread at a time (E's encoder thread puts, P's model runner reads).
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from atom.ec_transfer.base import EcStoreFull, dtype_name, handle_matches
from atom.utils import envs, mooncake_hca
from atom.utils.network import get_ip
from atom.utils.numa_utils import _physical_index

logger = logging.getLogger("atom")

try:
    from mooncake.engine import TransferEngine

    MOONCAKE_AVAILABLE = True
except ImportError:
    MOONCAKE_AVAILABLE = False

TRANSPORT = "mooncake"
DEFAULT_PROTOCOL = "rdma"
_WORD = 8  # bytes of the generation word that starts an entry
_ALIGN = 64  # entry start alignment in the pool
_INVALID = 0  # the word of an entry E has given up; no generation is 0


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
    GPU's local HCA."""
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
    phys_idx = None
    if protocol.strip().lower() != "tcp" and not configured and device.type == "cuda":
        phys_idx = _physical_index(device.index or 0)
    devices = mooncake_hca.ib_device_filter(
        protocol,
        configured,
        phys_idx,
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
    created: float


class MooncakeEcWriter:
    """E's registered pool: stores rows under a key for P to read."""

    def __init__(
        self,
        size_bytes: int,
        device: torch.device | str,
        *,
        protocol: str = DEFAULT_PROTOCOL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._buf = torch.zeros(size_bytes, dtype=torch.uint8, device=device)
        self._engine, self.session = _open_engine(self._buf.device, protocol)
        _register(self._engine, self._buf)
        self._lease_s = envs.ATOM_EC_LEASE_S
        self._clock = clock
        self._entries: deque[_Entry] = deque()
        # Random start: a word left inside another entry's rows never matches.
        self._generation = int.from_bytes(os.urandom(6), "little") + 1

    def _word(self, offset: int) -> torch.Tensor:
        return self._buf[offset : offset + _WORD].view(torch.int64)

    def _place(self, need: int) -> int:
        """Start of ``need`` free bytes, evicting expired entries oldest-first."""
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
        while any(e.start < end and start < e.end for e in self._entries):
            oldest = self._entries[0]
            if now - oldest.created < self._lease_s:
                self._entries.extendleft(reversed(victims))
                raise EcStoreFull(
                    f"pool of {self._buf.numel()} bytes is full: every entry in "
                    f"the way is inside its {self._lease_s} s lease"
                )
            victims.append(self._entries.popleft())
        if victims:
            for e in victims:
                self._word(e.start).fill_(_INVALID)
            _sync(self._buf)
        return start

    def put(self, key: int, rows: torch.Tensor) -> dict[str, Any]:
        """Store ``rows`` and return the handle P reads them by.

        ``address`` is the entry's generation word; the rows follow it.
        Raises ``EcStoreFull`` when no space is past its lease.
        """
        payload = rows.contiguous().view(-1).view(torch.uint8)
        need = -(-(_WORD + payload.numel()) // _ALIGN) * _ALIGN
        start = self._place(need)
        self._generation += 1
        body = start + _WORD
        self._buf[body : body + payload.numel()].copy_(payload)
        self._word(start).fill_(self._generation)
        _sync(self._buf)
        self._entries.append(_Entry(start, start + need, self._clock()))
        return {
            "transport": TRANSPORT,
            "key": key,
            "rows": rows.shape[0],
            "dtype": dtype_name(rows.dtype),
            TRANSPORT: {
                "session": self.session,
                "address": self._buf.data_ptr() + start,
                "length": payload.numel(),
                "generation": self._generation,
            },
        }


class MooncakeEcReader:
    """P's side: reads rows from E's pool into a registered buffer."""

    def __init__(
        self,
        size_bytes: int,
        device: torch.device | str,
        *,
        protocol: str = DEFAULT_PROTOCOL,
        kv_transfer_config: dict[str, Any] | None = None,
    ) -> None:
        """``kv_transfer_config``: this server's, whose ``ib_device``,
        ``ib_enable_alternate_hca`` and ``ib_hca_count`` the reader's engine
        takes as the KV connector's does."""
        kv = kv_transfer_config or {}
        # The reread generation word, then the rows.
        self._buf = torch.zeros(size_bytes + _WORD, dtype=torch.uint8, device=device)
        self._engine, _ = _open_engine(
            self._buf.device,
            protocol,
            kv.get("ib_device", ""),
            enable_alternate_hca=bool(kv.get("ib_enable_alternate_hca", False)),
            hca_count=int(kv.get("ib_hca_count", 8)),
        )
        _register(self._engine, self._buf)

    def get(
        self, handle: dict[str, Any], key: int, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """The rows E stored for ``key``, or None if P must encode itself."""
        if not handle_matches(handle, TRANSPORT, key, dtype):
            return None
        where = handle[TRANSPORT]
        session = where.get("session")
        address, length = where.get("address"), where.get("length")
        generation = where.get("generation")
        # A generation of 0 would match a given-up or never-written entry.
        if not (
            isinstance(session, str)
            and _positive_int(address)
            and _positive_int(generation)
            and _positive_int(length)
            and length % (handle["rows"] * dtype.itemsize) == 0
        ):
            logger.warning("Malformed image embedding handle: %s", where)
            return None
        if length > self._buf.numel() - _WORD:
            logger.warning(
                "Image embeddings of %d bytes exceed the %d byte read buffer",
                length,
                self._buf.numel() - _WORD,
            )
            return None
        ptr = self._buf.data_ptr()
        ret = self._engine.transfer_sync_read(
            session, ptr + _WORD, address + _WORD, length
        )
        if ret == 0:
            ret = self._engine.transfer_sync_read(session, ptr, address, _WORD)
        if ret != 0:
            logger.warning("Image embedding read from %s failed (ret=%d)", session, ret)
            return None
        if self._buf[:_WORD].view(torch.int64).item() != generation:
            return None
        rows = self._buf[_WORD : _WORD + length].view(dtype)
        return rows.view(handle["rows"], -1).clone()
