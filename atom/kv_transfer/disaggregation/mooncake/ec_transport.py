# SPDX-License-Identifier: MIT
"""Image embeddings from an encoder (E) to a prefill rank (P) over Mooncake.

E keeps each request's embedding rows in a pool of GPU memory registered with
a Mooncake TransferEngine and answers with ``ec_transfer_params``; P reads the
rows with ``transfer_sync_read``. An entry is ``[generation word | rows]``.
E reuses space oldest-first, and only once an entry's lease has expired; it
changes an entry's word before writing over the entry. P reads the rows, then
rereads the word: a word other than the one E answered with means the rows may
be another request's, and P encodes the images itself.
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

from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc
from atom.utils import envs
from atom.utils.network import get_ip
from atom.utils.numa_utils import _physical_index

logger = logging.getLogger("atom")

_WORD = 8  # bytes of the generation word that starts an entry
_ALIGN = 64  # entry start alignment in the pool
_INVALID = 0  # the word of an entry E has given up; no generation is 0


class EcPoolFull(RuntimeError):
    """Every entry the new rows would overwrite is still inside its lease."""


def _open_engine(device: torch.device, protocol: str) -> tuple[Any, str]:
    """A TransferEngine on the HCAs ``_select_ib_devices`` picks, and its session."""
    if not mc._MOONCAKE_AVAILABLE:
        raise RuntimeError(
            "Mooncake is not installed; image embedding transport needs it"
        )
    configured = envs.ATOM_MOONCAKE_IB_DEVICE
    phys_idx = None
    if protocol.strip().lower() != "tcp" and not configured and device.type == "cuda":
        phys_idx = _physical_index(device.index or 0)
    ib_device = ",".join(mc._select_ib_devices(protocol, configured, phys_idx))
    mc._configure_mooncake_transport(protocol)
    ip = get_ip()
    engine = mc.TransferEngine()
    ret = engine.initialize(ip, "P2PHANDSHAKE", protocol, ib_device)
    if ret != 0:
        raise RuntimeError(
            f"Mooncake TransferEngine.initialize() failed (ret={ret}) on ip={ip}, "
            f"protocol={protocol}, ib_device={ib_device}"
        )
    return engine, f"{ip}:{engine.get_rpc_port()}"


def _register(engine: Any, buf: torch.Tensor) -> None:
    ret = engine.register_memory(buf.data_ptr(), buf.numel())
    if ret != 0:
        raise RuntimeError(f"Mooncake register_memory failed (ret={ret})")


def _sync(buf: torch.Tensor) -> None:
    # A remote read sees GPU memory only once the stream that wrote it is done.
    if buf.is_cuda:
        torch.cuda.synchronize(buf.device)


@dataclass
class _Entry:
    start: int
    end: int
    created: float


class EcPool:
    """E's registered pool: stores rows under a ``cache_seed`` for P to read."""

    def __init__(
        self,
        size_bytes: int,
        device: torch.device | str,
        *,
        protocol: str = mc.MOONCAKE_DEFAULT_PROTOCOL,
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
            raise EcPoolFull(
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
                raise EcPoolFull(
                    f"pool of {self._buf.numel()} bytes is full: every entry in "
                    f"the way is inside its {self._lease_s} s lease"
                )
            victims.append(self._entries.popleft())
        for e in victims:
            self._word(e.start).fill_(_INVALID)
        _sync(self._buf)
        return start

    def put(self, cache_seed: int, rows: torch.Tensor) -> dict[str, Any]:
        """Store ``rows`` and return the ``ec_transfer_params`` P reads them by.

        ``address`` is the entry's generation word; the rows follow it.
        Raises ``EcPoolFull`` when no space is past its lease.
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
            "session": self.session,
            "cache_seed": cache_seed,
            "address": self._buf.data_ptr() + start,
            "length": payload.numel(),
            "generation": self._generation,
            "rows": rows.shape[0],
        }


class EcReader:
    """P's side: reads rows from E's pool into a registered buffer."""

    def __init__(
        self,
        size_bytes: int,
        device: torch.device | str,
        *,
        protocol: str = mc.MOONCAKE_DEFAULT_PROTOCOL,
    ) -> None:
        # The reread generation word, then the rows.
        self._buf = torch.zeros(size_bytes + _WORD, dtype=torch.uint8, device=device)
        self._engine, _ = _open_engine(self._buf.device, protocol)
        _register(self._engine, self._buf)

    def read(
        self, params: dict[str, Any], cache_seed: int, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """The rows E stored for ``cache_seed``, or None if P must encode itself."""
        if params["cache_seed"] != cache_seed:
            return None
        length = params["length"]
        if length > self._buf.numel() - _WORD:
            logger.warning(
                "Image embeddings of %d bytes exceed the %d byte read buffer",
                length,
                self._buf.numel() - _WORD,
            )
            return None
        session, address = params["session"], params["address"]
        ptr = self._buf.data_ptr()
        ret = self._engine.transfer_sync_read(
            session, ptr + _WORD, address + _WORD, length
        )
        if ret == 0:
            ret = self._engine.transfer_sync_read(session, ptr, address, _WORD)
        if ret != 0:
            logger.warning("Image embedding read from %s failed (ret=%d)", session, ret)
            return None
        if self._buf[:_WORD].view(torch.int64).item() != params["generation"]:
            return None
        rows = self._buf[_WORD : _WORD + length].view(dtype)
        return rows.view(params["rows"], -1).clone()
