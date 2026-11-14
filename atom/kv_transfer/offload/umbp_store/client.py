# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A client of one Mori UMBP standalone server, as the offload uses it.

Wraps ``mori.umbp.UMBPClient`` connected to a ``umbp_standalone_server`` that
something outside ATOM started (``auto_start=False``). The answers stay
UMBP's: one bool per key.

UMBP facts this wrapper is shaped by:

* Every I/O call is synchronous and answers one bool per key. True is a
  success; a put of a key that already exists is a success too (objects are
  immutable, and the stored bytes are kept). False does not say why: a miss,
  a full pool and a failed RPC (every key False) look the same.
* A call returns only after the server answered, and the server copies
  between its pool and the registered GPU memory itself (HIP IPC) before it
  answers. Once a call has returned, whatever it answered, no copy of it can
  still touch a caller buffer. A call that raised gives no such promise.
* There is no remove by key and no pinned or metadata-only object.
* ``deregister_memory`` of a standalone client drops every region that client
  registered, so a region is only deregistered once it is the last one.
* The Python client has no ``close``; dropping the last reference closes it.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any

logger = logging.getLogger("atom")

# Most keys one lookup RPC carries, to stay under the gRPC message limit.
_LOOKUP_BATCH_KEYS = 8192

# Clients whose registered memory may still be in use when they were closed
# (their pool kept a region registered): dropping one would deregister it
# under that use, so they are kept for the life of the process.
_RETAINED_CLIENTS: list[Any] = []


class CallRefused(RuntimeError):
    """This client refused a call before sending it to the server.

    Nothing was sent, so no copy can reach the call's buffers.
    """


def _new_umbp_client(address: str, startup_timeout_ms: int) -> Any:
    """A ``UMBPClient`` connected to the standalone server at ``address``.

    Imported here, never at load time: the scheduler and the CPU tests load
    this module without Mori.
    """
    from mori.umbp import UMBPClient, UMBPConfig, UMBPStandaloneProcessConfig

    process = UMBPStandaloneProcessConfig()
    process.address = address
    # One server per node, started outside ATOM: every rank connects to it.
    process.auto_start = False
    process.startup_timeout_ms = int(startup_timeout_ms)
    config = UMBPConfig()
    config.standalone_process = process
    return UMBPClient(config)


class UMBPStoreClient:
    """One ``UMBPClient`` of a standalone server."""

    def __init__(self, *, address: str, startup_timeout_ms: int = 10000) -> None:
        self.address = address
        self._stats_lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()
        # Registered regions, base -> size.
        self._regions: dict[int, int] = {}
        try:
            self._client = _new_umbp_client(address, startup_timeout_ms)
        except Exception as exc:
            raise RuntimeError(
                f"UMBP client could not connect to the standalone server at "
                f"{address}: {exc}. Start umbp_standalone_server on that address "
                "first; ATOM never starts it."
            ) from exc

    # -- memory registration ---------------------------------------------
    def register(self, ptr: int, nbytes: int) -> None:
        """Register a buffer the server may copy to or from.

        UMBP classifies the pointer itself: GPU memory is exported by HIP IPC.

        Raises:
            RuntimeError: UMBP refused it; every later put of it would fail.
        """
        client = self._live()
        if not client.register_memory(int(ptr), int(nbytes)):
            raise RuntimeError(
                f"UMBP could not register {int(nbytes)} bytes at {int(ptr):#x} "
                f"with the standalone server at {self.address}; GPU memory must "
                "be exportable by hipIpcGetMemHandle"
            )
        self._regions[int(ptr)] = int(nbytes)

    def unregister(self, ptr: int) -> None:
        """Forget a region; deregister once no other region is left.

        A standalone client deregisters all of its regions at once.
        """
        if self._regions.pop(int(ptr), None) is None:
            logger.warning("UMBP: the buffer at %#x is not registered", int(ptr))
            return
        if self._regions:
            logger.warning(
                "UMBP: the buffer at %#x stays registered until the client's "
                "other %d region(s) are released (no per-region deregister)",
                int(ptr),
                len(self._regions),
            )
            return
        client = self.__dict__.get("_client")
        if client is not None:
            client.deregister_memory(int(ptr))

    # -- data plane ------------------------------------------------------
    def exists(self, keys: list[str]) -> list[bool]:
        """Per key, whether the server holds it; order kept.

        Split into calls of at most ``_LOOKUP_BATCH_KEYS`` keys. UMBP answers a
        failed RPC as every key absent: a False is a miss or no answer.
        """
        client = self._live()
        results: list[bool] = []
        for start in range(0, len(keys), _LOOKUP_BATCH_KEYS):
            batch = list(keys[start : start + _LOOKUP_BATCH_KEYS])
            answers = [bool(answer) for answer in client.batch_exists(batch)]
            self._check_length("batch_exists", answers, batch)
            results.extend(answers)
            self._count("exists", batch, None, answers, count_false=False)
        return results

    def put(self, keys: list[str], ptrs: list[int], sizes: list[int]) -> list[bool]:
        """Store each buffer under its key; per key whether the server holds it.

        A key that already exists keeps its bytes and answers True.

        Raises:
            CallRefused: Duplicate keys, or a closed client; nothing was sent.
            Whatever ``batch_put_from_ptr`` raises, after which the server may
                still be reading the buffers.
        """
        self._check_unique(keys)
        client = self._live()
        answers = [
            bool(answer)
            for answer in client.batch_put_from_ptr(
                list(keys), [int(p) for p in ptrs], [int(s) for s in sizes]
            )
        ]
        self._check_length("batch_put_from_ptr", answers, keys)
        stored = sum(int(s) for s, ok in zip(sizes, answers) if ok)
        self._count("put", keys, stored, answers)
        return answers

    def get(self, keys: list[str], ptrs: list[int], sizes: list[int]) -> list[bool]:
        """Read each object into its buffer; per key whether it was read whole.

        Raises:
            CallRefused: Duplicate keys, or a closed client; nothing was sent.
            Whatever ``batch_get_into_ptr`` raises, after which the server may
                still be writing the buffers.
        """
        self._check_unique(keys)
        client = self._live()
        answers = [
            bool(answer)
            for answer in client.batch_get_into_ptr(
                list(keys), [int(p) for p in ptrs], [int(s) for s in sizes]
            )
        ]
        self._check_length("batch_get_into_ptr", answers, keys)
        read = sum(int(s) for s, ok in zip(sizes, answers) if ok)
        self._count("get", keys, read, answers)
        return answers

    def close(self) -> None:
        """Drop the client, unless a region it registered is still registered.

        The worker's pool unregisters its region first unless a copy may still
        reach it; dropping the client then would deregister that region under
        the copy, so the client is kept for the life of the process.
        """
        client = self.__dict__.pop("_client", None)
        if client is None:
            return
        if self._regions:
            _RETAINED_CLIENTS.append(client)
            logger.warning(
                "UMBP: %d registered region(s) may still be in use; the client "
                "of %s stays open for the life of the process",
                len(self._regions),
                self.address,
            )
            return
        # The binding has no close: the last reference closes the client.
        del client

    # -- accounting ------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Cumulative calls, keys, bytes and False answers, per operation."""
        with self._stats_lock:
            return {"counts": dict(self._counts), "failures": dict(self._failures)}

    def _count(self, op, keys, nbytes, answers, *, count_false=True) -> None:
        """Tally one call; ``nbytes`` is what it moved, None for a lookup.

        For a put that is the chunks the server holds once it returns: UMBP
        answers True for a key it already had, without moving its bytes.
        """
        false = sum(1 for ok in answers if not ok)
        with self._stats_lock:
            self._counts[f"{op}_calls"] += 1
            self._counts[f"{op}_keys"] += len(keys)
            if nbytes is not None:
                self._counts[f"{op}_bytes"] += int(nbytes)
            if count_false and false:
                self._failures[f"{op}:false"] += false

    def _live(self) -> Any:
        client = self.__dict__.get("_client")
        if client is None:
            raise CallRefused("UMBP client is closed")
        return client

    @staticmethod
    def _check_unique(keys: list[str]) -> None:
        if len(set(keys)) != len(keys):
            raise CallRefused("UMBP batch keys must be unique")

    @staticmethod
    def _check_length(call: str, answers: list, keys: list[str]) -> None:
        if len(answers) != len(keys):
            raise RuntimeError(
                f"UMBP {call} returned {len(answers)} results for {len(keys)} keys"
            )


__all__ = ["CallRefused", "UMBPStoreClient"]
