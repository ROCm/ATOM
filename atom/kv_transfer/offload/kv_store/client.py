# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the KV-store offload asks of a backend's store client.

A client moves bytes between registered caller buffers and the store, one
batch of unique keys per call, and answers per key. What an answer means --
which values are a success, whether a failed key may still be moved by the
store -- is the backend's: its worker reads them (``KVStoreOffloadConnector``'s
``_settle_put`` and ``_get_failure``). The one answer the shared layer reads
itself is ``exists``'s: a value equal to 1 (``True`` included) means the store
holds the key.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable
from typing import Any, Protocol


class CallRefused(RuntimeError):
    """The client refused a call before handing it to the store.

    Nothing was sent, so no transfer can reach the call's buffers or the
    store's memory.
    """


class KVStoreClient(Protocol):
    """A store client as the shared worker and pool use it."""

    def register(self, ptr: int, nbytes: int) -> None:
        """Register a buffer the store may move bytes to or from.

        Raises when the store refuses it.
        """

    def unregister(self, ptr: int) -> None:
        """Release a buffer ``register`` took."""

    def exists(self, keys: list[str]) -> list[Any]:
        """Per key, in order: 1 (or True) when the store holds it.

        Raises when the store answers for a different number of keys.
        """

    def put(self, keys: list[str], ptrs: list[int], sizes: list[int]) -> list[Any]:
        """Store each buffer under its key; per key the backend's answer.

        Raises :class:`CallRefused` when nothing was sent.
        """

    def get(self, keys: list[str], ptrs: list[int], sizes: list[int]) -> list[Any]:
        """Read each object into its buffer; per key the backend's answer.

        Raises :class:`CallRefused` when nothing was sent.
        """

    def close(self) -> None:
        """Release the connection."""

    def stats(self) -> dict[str, Any]:
        """Cumulative ``{"counts": {...}, "failures": {...}}`` of the calls."""


class CallStats:
    """A store client's cumulative calls, keys, bytes and failures per
    operation, safe to update from the save, load and scheduler threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()

    def snapshot(self) -> dict[str, Any]:
        """``{"counts": {...}, "failures": {...}}``, as ``stats`` answers."""
        with self._lock:
            return {"counts": dict(self._counts), "failures": dict(self._failures)}

    def count(
        self,
        op: str,
        keys: list[str],
        nbytes: int | None,
        failures: Iterable[str] = (),
    ) -> None:
        """Tally one call of ``op``: its keys, the bytes it moved (None for a
        lookup) and one ``failures`` label per key that failed."""
        with self._lock:
            self._counts[f"{op}_calls"] += 1
            self._counts[f"{op}_keys"] += len(keys)
            if nbytes is not None:
                self._counts[f"{op}_bytes"] += int(nbytes)
            for failure in failures:
                self._failures[f"{op}:{failure}"] += 1


def require_unique_keys(keys: list[str], store: str, why: str = "") -> None:
    """Refuse a batch that names a key twice, before it reaches ``store``.

    Raises:
        CallRefused: A duplicate key; nothing was sent.
    """
    if len(set(keys)) != len(keys):
        raise CallRefused(
            f"{store} batch keys must be unique" + (f"; {why}" if why else "")
        )


def require_answer_per_key(
    store: str, call: str, answers: list, keys: list[str]
) -> None:
    """Refuse an answer that is not one result per key.

    Raises:
        RuntimeError: ``store``'s ``call`` answered for a different number of
            keys.
    """
    if len(answers) != len(keys):
        raise RuntimeError(
            f"{store} {call} returned {len(answers)} results for {len(keys)} keys"
        )


__all__ = [
    "CallRefused",
    "CallStats",
    "KVStoreClient",
    "require_answer_per_key",
    "require_unique_keys",
]
