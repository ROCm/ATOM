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


__all__ = ["CallRefused", "KVStoreClient"]
