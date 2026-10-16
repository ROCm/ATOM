# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bookkeeping shared by every LMCache MP transfer: operation identity,
terminal detection, and the deadline past which an unprovable transfer
stops the engine instead of releasing memory the server may still use.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    LoadCompletionId,
    SaveCompletionId,
    SaveOperationId,
    SaveSourceGroupId,
)
from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
)
from atom.kv_transfer.offload.mp.deployment import _extra_config

logger = logging.getLogger("atom")

_OPERATION_TOMBSTONE_LIMIT = 4096


@dataclass
class _PendingLoad:
    completion: LoadCompletionId
    future: Any | None
    started_at: float = field(default_factory=lambda: time.monotonic())


@dataclass
class _PendingSave:
    completion: SaveCompletionId
    future: Any | None
    start: int
    end: int
    started_at: float = field(default_factory=lambda: time.monotonic())


_DEFAULT_TRANSFER_DEADLINE_S = 1200.0
# The scheduler watches the same operations from dispatch, a step or more
# before a worker submits them. The margin lets the worker, which can name the
# exact transfer, fail first.
_SCHEDULER_DEADLINE_MARGIN_S = 60.0


class LMCacheTransferUnprovable(RuntimeError):
    """An MP transfer never reached a terminal state within the deadline."""


def _transfer_deadline_s(config: Any) -> float:
    """Seconds any MP transfer may stay non-terminal before the engine stops.

    Engine memory under a transfer is released only on a terminal report: a
    timeout cannot prove that the server stopped its DMA, so freeing on a
    clock could hand live blocks to another request. A transfer that never
    reports is therefore fatal once this deadline passes -- fail-stop, rather
    than corrupt memory or wedge the pool with no fault to point at.
    """

    extra = _extra_config(config)
    deadline = float(
        extra.get("lmcache.mp.transfer_deadline_s", _DEFAULT_TRANSFER_DEADLINE_S)
    )
    if not deadline > 0:
        raise ValueError("lmcache.mp.transfer_deadline_s must be > 0")
    return deadline


def _enforce_transfer_deadline(
    operation: Any, started_at: float, deadline_s: float
) -> None:
    """Raise ``LMCacheTransferUnprovable`` once a transfer outlives the bound."""

    elapsed = time.monotonic() - started_at
    if elapsed < deadline_s:
        return
    message = (
        f"LMCache MP transfer {operation} is still not terminal after "
        f"{elapsed:.0f}s. The memory it reads or writes cannot be released "
        "without proof that the server stopped, so the engine is stopping. "
        "Raise lmcache.mp.transfer_deadline_s if the tier is only slow."
    )
    logger.error(message)
    raise LMCacheTransferUnprovable(message)


class _UnprovableSubmission:
    """Stand-in future for a submission whose transport call raised.

    The server may have taken the request before the connection failed, so
    this never becomes terminal on its own: the lease is kept and only the
    transfer deadline ends it.
    """

    def query(self) -> bool:
        return False

    def result(self, timeout: float | None = None) -> bool:
        del timeout
        return False


def _terminal_future_result(future: Any | None) -> tuple[bool, Any]:
    """Return ``(terminal, result)`` without blocking on a device future."""

    if future is None:
        return True, None
    try:
        if not future.query():
            return False, None
        return True, future.result(timeout=0)
    except Exception:
        # A query/result exception is not proof that the remote GPU stream has
        # quiesced. Keep the operation pending; the transfer deadline bounds a
        # future that keeps raising.
        logger.warning(
            "LMCache MP transfer future could not prove terminal state",
            exc_info=True,
        )
        return False, None


def _transfer_operation_id(kind: str, completion: Any) -> str:
    """Return a stable LMCache handle ID for one ATOM completion generation."""

    request_id = getattr(completion, "req_id", completion)
    generation = getattr(completion, "generation", None)
    if generation is None:
        return f"{kind}:{request_id}"
    return f"{kind}:{request_id}:{generation}"


def _remember_operation_tombstone(
    operation_id: str,
    tombstones: set[str],
    tombstone_order: deque[str],
) -> None:
    if operation_id in tombstones:
        return
    tombstones.add(operation_id)
    tombstone_order.append(operation_id)
    while len(tombstone_order) > _OPERATION_TOMBSTONE_LIMIT:
        tombstones.discard(tombstone_order.popleft())


def _chunk_ranges(start: int, end: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    """Split ``[start, end)`` into LMCache chunk token ranges."""

    return tuple(
        (chunk_start, min(chunk_start + chunk_size, end))
        for chunk_start in range(start, end, chunk_size)
    )


def _source_safe_completions(
    operation: SaveOperationId, ranges: Iterable[Sequence[int]]
) -> set[ConnectorCompletion]:
    """One PAGE source-safe completion per finished token range of a save."""

    return {
        ConnectorCompletion(
            DENSE_PAGE_SOURCE_SAFE_CHANNEL,
            SaveSourceGroupId(operation, (tuple(token_range),)),
            True,
        )
        for token_range in ranges
    }
