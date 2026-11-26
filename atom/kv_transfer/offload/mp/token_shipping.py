# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Incremental token shipping for LMCache MP transfers.

A transfer names its LMCache chunks by the request's token prefix, so every
save and load carries `token_ids[:end]`. The scheduler (`TokenShipper`) sends
only the tokens past what the workers already hold, as an `array('i')` slice
in a `ShippedTokens` that rides in `LMCacheReqMeta.token_ids`. Each worker
(`WorkerTokens`) appends it to its own `array('i')` copy of the request's
prefix (4 bytes a token).

A finished request's copy is dropped through the metadata's `token_drops`,
which counts as pending work (`has_drops`), so the drop is delivered even when
no transfer is left.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ShippedTokens:
    """What one transfer ships in place of its token prefix ``[0, end)``."""

    # Prefix length the workers already hold for this request; `tokens`
    # continues it. 0 starts the request's copy over.
    base: int
    # The transfer covers tokens ``[0, end)``.
    end: int
    # Tokens ``[base, base + len(tokens))`` (an `array('i')` slice).
    tokens: Any


class TokenShipper:
    """Scheduler side: what each request's workers hold."""

    def __init__(self) -> None:
        # req id -> (the request, prefix length shipped). The request itself,
        # to tell a reused id from it.
        self._shipped: dict[Any, tuple[Any, int]] = {}
        self._drops: list = []

    def has_drops(self) -> bool:
        """Whether a finished request's worker copy still waits for a metadata
        to carry its drop."""
        return bool(self._drops)

    def take_drops(self) -> tuple:
        """Request ids whose worker copy goes with the metadata being built."""
        drops, self._drops = tuple(self._drops), []
        return drops

    def ship(self, seq: Any, end: int) -> ShippedTokens:
        entry = self._shipped.get(seq.id)
        base = 0
        if entry is not None:
            if entry[0] is seq:
                base = entry[1]
            else:
                self.drop(seq.id)  # a reused id: start its copy over
        new = max(base, end)
        self._shipped[seq.id] = (seq, new)
        return ShippedTokens(base, end, seq.token_ids[base:new])

    def drop(self, req_id: Any, seq: Any = None) -> None:
        """Forget `req_id`'s copy (with `seq`: only if it is that request's).
        The workers drop theirs with the next metadata (`take_drops`)."""
        entry = self._shipped.get(req_id)
        if entry is None or (seq is not None and entry[0] is not seq):
            return
        del self._shipped[req_id]
        self._drops.append(req_id)


class WorkerTokens:
    """Worker side: each request's prefix, rebuilt from its shipments."""

    def __init__(self) -> None:
        self._held: dict[Any, array] = {}

    def resolve(self, metadata: Any) -> None:
        """Drop finished requests' copies, and replace each request's
        `ShippedTokens` with its token prefix (`TokenPrefix`, not a copy)."""
        held = self._held
        for req_id in getattr(metadata, "token_drops", ()) or ():
            held.pop(req_id, None)
        for req in getattr(metadata, "requests", None) or ():
            shipped = req.token_ids
            if not isinstance(shipped, ShippedTokens):
                continue
            if shipped.base == 0:
                tokens = held[req.req_id] = array("i")
            else:
                tokens = held.get(req.req_id)
                # The scheduler starts a copy over only at base 0, so any
                # other length is a desync: chunk keys from it would name
                # the wrong KV.
                if tokens is None or len(tokens) != shipped.base:
                    raise RuntimeError(
                        f"LMCache MP token copy for {req.req_id!r} holds "
                        f"{0 if tokens is None else len(tokens)} tokens, the "
                        f"scheduler shipped from {shipped.base}"
                    )
            tokens.extend(shipped.tokens)
            if len(tokens) < shipped.end:
                raise RuntimeError(
                    f"LMCache MP token copy for {req.req_id!r} holds "
                    f"{len(tokens)} tokens, the transfer needs {shipped.end}"
                )
            # No copy here: a rank builds the list only for a transfer it
            # submits (`token_list`).
            req.token_ids = TokenPrefix(tokens, shipped.end)


class TokenPrefix:
    """The first `end` tokens of a request's worker copy, not copied.

    The held array only grows past `end` (a reused request id gets a new
    array), so reading its first `end` tokens later -- on the transfer thread,
    after newer shipments extended it -- sees the same tokens.
    """

    __slots__ = ("_end", "_tokens")

    def __init__(self, tokens: array, end: int) -> None:
        self._tokens = tokens
        self._end = int(end)

    def __len__(self) -> int:
        return self._end

    def tolist(self) -> list[int]:
        return self._tokens[: self._end].tolist()


def token_list(token_ids: Any) -> list[int]:
    """The list `AtomMPTransferSpec` takes: built once, where a transfer is
    actually submitted."""
    if isinstance(token_ids, list):
        return token_ids
    tolist = getattr(token_ids, "tolist", None)
    return tolist() if callable(tolist) else list(token_ids)


__all__ = ["ShippedTokens", "TokenPrefix", "TokenShipper", "WorkerTokens", "token_list"]
