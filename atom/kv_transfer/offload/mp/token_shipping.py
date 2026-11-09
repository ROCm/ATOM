# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Incremental token shipping for LMCache MP transfers.

A transfer names its LMCache chunks by the request's token prefix, so every
save and load carries `token_ids[:end]`. Long prompts are saved a chunk at a
time: shipping the whole prefix with each one is quadratic in the prompt, and
the scheduler boxed it into a Python list and pickled it every time (77 ms per
transfer at 512k tokens).

Here the scheduler (`TokenShipper`) sends only the tokens past what the
workers already hold, as an `array('i')` slice (it pickles as raw bytes), in a
`ShippedTokens` that rides in `LMCacheReqMeta.token_ids`. Each worker shell
(`WorkerTokens`) appends it to its own `array('i')` copy of the request's
prefix (4 bytes a token) and hands the transfer a copy of the part it covers.

The scheduler only counts a shipment as delivered once the metadata carrying it
was built (`begin` / `commit` / `rollback`): a build that raises after shipping
would otherwise leave the workers a prefix behind and fail the request's next
transfer. A finished request's copy is dropped through the metadata's
`token_drops`, which makes the step work even when no transfer is left.
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
        self._snapshot: tuple[dict, list] | None = None

    def begin(self) -> None:
        """Start a metadata build: what `ship` and `drop` do is provisional."""
        self._snapshot = (dict(self._shipped), list(self._drops))

    def commit(self) -> None:
        """The metadata was built: its shipments are delivered."""
        self._snapshot = None

    def rollback(self) -> None:
        """The build failed and its metadata is lost: forget what it shipped."""
        if self._snapshot is not None:
            self._shipped, self._drops = self._snapshot
            self._snapshot = None

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
        `ShippedTokens` with its token prefix (an `array('i')`)."""
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
            # A copy, not the held array: the transfer thread reads it after
            # later shipments have extended the held one.
            req.token_ids = tokens[: shipped.end]


__all__ = ["ShippedTokens", "TokenShipper", "WorkerTokens"]
