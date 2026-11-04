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
(`WorkerTokens`) appends it to its copy of the request's prefix and hands the
transfer an ordinary list, so nothing past the shell knows about it. A
finished request's copy is dropped through the next shipment: drops ride on
it rather than on a metadata field of their own, since a worker copy only
grows through a shipment.
"""

from __future__ import annotations

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
    # Request ids whose worker copy is to be dropped, before this one is read.
    drops: tuple = ()


class TokenShipper:
    """Scheduler side: what each request's workers hold."""

    def __init__(self) -> None:
        # req id -> (the request, prefix length shipped). The request itself,
        # to tell a reused id from it.
        self._shipped: dict[Any, tuple[Any, int]] = {}
        self._drops: list = []

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
        drops, self._drops = tuple(self._drops), []
        return ShippedTokens(base, end, seq.token_ids[base:new], drops)

    def drop(self, req_id: Any, seq: Any = None) -> None:
        """Forget `req_id`'s copy (with `seq`: only if it is that request's).
        The workers drop theirs with the next shipment."""
        entry = self._shipped.get(req_id)
        if entry is None or (seq is not None and entry[0] is not seq):
            return
        del self._shipped[req_id]
        self._drops.append(req_id)


class WorkerTokens:
    """Worker side: each request's prefix, rebuilt from its shipments."""

    def __init__(self) -> None:
        self._held: dict[Any, list[int]] = {}

    def resolve(self, metadata: Any) -> None:
        """Replace each request's `ShippedTokens` with its token list."""
        held = self._held
        for req in getattr(metadata, "requests", None) or ():
            shipped = req.token_ids
            if not isinstance(shipped, ShippedTokens):
                continue
            for req_id in shipped.drops:
                held.pop(req_id, None)
            if shipped.base == 0:
                tokens = held[req.req_id] = []
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
            # Boxed once per token over the request's life, not per transfer.
            tokens.extend(shipped.tokens)
            if len(tokens) < shipped.end:
                raise RuntimeError(
                    f"LMCache MP token copy for {req.req_id!r} holds "
                    f"{len(tokens)} tokens, the transfer needs {shipped.end}"
                )
            # A copy, not the held list: the transfer thread reads it after
            # later shipments have extended the held one. The same list copy
            # an unpickled list would have cost.
            req.token_ids = tokens[: shipped.end]


__all__ = ["ShippedTokens", "TokenShipper", "WorkerTokens"]
