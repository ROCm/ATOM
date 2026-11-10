# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which UMBP servers this process no longer loads from, and why.

Replaces the Mooncake Store offload's taint marker, which is a pinned object
every client of the Store sees. UMBP has no pinned or metadata-only object,
so the mark here is **this process's only**: a worker that marks a server
stops loading from it, and so does anything else in the same process that
asks (another worker built in it, a scheduler sharing it). Other worker
processes, the scheduler process and other engines on the server are not
told, and nothing survives a restart of this process.

A worker marks its server when a put raised: such a call promises nothing
about the copy the server may still be making from the worker's buffer, and
whether that copy can land in an object the server has since handed to
another key is not something UMBP documents. Restart the server, then the
engine, to load from it again.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
# Server address -> why this process does not load from it.
_TAINTED: dict[str, str] = {}


def mark(address: str, why: str) -> bool:
    """Mark ``address`` for this process; True the first time only."""
    with _lock:
        if address in _TAINTED:
            return False
        _TAINTED[address] = why
        return True


def reason(address: str) -> str | None:
    """Why this process does not load from ``address``; None when it may."""
    with _lock:
        return _TAINTED.get(address)


__all__ = ["mark", "reason"]
