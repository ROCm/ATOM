# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A scheduler-process thread that sends and advances LMCache MP lookups.

Without it a lookup moves only when the scheduler runs a pass: its request is
sent from the head of the waiting queue, and each round trip (lookup ack, then
prefetch status) is noticed one pass later. During a prefill forward the engine
thread waits for the workers, so nothing moves for the whole forward. The
driver sends a lookup as soon as its request arrives and chains the status query
the moment the ack lands, so the answer is usually ready by admission.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any

logger = logging.getLogger("atom")

# Poll period while lookups are in flight. LMCache futures have no callbacks,
# and a round trip takes about a millisecond or more on the server.
_DRIVER_POLL_S = 0.001
# Wake period with nothing in flight, for the answered-but-unconsumed check.
_DRIVER_IDLE_S = 1.0


class _LockedAdapter:
    """Serializes every method call into the scheduler-side MP adapter.

    The adapter is not thread-safe (its client-hash cache and pending-lookup
    tables are plain containers). With the driver running, the scheduler thread
    and the driver both call it; this proxy makes each method call hold the
    lookup client's lock. Attribute reads pass through, so the caller must hold
    the same lock while it touches the adapter's containers.
    """

    def __init__(self, adapter: Any, lock: Any) -> None:
        object.__setattr__(self, "_locked_adapter", adapter)
        object.__setattr__(self, "_locked_lock", lock)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._locked_adapter, name)
        if not callable(value) or isinstance(value, type):
            return value
        lock = self._locked_lock

        @functools.wraps(value)
        def locked(*args: Any, **kwargs: Any) -> Any:
            with lock:
                return value(*args, **kwargs)

        return locked

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._locked_adapter, name, value)


class _LookupDriver:
    """Runs `_MPLookupClient._drive` on its own thread."""

    def __init__(self, client: Any, window: int) -> None:
        if window <= 0:
            raise ValueError("LMCache MP lookup window must be > 0")
        self._client = client
        self._window = int(window)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mp-lookup", daemon=True)

    @property
    def window(self) -> int:
        return self._window

    def start(self) -> None:
        self._thread.start()

    def notify(self) -> None:
        """New work was offered; run a pass now."""
        self._wake.set()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            # Cleared before the pass: an offer that lands during it sets the
            # flag again, so the wait below returns at once.
            self._wake.clear()
            try:
                busy = self._client._drive(self._window)
            except Exception:
                logger.warning("LMCache MP lookup driver pass failed", exc_info=True)
                busy = True
            self._wake.wait(_DRIVER_POLL_S if busy else _DRIVER_IDLE_S)


__all__ = ["_LockedAdapter", "_LookupDriver"]
