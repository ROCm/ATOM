# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Non-blocking tier lookups for the in-process ``lmcache_offload`` connector.

LMCache's ``LMCacheLookupClient`` is synchronous: it hashes the whole prompt in
Python and then waits for every TP worker's lookup-server thread, which competes
for the GIL with that worker's busy-polling main thread. On the scheduler thread
that cost every running decode a stall per admission -- measured at AgentX c64
(MiniMax-M3.1 TP4): 2687 lookups, 10.6 ms mean, 490 ms worst.

This wrapper gives the scheduler the asynchronous lookup contract the MP and
Mooncake Store clients already have (``submit`` / ``poll`` / ``is_pending`` /
``answered`` / ``discard`` / ``supports_async``), so ``prefetch_lookups`` sends
the lookups for the head of the waiting queue, admission passes over a request
whose answer is still in flight (``lookup_pending``, bounded by
``OFFLOAD_LOOKUP_DEFER_S``) and the consuming ``lookup`` returns at once.

Every call into the backend runs on one daemon thread, in submission order: the
backend's ZMQ request sockets and its memo dict are not thread-safe, and the
order keeps a ``clear_lookup_status`` behind the lookup it clears.

Worker pins. A worker pins what a lookup hit when it answers. The scheduler
unpins through connector metadata (``lookup_requests_in_step``): a consumed
answer goes through ``_fresh_tier_lookup`` exactly as before. An answer nobody
consumes -- its request finished or was admitted on a detached lookup, or the
consuming ``lookup`` gave up waiting -- is queued here and handed to the next
metadata build through ``take_unpins``.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from functools import partial

from atom.kv_transfer.offload.lookup_client import LookupClient
from atom.kv_transfer.offload.mooncake_store.async_lookup import DaemonExecutor

logger = logging.getLogger("atom")


class AsyncInprocLookup(LookupClient):
    def __init__(self, backend, capacity: int = 512, wait_s: float = 10.0):
        self.backend = backend
        self.token_database = getattr(backend, "token_database", None)
        self.capacity = capacity
        self.wait_s = wait_s
        self.lock = threading.RLock()
        self.executor = DaemonExecutor("lmc-lookup", limit=capacity)
        self.pending: dict[str, Future] = {}
        self.discarded: set[str] = set()
        self.discarded_hits: dict[str, int] = {}
        self.unpins: list[str] = []
        self.closed = False
        # submitted: sent ahead of admission; sync: consumed with nothing sent
        # ahead (the scheduler thread waits for a whole lookup); waited: sent
        # ahead but still in flight when consumed.
        self.stats = {"submitted": 0, "sync": 0, "waited": 0, "timeouts": 0, "unpins": 0}
        self._last_log = time.monotonic()

    def supports_async(self) -> bool:
        return True

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log >= 60.0:
            self._last_log = now
            logger.info("[OFFLOAD-ASYNC-LOOKUP] %s", self.stats)

    def _run(self, *args, **kwargs) -> Future | None:
        return self.executor.submit(partial(self.backend.lookup, *args, **kwargs))

    def submit(self, token_ids, lookup_id, **kwargs) -> bool:
        with self.lock:
            if (
                self.closed
                or lookup_id in self.pending
                or len(self.pending) >= self.capacity
            ):
                return False
            future = self._run(token_ids, lookup_id, **kwargs)
            if future is None:
                return False
            self.pending[lookup_id] = future
            self.stats["submitted"] += 1
            return True

    def is_pending(self, lookup_id) -> bool:
        with self.lock:
            f = self.pending.get(lookup_id)
            return f is not None and not f.done()

    def poll(self, lookup_id) -> bool:
        with self.lock:
            f = self.pending.get(lookup_id)
            return f is None or f.done()

    def answered(self, lookup_id) -> bool:
        with self.lock:
            f = self.pending.get(lookup_id)
            return f is not None and f.done() and not f.cancelled()

    def pending_ids(self):
        self.pump()
        with self.lock:
            return set(self.pending)

    def pump(self) -> None:
        with self.lock:
            for sid in list(self.discarded):
                f = self.pending.get(sid)
                if f is not None and not f.done():
                    continue
                self.discarded.discard(sid)
                self.pending.pop(sid, None)
                if f is None or f.cancelled():
                    continue
                try:
                    hit = f.result()
                except Exception:  # noqa: BLE001  # a failed lookup pinned nothing
                    continue
                if hit is not None:
                    self.discarded_hits[sid] = int(hit)
                self.unpins.append(sid)

    def lookup(self, token_ids, lookup_id, **kwargs):
        """The answer of ``lookup_id``'s submitted lookup, or a lookup made now.

        Either way on the backend thread; the caller waits at most ``wait_s``.
        Past it the answer is a non-answer (None), and the lookup's pins are
        released once it lands.
        """
        with self.lock:
            f = self.pending.get(lookup_id)
            if f is None:
                f = self._run(token_ids, lookup_id, **kwargs)
                if f is None:
                    return None
                self.pending[lookup_id] = f
                self.stats["sync"] += 1
            elif not f.done():
                self.stats["waited"] += 1
        self._maybe_log()
        try:
            hit = f.result(timeout=self.wait_s)
        except FutureTimeout:
            with self.lock:
                self.discarded.add(lookup_id)
                self.stats["timeouts"] += 1
            logger.warning(
                "LMCache tier lookup for %s did not answer in %.1fs", lookup_id, self.wait_s
            )
            return None
        except Exception:
            with self.lock:
                if self.pending.get(lookup_id) is f:
                    self.pending.pop(lookup_id)
            raise
        with self.lock:
            if self.pending.get(lookup_id) is f:
                self.pending.pop(lookup_id)
        return hit

    def clear_lookup_status(self, lookup_id) -> None:
        with self.lock:
            f = self.pending.get(lookup_id)
            if f is not None:
                # Submitted and never consumed: its pins go once it has landed.
                self.discarded.add(lookup_id)
        self.pump()
        self.executor.submit(partial(self.backend.clear_lookup_status, lookup_id))

    def discard(self, lookup_id) -> None:
        with self.lock:
            if lookup_id in self.pending:
                self.discarded.add(lookup_id)
        self.pump()

    def pop_discarded_hits(self) -> dict[str, int]:
        self.pump()
        with self.lock:
            hits, self.discarded_hits = self.discarded_hits, {}
            return hits

    def take_unpins(self) -> list[str]:
        """Answered lookups nobody consumed: the workers still pin their hits."""
        self.pump()
        with self.lock:
            out, self.unpins = self.unpins, []
            self.stats["unpins"] += len(out)
            return out

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for f in self.pending.values():
                f.cancel()
            self.pending.clear()
        self.executor.shutdown()
        close = getattr(self.backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001  # teardown
                logger.debug("LMCache lookup client close failed", exc_info=True)
