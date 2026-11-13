# SPDX-License-Identifier: MIT

"""Bounded arrival-time Store lookups, without blocking the scheduler thread.

Only completed Store answers are cached, scoped to one request lifecycle.
The worker rechecks/renews before loading; a stale answer never authorizes
reuse of missing bytes. One lookup executor serializes the arrival-time queries;
a lease renewal has an executor of its own and skips an occupied backend,
never waiting for arrival-time queries or earlier stalled renewals.
"""

import logging
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Lock, RLock

from atom.kv_transfer.offload.lookup_client import LookupClient, as_lookup_client

logger = logging.getLogger("atom")

# How long the scheduler thread waits for its own advisory renewal RPC.
# Occupied backends are skipped immediately (see `renew`); a Store RPC that
# is slow must not stall the scheduler and, through the DP lockstep, its peers.
RENEW_TIMEOUT_S = 1.0


class DaemonExecutor:
    """One worker thread that cannot hold the interpreter's exit.

    A `ThreadPoolExecutor` worker is joined at interpreter shutdown, and
    cancelling a running future does not interrupt its Store RPC, so a hung call
    would keep the engine process alive after `close()`. A daemon thread does
    not.
    """

    def __init__(self, name: str, limit: int | None = None) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False
        # Tasks queued and not yet taken by the worker, cancelled ones included:
        # a stalled Store RPC must not let submit/cancel cycles pile them up.
        self._limit = limit
        self._queued = 0
        self._queued_lock = threading.Lock()
        threading.Thread(target=self._run, name=name, daemon=True).start()

    def submit(self, fn, *args) -> Future | None:
        """None when `limit` tasks are already waiting for the worker."""
        future: Future = Future()
        if self._closed:
            future.cancel()
            return future
        with self._queued_lock:
            if self._limit is not None and self._queued >= self._limit:
                return None
            self._queued += 1
        item = [future, fn, args]
        # A cancelled task releases its arguments (the sequence) at once.
        future.add_done_callback(
            lambda _f, item=item: item.__setitem__(slice(1, 3), (None, ()))
        )
        self._queue.put(item)
        # Hand the GIL over so the worker starts the call now, as starting a
        # thread would have: a Store query is mostly waiting on the network.
        time.sleep(0)
        return future

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            with self._queued_lock:
                self._queued -= 1
            future, fn, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except BaseException as exc:  # noqa: BLE001  # handed to the waiter
                future.set_exception(exc)

    def shutdown(self) -> None:
        """Cancel what has not started; the worker may still be inside a call."""
        self._closed = True
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0].cancel()
        self._queue.put(None)


class AsyncStoreLookup(LookupClient):
    def __init__(self, backend, capacity=4096):
        self.backend = as_lookup_client(backend)
        self.capacity = capacity
        self.lock = RLock()
        self.executor = DaemonExecutor("store-query", limit=capacity)
        # Renewals ride their own worker so a burst of arrival-time queries
        # cannot queue one behind it; the lock keeps the synchronous Store
        # client (and the pool state around it) to one call at a time.
        self.renew_executor = DaemonExecutor("store-renew", limit=64)
        self.backend_lock = Lock()
        self.pending = {}
        # sid -> when its Store query was sent: the read lease the answer grants
        # runs from then, not from when the scheduler takes the answer.
        self.asked_at: OrderedDict = OrderedDict()
        self.discarded = set()
        self.discarded_hits = {}
        self.closed = False
        self._next_failure_warning = 0.0

    def supports_async(self):
        return True

    def _warn_lookup_failure(self, exc):
        now = time.monotonic()
        with self.lock:
            if now < self._next_failure_warning:
                return
            self._next_failure_warning = now + 60.0
        logger.warning("Mooncake Store lookup failed; answer unavailable: %s", exc)

    def _query(self, seq, sid, request_configs=None):
        with self.backend_lock:
            with self.lock:
                self.asked_at[sid] = time.monotonic()
                self.asked_at.move_to_end(sid)
                while len(self.asked_at) > 2 * self.capacity:
                    self.asked_at.popitem(last=False)
            kwargs = (
                {} if request_configs is None else {"request_configs": request_configs}
            )
            return self.backend.lookup(seq, lookup_id=sid, **kwargs)

    def query_started(self, sid):
        """When the Store query behind `sid`'s answer was sent; None if unknown."""
        with self.lock:
            return self.asked_at.get(sid)

    def submit(self, seq, sid, request_configs=None):
        with self.lock:
            if self.closed or sid in self.pending or len(self.pending) >= self.capacity:
                return False
            future = self.executor.submit(self._query, seq, sid, request_configs)
            if future is None:
                return False
            self.pending[sid] = future
            return True

    def pump(self):
        with self.lock:
            for sid in list(self.discarded):
                f = self.pending.get(sid)
                if f is None or f.done():
                    self.discarded.remove(sid)
                    self.pending.pop(sid, None)
                    if f is not None and not f.cancelled():
                        try:
                            hit = f.result()
                            if hit is not None:
                                self.discarded_hits[sid] = hit
                        except Exception:  # noqa: BLE001, S110
                            pass

    def pending_ids(self):
        self.pump()
        with self.lock:
            return set(self.pending)

    def is_pending(self, sid):
        with self.lock:
            return sid in self.pending

    def answered(self, sid):
        """Whether the lookup for `sid` has completed (read-only)."""
        with self.lock:
            f = self.pending.get(sid)
            return f is not None and f.done() and not f.cancelled()

    def poll(self, sid):
        with self.lock:
            f = self.pending.get(sid)
            return f is not None and f.done()

    def has_answer(self, sid):
        with self.lock:
            f = self.pending.get(sid)
            if f is None or not f.done() or f.cancelled():
                return False
            try:
                return f.result() is not None
            except Exception as exc:  # noqa: BLE001
                self._warn_lookup_failure(exc)
                return False

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def lookup(self, seq, lookup_id=None, request_configs=None):
        # Admission still reaches this after its defer window. A pending Store
        # query is a non-answer, not a reason to block the scheduler or a miss.
        sid = lookup_id or str(seq.id)
        self.submit(seq, sid, request_configs)
        with self.lock:
            f = self.pending.get(sid)
        if f is None or not f.done():
            return None
        try:
            result = f.result()
        except Exception as exc:  # noqa: BLE001
            self._warn_lookup_failure(exc)
            result = None
        with self.lock:
            if self.pending.get(sid) is f and f.done():
                self.pending.pop(sid)
        return result

    def clear_lookup_status(self, sid):
        with self.lock:
            self.asked_at.pop(sid, None)
            f = self.pending.pop(sid, None)
            if f is not None:
                f.cancel()
            self.discarded.discard(sid)
            self.discarded_hits.pop(sid, None)

    def discard(self, sid):
        with self.lock:
            if sid in self.pending:
                self.discarded.add(sid)
        self.pump()

    def pop_discarded_hits(self):
        self.pump()
        with self.lock:
            hits, self.discarded_hits = self.discarded_hits, {}
            return hits

    def _renew(self, seq, start, end):
        # Advisory renewal must not wait behind a stalled lookup or renewal.
        # Otherwise every request in a dispatch batch pays its own timeout.
        if not self.backend_lock.acquire(blocking=False):
            return 0
        try:
            return self.backend.renew(seq, start, end)
        finally:
            self.backend_lock.release()

    def renew(self, seq, start, end):
        """Renew the read leases of tokens ``[start, end)``; 0 if it is not done in time.

        Runs on the renewal executor and skips an occupied backend immediately.
        The scheduler thread waits at most `RENEW_TIMEOUT_S` for its own RPC.
        The load goes ahead whatever the answer (its all-or-nothing get fails on
        a chunk the Store lost and the request is prefilled), so a renewal that
        is late costs a lease, never correctness.
        """
        if self.backend_lock.locked():
            return 0
        future = self.renew_executor.submit(self._renew, seq, start, end)
        if future is None:  # earlier renewals are still stuck behind a slow RPC
            return 0
        try:
            return future.result(timeout=RENEW_TIMEOUT_S)
        except FutureTimeout:
            future.cancel()
            logger.warning(
                "Mooncake Store lease renewal for seq=%s did not finish in %.1fs; "
                "the load is dispatched without it",
                getattr(seq, "id", None),
                RENEW_TIMEOUT_S,
            )
            return 0

    def close(self):
        with self.lock:
            self.closed = True
            for f in self.pending.values():
                f.cancel()
            self.pending.clear()
        # Daemon workers: a hung Store RPC does not hold engine exit.
        self.executor.shutdown()
        self.renew_executor.shutdown()
        # And the backend's own workers (the Store's per-pool lookups).
        if self.backend is not None:
            self.backend.close()
