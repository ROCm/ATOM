# SPDX-License-Identifier: MIT

"""Bounded arrival-time Store lookups, without blocking the scheduler thread.

Only completed Store answers are cached, scoped to one request lifecycle.
The worker rechecks/renews before loading; a stale answer never authorizes
reuse of missing bytes. One lookup executor serializes the arrival-time queries;
a lease renewal has an executor of its own and takes the backend ahead of any
queued query, waiting only for the call in flight.
"""

import logging
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from threading import Condition, RLock

from atom.kv_transfer.offload.lookup_client import LookupClient

logger = logging.getLogger("atom")

# How long the scheduler thread waits for its own advisory renewal RPC: a slow
# Store RPC must not stall the scheduler and, through the DP lockstep, its peers.
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


class _BackendSlot:
    """One call at a time into the synchronous Store client; a renewal takes the
    slot ahead of every waiting query."""

    def __init__(self) -> None:
        self._cond = Condition()
        self._busy = False
        self._renewals = 0

    @contextmanager
    def take(self, renewal: bool = False):
        with self._cond:
            self._renewals += renewal
            while self._busy or (self._renewals and not renewal):
                self._cond.wait()
            self._renewals -= renewal
            self._busy = True
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()


class AsyncStoreLookup(LookupClient):
    def __init__(self, backend, capacity=4096):
        self.backend = backend
        self.capacity = capacity
        self.lock = RLock()
        self.executor = DaemonExecutor("store-query", limit=capacity)
        # Renewals ride their own worker so a burst of arrival-time queries
        # cannot queue one behind it; the slot keeps the synchronous Store
        # client (and the pool state around it) to one call at a time.
        self.renew_executor = DaemonExecutor("store-renew")
        self._renewing = False
        self.backend_slot = _BackendSlot()
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

    def _query(self, seq, sid):
        with self.backend_slot.take():
            with self.lock:
                self.asked_at[sid] = time.monotonic()
                self.asked_at.move_to_end(sid)
                while len(self.asked_at) > 2 * self.capacity:
                    self.asked_at.popitem(last=False)
            return self.backend.lookup(seq, lookup_id=sid)

    def query_started(self, sid):
        """When the Store query behind `sid`'s answer was sent; None if unknown."""
        with self.lock:
            return self.asked_at.get(sid)

    def submit(self, seq, sid):
        with self.lock:
            if self.closed or sid in self.pending or len(self.pending) >= self.capacity:
                return False
            future = self.executor.submit(self._query, seq, sid)
            if future is None:
                return False
            self.pending[sid] = future
            return True

    def submit_if(self, seq, sid, keep):
        """`submit`, if `keep()` still holds. Terminal cleanup takes the same
        lock, so a request cleaned up meanwhile never gets a lookup."""
        with self.lock:
            return keep() and self.submit(seq, sid)

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
                        except Exception as exc:  # noqa: BLE001
                            self._warn_lookup_failure(exc)

    def pending_ids(self):
        self.pump()
        with self.lock:
            return set(self.pending)

    def is_pending(self, sid):
        """Submitted and not answered yet."""
        with self.lock:
            f = self.pending.get(sid)
            return f is not None and not f.done()

    def answered(self, sid):
        """Whether the lookup for `sid` has completed (read-only)."""
        with self.lock:
            f = self.pending.get(sid)
            return f is not None and f.done() and not f.cancelled()

    def poll(self, sid):
        with self.lock:
            f = self.pending.get(sid)
            return f is not None and f.done()

    def lookup(self, seq, lookup_id=None):
        # Admission still reaches this after its defer window. A pending Store
        # query is a non-answer, not a reason to block the scheduler or a miss.
        sid = lookup_id or str(seq.id)
        self.submit(seq, sid)
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
        with self.backend_slot.take(renewal=True):
            return self.backend.renew(seq, start, end)

    def _renewal_done(self, _future) -> None:
        self._renewing = False

    def renew(self, seq, start, end):
        """Renew the read leases of tokens ``[start, end)``; 0 if it is not done in time.

        Runs on the renewal executor, ahead of any queued lookup. The scheduler
        thread waits at most `RENEW_TIMEOUT_S` for it, and skips renewal while
        an earlier one is still stuck behind a slow RPC.
        The load goes ahead whatever the answer (its all-or-nothing get fails on
        a chunk the Store lost and the request is prefilled), so a renewal that
        is late costs a lease, never correctness.
        """
        if self._renewing:
            return 0
        self._renewing = True
        future = self.renew_executor.submit(self._renew, seq, start, end)
        future.add_done_callback(self._renewal_done)
        try:
            return future.result(timeout=RENEW_TIMEOUT_S)
        except FutureTimeout:
            future.cancel()
            logger.warning(
                "Mooncake Store lease renewal for seq=%s did not finish in %.1fs; "
                "the load is dispatched without it",
                seq.id,
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
