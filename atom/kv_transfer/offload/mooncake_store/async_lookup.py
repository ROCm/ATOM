# SPDX-License-Identifier: MIT
"""Bounded arrival-time Store lookups, without blocking the scheduler thread.

Only completed Store answers are cached, scoped to one request lifecycle.
The worker rechecks/renews before loading; a stale answer never authorizes
reuse of missing bytes. One executor serializes the synchronous Store client.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import RLock


class AsyncStoreLookup:
    token_database = None
    driving = True

    def __init__(self, backend, capacity=4096):
        self.backend = backend
        self.capacity = capacity
        self.lock = RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="store-query")
        self.pending = {}
        self.discarded = set()
        self.discarded_hits = {}
        self.closed = False

    def supports_async(self):
        return True

    def offer(self, sid, request):
        with self.lock:
            if self.closed or sid in self.pending or len(self.pending) >= self.capacity:
                return False
            self.pending[sid] = self.executor.submit(self._query, request, sid)
            return True

    def _query(self, request, sid):
        seq = request()
        return None if seq is None else self.backend.lookup(seq, lookup_id=sid)

    def submit(self, seq, sid):
        return self.offer(sid, lambda: seq)

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
                        except Exception:
                            pass

    def pending_ids(self):
        self.pump()
        with self.lock:
            return set(self.pending)

    def is_pending(self, sid):
        with self.lock:
            return sid in self.pending

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
            except Exception:
                return False

    def take_answer(self, sid, tokens, before=None):
        with self.lock:
            if not self.has_answer(sid):
                return None
            if before is not None:
                before()
            return self.pending.pop(sid).result()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def lookup(self, seq, lookup_id=None):
        # Compatibility fallback after the scheduler's defer window. Normal
        # prefetch consumes with take_answer and never waits here.
        sid = lookup_id or str(seq.id)
        self.submit(seq, sid)
        with self.lock:
            f = self.pending.get(sid)
        if f is None:
            return None
        try:
            result = f.result(timeout=30)
        except Exception:
            result = None
        with self.lock:
            if self.pending.get(sid) is f and f.done():
                self.pending.pop(sid)
        return result

    def clear_lookup_status(self, sid):
        with self.lock:
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

    def renew(self, seq, start, end):
        # Serialize with lookups; renew is required only at load dispatch.
        return self.executor.submit(self.backend.renew, seq, start, end).result()

    def close(self):
        with self.lock:
            self.closed = True
            for f in self.pending.values():
                f.cancel()
            self.pending.clear()
        self.executor.shutdown(wait=True, cancel_futures=True)
