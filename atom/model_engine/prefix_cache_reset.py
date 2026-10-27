# SPDX-License-Identifier: MIT
"""Admission fence for an idle prefix-index reset.

Track terminal scheduler outputs, not final stream chunks or frontend aborts:
those can precede scheduler cleanup. Failed sends remain conservatively busy.
"""

from contextlib import contextmanager
from functools import wraps
from threading import Lock, RLock


class PrefixCacheBusyError(RuntimeError):
    pass


class PrefixCacheResetGate:
    def __init__(self):
        self.admission = RLock()
        self._state_lock = Lock()
        self._active = set()

    @contextmanager
    def submitting(self, seqs):
        with self.admission:
            with self._state_lock:
                self._active.update(seq.id for seq in seqs)
            yield

    def finished(self, seq_id):
        with self._state_lock:
            self._active.discard(seq_id)

    @contextmanager
    def resetting(self):
        if not self.admission.acquire(blocking=False):
            raise PrefixCacheBusyError("Request admission is in progress")
        try:
            with self._state_lock:
                if self._active:
                    raise PrefixCacheBusyError(
                        f"{len(self._active)} requests have not finished scheduler cleanup"
                    )
            yield
        finally:
            self.admission.release()


def fence_prefix_cache_admission(fn):
    @wraps(fn)
    def wrapped(self, seqs):
        gate = getattr(self, "_prefix_cache_reset_gate", None)
        if gate is None:  # lightweight routing test doubles
            return fn(self, seqs)
        with gate.submitting(seqs):
            return fn(self, seqs)

    return wrapped
