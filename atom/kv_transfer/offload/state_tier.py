# SPDX-License-Identifier: MIT
"""Worker-side spill and load driver for the state offload tier.

Its own executor, separate from the KV connector's `_load_executor` and
`_save_executor`. The reason is the one recorded at `connector.py:83-88`: a
load is on the TTFT critical path -- a parked sequence is waiting for it --
and must never queue behind a backlog of fire-and-forget spills.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class StateOffloadTier:
    def __init__(self, codec, index, *, max_workers: int = 1) -> None:
        self.codec = codec
        self.index = index
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state"
        )
        self._lock = threading.Lock()
        self._done: set[str] = set()
        self._failed: set[str] = set()
        self._inflight: set = set()

    def _register(self, fut) -> None:
        """Add *fut* to the inflight set and attach a callback that removes it
        on completion.  The callback fires on the worker thread (or inline on
        the submitting thread if the future is already done), so we must not
        call ``fut.result()`` while holding ``self._lock`` to avoid a deadlock.
        """
        with self._lock:
            self._inflight.add(fut)

        def _discard(f):
            with self._lock:
                self._inflight.discard(f)

        fut.add_done_callback(_discard)

    def submit_spill(self, h: int, entry_index: int, staging_slot: int) -> None:
        """`entry_index` is what the codec packs, `staging_slot` what the ring
        releases. Two index spaces: the staging entries sit past the pool's
        range in the arena (`num_groups + slot`), while the ring counts from 0.
        """
        fut = self._executor.submit(self._do_spill, h, entry_index, staging_slot)
        self._register(fut)

    def submit_load(self, req_id: str, h: int, group: int) -> None:
        self.index.loads_attempted += 1
        fut = self._executor.submit(self._do_load, req_id, h, group)
        self._register(fut)

    def drain(self) -> None:
        """Block until every submitted transfer has settled. Tests and shutdown
        only -- the serving path polls `get_finished` instead."""
        with self._lock:
            snapshot = set(self._inflight)
        for fut in snapshot:
            fut.result()

    def get_finished(self) -> tuple[set[str], set[str]]:
        with self._lock:
            done, failed = set(self._done), set(self._failed)
            self._done.clear()
            self._failed.clear()
        return done, failed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _do_spill(self, h: int, entry_index: int, staging_slot: int) -> None:
        try:
            if self.codec.put(h, entry_index):
                self.index.confirm_spill(h)
        except Exception:  # a spill is best effort by design
            logger.warning("state offload: spill of hash %d failed", h, exc_info=True)
        finally:
            # Always: a leaked slot shrinks the ring permanently and the
            # feature quietly stops spilling.
            self.index.release_staging(staging_slot)

    def _do_load(self, req_id: str, h: int, group: int) -> None:
        # A load target is a real pool group, not a staging entry: the bytes
        # land where the resuming request will read them. Only the spill
        # direction needs the staging indirection.
        ok = False
        try:
            ok = bool(self.codec.get(h, group))
        except Exception:  # a failed load is a normal path
            logger.warning("state offload: load of hash %d failed", h, exc_info=True)
        with self._lock:
            if ok:
                self._done.add(req_id)
            else:
                self.index.loads_failed += 1
                # So the next request does not repeat the attempt.
                self.index.forget(h)
                self._failed.add(req_id)
