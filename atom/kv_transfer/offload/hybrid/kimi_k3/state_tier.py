# SPDX-License-Identifier: MIT
"""Worker-side store and load driver for the state offload tier.

The store leg runs on its own executor so a checkpoint's D2H never blocks the
engine thread. The load leg does not: `load_state` runs synchronously on the KV
load task that called it, which is what fuses the two legs of one request into a
single dispatch and a single completion.

This class **reports, and the engine applies**: `StateOffloadIndex` lives in the
engine process, so `take_store_reports`/`take_hash_verdicts` hand their sets to
the connector rather than index a hash here. The failed-hash set resolves the
aggregator's quorum -- without it a partial store pins the hash forever.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

logger = logging.getLogger(__name__)


class StateOffloadTier:
    """Moves bytes; decides nothing, and holds no index.

    No index here: `StateOffloadIndex` lives in the engine process, so every
    counter and hash retraction is applied there from the reports below.
    Neither side can hold a second opinion about what is stored.
    """

    def __init__(self, codec, *, max_workers: int = 1) -> None:
        self.codec = codec
        # Only the store leg is queued. A store is off the TTFT critical path
        # and must not run on the engine thread; a load IS the critical path and
        # runs inline on the KV load task, so it can never queue behind a store.
        self._store_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state-store"
        )
        self._lock = threading.Lock()
        # hash -> whether its `get` produced bytes, drained by
        # `take_hash_verdicts`. Written from whichever KV load task ran the
        # state leg, read from the engine-facing thread, hence the lock.
        self._hash_verdicts: dict[int, bool] = {}
        self._inflight: set = set()
        # Store reports, drained by `take_store_reports`. Sets of
        # `StateStoreOperationId`, not bare hashes: the engine settles the pin
        # for that exact generation, and the aggregator would tombstone a bare
        # hash after its first store.
        #
        # `_source_released` is separate from `_indexed`: the source units are
        # free the instant the D2H drains, while whether the CPU put succeeded
        # is decided afterwards and cannot touch them. Reporting only the second
        # would hold an image out of the pool across a CPU-only operation.
        self._source_released: set = set()
        self._indexed: set = set()
        self._index_failed: set = set()
        # op -> monotonic at submission, for `oldest_store_age_s`.
        self._store_submitted_at: dict = {}

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

    def oldest_store_age_s(self) -> float:
        """How long the oldest unfinished store has been outstanding.

        Zero when nothing is in flight. This is the number that says a backend
        has stopped rather than being busy, and the one the in-flight *count*
        cannot express.
        """
        with self._lock:
            if not self._store_submitted_at:
                return 0.0
            oldest = min(self._store_submitted_at.values())
        return max(0.0, monotonic() - oldest)

    def submit_store(self, op, unit_ids) -> None:
        """Pack the checkpoint image in `unit_ids` for LMCache, under `op`.

        `op` is a `StateStoreOperationId`; the bytes are keyed by
        `op.prefix_hash` and the report is keyed by the whole operation, so
        two attempts at one prefix write the same entry but settle their own
        pins.

        No `ready_event`: the units are reserved out of the KV pool and pinned
        by the engine for the length of this transfer, so nothing on the compute
        stream is writing them and the packer gathers straight from where they
        sit.
        """
        with self._lock:
            self._store_submitted_at[op] = monotonic()
        self._register(self._store_executor.submit(self._do_store, op, unit_ids))

    def load_state(self, prefix_hash: int, slot: int) -> bool:
        """Fetch `prefix_hash` into pool slot `slot`. Synchronous, no report.

        Runs on the caller's thread -- the KV load task that owns this request --
        so the two legs of one load are one dispatch with one completion, and
        the state leg cannot queue behind a store.

        A miss is a normal path, not an error: LMCache's LRU can drop bytes
        under a hash the engine's index still advertises. Raising here would
        escape into that task and could strand every request parked on a later
        load, so the exception is swallowed and recorded as a miss instead. The
        engine owns the index and retracts the hash from the verdict below.
        """
        h = int(prefix_hash)
        ok = False
        try:
            ok = bool(self.codec.get(h, int(slot)))
        except Exception:  # deliberately blind; see the docstring
            logger.warning("state offload: load of hash %d failed", h, exc_info=True)
        with self._lock:
            # Failure-dominant: a hash this rank already missed stays missed
            # even if a later request finds it back.
            self._hash_verdicts[h] = self._hash_verdicts.get(h, True) and ok
        return ok

    def take_hash_verdicts(self) -> dict[int, bool]:
        """`{hash: the `get` produced bytes}` for every load run since the last
        call.

        Successes are reported as well as misses because the TP aggregator only
        acts on a key every rank reported: a miss-only channel would never reach
        quorum under TP>1 and the hash would stay advertised forever, parking
        every later request over that prefix against a `get` that must miss.
        The quorum is failure-dominant, so one rank's miss is what retracts.
        """
        with self._lock:
            verdicts = self._hash_verdicts
            self._hash_verdicts = {}
        return verdicts

    def drain(self) -> None:
        """Block until every submitted store has settled. Tests and shutdown
        only -- the serving path polls `take_store_reports` instead."""
        with self._lock:
            snapshot = set(self._inflight)
        for fut in snapshot:
            fut.result()

    def shutdown(self) -> None:
        self._store_executor.shutdown(wait=True)

    def _do_store(self, op, unit_ids) -> None:
        stored = False
        released = False

        def _source_released() -> None:
            # Fires from `codec.put` once `pack`'s gather+D2H drain, before
            # `batched_put`: the GPU has stopped reading the units, so hand them
            # back now instead of holding a whole image out of the pool across
            # the CPU put. Under `self._lock` (the engine thread drains the set);
            # the flag makes the end-of-store backstop a no-op so the release is
            # emitted once -- a second emission would double-unpin engine-side.
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                self._source_released.add(op)

        try:
            stored = bool(
                self.codec.put(
                    int(op.prefix_hash),
                    unit_ids,
                    on_source_released=_source_released,
                )
            )
        except Exception:  # deliberately blind
            # `codec.put` reaches into LMCache, whose failure modes are its own.
            # A store that cannot happen must cost one checkpoint's CPU copy --
            # not this worker thread, whose death would strand every request
            # parked on a later load.
            logger.warning(
                "state offload: store of hash %d (generation %d) failed",
                op.prefix_hash,
                op.generation,
                exc_info=True,
            )
        with self._lock:
            # Backstop, not the primary path: on success the callback already
            # published the release, so this must NOT re-add it (that second
            # emission is the double-unpin). It fires only on paths the callback
            # never reached -- a refused allocation never read the units, and a
            # throwing `pack` drained the device first
            # (`StagedTransfer._drain_device`) -- where withholding it would hold
            # an image out of the pool until the stale reclaimer noticed.
            self._store_submitted_at.pop(op, None)
            if not released:
                released = True
                self._source_released.add(op)
            # Report, never apply: `StateOffloadIndex` lives in the engine
            # process; the engine applies these via KVConnectorOutput.
            if stored:
                self._indexed.add(op)
            else:
                # The failure channel lets the aggregator take quorum on
                # `indexed | index_failed` rather than await a second report
                # that will never come from this rank.
                self._index_failed.add(op)

    def take_store_reports(self) -> tuple[set, set]:
        """`(operations stored, operations failed)` since the last call.

        An operation appears in exactly one of the two. The aggregator's quorum
        over them is failure-dominant, so a partial store resolves in the same
        step rather than pinning the key.
        """
        with self._lock:
            indexed = set(self._indexed)
            index_failed = set(self._index_failed)
            self._indexed.clear()
            self._index_failed.clear()
        return indexed, index_failed

    def take_source_releases(self) -> set:
        """Operations whose PAGE units the GPU has finished reading.

        Drained apart from `take_store_reports`, and usually in the same step:
        both are reported once `_do_store` returns, but the release is what
        hands the units back and the store report is what indexes the hash.
        """
        with self._lock:
            released = set(self._source_released)
            self._source_released.clear()
        return released
