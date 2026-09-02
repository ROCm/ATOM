# SPDX-License-Identifier: MIT
"""Worker-side store and load driver for the state offload tier.

One executor of its own, separate from the KV connector's, so state and KV
transfers cannot block each other -- but a single one shared by this tier's own
stores and loads; see `submit_load`.

This class **reports, and the engine applies**. `StateOffloadIndex` lives in the
engine process, so nothing here can index a hash directly: `take_store_reports`
and `get_finished` hand their sets to the connector. The failed-hash set exists
to resolve the aggregator's quorum -- without it a partial store pins the hash
forever.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic
from typing import Any

logger = logging.getLogger(__name__)

#: A load waiting longer than this for its lane is worth a line: it is time
#: added straight to TTFT, and the in-flight *count* cannot show it.
_LOAD_WAIT_WARN_MS = 50.0


class StateOffloadTier:
    """Moves bytes; decides nothing, and holds no index.

    No index here: `StateOffloadIndex` lives in the engine process, so every
    counter and hash retraction is applied there from the reports below.
    Neither side can hold a second opinion about what is stored.
    """

    def __init__(self, codec, *, max_workers: int = 1, staging_lanes: int = 2) -> None:
        self.codec = codec
        # Two lanes, not one queue. A load is on the TTFT critical path and a
        # store is not, but a single serial executor made that ordering
        # unenforceable: a load submitted in a later scheduler step queued
        # behind every store already sitting in front of it, and one store
        # stuck in gather/D2H blocked every later load for as long as it ran.
        # Putting same-step loads first in the submit order does not help --
        # it cannot overtake work already queued.
        self._load_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state-load"
        )
        self._store_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state-store"
        )
        # What the lanes share is HBM, not a queue. `StagedTransfer` keeps its
        # staging buffer in `threading.local` and sizes it to a whole entry
        # (~55 MiB on K3), so each lane inside a transfer holds one. The
        # budget is stated here rather than left implicit in the thread count:
        # 2 gives each lane its own buffer, so a load never waits on a store at
        # all; 1 makes them share one, halving the standing HBM at the cost of
        # a load waiting out the single in-flight store. Either way a load
        # never queues behind a *backlog* of stores, which is the actual
        # head-of-line problem.
        self._staging_budget = threading.BoundedSemaphore(max(1, int(staging_lanes)))
        self._lock = threading.Lock()
        self._done: set[str] = set()
        self._failed: set[str] = set()
        self._inflight: set = set()
        # The store path's reports, drained by `take_store_reports`. Sets of
        # `StateStoreOperationId`, not of bare hashes: the engine settles the
        # pin for that exact generation, and the aggregator would tombstone a
        # bare hash after the first store of it.
        #
        # `_source_released` is a separate report from `_indexed` because the
        # two answer different questions and land at different times. The
        # source is the KV pool's PAGE units and is free the instant the D2H
        # drains; whether the CPU put succeeded is decided afterwards and
        # cannot touch them. Reporting only the second would hold an image out
        # of the pool across an operation that does not need it.
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

    def submit_load(self, req_id: str, h: int, slot: int) -> None:
        """Fetch `h` into pool slot `slot` for the parked request `req_id`.

        Its own lane, so a load is never behind a backlog of stores. What the
        two lanes still share is the staging-memory budget; see `__init__`.
        `_do_load` warns when that remaining wait crosses `_LOAD_WAIT_WARN_MS`.
        """
        self._register(
            self._load_executor.submit(self._do_load, req_id, h, slot, monotonic())
        )

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
        self._load_executor.shutdown(wait=True)
        self._store_executor.shutdown(wait=True)

    def _do_store(self, op, unit_ids) -> None:
        stored = False
        released = False

        def _source_released() -> None:
            # Fires from `codec.put` the moment `pack`'s gather+D2H have
            # drained (state_object.put), before `batched_put`. Publish the
            # release *here*, early: the PAGE units are the KV pool's and the
            # GPU has stopped reading them, so hand them back now instead of
            # holding a whole image out of the pool across the CPU put. Under
            # `self._lock` because `take_source_releases` drains the set from
            # the engine thread; the flag makes the end-of-store backstop a
            # no-op so the release is emitted exactly once (a second emission
            # would double-unpin on the engine side).
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                self._source_released.add(op)

        try:
            with self._staging_budget:
                stored = bool(
                    self.codec.put(
                        int(op.prefix_hash),
                        unit_ids,
                        on_source_released=_source_released,
                    )
                )
        except Exception:  # deliberately blind, see below
            # Deliberately blind. `codec.put` reaches into LMCache, whose
            # failure modes are its own, and a store that cannot happen must
            # cost one checkpoint's CPU copy -- not this worker thread, whose
            # death would strand every request parked on a later load.
            logger.warning(
                "state offload: store of hash %d (generation %d) failed",
                op.prefix_hash,
                op.generation,
                exc_info=True,
            )
        with self._lock:
            # Backstop, not the primary path: on success the D2H callback above
            # already published the release early, so this must NOT add it again
            # -- a second emission across two engine drains is the double-unpin.
            # It still fires on the paths the callback never reached: a refused
            # allocation never read the units, and a throwing `pack` drained the
            # device before it propagated (`StagedTransfer._drain_device`), so
            # the GPU has stopped reading them there too. Withholding it on those
            # paths would hold an image out of the KV pool until the stale
            # reclaimer noticed.
            self._store_submitted_at.pop(op, None)
            if not released:
                released = True
                self._source_released.add(op)
            # Report, never apply: `StateOffloadIndex` lives in the engine
            # process and this runs in a spawned runner. The engine applies
            # these via KVConnectorOutput.
            if stored:
                self._indexed.add(op)
            else:
                # The failure channel lets the aggregator take quorum on
                # `indexed | index_failed` instead of waiting for a second
                # report that will never come from this rank.
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

    def _do_load(self, req_id: str, h: int, slot: int, submitted_at=None) -> None:
        # The bytes land in the committed slot, where the resuming request
        # reads them.
        #
        # A miss is a normal path, not an error: LMCache's LRU can drop bytes
        # under a hash the engine's index still advertises. Retracting that
        # claim is the engine's job -- it owns the index -- and it does it from
        # the report below.
        if submitted_at is not None:
            waited_ms = max(0.0, monotonic() - submitted_at) * 1000.0
            if waited_ms >= _LOAD_WAIT_WARN_MS:
                logger.warning(
                    "state offload: a state load waited %.0fms for its lane "
                    "(oldest store outstanding %.1fs). This is TTFT.",
                    waited_ms,
                    self.oldest_store_age_s(),
                )
        ok = False
        try:
            with self._staging_budget:
                ok = bool(self.codec.get(h, slot))
        except Exception:  # a failed load is a normal path
            # Same reasoning as `_do_store`, and here a miss is expected:
            # LMCache's LRU can drop bytes under a hash the index still
            # advertises. The report below is what retracts the claim.
            logger.warning("state offload: load of hash %d failed", h, exc_info=True)
        with self._lock:
            if ok:
                self._done.add(req_id)
            else:
                self._failed.add(req_id)


class _JointPark:
    """One park for the KV load and the state load of the same request.

    Both completions must land before unpark. Waking on the state transfer
    alone lets the model read KV blocks that are not yet filled, which is
    silent rather than an error.

    Either side failing fails the pair: half a load leaves state claiming a
    prefix whose KV never arrived, and `failed_loading` already means "wake for
    recompute using the blocks already allocated", which is exactly right here.
    """

    def __init__(self) -> None:
        self._need: dict[Any, set[str]] = {}
        self._failed: set = set()
        self._ready: set = set()
        self._ready_failed: set = set()
        # The two legs do not report the same identity. The KV leg reports
        # whatever `_load_completion_id` yields -- a typed `LoadOperationId`
        # whenever the scheduler issued one, which for an offload load is
        # always -- while the state tier is keyed by request and reports the
        # bare id. The park is filed under the KV identity, because that is
        # what has to reach the engine on `finished_loading`; this maps the
        # bare id onto it so a state report can find its own park.
        self._alias: dict = {}
        self._alias_of: dict = {}

    def arm(
        self,
        req_id: str,
        *,
        needs_kv: bool,
        needs_state: bool,
        kv_id=None,
    ) -> None:
        """Park `req_id`, filed under `kv_id` when the KV leg reports one.

        `kv_id` must be exactly what the KV worker will put on
        `finished_loading`/`failed_loading` for this load, because that report
        is matched by equality and nothing translates it on the way in.
        """
        need = set()
        if needs_kv:
            need.add("kv")
        if needs_state:
            need.add("state")
        key = req_id if kv_id is None else kv_id
        self._need[key] = need
        if key != req_id:
            self._alias[req_id] = key
            self._alias_of[key] = req_id
        if not need:
            self._release(key)

    def settle_kv(self, ident, ok: bool) -> None:
        self._settle(ident, "kv", ok)

    def settle_state(self, ident, ok: bool) -> None:
        self._settle(ident, "state", ok)

    def _resolve(self, ident):
        """The park key for either leg's identity. Identity when unarmed."""
        return self._alias.get(ident, ident)

    def _settle(self, ident, leg: str, ok: bool) -> None:
        key = self._resolve(ident)
        need = self._need.get(key)
        if need is None:
            return
        need.discard(leg)
        if not ok:
            self._failed.add(key)
        if need:
            return
        self._release(key)

    def _release(self, key) -> None:
        self._need.pop(key, None)
        bare = self._alias_of.pop(key, None)
        if bare is not None:
            self._alias.pop(bare, None)
        if key in self._failed:
            self._failed.discard(key)
            self._ready_failed.add(key)
        else:
            self._ready.add(key)

    def waits_for(self, ident) -> bool:
        """Whether this park still owes `ident`'s request a leg.

        Asked before settling: the legs report through channels a single-leg
        request also uses, and `_settle` ignores unknown ids, which is
        indistinguishable from a leg that landed. Accepts either leg's
        identity, so the caller does not have to know which channel it is
        draining.
        """
        return self._resolve(ident) in self._need

    def take_ready(self) -> tuple[set[str], set[str]]:
        ready, failed = set(self._ready), set(self._ready_failed)
        self._ready.clear()
        self._ready_failed.clear()
        return ready, failed
