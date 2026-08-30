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
from typing import Any

logger = logging.getLogger(__name__)


class StateOffloadTier:
    """Moves bytes; decides nothing, and holds no index.

    No index here: `StateOffloadIndex` lives in the engine process, so every
    counter and hash retraction is applied there from the reports below.
    Neither side can hold a second opinion about what is stored.
    """

    def __init__(self, codec, *, max_workers: int = 1) -> None:
        self.codec = codec
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state"
        )
        self._lock = threading.Lock()
        self._done: set[str] = set()
        self._failed: set[str] = set()
        self._inflight: set = set()
        # The store path's two reports, drained by `take_store_reports`.
        self._indexed: set[int] = set()
        self._index_failed: set[int] = set()

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

    def submit_store(self, h: int, unit_ids) -> None:
        """Pack the checkpoint image in `unit_ids` under `h`, for LMCache.

        No `ready_event`: the units are reserved out of the KV pool and pinned
        by the engine for the length of this transfer, so nothing on the compute
        stream is writing them and the packer gathers straight from where they
        sit.
        """
        self._register(self._executor.submit(self._do_store, h, unit_ids))

    def submit_load(self, req_id: str, h: int, slot: int) -> None:
        """Fetch `h` into pool slot `slot` for the parked request `req_id`.

        Shares one serial executor with the stores: `StagedTransfer` keeps its
        staging buffer in `threading.local`, so a second thread costs a second
        resident buffer per rank. A load is on the TTFT critical path and a
        store is not, so if stores become frequent this queue wants a priority;
        measure `state_load_queue_wait_ms` before adding one.
        """
        self._register(self._executor.submit(self._do_load, req_id, h, slot))

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

    def _do_store(self, h: int, unit_ids) -> None:
        stored = False
        try:
            stored = bool(self.codec.put(h, unit_ids))
        except Exception:  # deliberately blind, see below
            # Deliberately blind. `codec.put` reaches into LMCache, whose
            # failure modes are its own, and a store that cannot happen must
            # cost one checkpoint's CPU copy -- not this worker thread, whose
            # death would strand every request parked on a later load.
            logger.warning("state offload: store of hash %d failed", h, exc_info=True)
        with self._lock:
            # Report, never apply: `StateOffloadIndex` lives in the engine
            # process and this runs in a spawned runner. The engine applies
            # these via KVConnectorOutput.
            if stored:
                self._indexed.add(int(h))
            else:
                # The failure channel lets the aggregator take quorum on
                # `indexed | index_failed` instead of waiting for a second
                # report that will never come from this rank.
                self._index_failed.add(int(h))

    def take_store_reports(self) -> tuple[set[int], set[int]]:
        """`(hashes stored, hashes failed)` since the last call.

        A hash appears in exactly one of the two per store. The aggregator's
        quorum over them is failure-dominant, so a partial store resolves in
        the same step rather than pinning the key.
        """
        with self._lock:
            indexed = set(self._indexed)
            index_failed = set(self._index_failed)
            self._indexed.clear()
            self._index_failed.clear()
        return indexed, index_failed

    def _do_load(self, req_id: str, h: int, slot: int) -> None:
        # The bytes land in the committed slot, where the resuming request
        # reads them.
        #
        # A miss is a normal path, not an error: LMCache's LRU can drop bytes
        # under a hash the engine's index still advertises. Retracting that
        # claim is the engine's job -- it owns the index -- and it does it from
        # the report below.
        ok = False
        try:
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
