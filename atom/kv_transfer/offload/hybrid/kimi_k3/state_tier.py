# SPDX-License-Identifier: MIT
"""Worker-side spill and load driver for the state offload tier.

One executor of its own, separate from the KV connector's `_load_executor` and
`_save_executor` so that state transfers and KV transfers cannot block each
other -- but a single one, shared by this tier's own spills and loads. See
`submit_load` for why the KV connector's load/save split is not copied here.

This class **reports, and the engine applies**. It runs in a spawned runner
process while `StateOffloadIndex` lives in the engine process, so this side
cannot free a staging slot or index a hash directly. `take_spill_reports` and
`get_finished` hand their sets to the connector, which emits the first as
`ConnectorCompletion`s on its own channels and merges the second into
`finished_loading`/`failed_loading`. The failed-hash set exists to resolve the
aggregator's quorum: without it a partial store would pin the hash forever.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

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
        # The spill path's three reports, drained by `take_spill_reports`.
        self._indexed: set[int] = set()
        self._index_failed: set[int] = set()
        self._released: set[int] = set()

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

    def submit_spill(
        self, h: int, entry_index: int, staging_slot: int, ready_event=None
    ) -> None:
        """`entry_index` is what the codec packs, `staging_slot` what the ring
        releases. Two index spaces: the staging entries sit past the pool's
        range in the arena (`num_groups + slot`), while the ring counts from 0.

        `ready_event` fences the D2D staging copy that `AttentionBackend.build`
        issued on the forward's compute stream. This worker packs on its own
        stream, so without the wait it reads the entry's previous occupant.
        """
        fut = self._executor.submit(
            self._do_spill, h, entry_index, staging_slot, ready_event
        )
        self._register(fut)

    def submit_load(self, req_id: str, h: int, group: int) -> None:
        """Fetch `h` into pool group `group` for the parked request `req_id`.

        Shares one serial executor with the spills, unlike the KV connector's
        split pair: `StagedTransfer` keeps its staging buffer in
        `threading.local`, so a second thread costs a second resident buffer
        per rank out of the HBM the state pool is already short of. The queue
        it would shorten is at most `OFFLOAD_STATE_STAGING_GROUPS` deep.
        """
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

    def _do_spill(
        self, h: int, entry_index: int, staging_slot: int, ready_event=None
    ) -> None:
        stored = False
        try:
            if ready_event is not None:
                ready_event.synchronize()
            stored = bool(self.codec.put(h, entry_index))
        except Exception:  # a spill is best effort by design
            logger.warning("state offload: spill of hash %d failed", h, exc_info=True)
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
            # Always, stored or not: a leaked slot shrinks the ring
            # permanently and the feature quietly stops spilling.
            self._released.add(int(staging_slot))

    def take_spill_reports(self) -> tuple[set[int], set[int], set[int]]:
        """`(hashes stored, staging slots free, hashes failed)` since the last call.

        A hash appears in exactly one of the first or third set per spill.
        The aggregator's quorum over the two is failure-dominant, so a partial
        store resolves in the same step rather than pinning the key.
        """
        with self._lock:
            indexed = set(self._indexed)
            index_failed = set(self._index_failed)
            released = set(self._released)
            self._indexed.clear()
            self._index_failed.clear()
            self._released.clear()
        return indexed, released, index_failed

    def _do_load(self, req_id: str, h: int, group: int) -> None:
        # A load target is a real pool group, not a staging entry: the bytes
        # land where the resuming request will read them. Only the spill
        # direction needs the staging indirection.
        #
        # A miss is a normal path, not an error: LMCache's LRU can drop bytes
        # under a hash the engine's index still advertises. Retracting that
        # claim is the engine's job -- it owns the index -- and it does it from
        # the report below.
        ok = False
        try:
            ok = bool(self.codec.get(h, group))
        except Exception:  # a failed load is a normal path
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
        self._need: dict[str, set[str]] = {}
        self._failed: set[str] = set()
        self._ready: set[str] = set()
        self._ready_failed: set[str] = set()

    def arm(self, req_id: str, *, needs_kv: bool, needs_state: bool) -> None:
        need = set()
        if needs_kv:
            need.add("kv")
        if needs_state:
            need.add("state")
        self._need[req_id] = need
        if not need:
            self._ready.add(req_id)

    def settle_kv(self, req_id: str, ok: bool) -> None:
        self._settle(req_id, "kv", ok)

    def settle_state(self, req_id: str, ok: bool) -> None:
        self._settle(req_id, "state", ok)

    def _settle(self, req_id: str, leg: str, ok: bool) -> None:
        need = self._need.get(req_id)
        if need is None:
            return
        need.discard(leg)
        if not ok:
            self._failed.add(req_id)
        if need:
            return
        del self._need[req_id]
        if req_id in self._failed:
            self._failed.discard(req_id)
            self._ready_failed.add(req_id)
        else:
            self._ready.add(req_id)

    def waits_for(self, req_id: str) -> bool:
        """Whether this park still owes `req_id` a leg.

        Asked before settling: the legs report through channels a single-leg
        request also uses, and `_settle` ignores unknown ids, which is
        indistinguishable from a leg that landed.
        """
        return req_id in self._need

    def take_ready(self) -> tuple[set[str], set[str]]:
        ready, failed = set(self._ready), set(self._ready_failed)
        self._ready.clear()
        self._ready_failed.clear()
        return ready, failed
