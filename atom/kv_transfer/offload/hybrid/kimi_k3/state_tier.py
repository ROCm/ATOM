# SPDX-License-Identifier: MIT
"""Worker-side store and load driver for the state offload tier.

Its own executors, separate from the KV connector's, so state and KV transfers
cannot block each other; see `__init__` for the store/load lane split.

This class **reports, and the engine applies**: `StateOffloadIndex` lives in the
engine process, so `take_store_reports`/`get_finished` hand their sets to the
connector rather than index a hash here. The failed-hash set resolves the
aggregator's quorum -- without it a partial store pins the hash forever.
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
        # store is not; a single serial executor made that unenforceable -- a
        # later-step load queued behind every store in front of it, and one
        # store stuck in gather/D2H blocked every later load. Submit order
        # cannot overtake work already queued.
        self._load_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state-load"
        )
        self._store_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state-store"
        )
        # `staging_lanes` gates *concurrency* between the load and store lanes,
        # not standing HBM. `StagedTransfer._tls = threading.local()` keys the
        # ~55 MiB staging buffer per thread, both executors are `max_workers=1`,
        # and `release_after_transfer` defaults off -- so standing HBM is exactly
        # two buffers (one per executor thread) whether this is 1 or 2. The knob
        # only decides whether a load and a store may run at once:
        #   2 (default): they overlap; the semaphore is never contended.
        #   1: a load waits out any in-flight store, re-introducing the
        #      head-of-line coupling the two-executor split exists to remove,
        #      and saves zero HBM.
        # Do not read this as an HBM lever; it is a lane-serialisation switch.
        self._staging_budget = threading.BoundedSemaphore(max(1, int(staging_lanes)))
        self._lock = threading.Lock()
        self._done: set[str] = set()
        self._failed: set[str] = set()
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
            with self._staging_budget:
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

    def _do_load(self, req_id: str, h: int, slot: int, submitted_at=None) -> None:
        # The bytes land in the committed slot, where the resuming request
        # reads them.
        #
        # A miss is a normal path, not an error: LMCache's LRU can drop bytes
        # under a hash the engine's index still advertises. Retracting that
        # claim is the engine's job -- it owns the index -- and it does it from
        # the report below.
        ok = False
        try:
            with self._staging_budget:
                # Measure the wait *after* acquiring the lane, not before. The
                # semaphore acquire is the lane contention this metric exists to
                # surface: under `staging_lanes=1` a load blocks here for a full
                # ~55 MiB gather+D2H behind an in-flight store. Sampling
                # `submitted_at` above the `with` timed only the executor-queue
                # wait and was structurally blind to the acquire -- the one thing
                # it was added to see. Now `waited_ms` spans both.
                if submitted_at is not None:
                    waited_ms = max(0.0, monotonic() - submitted_at) * 1000.0
                    if waited_ms >= _LOAD_WAIT_WARN_MS:
                        logger.warning(
                            "state offload: a state load waited %.0fms for its "
                            "lane (oldest store outstanding %.1fs). This is TTFT.",
                            waited_ms,
                            self.oldest_store_age_s(),
                        )
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
        # Per-leg outcome, kept apart from the single failure-dominant `_failed`
        # flag (finding #6). `_failed` fires when *either* leg fails, which is
        # right for the KV wake -- half a load must recompute -- but wrong for
        # the state index: a KV chunk that LMCache's LRU dropped fails the pair
        # while the state H2D that landed is still intact, and routing that
        # through `fail_load` would `forget` a hash whose bytes are present,
        # denying it to every later request over the prefix. `_state_failed`
        # records the keys whose *state* leg specifically missed, so `_release`
        # can tell "keep the state hash, only KV recompute" (abandon) from "the
        # state bytes are really gone" (fail/forget). `_dispositions` carries
        # that verdict per released key out to the worker's completion channel.
        self._state_failed: set = set()
        self._dispositions: dict = {}
        # The two legs report different identities: the KV leg reports its typed
        # `LoadOperationId` (always issued for an offload load), the state tier
        # the bare request id. The park is filed under the KV identity -- that
        # is what reaches the engine on `finished_loading` -- and this maps the
        # bare id onto it so a state report can find its own park.
        self._alias: dict = {}
        self._alias_of: dict = {}
        # Monotonic stamp per armed key, for `reclaim_stale_parks` (finding #3).
        # A park is only ever released by both legs reporting; a report lost to
        # an aborted request, a killed worker thread, or a dropped completion
        # would otherwise leave the key in `_need`/`_alias`/`_alias_of` for the
        # life of the process -- and `_settle_joint` would swallow every later
        # KV completion that reuses the stale `kv_id`. `abort` is the prompt
        # exit; this stamp is the backstop the reconciler sweeps.
        self._armed_at: dict = {}

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
        # Evict any bookkeeping a prior park for this same `req_id` left behind
        # before filing the new one. A preempt/requeue re-arms the request,
        # usually under a fresh `kv_id`; without this the old key's `_need`
        # entry and its reverse `_alias_of[old_key] -> req_id` survive. A stray
        # late `settle` for the old key would then `_release(old_key)`, whose
        # `_alias_of` -> `_alias` cleanup pops `_alias[req_id]` -- the live
        # mapping the re-armed park's state leg needs -- so that park's state
        # report resolves to the bare id, finds no `_need`, and never unparks.
        # `_purge` drops the stale key everywhere without emitting a
        # ready/failed/disposition (an eviction, not a settled outcome).
        self._purge(self._alias.pop(req_id, None))
        self._purge(req_id)
        need = set()
        if needs_kv:
            need.add("kv")
        if needs_state:
            need.add("state")
        key = req_id if kv_id is None else kv_id
        self._need[key] = need
        self._armed_at[key] = monotonic()
        if key != req_id:
            self._alias[req_id] = key
            self._alias_of[key] = req_id
        if not need:
            self._release(key)

    def _purge(self, key) -> None:
        """Drop a key from every park structure without settling it.

        Used only by re-arm to evict a stale prior park; unlike `_release` it
        produces no `_ready`/`_ready_failed`/`_dispositions` entry, so an evicted
        load never masquerades as a completed one. No-op on None (nothing was
        aliased) and on an unknown key."""
        if key is None or key not in self._need:
            return
        self._need.pop(key, None)
        self._armed_at.pop(key, None)
        bare = self._alias_of.pop(key, None)
        if bare is not None:
            self._alias.pop(bare, None)
        self._failed.discard(key)
        self._state_failed.discard(key)
        self._ready.discard(key)
        self._ready_failed.discard(key)
        self._dispositions.pop(key, None)

    def reclaim_stale_parks(self, max_age_s: float) -> int:
        """Evict parks armed longer than `max_age_s` ago; return the count.

        The park's abort/expiry/eviction exit (finding #3). A joint park is
        released only by both legs reporting; a report lost to an aborted request
        (the KV leg cancelled scheduler-side, never reporting, while the state
        tier's leg still lands -- half a report cannot release the pair), a
        worker thread killed mid-transfer, or a completion dropped between worker
        and engine would otherwise leave the key in `_need` (with its
        `_alias`/`_alias_of`) forever, and `_settle_joint` would swallow every
        later KV completion reusing that stale `kv_id`. The abort signal is
        scheduler-side and cannot reach this worker-side park synchronously, so
        expiry, not a synchronous abort call, is the exit; the same-request
        re-admission shape is separately handled by `arm`'s pre-file `_purge`.

        Like the engine's pin and orphan-slot reconcilers it cannot tell a lost
        report from a merely slow one, so the caller passes the same
        save-abandon window; a park that outlives it is treated as never going
        to complete. Eviction (no ready/failed) is safe: a load whose report
        truly arrives after this finds its key gone and passes through
        `_settle_joint` as an unarmed id, which the engine drops for the
        long-departed request rather than manufacturing a restore.
        """
        cutoff = monotonic() - max_age_s
        stale = [key for key, at in self._armed_at.items() if at <= cutoff]
        for key in stale:
            self._purge(key)
        return len(stale)

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
            if leg == "state":
                # Only a state-leg miss is evidence the bytes are gone (finding
                # #6). A KV-leg failure leaves `_state_failed` untouched, so the
                # disposition `_release` files says "state intact" and the index
                # abandons rather than forgets.
                self._state_failed.add(key)
        if need:
            return
        self._release(key)

    def _release(self, key) -> None:
        self._need.pop(key, None)
        self._armed_at.pop(key, None)
        bare = self._alias_of.pop(key, None)
        if bare is not None:
            self._alias.pop(bare, None)
        # File the state-index verdict for this key before dropping the per-leg
        # bookkeeping (finding #6): True means the state bytes are intact (the
        # index should keep the hash -- `abandon_load`), False means the state
        # leg itself missed (the index should drop it -- `fail_load`). Every
        # released key gets one, success or failure, so the worker emits a
        # completion the TP aggregator can bring to quorum alongside the KV load
        # report rather than leaking a partially-reported key.
        self._dispositions[key] = key not in self._state_failed
        self._state_failed.discard(key)
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

    def take_dispositions(self) -> dict:
        """State-index verdict per released key since the last drain (finding #6).

        `{key: state_intact}` -- True keeps the hash (abandon), False forgets it
        (fail). Drained together with `take_ready`: the worker turns each into a
        `STATE_LOAD_DISPOSITION_CHANNEL` completion so the engine can override
        the failure-dominant default of `fail_load` for a load whose only failed
        leg was the KV chunk.
        """
        out = self._dispositions
        self._dispositions = {}
        return out
