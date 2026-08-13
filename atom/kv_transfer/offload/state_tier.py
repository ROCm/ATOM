# SPDX-License-Identifier: MIT
"""Worker-side spill and load driver for the state offload tier.

Its own executor, separate from the KV connector's `_load_executor` and
`_save_executor`. The reason is the one recorded at `connector.py:83-88`: a
load is on the TTFT critical path -- a parked sequence is waiting for it --
and must never queue behind a backlog of fire-and-forget spills.

On the spill path this class **reports, and the engine applies**. It runs in a
spawned runner process; `StateOffloadIndex` lives in the engine process, so
this side cannot free a staging slot or index a hash directly. `take_spill_reports`
hands both sets to `LMCacheOffloadConnector.get_finished`, which puts them on
`KVConnectorOutput` for the engine to apply in
`Scheduler._update_from_kv_xfer_finished`.
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
        # The spill path's two reports, drained by `take_spill_reports`.
        self._indexed: set[int] = set()
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

        `ready_event` is the producer fence for the D2D staging copy, which
        `AttentionBackend.build()` issued on the forward's compute stream.
        This worker packs on its own stream, so without the wait it reads the
        staging entry's previous occupant. Same reasoning, same shape as the
        KV path's `save_ready_event` (`connector.py:236-240`).
        """
        fut = self._executor.submit(
            self._do_spill, h, entry_index, staging_slot, ready_event
        )
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
            # Always, stored or not: a leaked slot shrinks the ring
            # permanently and the feature quietly stops spilling.
            self._released.add(int(staging_slot))

    def take_spill_reports(self) -> tuple[set[int], set[int]]:
        """`(hashes stored, staging slots free)` since the last call."""
        with self._lock:
            indexed, released = set(self._indexed), set(self._released)
            self._indexed.clear()
            self._released.clear()
        return indexed, released

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


def clamp_state_boundary(state_blocks: int, kv_loaded_blocks: int) -> int:
    """`P <= L`: a state boundary may not claim history the KV does not cover.

    Today this holds for free because both derive from `block_hashes`. Once the
    tier admits spilled hashes and the KV load length is decided on a separate
    path, it does not, and the failure is silent: state is the compressed
    history of [0,P), so with P > L the forward reads a compressed prefix whose
    raw KV was never loaded and produces wrong output without raising.

    Clamping to 0 means the sequence recomputes -- the existing path.
    """
    return max(0, min(int(state_blocks), int(kv_loaded_blocks)))


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

    def take_ready(self) -> tuple[set[str], set[str]]:
        ready, failed = set(self._ready), set(self._ready_failed)
        self._ready.clear()
        self._ready_failed.clear()
        return ready, failed


def should_load_state(hit_tokens: int, floor_tokens: int) -> bool:
    """Whether a state hit of `hit_tokens` is worth an H2D.

    Mirrors KV's OFFLOAD_MIN_LOAD_TOKENS (`connector.py:526`). Two jobs: a
    short prefix does not repay the round trip, and the same floor bounds what
    a false positive costs -- the index cannot know LMCache's LRU dropped the
    bytes until the load misses.
    """
    hit_tokens = int(hit_tokens)
    return hit_tokens > 0 and hit_tokens >= int(floor_tokens)
