# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Composite KV connector — run several sub-connectors behind one interface.

The canonical use case is a prefill node that must do two things with the same
KV at once:

* **moriio** (``kv_role: kv_producer``) — RDMA-send the KV to a remote decode
  node for P/D disaggregation;
* **lmcache_offload** (``kv_role: offload``) — save the KV to CPU/NVMe so a
  future request that shares the prefix can skip recompute.

A single engine selects exactly one connector (``KVConnectorFactory`` reads one
``kv_connector`` name). ``MultiConnector`` is that one connector; it owns a list
of real sub-connectors and merges their results so the engine, scheduler, and
output aggregator stay unchanged.

Config::

    --kv-transfer-config '{
      "kv_connector": "multi",
      "connectors": [
        {"kv_connector": "moriio", "kv_role": "kv_producer", "proxy_ip": "...", ...},
        {"kv_connector": "lmcache_offload", "kv_role": "offload"}
      ]
    }'

Merge strategy mirrors vLLM's ``MultiConnector``, adapted to ATOM's
``base.py`` interface:

* ``get_num_new_matched_tokens`` — **first-hit-wins**: the first sub-connector
  that reports a prefix match owns the load for that request.
* ``update_state_after_alloc`` / ``request_finished`` — fan out to **all** subs
  (moriio sets up its send, offload sets up its save; both must run).
* ``build_connector_meta`` — returns :class:`MultiConnectorMetadata` carrying one
  sub-metadata per connector, in connector order. The worker de-multiplexes by
  index in ``start_load_kv``.
* ``get_finished`` — union the completion sets, **but** see the send/save
  pairing below.
* ``_state_tier`` — the state offload tier, if a sub built one, re-exposed on
  the composite by ``_adopt_state_tier`` at ``register_kv_caches`` time (which
  also refuses a config that lists two offload subs). Mirroring the sub's tier
  on the composite keeps the attribute defined, so a probe for it resolves to
  the real tier instead of raising ``AttributeError``.

Send/save pairing (the one tricky correctness point)
----------------------------------------------------
On a producer node the scheduler frees a finished request's blocks as soon as it
sees ``finished_sending`` (``scheduler.py``: producer path), and it can *also*
free on ``finished_saving`` when the connector does not defer. If offload is
still reading those blocks for its save when the moriio send completes (or vice
versa), the free would corrupt the in-flight transfer. So when a request needs
**both** a send and one or more saves, ``MultiConnector`` withholds *both*
completion signals until every known save is done, then emits them together.
The scheduler's ``finished_sending`` handler frees first; the
``finished_saving`` handler then finds nothing to free and no-ops. This is the
analogue of vLLM's ``_extra_async_saves`` refcount.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorMetadata,
    KVConnectorOutput,
    SaveCompletionId,
    StateStoreOperationId,
    completion_req_key,
    connector_metadata_has_work,
)

logger = logging.getLogger("atom")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_subconnectors(config: Any, role: str) -> list:
    """Instantiate each sub-connector listed in ``kv_transfer_config.connectors``.

    Each entry is a full ``kv_transfer_config`` dict (with its own
    ``kv_connector`` name). We shallow-copy the engine config, swap in the
    sub-dict, and route through the normal factory — no recursion, since each
    sub names a concrete backend (moriio / lmcache_offload / ...), not ``multi``.
    """
    # Imported lazily: the factory module registers backends at import time and
    # we must not create an import cycle with it.
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    kvc = getattr(config, "kv_transfer_config", None) or {}
    subs = kvc.get("connectors")
    if not subs:
        raise ValueError(
            "multi connector requires a non-empty 'connectors' list in "
            "kv_transfer_config"
        )

    connectors = []
    for i, sub in enumerate(subs):
        if not isinstance(sub, dict) or "kv_connector" not in sub:
            raise ValueError(
                f"connectors[{i}] must be a dict with a 'kv_connector' key, "
                f"got {sub!r}"
            )
        if sub["kv_connector"] == "multi":
            raise ValueError("multi connector cannot nest another 'multi'")
        cfg_i = copy.copy(config)
        cfg_i.kv_transfer_config = sub
        connectors.append(KVConnectorFactory.create_connector(cfg_i, role=role))
        logger.debug(
            "multi: built sub-connector[%d] backend=%s role=%s",
            i,
            sub["kv_connector"],
            role,
        )
    return connectors


def _normalize_finished(finished: Any) -> KVConnectorOutput:
    """Coerce a sub-connector's ``get_finished()`` result to KVConnectorOutput.

    Legacy P/D connectors (moriio/mooncake) return a ``(done_sending,
    done_recving)`` tuple; the offload connector already returns a full
    :class:`KVConnectorOutput`.
    """
    if isinstance(finished, KVConnectorOutput):
        return finished
    done_sending, done_recving = finished
    return KVConnectorOutput(
        finished_sending=set(done_sending or ()),
        finished_recving=set(done_recving or ()),
    )


def _first_with(connectors: list, name: str):
    """Return the first sub-connector exposing attribute/method *name*, or None."""
    for c in connectors:
        if hasattr(c, name):
            return c
    return None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class MultiConnectorMetadata(ConnectorMetadata):
    """Carries one sub-connector metadata per connector, in connector order.

    Subclasses :class:`ConnectorMetadata` so existing ``isinstance`` checks and
    the worker dispatch path accept it unchanged. The worker reads ``metas`` and
    routes ``metas[i]`` to ``connectors[i].start_load_kv``.
    """

    def __init__(self, metas: list) -> None:
        super().__init__()
        self.metas = list(metas)

    def has_work(self) -> bool:
        """Ask the subs; this wrapper holds none of the work itself.

        Its own base fields are always empty -- everything lives in `metas` --
        so answering from them alone drops every step whose only work belongs
        to a sub. Delegating rather than mirroring the subs' fields is the
        point: the aggregating properties below exist for the idle-dispatch
        path and have to name each field, and `state_loads` was missed there,
        which silently parked every state-only load run under `multi`.
        """
        return super().has_work() or any(
            connector_metadata_has_work(m) for m in self.metas
        )

    @property
    def requests(self):
        """Aggregate of sub-metas' ``requests`` (offload uses this attribute).

        ``EngineCore._dispatch_idle_offload_work`` gates its idle dispatch on a
        truthy ``meta.requests``; exposing it here keeps offload's idle
        save/load flowing when offload runs inside a ``multi`` connector.
        """
        agg: list = []
        for m in self.metas:
            sub = getattr(m, "requests", None)
            if sub:
                agg.extend(sub)
        return agg

    @property
    def lookup_requests_in_step(self):
        """Aggregate of sub-metas' pending lookup-pin releases.

        Same reason as ``requests``: the idle dispatch gates on this, and a
        metadata dropped for looking empty takes the sub-meta's only unpin
        with it.
        """
        agg: list = []
        for m in self.metas:
            sub = getattr(m, "lookup_requests_in_step", None)
            if sub:
                agg.extend(sub)
        return agg


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


class MultiConnector(KVConnectorBase):
    """Worker-side composite connector (one instance per TP rank)."""

    def __init__(self, config: Any) -> None:
        self._connectors = _build_subconnectors(config, role="worker")
        # Producer if any sub is a producer (moriio kv_producer drives the
        # scheduler's producer-side deferred-free path).
        self.is_producer = any(
            getattr(c, "is_producer", False) for c in self._connectors
        )

        pp_rank = getattr(
            getattr(config, "parallel_config", None),
            "pipeline_parallel_rank",
            0,
        )
        self._pp_is_head = pp_rank == 0

        # Send/save pairing state, all keyed by str(req_id). See module
        # docstring. The values below are completion identities, not keys: a
        # pending save is named by its SaveOperationId, or by the bare req_id
        # when the metadata carries none -- whichever the worker will report.
        self._pending_save_ops: dict[str, set[SaveCompletionId]] = {}
        self._sent: dict[str, Any] = {}
        self._saved: dict[str, set[SaveCompletionId]] = {}
        # The state tier of whichever sub owns one. Adopted in
        # `register_kv_caches` via `_adopt_state_tier`; set to None here so the
        # attribute exists before the subs register -- a probe for it must
        # resolve, not raise `AttributeError`.
        self._state_tier = None

    @property
    def _pairs_send_and_save(self) -> bool:
        """Whether this rank has a send to pair its saves against.

        Only a producer's PP stage 0 does: mooncake reports done_sending on
        stage 0 alone (via ``_record_release``). Every other rank passes both
        completions straight through and must keep no pairing state.
        """
        return self.is_producer and self._pp_is_head

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        for c in self._connectors:
            c.register_kv_caches(kv_caches, transfer_tensors, num_blocks)
        self._adopt_state_tier()

    def _adopt_state_tier(self) -> None:
        """Take over the one sub-connector's state tier, or refuse two.

        Nothing in ``_build_subconnectors`` stops a config from listing
        ``lmcache_offload`` twice, which would leave two live tiers and no
        answer to "which one packs this spill". Picking the first is wrong
        rather than arbitrary: a hash could be reported indexed by a tier that
        never stored it, then fetched from one that cannot produce it. Raising
        at model load costs nothing and is loud.
        """
        tiers = [
            c for c in self._connectors if getattr(c, "_state_tier", None) is not None
        ]
        if len(tiers) > 1:
            names = [type(c).__name__ for c in tiers]
            raise ValueError(
                f"multi connector: {len(tiers)} sub-connectors built a state "
                f"offload tier ({names}); exactly one may. List the offload "
                "backend once in kv_transfer_config.connectors."
            )
        self._state_tier = tiers[0]._state_tier if tiers else None

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        metas = getattr(metadata, "metas", None)
        if metas is None:
            logger.warning(
                "multi: start_load_kv got %s, expected MultiConnectorMetadata",
                type(metadata).__name__,
            )
            return
        for c, m in zip(self._connectors, metas):
            if m is None:
                continue
            # Remember what offload is about to save, so get_finished can hold
            # the send until it finishes.
            if self._pairs_send_and_save:
                reqs = getattr(m, "requests", None)
                if reqs:
                    for req in reqs:
                        has_save = (
                            getattr(req, "save_spec", None) is not None
                            or getattr(req, "slot_save_spec", None) is not None
                        )
                        if not has_save:
                            continue
                        operation = getattr(req, "save_operation", None)
                        self._pending_save_ops.setdefault(
                            completion_req_key(req.req_id), set()
                        ).add(operation if operation is not None else req.req_id)
            c.start_load_kv(m)

    def get_finished(self) -> KVConnectorOutput:
        recv: set = set()
        failed: set = set()
        loaded: set = set()
        load_failed: set = set()
        send_now: list = []
        save_now: list = []
        completions: set = set()
        for c in self._connectors:
            o = _normalize_finished(c.get_finished())
            recv |= o.finished_recving
            failed |= o.failed_recving
            loaded |= o.finished_loading
            load_failed |= o.failed_loading
            send_now.extend(o.finished_sending)
            save_now.extend(o.finished_saving)
            completions |= o.connector_completions

        out = KVConnectorOutput(
            finished_recving=recv,
            failed_recving=failed,
            finished_loading=loaded,
            failed_loading=load_failed,
            connector_completions=completions,
        )

        if not self._pairs_send_and_save:
            out.finished_sending = set(send_now)
            out.finished_saving = set(save_now)
            return out

        # Pair each request's send and save before releasing either.
        for r in send_now:
            self._sent[str(r)] = r
        # State-tier store completions (`StateStoreOperationId`: a
        # (prefix_hash, generation) pair) have no send counterpart to pair
        # against. Parking them in `self._saved` leaked for the life of the
        # process -- their key never enters `self._sent`, so the pop below never
        # fired. They are terminal on their own: release immediately. Match on
        # the exact type, not `hasattr(r, "req_id")` -- a bare `ReqId` save
        # completion (a plain str/int) has no `req_id` attribute either and must
        # still go through send/save pairing on a producer node.
        state_saves: set = set()
        for r in save_now:
            if isinstance(r, StateStoreOperationId):
                state_saves.add(r)
                continue
            key = completion_req_key(r)
            self._saved.setdefault(key, set()).add(r)
            pending_ops = self._pending_save_ops.get(key)
            if pending_ops is not None:
                pending_ops.discard(r)
                if not pending_ops:
                    self._pending_save_ops.pop(key, None)

        rel_send: set = set()
        rel_save: set = set()
        for key, raw in list(self._sent.items()):
            if self._pending_save_ops.get(key):
                continue  # hold: save still in flight for this request
            rel_send.add(raw)
            del self._sent[key]
            rel_save.update(self._saved.pop(key, set()))

        out.finished_sending = rel_send
        out.finished_saving = rel_save | state_saves
        return out

    def get_finished_recv_blocks(self) -> list[int]:
        blocks: list[int] = []
        for c in self._connectors:
            blocks.extend(c.get_finished_recv_blocks())
        return blocks

    def close(self) -> None:
        """Tear down every sub-connector at worker teardown.

        `ModelRunner.exit()` resolves `getattr(connector, "close", None)` on the
        composite under `multi`; without this forwarder it returns None and no
        sub is joined -- the offload sub's non-daemon `lmc-state-store` /
        `lmc-state-load` / `offload-save` threads keep copying out of the KV pool
        that `destroy_dist_env()` is about to release. Guard per sub (`getattr`):
        a producer sub such as moriio need not implement `close`.
        """
        for c in self._connectors:
            fn = getattr(c, "close", None)
            if callable(fn):
                fn()


# ---------------------------------------------------------------------------
# Scheduler side
# ---------------------------------------------------------------------------


class MultiConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler-side composite connector."""

    def __init__(self, config: Any) -> None:
        self._connectors = _build_subconnectors(config, role="scheduler")
        self.is_producer = any(
            getattr(c, "is_producer", False) for c in self._connectors
        )
        # Opt into the scheduler's offload suffix-prefill path if any sub is the
        # offload backend (Scheduler._is_offload_connector reads this).
        self.is_offload = any(getattr(c, "is_offload", False) for c in self._connectors)
        # The sub that won `get_num_new_matched_tokens` for each in-flight
        # request, keyed by `seq.id`. The load-side decisions
        # (`should_park_for_load_after_alloc` and the two chunk/partial-prefill
        # siblings) belong to whichever sub armed the load, not to whichever sub
        # hosts the state tier -- see `_load_owner` (review round 5, finding 1).
        self._load_winner: dict = {}

    def _load_owner(self, seq: Any):
        """The sub that armed this request's KV load, or None.

        Recorded at `get_num_new_matched_tokens` time (first-hit-wins, losers
        cancelled). The three load-side questions below --
        `should_park_for_load_after_alloc`, `adjust_prefill_chunk_after_alloc`,
        `should_park_partial_prefill_for_load` -- ask "is there a load in flight
        for this sequence, and must the forward wait for it?" That answer is the
        winner's alone. Routing them through `_state_tier_sub()` instead (an
        earlier fix) named the tier sub even when a *different* sub won the load
        -- e.g. moriio winning ahead of `kimi_k3` under `[moriio, kimi_k3]`: the
        tier sub armed nothing, its `_load_specs.get(sid)` was None, so it
        answered "don't park" and the prefill ran over moriio's in-flight KV.
        The tier sub still owns the state-face forwarders; only these three
        load-side questions follow the load's owner.
        """
        return self._load_winner.get(getattr(seq, "id", None))

    def _state_tier_sub(self):
        """The one sub-connector that actually hosts the state offload tier.

        Selection is by real capability (`has_state_tier`), never by method
        presence: `LMCacheOffloadConnectorScheduler` defines the entire state
        face on every layout, so `_first_with(..., "enqueue_state_stores")`
        would pick a non-tier offload shell -- whose `_impl` has no tier and
        returns False / empty for every state call -- ahead of the shell that
        owns the tier. At most one sub can host the tier: two `lmcache_offload`
        sub-connectors are refused at startup (`_offload_subconfig`), so "first"
        here is "only". Returns None when no sub hosts a tier -- the legal
        `[producer]`-only shape -- and the state-face forwarders then fall to
        their no-tier defaults.

        This selects the tier owner for the *state-face* forwarders only
        (`enqueue_state_*`, `take_state_*`, `save_abandon_timeout_s`). The
        load-side questions route through `_load_owner` instead -- the sub that
        armed the load, which need not be the tier owner.
        """
        for c in self._connectors:
            if getattr(c, "has_state_tier", False):
                return c
        return None

    # -- base interface -----------------------------------------------------

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        """First-hit-wins, and undo the losers' armed loads.

        A sub's lookup is not side-effect-free: an offload sub arms a KV load
        for any prefix it matches -- it takes an LMCache lookup pin, records a
        `_load_specs` entry and a `_lookup_in_step` id, so `update_state_after_
        alloc` (which the composite fans to *every* sub) later flips
        `can_load=True` and recv-queues that request. Only the first matching
        sub owns the load. If a second sub also matched -- e.g. moriio winning
        ahead of an offload sub -- the loser's armed load fires into the same
        block table on `update_state_after_alloc`: a second writer over the
        winner's KV, plus a `finished_loading` the scheduler never accounted. So
        once a winner is chosen, cancel every other sub's pending load.
        `cancel_pending_load` is idempotent and guarded by `_load_lifecycles`,
        so a sub that armed nothing (a miss, or moriio which has no such method)
        is a no-op.

        Record the winner under `seq.id`: the load-side decisions
        (`should_park_for_load_after_alloc` and siblings) belong to it, not to
        the tier sub (finding 1). Cleared by `request_finished` /
        `cancel_pending_load`.
        """
        result = (0, False)
        winner = None
        for c in self._connectors:
            toks, needs_load = c.get_num_new_matched_tokens(seq)
            if winner is None and toks > 0:
                result = (toks, needs_load)
                winner = c
        sid = getattr(seq, "id", None)
        if winner is not None:
            self._load_winner[sid] = winner
            for c in self._connectors:
                if c is not winner:
                    fn = getattr(c, "cancel_pending_load", None)
                    if callable(fn):
                        fn(seq)
        else:
            self._load_winner.pop(sid, None)
        return result

    def cancel_pending_load(self, seq: Any) -> None:
        """Forward a load cancellation to every sub that arms loads.

        The scheduler calls this on `self.kv_connector` -- the composite under
        `multi` -- when a parked load is abandoned (park timeout, request
        finished before its transfer). Only offload subs implement it, and the
        composite had no forwarder, so under `multi` the cancel never reached
        the offload sub: its `_load_specs`/`_reqs_need_recv`/lookup pin leaked
        and the abandoned request stayed recv-queued. Idempotent per sub (the
        `_load_lifecycles` guard), so fanning to all is safe.
        """
        self._load_winner.pop(getattr(seq, "id", None), None)
        for c in self._connectors:
            fn = getattr(c, "cancel_pending_load", None)
            if callable(fn):
                fn(seq)

    def build_connector_meta(self) -> MultiConnectorMetadata:
        return MultiConnectorMetadata(
            metas=[c.build_connector_meta() for c in self._connectors]
        )

    def update_state_after_alloc(self, seq: Any) -> None:
        for c in self._connectors:
            c.update_state_after_alloc(seq)

    def request_finished(self, seq: Any) -> None:
        self._load_winner.pop(getattr(seq, "id", None), None)
        for c in self._connectors:
            if hasattr(c, "request_finished"):
                c.request_finished(seq)

    def abandon_save(self, req_id: Any) -> None:
        # Reclamation of a stalled offload save (see
        # `DenseOffloadConnector.abandon_save`). Only the offload sub tracks
        # `_save_inflight`; forward to whichever sub implements it. Idempotent
        # (pop-with-default), so fanning to all is harmless.
        for c in self._connectors:
            fn = getattr(c, "abandon_save", None)
            if callable(fn):
                fn(req_id)

    # -- offload-specific methods, forwarded to the owning sub --------------
    # The scheduler guards every one of these with hasattr(), so MultiConnector
    # only needs to expose them when a sub-connector implements them.

    def should_park_for_load_after_alloc(self, seq: Any) -> bool:
        # Route to the sub that armed this request's load (`_load_owner`), not
        # the tier sub: the question is whether the forward must wait for the
        # in-flight load, and only its owner knows. An offload owner refines via
        # `_decide_load_after_alloc` (kimi_k3's joint-boundary clamp); an owner
        # that armed a load but does not refine (a P/D producer) parks -- the
        # scheduler's own absent-hook default -- so the forward waits rather than
        # running over the load's blocks. No owner means no load: don't park.
        c = self._load_owner(seq)
        if c is not None and hasattr(c, "should_park_for_load_after_alloc"):
            return c.should_park_for_load_after_alloc(seq)
        return c is not None

    def adjust_prefill_chunk_after_alloc(self, seq: Any, chunk: int) -> int:
        # The prefill chunk is sized to the load its owner armed (kimi_k3 clamps
        # it to the joint boundary); a sub that armed nothing must not resize it.
        # Route to the load owner; unchanged if it does not resize.
        c = self._load_owner(seq)
        if c is not None and hasattr(c, "adjust_prefill_chunk_after_alloc"):
            return c.adjust_prefill_chunk_after_alloc(seq, chunk)
        return chunk

    def enqueue_state_loads(self, loads) -> bool:
        """First sub that can carry them owns them; False if none can.

        Only one sub may host the tier (the worker raises at model load when
        two do), so "first" is also "only".

        The False is not a formality: every load here belongs to a parked
        request only a report can wake, and the caller's `hasattr` guard cannot
        catch a swallowed one because this method always exists.

        Select by `has_state_tier`, not method presence -- see
        `_state_tier_sub`.
        """
        c = self._state_tier_sub()
        if c is None:
            return False
        return bool(c.enqueue_state_loads(loads))

    def enqueue_state_stores(self, stores) -> bool:
        """Symmetric to `enqueue_state_loads`: the one sub that hosts the tier
        owns the stores; False if none can.

        Without this forwarder the engine's `getattr(connector,
        "enqueue_state_stores")` misses on the shell, so it takes the "did not
        carry" branch and releases each store's PAGE units *before* the D2H --
        the CPU tier can never fill under `kv_connector: multi`, even with an
        offload sub-connector configured. `_adopt_state_tier` already guarantees
        at most one tier, so "first" is "only" here too.

        Select by `has_state_tier`, not method presence -- see
        `_state_tier_sub`.
        """
        c = self._state_tier_sub()
        if c is None:
            return False
        return bool(c.enqueue_state_stores(stores))

    def take_state_reports(self):
        """`(indexed, failed)` from the tier sub, else two empty sets.

        The engine unpacks exactly this 2-tuple (`indexed, failed = take()` in
        `scheduler._update_from_kv_xfer_finished`). Missing here, the engine
        never drains store reports, so pins never settle and stored hashes are
        never indexed -- the CPU tier fills but nothing can be found in it.

        Select by `has_state_tier`, not method presence -- see
        `_state_tier_sub`.
        """
        c = self._state_tier_sub()
        if c is None:
            return set(), set()
        return c.take_state_reports()

    def take_state_load_survived(self) -> set:
        """Requests whose state bytes outlived a failed joint load, from the
        tier sub; empty set if none.

        `Scheduler._update_from_kv_xfer_finished` reads this off the composite,
        and its `getattr(..., None)` default is indistinguishable from "nothing
        survived" -- so without the forwarder every survivor under
        `kv_connector: multi` settled as a failure, forgetting a hash whose
        bytes are present.

        Select by `has_state_tier`, not method presence -- see
        `_state_tier_sub`.
        """
        c = self._state_tier_sub()
        if c is None:
            return set()
        return c.take_state_load_survived()

    def take_state_source_releases(self) -> set:
        """Stores whose PAGE units the GPU has finished reading, from the tier
        sub; empty set if none.

        Its own forwarder rather than a third element of `take_state_reports`,
        for the reason the offload connector records: that tuple's arity is a
        contract with the caller, and widening it once already wedged the pool.

        Select by `has_state_tier`, not method presence -- see
        `_state_tier_sub`.
        """
        c = self._state_tier_sub()
        if c is None:
            return set()
        return c.take_state_source_releases()

    def should_park_partial_prefill_for_load(self, seq: Any) -> bool:
        # Called for every running sequence, so most have no load owner -> False
        # (don't park). Parking a partial prefill to await its load is the
        # load owner's decision; a sub that armed nothing must not answer it.
        c = self._load_owner(seq)
        if c is not None and hasattr(c, "should_park_partial_prefill_for_load"):
            return c.should_park_partial_prefill_for_load(seq)
        return False

    def should_defer_free(self, seq: Any) -> bool:
        # Defer if ANY sub wants to defer (so neither a pending save nor a
        # pending send loses its blocks early).
        return any(
            hasattr(c, "should_defer_free") and c.should_defer_free(seq)
            for c in self._connectors
        )

    def has_pending_work(self) -> bool:
        # Scheduler-side only: the send/save pairing state lives on the worker
        # instance, and a pending send is already visible to the engine through
        # the scheduler's deferred_free_blocks.
        return any(
            c.has_pending_work()
            for c in self._connectors
            if hasattr(c, "has_pending_work")
        )

    def process_completions(self, output: KVConnectorOutput) -> KVConnectorOutput:
        """Let the one offload sub apply its own completions and normalize output.

        Only offload defines this. Without the fan-out its save/load
        bookkeeping never clears and raw operation ids reach the scheduler,
        which looks requests up by bare id.

        `OffloadSchedulerMixin.process_completions` is *destructive* and cannot
        partition: it replaces `finished_loading`/`failed_loading`/
        `finished_saving` with only the operations it recognises and `.clear()`s
        `connector_completions` wholesale, over the full sets it is handed. Two
        offload subs would each be handed the other's completions -- one sub
        retiring the other's `_save_inflight` (both key by `str(seq.id)`), whose
        `_maybe_release_deferred` then frees blocks the other is still reading,
        plus a WARNING per foreign completion at steady state. There is no shared
        key by which the composite could split the sets per sub.

        That case is now unrepresentable: `_offload_subconfig` refuses two
        `lmcache_offload` sub-connectors at startup, and only `lmcache_offload`
        subs define `process_completions`, so `handlers` is 0 or 1. The direct
        call is byte-for-byte the single-offload (`[producer, offload]`)
        behaviour. `>1` is guarded loudly in case that refusal is ever bypassed
        -- silently corrupting saves is the worse failure.
        """
        handlers = [
            handler
            for c in self._connectors
            if callable(handler := getattr(c, "process_completions", None))
        ]
        if not handlers:
            return output
        if len(handlers) > 1:
            raise RuntimeError(
                "multi has >1 offload sub-connector with process_completions; "
                "their completion sets cannot be partitioned per sub and this "
                "composite is refused at startup (_offload_subconfig)."
            )
        return handlers[0](output)

    def save_finished(self, req_id: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "save_finished"):
                c.save_finished(req_id)

    def load_failed(self, req_id: Any) -> None:
        for c in self._connectors:
            if hasattr(c, "load_failed"):
                c.load_failed(req_id)

    def save_abandon_timeout_s(self) -> float:
        """The reclaim window for the offload leg, else 0.0 (disabled).

        `Scheduler._save_abandon_timeout_s` reads this off `self.kv_connector`
        -- the composite under `multi` -- and `getattr`-defaults to 0.0 when it
        is absent. 0.0 silently switches off all three leak reclaimers
        (`_reconcile_stalled_deferred_saves`, `reclaim_stale_state_store_pins`,
        `reconcile_orphan_load_slots`), so a missing forwarder here is not a
        no-op: one lost save report then keeps `has_pending_kv_work()` True
        forever, one dropped store completion pins a checkpoint image out of the
        pool, one orphaned load slot wedges `can_allocate`'s state gate -- with
        no fault to point at. The window is a global LMCache property
        (`LMCACHE_EC_PIN_TIMEOUT_SEC + 30`), identical across offload subs;
        prefer the tier sub, then any offload sub. Select by capability, never
        method presence -- see `_state_tier_sub`.
        """
        c = self._state_tier_sub() or _first_with(
            self._connectors, "save_abandon_timeout_s"
        )
        return c.save_abandon_timeout_s() if c is not None else 0.0

    def release_stalled_save(self, seq: Any) -> None:
        """Forward a stalled-save release to every sub that tracks saves.

        `Scheduler._connector_release_stalled_save` calls this on
        `self.kv_connector` when `_reconcile_stalled_deferred_saves` reclaims a
        save the connector never reported done. Only the offload sub tracks
        `_save_inflight`; fan to whichever subs implement it (idempotent
        pop-with-default, like `abandon_save`), so a producer sub without saves
        is a no-op. Absent here, the connector's stall clock never advances and
        the offload save loop wedges permanently.
        """
        for c in self._connectors:
            fn = getattr(c, "release_stalled_save", None)
            if callable(fn):
                fn(seq)

    def get_statistics(self) -> dict[str, int]:
        """Merge offload metrics across subs, summing shared counters.

        `engine_utility` reads this off `self.kv_connector` under an `hasattr`
        guard and renders `{}` when it is absent -- so under `multi` the whole
        offload metrics block silently reads empty. Sum overlapping int keys
        across every sub that reports (only offload subs do), rather than
        first-hit. At most one offload sub is legal (`_offload_subconfig`), but
        summing stays correct for that one and for any future producer that
        grows int counters.
        """
        merged: dict[str, int] = {}
        for c in self._connectors:
            fn = getattr(c, "get_statistics", None)
            if not callable(fn):
                continue
            for k, v in fn().items():
                merged[k] = merged.get(k, 0) + v
        return merged
