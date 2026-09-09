# SPDX-License-Identifier: MIT
"""Scheduler-side bookkeeping for the state-cache offload tier.

One thing lives here and it touches no device: the set of hashes believed to be
in LMCache, plus the loads offered against them. The bytes and the transfers are
the worker's business (`kv_transfer/offload/hybrid/kimi_k3/state_tier.py`).

The set is in memory and never persisted: `LocalDiskBackend.__init__` starts
from an empty dict and never scans its directory, so an index recovered from
disk would be a pure false-positive generator. Index and bytes share one server
lifetime.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, replace

logger = logging.getLogger("atom")


class StateOffloadIndex:
    """What is believed to be in LMCache, and what is being fetched back.

    `hashes` answers the membership half of the tier's vote. It is deliberately
    optimistic: LMCache's own LRU can drop bytes at any time, so a hash here
    means "was stored once", never "is still there". The false positive costs
    one lookup and a park/unpark, and is handled by the `failed_loading` path
    (`fail_load` -> `forget`).
    """

    def __init__(self, *, can_store: bool = True, can_load: bool = True) -> None:
        # The two legs are separately granted. A `kv_producer` role saves and
        # never loads, a `kv_consumer` loads and never saves, and offering a
        # load the worker will not serve parks the request that took it.
        self.can_store = bool(can_store)
        self.can_load = bool(can_load)
        # The optimistic membership index (see the class docstring). It only
        # grows on `note_stored` and shrinks on `forget` (a failed load), so a
        # long-lived server that stored many distinct prefixes would grow it
        # without bound. Cap it: `hashes` stays a plain set -- membership stays
        # O(1) and every caller/test keeps `in` / `len` / `== set()` /
        # `discard` -- and `_hash_lru` records insertion order beside it purely
        # so the oldest hash can be dropped on overflow. The two move in
        # lockstep (every add and every forget touches both), so `_hash_lru`
        # never diverges in length from `hashes`. Dropping the oldest hash is
        # safe for the same reason the index is optimistic: a false "not stored"
        # only costs one recompute, never wrong output.
        self.hashes: set[int] = set()
        self._hash_lru: OrderedDict[int, None] = OrderedDict()
        # ~1M hashes -> a few tens of MB. Well above any working set that a
        # tier of realistic capacity actually holds, so it bounds a leak
        # without evicting live entries in normal operation.
        self._hash_cap = 1 << 20
        self.hashes_evicted = 0
        # req_id -> hash, for loads offered and not yet settled. Keyed by
        # request because that is what comes back, on the same
        # `finished_loading`/`failed_loading` channel a KV load uses.
        self.pending_loads: dict = {}
        self.loads_attempted = 0
        self.loads_completed = 0
        self.loads_failed = 0
        # Store-side counters. Until this commit the only store-side signal was
        # `indexed` growing, which is why a shell that refused every store
        # (`enqueue_state_stores` never forwarded) looked identical to a tier
        # with nothing to do -- 94 refusals produced one warning line and no
        # number anywhere. `stores_refused` is the probe that would have said
        # so on the first pass, and it is deliberately apart from
        # `stores_failed`: refused means nobody tried, failed means the worker
        # tried and could not.
        self.stores_attempted = 0
        self.stores_completed = 0
        self.stores_failed = 0
        self.stores_refused = 0
        # Stores that reported success after the stale reclaimer had already
        # taken their source units back. Not counted as completed: nothing can
        # say the worker had stopped reading, so the image is forfeited rather
        # than indexed. Any non-zero value means the reclaim window is firing
        # on live transfers and is set too low.
        self.stores_untrusted = 0

    # ------------------------------- stores -------------------------------- #
    def note_stored(self, h: int) -> None:
        """A store landed in LMCache and the hash is now worth voting for.

        Called from the report the worker sends back, never at submission: a
        hash advertised before its bytes exist parks the next request over that
        prefix against a `get` that must miss.
        """
        h = int(h)
        self.hashes.add(h)
        # Reinsert at the young end whether or not it was already present, so a
        # re-stored prefix is treated as freshly used, then trim the cold end.
        self._hash_lru.pop(h, None)
        self._hash_lru[h] = None
        while len(self._hash_lru) > self._hash_cap:
            old, _ = self._hash_lru.popitem(last=False)
            self.hashes.discard(old)
            self.hashes_evicted += 1

    def forget(self, h: int) -> None:
        """Drop a hash whose load failed, so the next request does not retry."""
        self.hashes.discard(h)
        self._hash_lru.pop(h, None)

    def could_serve(self, h: int) -> bool:
        """Whether a load for `h` could be offered at all: this tier can load
        AND believes it holds `h`.

        The membership+capability half of `request_load`'s guard (minus its
        per-request in-flight check), factored out so the admission-path voters
        -- `PageUnitCheckpointCoordinator._reachable` and
        `BlockManager._tier_can_serve` -- test exactly what `request_load` will
        accept and cannot drift from it. Mirroring only `h in hashes` there,
        without `can_load`, made a `kv_producer` (can_load=False, but `hashes`
        populated from its own stores) vote a tier hit it would then refuse: the
        right-to-left resumable scan stopped at the tier rung, skipped a
        still-resident HBM rung, and `request_load` returned False -- the
        boundary disowned and the HBM checkpoint forfeited to a full recompute.
        """
        return self.can_load and h in self.hashes

    # -------------------------------- loads -------------------------------- #
    def request_load(self, req_id, h: int) -> bool:
        """Offer to fetch `h` back for `req_id`. False if this tier cannot.

        The guard between believing and delivering: a load is resolved only by
        a worker report, so offering one for a hash never stored would park the
        request against bytes no `get` can produce.
        """
        if not self.could_serve(h):
            # A store-only role (can_load False) or a hash never stored. Voting
            # for either parks the request against a report that never comes;
            # refuse so the boundary is disowned and recomputed.
            return False
        if req_id in self.pending_loads:
            # One request, one outstanding load: reports are keyed by request
            # id, so the first completion would unpark while the second is
            # still writing. No in-tree path reaches this today, and the
            # refusal costs only a disown.
            logger.warning(
                "state offload: request %s already has a load in flight; "
                "refusing a second one and letting the boundary be disowned.",
                req_id,
            )
            return False
        self.pending_loads[req_id] = int(h)
        self.loads_attempted += 1
        return True

    def complete_load(self, req_id) -> None:
        """The bytes landed in the request's slot.

        The hash stays indexed: a load reads LMCache, it does not consume it,
        and the next request over the same prefix must still find it.
        """
        if self.pending_loads.pop(req_id, None) is not None:
            self.loads_completed += 1

    def fail_load(self, req_id) -> None:
        """No bytes came back. Retract the claim as well as counting it.

        A miss is the only evidence that LMCache's LRU dropped what this index
        advertises, so it has to be what un-advertises it. Leaving the hash makes
        every later request over that prefix park, miss and recompute.
        """
        h = self.pending_loads.pop(req_id, None)
        if h is None:
            return
        self.loads_failed += 1
        self.forget(h)

    def abandon_load(self, req_id) -> None:
        """The request went away before its bytes did. Neither outcome.

        Not `fail_load`: an abort says nothing about the bytes, and forgetting
        a loadable hash sends the next request back to a full recompute.
        """
        self.pending_loads.pop(req_id, None)

    def stats(self) -> dict[str, int]:
        """Counters for the periodic `state checkpoints:` line.

        Read the load counters together: `failed / attempted` is this index's
        false-positive rate, and `attempted - completed - failed` is what is in
        flight or was abandoned by an aborted request. Read the store counters
        the same way, and read `stores_completed` against `checkpoints_kept`:
        the gap is how much of what HBM keeps the CPU tier never received.
        """
        return {
            # Store leg. `attempted - completed - failed` is in flight;
            # `refused` is a wiring fault, not backpressure -- any non-zero
            # value means the connector never took the store at all.
            "stores_attempted": self.stores_attempted,
            "stores_completed": self.stores_completed,
            "stores_failed": self.stores_failed,
            "stores_refused": self.stores_refused,
            "stores_untrusted": self.stores_untrusted,
            # Load leg.
            "loads_attempted": self.loads_attempted,
            "loads_completed": self.loads_completed,
            "loads_failed": self.loads_failed,
            "indexed": len(self.hashes),
            # Non-zero means the index hit its `_hash_cap` and is dropping the
            # coldest hashes. Harmless (a dropped hash just misses one reuse),
            # but a climbing value means the cap is smaller than the live prefix
            # working set and reuse is being left on the table.
            "hashes_evicted": self.hashes_evicted,
        }


#: Offload layouts whose worker half builds a `StateOffloadTier`. Not every
#: `lmcache_offload` does: `dense` has no per-request state at all and `hybrid`
#: (DSV4) keeps its own in the SLOT sidecar, so only K3 offloads state.
_STATE_TIER_LAYOUTS = frozenset({"kimi_k3"})

#: Roles from which the worker will save / load. Mirrors `_do_save` / `_do_load`
#: in `OffloadWorkerMixin`; the state legs ride those same halves.
_SAVE_ROLES = frozenset({"offload", "kv_both", "kv_producer"})
_LOAD_ROLES = frozenset({"offload", "kv_both", "kv_consumer"})


@dataclass(frozen=True)
class StateTierCapability:
    """What the worker half will actually do for the state tier.

    Derived from configuration, not from the connector's public name. The name
    only says which connector class is constructed; whether that class builds a
    `StateOffloadTier` depends on the layout it resolves, the pipeline depth,
    and the role it was given. The engine used to install a `StateOffloadIndex`
    for any `lmcache_offload`, so a K3 run under PP>1 -- or any `dense`/`hybrid`
    run -- got an index against a worker that would never build the tier. That
    mismatch is what left stores emitted with nowhere to go, and it is the
    common root of the tier-none fallback paths.

    `reason` is filled in whenever the tier is off, so the log names which check
    refused it rather than leaving the operator to guess.
    """

    can_store_state: bool
    can_load_state: bool
    reason: str = ""

    @property
    def hosts_state_tier(self) -> bool:
        """Whether to build the engine-side index at all.

        Either leg is enough to need one: a store-only role still has to hold
        pins and settle reports, and a load-only role still has to vote.
        """
        return self.can_store_state or self.can_load_state


def _offload_subconfig(cfg: dict) -> tuple[dict | None, str]:
    """The one offload sub-connector under `multi`, or a reason there is none.

    Exactly one, as asked: the tier's bytes ride a specific connector's worker
    half and its completions ride that connector's `get_finished`, so two
    providers would each hold half an answer and a request parked on the wrong
    one would never be reported.

    Two offload sub-connectors (e.g. `[lmcache_offload(dense),
    lmcache_offload(kimi_k3)]`) is refused *loudly* here, the one place the
    composite is inspected, rather than degraded to a no-tier fallback. A silent
    fallback disabled the state tier but left the KV load path live: dense won
    `get_num_new_matched_tokens` and queued its H2D, the tier never armed, and
    the prefill forward ran over the block table dense's worker was still
    writing -- torn KV, silent wrong output, and a `finished_loading` the
    scheduler never accounted (review round 5, finding 0). Making it a startup
    `ValueError` keeps that half-configured composite unreachable and names the
    cause where the operator can act on it. A composite with *no* offload sub is
    still a soft no-tier (`None`), the legal `[producer]`-only shape.
    """
    subs = [s for s in (cfg.get("connectors") or ()) if isinstance(s, dict)]
    offload = [s for s in subs if s.get("kv_connector") in _STATE_TIER_BACKENDS]
    if not offload:
        return None, "multi lists no offload connector"
    if len(offload) > 1:
        raise ValueError(
            f"kv_transfer_config: `multi` lists {len(offload)} offload "
            "connectors, but the state tier's bytes and completions ride one "
            "connector's worker half -- two would each hold half an answer and "
            "a request parked on the wrong one would never be reported. Use at "
            "most one `lmcache_offload` sub-connector."
        )
    return offload[0], ""


class _SubConnectorView:
    """`config` with one `multi` sub-connector's transfer config in front.

    Attribute lookups fall through to the real config, so the model fields the
    layout selector reads are unchanged; only `kv_transfer_config` is replaced.
    """

    def __init__(self, config, sub: dict) -> None:
        self._config = config
        self.kv_transfer_config = sub

    def __getattr__(self, name):
        # __getattr__ only fires for names normal lookup misses, so if `_config`
        # itself is unbound -- a pickle/deepcopy probe, or a raise between
        # `__new__` and the assignment in `__init__` -- reading `self._config`
        # here re-enters __getattr__ for "_config" and recurses to
        # RecursionError. This view is built at `BlockManager.__init__`, so that
        # blast radius is engine startup. Fail as an ordinary missing attribute.
        if name == "_config":
            raise AttributeError(name)
        return getattr(self._config, name)


def state_tier_capability(config) -> StateTierCapability:
    """What the worker will do for the state tier, decided from config alone.

    A capability descriptor rather than a name check, because the two are not
    the same question and the worker refuses on conditions the name cannot see
    (no transfer config, a `multi` composite with no offload sub, `pp_size > 1`,
    an unknown explicit override, a layout that offloads no state, or a `kv_role`
    that neither saves nor loads). Config alone because this runs in the engine
    process at
    `BlockManager.__init__`, before any worker connector exists -- so it must
    agree with the worker by construction, which is why the layout comes from
    the same `select_offload_layout` the worker uses and the roles from the
    same sets `OffloadWorkerMixin` reads.

    A false negative (tier off) is much cheaper than a false positive: off
    costs reuse, on-against-a-worker-that-refuses parks requests against loads
    nobody can report.
    """
    from atom.kv_transfer.offload.config import select_offload_layout

    none = StateTierCapability(False, False, "")
    cfg = getattr(config, "kv_transfer_config", None) or {}
    if not isinstance(cfg, dict):
        return replace(none, reason="no kv_transfer_config")

    name = cfg.get("kv_connector", "moriio")
    layout_config = config
    if name == "multi":
        sub, why = _offload_subconfig(cfg)
        if sub is None:
            return replace(none, reason=why)
        cfg = sub
        name = cfg.get("kv_connector")
        # The layout selector reads `config.kv_transfer_config`, and under
        # `multi` the interesting one is the sub-connector's -- an
        # `offload_layout` override lives there, not on the composite. Model
        # fields still come from the real config.
        layout_config = _SubConnectorView(config, cfg)
    if name not in _STATE_TIER_BACKENDS:
        return replace(none, reason=f"connector {name!r} hosts no state tier")

    # The worker refuses PP outright: `CacheEngineKey` has no PP component, so
    # two stages at the same TP rank would overwrite each other's entries.
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    if pp_size > 1:
        return replace(none, reason=f"pipeline_parallel_size={pp_size}")

    try:
        layout = select_offload_layout(layout_config)
    except ValueError as exc:  # an unknown explicit override
        return replace(none, reason=str(exc))
    if layout not in _STATE_TIER_LAYOUTS:
        return replace(none, reason=f"layout {layout!r} offloads no state")

    role = cfg.get("kv_role", "offload")
    can_store = role in _SAVE_ROLES
    can_load = role in _LOAD_ROLES
    if not (can_store or can_load):
        return replace(none, reason=f"kv_role {role!r} neither saves nor loads")
    return StateTierCapability(can_store, can_load)


def state_tier_chunk_tokens(config) -> int:
    """The LMCache chunk size in tokens for the joint KV leg, or 0.

    Decided from config for the same reason `state_tier_capability` is: this
    runs in the engine process at `BlockManager.__init__`, and the scheduler
    holds whatever `get_kvconnector` returned -- a connector object without
    `chunk_size` would zero this and disable the joint KV load silently. The KV
    leg moves whole LMCache chunks, so a joint boundary has to be a multiple of
    this.

    Kept out of `state_tier_capability` on purpose: that check is pure and has a
    second caller (`hosts_state_tier`), and this reaches into LMCache -- an
    optional dependency whose absence must not warn on an engine that hosts no
    tier. Call this only when the capability says a tier is hosted; the WARNING
    below is then worth printing, because a missing chunk size really does
    disable the joint KV load.
    """
    try:
        from atom.kv_transfer.offload.config import build_lmcache_config

        kvcfg = getattr(config, "kv_transfer_config", None) or {}
        # Under `multi` the lmcache.* keys (chunk_size, offload_layout) live on
        # the offload SUB-connector, not the composite. Passing the raw
        # composite here read a zero chunk grid. Unwrap the sub the same way
        # `state_tier_capability` does, so the gate's chunk size and the tier
        # the capability check builds stay consistent.
        if isinstance(kvcfg, dict) and kvcfg.get("kv_connector") == "multi":
            sub, _why = _offload_subconfig(kvcfg)
            if sub is not None:
                kvcfg = sub
        return int(build_lmcache_config(kvcfg).chunk_size)
    except Exception:
        # Blind on purpose: this runs at model load, the import reaches a
        # third-party package that may be absent entirely, and the only
        # consequence of not knowing the chunk size is that the joint KV load
        # stays off. Refusing to start would be worse.
        logger.warning(
            "state offload: could not read the LMCache chunk size; the joint "
            "KV load needs it and stays off",
            exc_info=True,
        )
        return 0


#: The connector backends whose worker half can build a `StateOffloadTier`.
#: `multi` qualifies when it lists exactly one.
_STATE_TIER_BACKENDS = frozenset({"lmcache_offload"})
