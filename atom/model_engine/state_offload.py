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

logger = logging.getLogger("atom")


class StateOffloadIndex:
    """What is believed to be in LMCache, and what is being fetched back.

    `hashes` answers the membership half of the tier's vote. It is deliberately
    optimistic: LMCache's own LRU can drop bytes at any time, so a hash here
    means "was stored once", never "is still there". The false positive costs
    one lookup and a park/unpark, and is handled by the `failed_loading` path
    (`fail_load` -> `forget`).
    """

    def __init__(self) -> None:
        self.hashes: set[int] = set()
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
        self.hashes.add(int(h))

    def forget(self, h: int) -> None:
        """Drop a hash whose load failed, so the next request does not retry."""
        self.hashes.discard(h)

    # -------------------------------- loads -------------------------------- #
    def request_load(self, req_id, h: int) -> bool:
        """Offer to fetch `h` back for `req_id`. False if this tier cannot.

        The guard between believing and delivering: a load is resolved only by
        a worker report, so offering one for a hash never stored would park the
        request against bytes no `get` can produce.
        """
        if h not in self.hashes:
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
        }


# The connector backends whose worker half builds a `StateOffloadTier`. Only
# the offload backend does; `multi` qualifies when it lists one.
_STATE_TIER_BACKENDS = frozenset({"lmcache_offload"})


def kv_connector_hosts_state_tier(kv_transfer_config) -> bool:
    """Whether the configured KV connector can actually run the state tier.

    The tier is not standalone: its bytes ride the KV connector's worker half
    and its completions ride that connector's `get_finished`. Against any other
    backend a load would be offered and never reported, parking the request
    that took it -- so this gates whether the index is built at all, and a
    false negative (tier off) is much cheaper than a false positive.

    A name check rather than a capability probe: this runs in the engine
    process at `BlockManager.__init__`, before any worker connector exists.
    """
    cfg = kv_transfer_config or {}
    if not isinstance(cfg, dict):
        return False
    name = cfg.get("kv_connector", "moriio")
    if name == "multi":
        subs = cfg.get("connectors") or ()
        return any(
            isinstance(s, dict) and s.get("kv_connector") in _STATE_TIER_BACKENDS
            for s in subs
        )
    return name in _STATE_TIER_BACKENDS
