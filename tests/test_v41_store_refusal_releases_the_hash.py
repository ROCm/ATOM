"""An offered boundary that is never reported blocks its prefix forever.

`collect_frontier_stores` records every offered boundary in `_pending_stores`
and refuses to offer a prefix hash that is already in flight. Only a report --
stored or failed -- clears it. So a path that drops stores without reporting
them does not merely lose those boundaries: it takes that prefix out of the
tier for the life of the process, and the symptom is a tier that looks cold.
"""

import pytest

from atom.plugin.vllm.kv_transfer.v41_state import V41Store


class _Leg:
    """Stands in for the worker leg so the connector half can be driven alone."""

    def __init__(self):
        self.seen = []

    def snapshot_and_submit(self, stores):
        self.seen.extend(stores)
        return {}


def _connector(leg):
    from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector

    c = object.__new__(AtomLMCacheOffloadConnector)
    c._is_deepseek_v41 = True
    c._v41_leg = leg
    c._v41_refusals = []
    c._pending_kda_stores = []
    return c


class _Meta:
    def __init__(self, stores):
        self.kda_stores = stores


_STORES = [V41Store(op_id=7, prefix_hash=0xABC, req_id="r0", boundary=8192)]


def test_a_leg_that_is_up_submits():
    leg = _Leg()
    c = _connector(leg)
    c._stash_kda_stores(_Meta(list(_STORES)))
    assert [s.op_id for s in leg.seen] == [7]
    assert c._v41_refusals == []


def test_a_leg_that_is_down_refuses_rather_than_drops():
    c = _connector(None)
    c._stash_kda_stores(_Meta(list(_STORES)))
    # The op id has to reach the scheduler half; a bare log does not.
    assert [op_id for _, op_id in c._v41_refusals] == [7]


def test_a_refusal_lets_the_same_prefix_be_offered_again():
    """The end of the chain, and the part that actually matters.

    Reporting the failure is only useful if it clears the in-flight record.
    Drive the planner directly: offer, fail, offer again.
    """
    planner = pytest.importorskip(
        "atom.plugin.vllm.kv_transfer.v41_state"
    ).V41BoundaryPlanner
    p = object.__new__(planner)
    p._pending_stores = {}
    p._world_size = 1

    class _Index:
        hashes = frozenset()

        def note_stored(self, h):  # pragma: no cover - not reached on failure
            raise AssertionError("a failed store must not claim the boundary")

    p._index = _Index()

    from atom.plugin.vllm.kv_transfer.v41_state import _PendingStore

    p._pending_stores[7] = _PendingStore(0xABC)
    assert p._offer_in_flight(0xABC)

    p.absorb_reports(stored={}, failed={7: 1})

    assert not p._offer_in_flight(0xABC), "a reported failure must free the prefix"
    assert 7 not in p._pending_stores


def test_an_unreported_store_is_what_blocks_it():
    """The negative control: without the report the prefix stays blocked.

    This is the behaviour the fix exists to avoid, so it is asserted rather
    than assumed -- if in-flight records ever expire on their own, this test
    fails and the refusal path can be simplified instead of kept.
    """
    from atom.plugin.vllm.kv_transfer.v41_state import (
        V41BoundaryPlanner,
        _PendingStore,
    )

    p = object.__new__(V41BoundaryPlanner)
    p._pending_stores = {7: _PendingStore(0xABC)}
    assert p._offer_in_flight(0xABC)
