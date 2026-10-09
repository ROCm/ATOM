"""`has_pending_block_frees`, the hook vLLM 0.31 asks before it preempts.

Every failure here is silent in production: the default `False` costs a
preemption that one step would have avoided, and nothing logs it. So each
source of pending frees gets a test that fails if the hook stops seeing it,
and one that fails if the hook reports work when there is none -- a hook that
always said True would pass the first four and starve the scheduler instead.
"""

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector


class _Scheduler:
    def __init__(self, leases=False):
        self._leases = leases

    def has_active_save_leases(self) -> bool:
        return self._leases


class _Planner:
    def __init__(self, pending=False):
        self._pending = pending

    def has_pending_work(self) -> bool:
        return self._pending


def _connector(**state):
    """A connector with only the fields the hook reads, bypassing __init__."""
    c = object.__new__(AtomLMCacheOffloadConnector)
    c._deferred_frees = state.get("deferred", set())
    c._releases_in_flight = state.get("in_flight", set())
    c._kda_planner = state.get("planner")
    c._scheduler = state.get("scheduler")
    return c


def test_nothing_pending_reports_nothing():
    # The one that keeps the others honest: a hook stuck at True would make
    # the scheduler wait for frees that are not coming.
    assert _connector(scheduler=_Scheduler(leases=False)).has_pending_block_frees() is False


def test_a_deferred_free_is_pending():
    assert _connector(deferred={"r0"}, scheduler=_Scheduler()).has_pending_block_frees()


def test_a_release_in_flight_is_pending():
    assert _connector(in_flight={"r0"}, scheduler=_Scheduler()).has_pending_block_frees()


def test_a_pinned_boundary_block_is_pending():
    assert _connector(
        planner=_Planner(pending=True), scheduler=_Scheduler()
    ).has_pending_block_frees()


def test_an_active_save_lease_is_pending():
    # The source a reader would miss: the request is finished and its table
    # returned, so nothing in vLLM's view of it says these blocks are held.
    assert _connector(scheduler=_Scheduler(leases=True)).has_pending_block_frees()


@pytest.mark.parametrize("scheduler", [None, object()])
def test_a_scheduler_without_the_query_is_not_an_error(scheduler):
    # Not every offload backend carries block leases; the hook must answer
    # rather than raise, or an allocation failure becomes a crash.
    assert _connector(scheduler=scheduler).has_pending_block_frees() is False


def test_vllm_calls_it_on_the_base_class():
    # The hook only runs if vLLM's base declares it; a rename upstream would
    # leave this override orphaned and silently unreachable.
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    assert hasattr(KVConnectorBase_V1, "has_pending_block_frees")
