# SPDX-License-Identifier: MIT
"""The plugin connector must not block vLLM's `schedule()` on a tier lookup."""

from types import SimpleNamespace
from unittest.mock import Mock

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector


def _connector(*, pending: bool, async_capable: bool = True, detachable=False):
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    seq = SimpleNamespace(set_num_cached_tokens=Mock())
    c._seqs = SimpleNamespace(get_or_create=lambda request: seq)
    c._requests = {}
    c._promised_loads = {}
    c._kda_planner = None
    c._v41_planner = None
    scheduler = SimpleNamespace(
        get_num_new_matched_tokens=Mock(return_value=(512, True)),
        should_park_for_load_after_alloc=Mock(return_value=True),
    )
    if async_capable:
        scheduler.prefetch_lookups = Mock()
        scheduler.lookup_pending = Mock(return_value=pending)
        scheduler.can_detach_lookup = Mock(return_value=detachable)
        scheduler.detach_lookup = Mock(return_value=True)
    c._scheduler = scheduler
    return c, seq


def test_a_lookup_in_flight_defers_the_request_a_step():
    """None tells vLLM to pass over the request and ask again next step."""
    c, seq = _connector(pending=True)
    assert c.get_num_new_matched_tokens(SimpleNamespace(request_id="r"), 0) == (
        None,
        False,
    )
    c._scheduler.prefetch_lookups.assert_called_once_with([seq])
    c._scheduler.get_num_new_matched_tokens.assert_not_called()
    assert c._promised_loads == {}


def test_an_answered_lookup_is_consumed_as_before():
    c, seq = _connector(pending=False)
    assert c.get_num_new_matched_tokens(SimpleNamespace(request_id="r"), 0) == (
        512,
        True,
    )
    c._scheduler.get_num_new_matched_tokens.assert_called_once_with(seq)
    assert c._promised_loads["r"] == [0, 0, 512, False]


def test_a_scheduler_without_async_lookups_stays_synchronous():
    c, _ = _connector(pending=True, async_capable=False)
    assert c.get_num_new_matched_tokens(SimpleNamespace(request_id="r"), 0) == (
        512,
        True,
    )


def test_a_load_ruled_out_by_the_local_hit_is_admitted_without_waiting():
    """The lookup still goes out -- for recency and the save floor -- but
    nothing waits on it: no answer could make this request load."""
    c, seq = _connector(pending=True, detachable=True)
    assert c.get_num_new_matched_tokens(SimpleNamespace(request_id="r"), 6912) == (
        0,
        False,
    )
    c._scheduler.detach_lookup.assert_called_once_with(seq, 6912)
    c._scheduler.prefetch_lookups.assert_not_called()
    c._scheduler.get_num_new_matched_tokens.assert_not_called()
