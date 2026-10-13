"""A dispatched load that never reports is withdrawn, not waited on forever.

`_check_promised_loads` covers the step before this one: a promise that
never becomes a dispatched load. It settles its entry at dispatch, so
nothing watched the gap between a load going out and its result coming
back -- and a load that is dispatched, performed, and then never reported
leaves the request parked for good.

Measured on DSpark: 12 state loads armed against 11 completed, three runs
out of three, the odd one retrieving its 1280 tokens and never arriving.
One request was re-examined 617k times while the client waited for it.

Two earlier guards were written for this and neither could fire. Both aged
their counters inside `_join_kda`, and the stall is quiescent: the request
is parked, nothing runs, the worker produces no output and the join is
never called. A bound counted there advances only while the engine is busy,
which is exactly when it is not needed.
"""

from types import SimpleNamespace

from atom.plugin.vllm.kv_transfer.connector import (
    _DISPATCHED_LOAD_GRACE_STEPS,
    AtomLMCacheOffloadConnector,
)


def _conn():
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._dispatched_loads = {}
    c._dispatched_load_abandoned = 0
    c._seqs = SimpleNamespace(get=lambda _rid: None)
    return c


def _dispatched(rid="r1", start=0, end=1280):
    return SimpleNamespace(
        requests=[
            SimpleNamespace(
                req_id=rid,
                load_operation=object(),
                load_spec=SimpleNamespace(
                    hbm_cached_tokens=start, lmcache_cached_tokens=end
                ),
            )
        ]
    )


def test_a_dispatched_load_that_never_reports_is_withdrawn():
    c = _conn()
    assert c._reap_dispatched_loads(_dispatched()) == []
    empty = SimpleNamespace(requests=[])
    for _ in range(_DISPATCHED_LOAD_GRACE_STEPS - 1):
        assert c._reap_dispatched_loads(empty) == []
    out = c._reap_dispatched_loads(empty)
    assert [r for r, _ in out] == ["r1"]
    assert c._dispatched_load_abandoned == 1
    assert c._dispatched_loads == {}


def test_a_load_that_reports_is_not_withdrawn():
    """`finished_recving` discharges the entry, so the clock never runs out."""
    c = _conn()
    c._reap_dispatched_loads(_dispatched())
    c._dispatched_loads.pop("r1")            # what the report does
    empty = SimpleNamespace(requests=[])
    for _ in range(_DISPATCHED_LOAD_GRACE_STEPS + 5):
        assert c._reap_dispatched_loads(empty) == []
    assert c._dispatched_load_abandoned == 0


def test_a_save_only_request_is_not_mistaken_for_a_load():
    """A save rides the same list; treating it as a dispatched load would
    withdraw a load that was never made."""
    c = _conn()
    meta = SimpleNamespace(requests=[
        SimpleNamespace(req_id="r2", load_operation=None, load_spec=None)
    ])
    for _ in range(_DISPATCHED_LOAD_GRACE_STEPS + 2):
        assert c._reap_dispatched_loads(meta) == []
    assert c._dispatched_loads == {}


def test_the_reaper_runs_where_the_stall_cannot_stop_it():
    import inspect

    from atom.plugin.vllm.kv_transfer import connector as mod

    src = inspect.getsource(mod.AtomLMCacheOffloadConnector.build_connector_meta)
    assert "_reap_dispatched_loads(inner)" in src
