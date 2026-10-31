# SPDX-License-Identifier: MIT
"""The connector drives its boundary planners through one loop per hook.

Every scheduler hook used to name the KDA and the V4.1 planner in mirrored
branches, which is how a rule reaches one leg and not the other. These hold
each hook to calling every planner the connector has, in the order the hook
needs -- with two planners present, which no supported configuration builds,
so that a hook that names one of them by hand shows up as the other missing.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector


class _Recorder:
    """A planner that records each hook the connector calls on it.

    ``returns`` maps a hook name to what that hook answers; any other hook
    answers None.
    """

    def __init__(self, leg_name, calls, **returns):
        self.leg_name = leg_name
        self._calls = calls
        self._returns = returns

    def __getattr__(self, hook):
        if hook.startswith("_"):
            raise AttributeError(hook)

        def record(*args, **kwargs):
            self._calls.append((self.leg_name, hook, args, kwargs))
            return self._returns.get(hook)

        return record


def _connector(calls, kda=None, v41=None):
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._kda_planner = _Recorder("kda", calls, **(kda or {}))
    c._v41_planner = _Recorder("v41", calls, **(v41 or {}))
    return c


def _hooks(calls):
    return [(leg, hook) for leg, hook, *_ in calls]


# -- every hook reaches every planner ----------------------------------------


def _lookup_connector(calls, lookup):
    c = _connector(calls)
    seq = SimpleNamespace(id="r0", set_num_cached_tokens=Mock())
    c._seqs = SimpleNamespace(get_or_create=lambda request: seq)
    c._requests, c._promised_loads, c._detach_after_alloc = {}, {}, {}
    c._scheduler = SimpleNamespace(get_num_new_matched_tokens=lookup)
    return c


def test_the_pool_and_every_lookup_reach_every_planner():
    calls = []

    def lookup(seq):
        calls.append(("scheduler", "lookup"))
        return 0, False

    c = _lookup_connector(calls, lookup)
    c.bind_gpu_block_pool("pool")
    assert c.get_num_new_matched_tokens(SimpleNamespace(request_id="r0"), 0) == (
        0,
        False,
    )
    assert _hooks(calls) == [
        ("kda", "bind_gpu_block_pool"),
        ("v41", "bind_gpu_block_pool"),
        ("kda", "begin_lookup"),
        ("v41", "begin_lookup"),
        ("scheduler", "lookup"),
        ("kda", "end_lookup"),
        ("v41", "end_lookup"),
    ]


def test_a_failed_lookup_still_disarms_every_planner():
    calls = []

    def lookup(seq):
        raise RuntimeError("tier down")

    c = _lookup_connector(calls, lookup)
    with pytest.raises(RuntimeError):
        c.get_num_new_matched_tokens(SimpleNamespace(request_id="r0"), 0)
    assert _hooks(calls)[-2:] == [("kda", "end_lookup"), ("v41", "end_lookup")]


@pytest.mark.parametrize("external", [512, 0])
def test_an_admitted_hit_reaches_every_planner(external):
    calls = []
    c = _connector(calls)
    c._attn_group_id = 0
    c._config = SimpleNamespace(kv_cache_block_size=128)
    c._requests, c._promised_loads, c._detach_after_alloc = {}, {}, {}
    seq = SimpleNamespace(num_cached_tokens=256, set_block_table=lambda ids: None)
    c._seqs = SimpleNamespace(get_or_create=lambda request: seq)
    c._scheduler = SimpleNamespace(update_state_after_alloc=lambda seq: None)
    c.update_state_after_alloc(
        SimpleNamespace(request_id="r0"), ([1, 2, 3], [4, 5, 6]), external
    )
    if not external:
        assert calls == []
        return
    expected = {
        "group_blocks": ([1, 2, 3], [4, 5, 6]),
        "num_total_computed": 768,
        "num_external_tokens": 512,
        "attention_group_id": 0,
        "attention_block_size": 128,
    }
    assert [(leg, hook, kwargs) for leg, hook, _, kwargs in calls] == [
        ("kda", "admit_load", expected),
        ("v41", "admit_load", expected),
    ]


def test_a_step_forgets_before_it_collects_and_settles_after():
    """Planner work runs after the inner build and the stale-save reconcile, a
    request is forgotten before the stores are collected, and all of it before
    the transfer deadlines and the releases."""
    calls = []
    c = _connector(
        calls,
        kda={"collect_step_stores": ["kda-store"], "take_loads": ["kda-load"]},
        v41={"collect_step_stores": [], "take_loads": []},
    )
    c._seqs = SimpleNamespace(get=lambda req_id: None)
    c._requests = {"n": object()}
    c._scheduler = SimpleNamespace(
        virtual_block_size=256,
        build_connector_meta=lambda: calls.append(("scheduler", "build")) or "inner",
        enforce_transfer_deadlines=lambda: calls.append(("scheduler", "deadlines")),
    )
    c._settle_promised_loads = lambda inner: []
    c._reconcile_stale_saves = lambda: calls.append(("connector", "reconcile"))
    c._collect_releases = lambda: calls.append(("connector", "releases")) or []
    scheduler_output = SimpleNamespace(
        preempted_req_ids=["p"],
        finished_req_ids={"f"},
        scheduled_new_reqs=[SimpleNamespace(req_id="n", num_computed_tokens=512)],
        scheduled_cached_reqs=None,
    )

    meta = c.build_connector_meta(scheduler_output)

    per_leg = [
        "forget_request",
        "forget_request",
        "collect_step_stores",
        "take_loads",
        "log_stats",
    ]
    assert _hooks(calls) == [
        # Preemption is settled first, by every planner.
        ("kda", "abandon_pending_load"),
        ("kda", "forget_request"),
        ("v41", "abandon_pending_load"),
        ("v41", "forget_request"),
        ("scheduler", "build"),
        ("connector", "reconcile"),
        *[("kda", hook) for hook in per_leg],
        *[("v41", hook) for hook in per_leg],
        ("scheduler", "deadlines"),
        ("connector", "releases"),
    ]
    for leg in ("kda", "v41"):
        # Past the preemption hooks: the step's own forgets, then the stores.
        step = [entry[1:] for entry in calls if entry[0] == leg][2:]
        assert {args[0] for hook, args, _ in step if hook == "forget_request"} == {
            "p",
            "f",
        }
        ((args, _),) = [
            (args, kwargs)
            for hook, args, kwargs in step
            if hook == "collect_step_stores"
        ]
        assert args == (scheduler_output, {"n": 512}, c._requests, {"p", "f"})
    assert meta.kda_stores == ["kda-store"] and meta.kda_loads == ["kda-load"]


@pytest.mark.parametrize("busy", [None, "kda", "v41"])
def test_pending_work_and_frees_are_asked_of_every_planner(busy):
    calls = []
    c = _connector(
        calls,
        kda={"has_pending_work": busy == "kda", "has_pending_frees": busy == "kda"},
        v41={"has_pending_work": busy == "v41", "has_pending_frees": busy == "v41"},
    )
    c._scheduler = None
    c._deferred_frees, c._deferred_free_at, c._releases_in_flight = set(), {}, set()
    assert c.has_pending_push_work() is (busy is not None)
    assert c.has_pending_block_frees() is (busy is not None)
    asked = [(leg, hook) for leg, hook, *_ in calls]
    # `any` stops at the first planner that answers yes.
    if busy == "kda":
        assert asked == [("kda", "has_pending_work"), ("kda", "has_pending_frees")]
    else:
        assert asked == [
            ("kda", "has_pending_work"),
            ("v41", "has_pending_work"),
            ("kda", "has_pending_frees"),
            ("v41", "has_pending_frees"),
        ]
    horizons = {args for _, hook, args, _ in calls if hook == "has_pending_frees"}
    assert horizons == {(c._PENDING_FREE_HORIZON_S,)}


def test_worker_reports_reach_every_planner_before_the_loads_resolve():
    calls = []
    c = _connector(calls)
    c._gpu_block_pool = None
    c._promised_loads, c._releases_in_flight = {}, set()
    c._state_load_failure_reports, c._load_failure_reports = {}, {}
    c._save_reports, c._world_size = {}, 1
    c._scheduler = SimpleNamespace(
        load_failed_by_request=lambda rid: calls.append(("scheduler", "failed")),
        load_finished_by_request=lambda rid: calls.append(("scheduler", "finished")),
    )
    meta = SimpleNamespace(
        completions=(),
        state_stored={7: 1},
        state_store_failed={8: 1},
        state_load_failed={"r0": 1},
        load_failed={},
        saved={},
    )
    c.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=meta,
            finished_recving={"r0"},
            failed_recving=set(),
            finished_sending=(),
        )
    )
    assert calls == [
        ("kda", "absorb_reports", ({7: 1}, {8: 1}), {}),
        ("v41", "absorb_reports", ({7: 1}, {8: 1}), {}),
        ("kda", "on_load_result", ("r0", False), {}),
        ("v41", "on_load_result", ("r0", False), {}),
        ("scheduler", "failed"),
    ]


def test_a_finished_request_is_forgotten_by_every_planner():
    calls = []
    c = _connector(calls)
    c._promised_loads, c._detach_after_alloc = {}, {}
    c._requests = {"r0": object()}
    c._seqs = SimpleNamespace(get=lambda req_id: None)
    assert c.request_finished_all_groups(SimpleNamespace(request_id="r0"), ([],)) == (
        False,
        None,
    )
    assert calls == [
        ("kda", "abandon_pending_load", ("r0",), {}),
        ("kda", "forget_request", ("r0",), {}),
        ("v41", "abandon_pending_load", ("r0",), {}),
        ("v41", "forget_request", ("r0",), {}),
    ]
