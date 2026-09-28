# SPDX-License-Identifier: MIT

"""`gc.freeze()` is what removes the collector's cost; these pin what it costs.

Freezing is the only knob here that changes what the collector *may not touch*,
so the two things worth a test are the two ways it can be wrong: freezing too
much (garbage made permanent, or never handed back) and freezing too little
(the safety net gone, which would make it `gc.disable()` in disguise).

Each case carries the failure shape it guards, because a happy-path assertion
would pass against a `freeze_gc_heap` that did nothing at all.
"""

from __future__ import annotations

import gc
import json

import pytest

from atom.utils.gc_utils import (
    _owner,
    arm_reclaim_watch,
    freeze_gc_heap,
    gc_census,
    reclaim_watch,
    tune_gc,
    unfreeze_gc_heap,
)


class _Cycle:
    """A reference cycle: unreachable by refcount, only the collector frees it."""

    def __init__(self):
        self.self_ref = self


@pytest.fixture(autouse=True)
def _leave_gc_as_found():
    """Every test here mutates interpreter-global state.

    Thresholds and the reclaim watch are restored too: they are process-wide,
    and a test that left them raised would silently change how often the
    collector runs for every test after it in the same process.
    """
    from atom.utils import gc_utils

    was_enabled = gc.isenabled()
    thresholds = gc.get_threshold()
    baseline, warned = gc_utils._reclaim_baseline, gc_utils._reclaim_warned
    yield
    gc.unfreeze()
    gc.collect()
    gc.set_threshold(*thresholds)
    gc_utils._reclaim_baseline, gc_utils._reclaim_warned = baseline, warned
    if was_enabled:
        gc.enable()


def test_freezing_takes_the_live_heap_out_of_the_collectors_reach():
    live = [object() for _ in range(64)]
    gc.collect()
    before = gc.get_freeze_count()

    freeze_gc_heap("test")

    assert gc.get_freeze_count() > before, "nothing was frozen"
    # Control: what the frozen set is for. `gc.get_objects()` reports only what
    # a collection would still walk, so the drop is the cost that goes away.
    assert len(gc.get_objects()) < len(live), (
        "the live heap is still visible to the collector, so freezing bought " "nothing"
    )


def test_new_objects_are_still_collected_after_a_freeze():
    """This is what separates freezing from `gc.disable()`.

    Freezing forfeits only what was alive at that instant. A cycle created by
    later code -- a code path added next year -- must still be reclaimed, or
    this becomes an unbounded leak instead of a bounded one.
    """
    freeze_gc_heap("test")

    # `gc.collect()` runs even when the collector is disabled, so calling it
    # proves only that the object is not frozen -- it would pass just as well
    # against a `freeze_gc_heap` that also called `gc.disable()`. What has to
    # be shown is that a collection still fires *on its own*.
    assert gc.isenabled(), "freezing disabled the collector"

    fired: list[int] = []
    gc.callbacks.append(
        lambda phase, info: fired.append(1) if phase == "stop" else None
    )
    try:
        threshold = gc.get_threshold()[0]
        for _ in range(threshold * 4):
            _Cycle()  # dropped immediately; only the collector can free it
            if fired:
                break
    finally:
        gc.callbacks.pop()

    assert fired, (
        "no automatic collection ran after the freeze -- the safety net for "
        "cycles written by later code is gone, which makes this gc.disable()"
    )


def test_garbage_alive_at_freeze_time_is_not_made_permanent():
    """`gc.freeze()` moves every generation across exactly as it finds it, so
    freezing without collecting first would make current garbage permanently
    unreclaimable. `freeze_gc_heap` collects all three generations first."""
    _Cycle()  # garbage right now, but no collection has run to notice
    freeze_gc_heap("test")

    unfreeze_gc_heap()
    # Nothing left for a collection to find: the freeze helper already took it.
    assert gc.collect() == 0, "a cycle was carried into the permanent generation"


def test_unfreezing_makes_the_startup_heap_reclaimable_again():
    """Required on engine shutdown. Without it an engine torn down inside a
    live interpreter leaves its weights unreachable *and* uncollectable, which
    presents as a GPU memory leak rather than as anything about GC."""
    doomed = _Cycle()
    gc.collect()  # promote it, so it is part of the heap being frozen
    freeze_gc_heap("test")
    del doomed

    # Control: frozen, it is beyond the collector's reach.
    assert gc.collect() == 0, "the frozen object was collected; nothing to prove"

    unfreeze_gc_heap()
    assert gc.get_freeze_count() == 0
    assert gc.collect() > 0, "unfreezing did not hand the object back"


def test_freezing_twice_is_additive_and_harmless():
    """The disaggregated decode path freezes a second time, once its block pool
    exists -- it is built later in `DecodeEngineCore.__init__`, after the base
    freeze has already run."""
    freeze_gc_heap("first")
    first = gc.get_freeze_count()
    later = [object() for _ in range(64)]
    freeze_gc_heap("second")

    assert gc.get_freeze_count() > first, "the second freeze caught nothing"
    assert len(later) == 64  # and did not disturb what it froze


def test_the_frozen_count_cannot_be_mirrored_so_nothing_scrape_side_reads_it():
    """`gc.get_freeze_count()` is too slow for a scrape -- `_gc_metrics` has
    the measurement -- and caching it here, the obvious answer, is wrong: the
    count is not a function of this module's calls, which is what the middle of
    this test shows. Hence no `atom:gc_frozen_objects` gauge to keep in step.
    """
    freeze_gc_heap("test")
    unfreeze_gc_heap()
    assert gc.get_freeze_count() == 0

    gc.collect()  # touches nothing this module owns

    assert gc.get_freeze_count() > 0, (
        "CPython no longer repopulates the permanent generation on its own -- "
        "a mirror maintained at freeze/unfreeze would now be safe, and this "
        "test is the reason there isn't one"
    )


def test_every_serving_frontend_applies_the_gc_policy():
    """The axis, not one instance of it.

    This coverage has gone stale twice already: #1980 reached the API server
    but not the disaggregated EngineCores, whose `run_engine` does not call the
    base one; and the first version of this change reached those but not
    atomesh, which builds its own engine and never runs the FastAPI lifespan.
    Both misses are silent -- the process simply keeps its 200ms pauses.

    So the rule is checked rather than the instances: a process that builds an
    engine and then serves has to apply the policy.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "atom"
    frontends = sorted(
        p
        for p in root.rglob("*.py")
        # `examples/` are batch scripts: they exit, they do not serve.
        if ".create_engine(" in p.read_text() and "examples/" not in p.as_posix()
    )
    assert frontends, "no engine frontend found; this test has stopped checking"
    missing = [
        p.relative_to(root).as_posix()
        for p in frontends
        # The call, not the import: an unused import passes for the name.
        if "freeze_gc_heap(" not in p.read_text()
    ]
    assert not missing, (
        f"these build an engine and serve, but never apply the GC policy: "
        f"{missing}. Add tune_gc/maybe_attach_gc_debug_callback/freeze_gc_heap "
        f"once startup is done, as atom/entrypoints/openai/api_server.py does."
    )


def test_a_process_has_exactly_one_name():
    """`ps`, the freeze line and the debug callback all take a name, and they
    have to be the same string: under dp>1 a name that omits the dp rank makes
    every rank's logs identical, which is the case worth telling apart."""
    from types import SimpleNamespace as NS

    from atom.utils import engine_process_name, worker_process_name

    def cfg(pp=1, dp=1, pp_rank=0, dp_rank=0):
        return NS(
            pipeline_parallel_size=pp,
            parallel_config=NS(
                pipeline_parallel_rank=pp_rank,
                data_parallel_size=dp,
                data_parallel_rank=dp_rank,
            ),
        )

    assert engine_process_name(cfg()) == "EngineCore"
    assert engine_process_name(cfg(dp=4, dp_rank=2)) == "EngineCore_DP2"
    assert engine_process_name(cfg(pp=2, pp_rank=1)) == "EngineCore_PP1"

    assert worker_process_name(cfg(), 3) == "TP3"
    assert worker_process_name(cfg(dp=4, dp_rank=2), 3) == "DP2TP3"
    # A worker can be built without a config; naming must not be what fails.
    assert worker_process_name(None, 3) == "TP3"

    # Control: dp ranks must not collide, which is what a rank-blind name does.
    names = {worker_process_name(cfg(dp=4, dp_rank=r), 0) for r in range(4)}
    assert len(names) == 4, f"dp ranks share a name: {names}"


def test_the_worker_rpc_returns_something():
    """`AsyncIOProc.busy_loop` replies only `if out is not None`, so an RPC
    target that returns None hangs its `wait_out=True` caller forever. The
    EngineCore freezes its workers through exactly such a call, and a server
    started with the first version of it never reached "ready".

    Read from source rather than imported: `model_runner` pulls in aiter, which
    the non-GPU CI runner does not have, and a contract about what the code
    says needs no runtime anyway.
    """
    import ast
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "atom"
        / "model_engine"
        / "model_runner.py"
    ).read_text()
    fn = next(
        (
            node
            for cls in ast.parse(src).body
            if isinstance(cls, ast.ClassDef) and cls.name == "ModelRunner"
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "freeze_gc_heap"
        ),
        None,
    )
    assert fn is not None, "ModelRunner.freeze_gc_heap is gone; who freezes workers?"
    assert any(
        isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn)
    ), (
        "ModelRunner.freeze_gc_heap returns None, which deadlocks the "
        "EngineCore's call_func(..., wait_out=True)"
    )


def test_the_env_gate_turns_it_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_GC_FREEZE", False)
    gc.collect()
    before = gc.get_freeze_count()
    freeze_gc_heap("test")
    assert gc.get_freeze_count() == before


class _Counted:
    """A type distinctive enough that the census cannot find it by accident."""


class _HostileDict(dict):
    """A dict subclass whose iteration raises, standing in for any mapping
    that computes its keys. The census walks whatever the process allocated,
    so it meets these; it must not run their code to describe them."""

    def __iter__(self):
        raise AssertionError("the census iterated a subclass it should skip")


def test_the_census_counts_by_type():
    """The whole point is naming the type that dominates, so a census that
    reported only totals -- which the generation sizes already give -- would
    pass a shape-only assertion."""
    held = [_Counted() for _ in range(37)]
    gc.collect()  # promote to gen 2, which is what the census reads

    # Untruncated: `top` is a display bound, and in a full test run this
    # process holds thousands of distinct types, so a ranked slice drops a
    # 37-instance one and the assertion below fails for a reason that has
    # nothing to do with counting.
    counts = {r["type"]: r["count"] for r in gc_census(top=10**6)["by_type"]}

    assert counts.get("_Counted", 0) >= 37
    assert len(held) == 37


def test_the_census_never_reads_what_a_container_holds():
    """The tracked dicts on this process are overwhelmingly parsed request
    state -- tool schemas, metadata maps, `extra_body` -- whose keys the client
    chose. An earlier version fingerprinted containers by their contents, which
    put those keys in the body of an endpoint that has no authentication and no
    rate limit, so one tenant could read another's.

    `_HostileDict` is the enforcement rather than the illustration: it raises
    if anything iterates it, so reintroducing the fingerprint fails here
    instead of in production.
    """
    client_supplied = "acme_internal_tool_id"
    held = [{client_supplied: i, "payload": [i]} for i in range(400)]
    hostile = [_HostileDict(a=1) for _ in range(4)]
    gc.collect()

    census = gc_census(top=10**6)  # must not raise

    # Serialised, because the endpoint returns it as JSON: the key must not
    # reach the client through any field, not merely through the one removed.
    assert client_supplied not in json.dumps(census)
    assert len(held) == 400 and len(hostile) == 4


@pytest.mark.parametrize("top", [0, -1, -100])
def test_a_nonsensical_row_bound_does_not_silently_shorten_the_answer(top):
    """Slicing is the trap: `[:0]` returns nothing and `[:-1]` drops the
    smallest row, and both read as a complete census rather than as a bad
    argument. The bound is caller-controlled -- it is a query parameter."""
    held = [_Counted() for _ in range(37)]
    gc.collect()

    rows = gc_census(top=top)["by_type"]

    assert len(rows) == 1, "a clamped bound must still return the ranking's head"
    assert len(held) == 37


def test_a_negative_type_bound_yields_no_types_rather_than_almost_all():
    """`[:-3]` is "all but the last three", which is not what anyone typing a
    negative number meant, and the row still looks well formed."""
    census = gc_census(top=5, types_per_owner=-3)

    assert all(row["types"] == [] for row in census["by_owner"])


def test_no_env_means_no_process_touches_its_thresholds(monkeypatch):
    """There is no per-process default. Raising these does not make a pass
    cheaper, only rarer, so the same scan lands in fewer and longer pauses --
    a trade nothing here has measured, and one the API server and a worker
    would not make on the same terms anyway."""
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_GC_THRESHOLD", "")
    gc.set_threshold(700, 10, 10)

    tune_gc()

    assert gc.get_threshold() == (700, 10, 10)


def test_the_env_sets_the_thresholds(monkeypatch):
    """The one way to change them, and it has to work without a redeploy."""
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_GC_THRESHOLD", "123,4,5")
    gc.set_threshold(700, 10, 10)

    tune_gc()

    assert gc.get_threshold() == (123, 4, 5)


@pytest.mark.parametrize(
    "value",
    [
        "20000,fifty,50",  # not a number
        "20000,50,50,50",  # one too many -- set_threshold raises on this
        "999",  # one too few -- set_threshold would half-apply it
        ",,",
        "  ",
        # Out of range. CPython accepts each of these and stores it: `0,...`
        # turns automatic collection off, and negatives are nonsense.
        "0,0,0",
        "0,10,10",
        "-1,10,10",
        "700,-1,10",
        # Past C long. `set_threshold` raises OverflowError here, which is an
        # ArithmeticError -- so a guard catching only ValueError and TypeError
        # lets through the one exception this call actually produces.
        "70000000000000000000,10,10",
        "9223372036854775808,10,10",
        # The same overflow further along the tuple, which is a different bug:
        # `set_threshold` converts and stores one argument at a time, so these
        # raise only after the earlier values are live. Measured on 3.12.3,
        # they left (1, 10, 10) and (100000, 50, 10) behind -- the first
        # collecting on nearly every allocation -- while the log said the value
        # had been ignored. Overflowing only `t0` cannot catch this: nothing is
        # written before it fails.
        "1,9223372036854775808,10",
        "100000,50,9223372036854775808",
        "700,50,70000000000000000000",
    ],
)
def test_a_malformed_env_leaves_the_thresholds_alone(monkeypatch, value):
    """A typo is a mistake to surface. Running on half-parsed numbers would
    present as an unexplained performance change, not as a typo -- and running
    on none at all is worse: `tune_gc` is called unguarded in four processes,
    and in a worker a raise lands between "load model runner success" and
    "ready", where `wait_server_ready.sh` reports nothing."""
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_GC_THRESHOLD", value)
    gc.set_threshold(700, 10, 10)

    tune_gc()

    assert gc.get_threshold() == (700, 10, 10)


@pytest.mark.parametrize("value", ["700,0,0", "1,1,1", "100000,50,50"])
def test_an_aggressive_setting_is_the_operators_call(monkeypatch, value):
    """The negative control for the range check, and the reason it is not
    `all(x >= 1)`. Only `t0` gates collection; `t1` and `t2` are ratios, where
    zero means gen-1 and gen-2 run on *every* gen-0 pass. That is expensive
    here -- gen 2 was measured at 979 ms in a worker -- but it is a coherent
    thing to ask a tuning knob for, and refusing it needs a reason this has."""
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_GC_THRESHOLD", value)
    gc.set_threshold(700, 10, 10)

    tune_gc()

    assert gc.get_threshold() == tuple(int(x) for x in value.split(","))


def test_the_watch_is_quiet_while_nothing_is_reclaimed():
    """The expected steady state for this process, and what says that spacing
    its collections out would be free."""
    gc.collect()
    arm_reclaim_watch()

    assert reclaim_watch("test") == 0


def test_the_watch_notices_a_cycle_being_reclaimed():
    """The positive control. Without it the test above passes against a watch
    that returns zero unconditionally -- which is exactly what a watch whose
    baseline was armed at the wrong moment would do."""
    gc.collect()
    arm_reclaim_watch()
    _Cycle()  # unreachable by refcount; only the collector frees it
    gc.collect()

    assert reclaim_watch("test") > 0


def test_the_watch_warns_once():
    """At four thousand streams a per-check line would bury the log. The
    counter in /metrics is the continuous signal; the warning only has to make
    someone look at it."""
    gc.collect()
    arm_reclaim_watch()
    _Cycle()
    gc.collect()

    first = reclaim_watch("test")
    from atom.utils import gc_utils

    warned_after_first = gc_utils._reclaim_warned
    second = reclaim_watch("test")

    assert first > 0 and second > 0  # still reports, just does not re-warn
    assert warned_after_first is True


def test_an_unarmed_watch_reports_nothing_rather_than_guessing():
    """Before the baseline exists the counters carry startup's own collections
    -- 18,826 objects in one measured run -- and reporting those would fire the
    warning on every process at boot."""
    from atom.utils import gc_utils

    gc_utils._reclaim_baseline = None

    assert reclaim_watch("test") == 0


def _fn_from(path: str):
    """A function whose code claims to come from `path`, to test attribution
    without importing the library it names."""
    import types

    return types.FunctionType(compile("x = 1", path, "exec"), {})


def test_objects_are_attributed_to_the_library_that_built_them():
    """The type table says "22.8 coroutines per stream"; the question it does
    not answer is whose they are, which is the one that decides whether the
    count can be reduced at all."""
    assert _owner(_fn_from("/x/site-packages/fastapi/routing.py")) == "fastapi"
    assert _owner(_fn_from("/x/site-packages/starlette/responses.py")) == "starlette"
    assert _owner(_fn_from("/app/ATOM/atom/entrypoints/openai/api_server.py")) == "atom"
    # Longest-match ordering: a starlette path must not fall into a shorter one.
    assert _owner(_fn_from("/x/site-packages/uvicorn/protocols/http/h11_impl.py")) == (
        "uvicorn"
    )
    # Instances are attributed by their class's module, not by any code object.
    assert _owner(gc.callbacks) == "unattributable"  # a plain list
    assert _owner(_Counted()).startswith(("module:", "stdlib"))


def test_a_module_prefix_does_not_swallow_a_longer_name():
    """`atom` must not claim `atomic_whatever`. Matching on the root package
    and not `startswith` is the difference, and nothing in this repo's own
    imports would have shown it."""

    class _Impostor:
        pass

    _Impostor.__module__ = "atomicwrites.core"

    assert _owner(_Impostor()) == "module:atomicwrites"


def test_a_container_is_reported_as_unattributable_rather_than_guessed():
    """A tuple of two cells belongs to whoever built the closure, and that is
    not recoverable from the tuple. Guessing would make the breakdown read as
    precision it does not have."""
    for obj in ({}, [], (), set(), frozenset()):
        assert _owner(obj) == "unattributable"


def test_attribution_does_not_touch_the_object_it_attributes():
    """`getattr(obj, "__code__", None)` looks harmless and is not: it runs the
    object's `__getattr__` and its descriptors, so a census would execute the
    code it is measuring. A torch module's deprecation shim firing a warning
    from inside this probe is how it was found, so the guard is a dispatch on
    `type(obj)` and this pins it."""
    touched = []

    class _Watched:
        def __getattr__(self, name):
            touched.append(name)
            raise AttributeError(name)

    who = _owner(_Watched())

    assert touched == [], f"attribution read {touched} off the object"
    assert who != "error"


def test_one_hostile_object_does_not_take_down_the_census():
    """Whatever the process allocated ends up here, including objects whose
    class attributes raise. It is the diagnostic path; it must not be the
    reason a debug call fails."""

    class _Meta(type):
        @property
        def __module__(cls):
            raise RuntimeError("nope")

    class _Hostile(metaclass=_Meta):
        pass

    assert _owner(_Hostile()) == "error"


def test_the_owner_breakdown_accounts_for_every_object():
    """Sums to the generation size. A breakdown that silently dropped a third
    of the set would still look like a clean answer, and the whole point of it
    is deciding what fraction is reducible."""
    held = [_Counted() for _ in range(50)]
    gc.collect()

    census = gc_census(top=5, types_per_owner=1)
    total = sum(row["count"] for row in census["by_owner"])

    assert total == census["generations"]["2"]
    assert len(held) == 50


def test_the_free_counters_ride_along():
    """`collected` is the number that says whether raising thresholds defers
    real work, and it costs nothing -- a caller that walked every object to
    learn it would be paying a second time."""
    census = gc_census(top=1)

    assert len(census["stats"]) == 3
    assert all("collected" in s for s in census["stats"])
    assert set(census["generations"]) == {"0", "1", "2"}
    assert len(census["thresholds"]) == 3
