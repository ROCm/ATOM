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

import pytest

from atom.utils.gc_utils import freeze_gc_heap, unfreeze_gc_heap


class _Cycle:
    """A reference cycle: unreachable by refcount, only the collector frees it."""

    def __init__(self):
        self.self_ref = self


@pytest.fixture(autouse=True)
def _leave_gc_as_found():
    """Every test here mutates interpreter-global state."""
    was_enabled = gc.isenabled()
    yield
    gc.unfreeze()
    gc.collect()
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
    """The disaggregated decode path freezes a second time, once its scheduler
    exists -- its block pool is built after `EngineCore.__init__` returns."""
    freeze_gc_heap("first")
    first = gc.get_freeze_count()
    later = [object() for _ in range(64)]
    freeze_gc_heap("second")

    assert gc.get_freeze_count() > first, "the second freeze caught nothing"
    assert len(later) == 64  # and did not disturb what it froze


def test_the_worker_rpc_returns_something():
    """`AsyncIOProc.busy_loop` replies only `if out is not None`, so an RPC
    target that returns None hangs its `wait_out=True` caller forever. The
    EngineCore freezes its workers through exactly such a call, and a server
    started with the first version of it never reached "ready".

    Checked on the unbound function so this needs no GPU and no runner.
    """
    import inspect

    from atom.model_engine.model_runner import ModelRunner

    src = inspect.getsource(ModelRunner.freeze_gc_heap)
    assert "return " in src, (
        "ModelRunner.freeze_gc_heap returns None, which deadlocks the "
        "EngineCore's call_func(..., wait_out=True)"
    )
    assert inspect.signature(ModelRunner.freeze_gc_heap).return_annotation is not None


def test_the_env_gate_turns_it_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_GC_FREEZE", False)
    gc.collect()
    before = gc.get_freeze_count()
    freeze_gc_heap("test")
    assert gc.get_freeze_count() == before
