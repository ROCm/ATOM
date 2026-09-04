# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The --profiler-delay-iters / --profiler-max-iters window.

The counters and the code that drives them are covered together: these
build a real `EngineUtilityHandler`, so a window that stopped being wired
to a forward would fail here. The RPCs are the only thing faked, since the
handler reaches the workers through `runner_mgr.call_func` and recording
the names it sends is enough to say when the profiler started and stopped.

The last tests are a drift guard. `_FORWARD_FUNCS` is a hand-maintained
allowlist, and an engine variant that dispatches a forward under a new name,
or without waiting for it, would silently stop advancing the window -- the
trace would just run long. `prefill_forward` arrived that way with PD
disagg, so this is not hypothetical.
"""

import ast
import pathlib
import queue
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

from atom.model_engine.engine_utility import EngineUtilityHandler

with stubbed_aiter():
    from atom.model_engine.async_proc import AsyncIOProcManager

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE_LOOP_PATHS = [
    REPO_ROOT / "atom" / "model_engine" / "engine_core.py",
    REPO_ROOT / "atom" / "model_engine" / "pp_engine_core.py",
]

# Engine RPCs that do not run a model forward. Kept beside `_FORWARD_FUNCS`
# so that every name the engine sends is accounted for by one list or the
# other, and a new one fails the test until someone decides which it is.
NOT_FORWARD = frozenset(
    {
        "allocate_kv_cache",
        "async_proc_aggregation",
        "capture_cudagraph",
        "create_decode_stream_pool",
        "create_prefill_stream_pool",
        "exit",
        "export_kv_cache_ipc_handle",
        "export_model_weight_ipc_handles",
        "flush_pp_send",
        "freeze_gc_heap",
        "get_num_blocks",
        "import_kv_cache_ipc_handle",
        "import_model_weight_ipc_handles",
        "process_kvconnector_output",
    }
)


class FakeRunnerMgr:
    """Records the RPC names the handler sends, in order."""

    def __init__(self):
        self.calls = []

    def call_func(self, func_name, *args, wait_out=False):
        self.calls.append(func_name)
        # Return what the real runners return: start_profiler answers with a
        # bare True, stop_profiler with the trace info.
        if func_name == "start_profiler":
            return True
        return {"trace_dir": "/tmp/traces", "elapsed": 0.0}


def make_handler(delay=0, max_iters=0, scheduler=None):
    runner_mgr = FakeRunnerMgr()
    handler = EngineUtilityHandler(
        runner_mgr,
        queue.Queue(),
        scheduler=scheduler,
        profiler_delay_iters=delay,
        profiler_max_iters=max_iters,
    )
    return handler, runner_mgr


def run_steps(handler, count):
    for _ in range(count):
        handler.profiler_step()


def rpc_steps(delay=0, max_iters=0, steps=20):
    """Run a window and report which step each RPC landed on.

    Step 0 is the `start_profile` request itself, before any forward. A name
    absent from the result was never sent, so an equality check on the whole
    mapping also pins the cases that must never auto-stop.
    """
    handler, mgr = make_handler(delay, max_iters)
    handler._handle_start_profile({})
    landed = dict.fromkeys(mgr.calls, 0)
    for step in range(1, steps + 1):
        already_sent = len(mgr.calls)
        handler.profiler_step()
        landed.update(dict.fromkeys(mgr.calls[already_sent:], step))
    return landed


# ── Window behaviour ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "delay, max_iters, expected",
    [
        # Neither knob set: today's behaviour, records until stop_profile.
        (0, 0, {"start_profiler": 0}),
        (0, 3, {"start_profiler": 0, "stop_profiler": 3}),
        (2, 0, {"start_profiler": 2}),
        # The step that starts the profiler is itself over and unrecorded, so
        # the window is the three that follow it. This is the off-by-one.
        (2, 3, {"start_profiler": 2, "stop_profiler": 5}),
    ],
)
def test_window_boundaries(delay, max_iters, expected):
    assert rpc_steps(delay, max_iters) == expected


@pytest.mark.parametrize("delay", [0, 5])
def test_second_start_profile_is_rejected(delay):
    handler, mgr = make_handler(delay=delay, max_iters=10)

    handler._handle_start_profile({})
    first = handler.output_queue.get_nowait()[1]["result"]
    # The endpoint runs `"error" in result` and `"message" in result` on this,
    # so every branch has to answer with a dict rather than the RPC's own
    # return value, which is a bare True.
    assert isinstance(first, dict)
    assert "error" not in first and first["message"]
    handler._handle_start_profile({})
    assert "error" in handler.output_queue.get_nowait()[1]["result"]

    run_steps(handler, delay + 10)
    assert mgr.calls.count("start_profiler") == 1, "rejected call must not re-arm"

    if delay:
        # The endpoint forwards this, so the wording lives in the handler only.
        assert first["armed_after_iters"] == delay
        assert f"{delay} engine steps" in first["message"]


def test_windows_reset_between_requests():
    handler, mgr = make_handler(delay=1, max_iters=2)
    one_window = ["start_profiler", "stop_profiler"]

    handler._handle_start_profile({})
    run_steps(handler, 3)
    assert mgr.calls == one_window

    # The second window has to re-arm the delay and restart the recorded
    # count, or its stop lands early instead of on the third step again.
    handler._handle_start_profile({})
    run_steps(handler, 2)
    assert mgr.calls == one_window + ["start_profiler"]
    handler.profiler_step()
    assert mgr.calls == one_window * 2

    # An explicit stop cancels a pending delay, so nothing opens behind it.
    handler._handle_start_profile({})
    handler._handle_stop_profile({})
    run_steps(handler, 10)
    assert mgr.calls == one_window * 2 + ["stop_profiler"]


def test_call_func_ticks_only_on_completed_forwards():
    """The glue between the RPC layer and the window.

    `AsyncIOProcManager.__init__` spawns the TP workers, so the instance is
    built without it and given only what `call_func` reads.
    """
    forwards = sorted(AsyncIOProcManager._FORWARD_FUNCS)
    mgr = AsyncIOProcManager.__new__(AsyncIOProcManager)
    mgr.label = "test"
    mgr.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda msg: None)
    mgr.outputs_queue = queue.Queue()
    ticks = []
    mgr.on_forward_end = lambda: ticks.append(1)

    for name in forwards:
        mgr.outputs_queue.put({})
        mgr.call_func(name, wait_out=True)
    assert len(ticks) == len(forwards)

    mgr.outputs_queue.put({})
    mgr.call_func("start_profiler", wait_out=True)
    mgr.call_func("forward")
    assert len(ticks) == len(forwards), "only a waited-on forward may tick"


def test_scheduler_detailed_aggregates_track_the_recorded_window():
    from atom.model_engine.scheduler import Scheduler

    scheduler = Scheduler(MockConfig())
    handler, _ = make_handler(delay=1, max_iters=2, scheduler=scheduler)

    handler._handle_start_profile({})
    assert scheduler.profile_active is False, "armed is not yet recording"

    handler.profiler_step()
    assert scheduler.profile_active is True

    run_steps(handler, 2)
    assert scheduler.profile_active is False


# ── Drift guard on _FORWARD_FUNCS ─────────────────────────────────────────


def engine_rpc_calls():
    """Every `(file, line, name, waits)` the engine loops send via call_func.

    Pure AST: the engine modules import torch and the AITER kernels, which
    the CPU gate this runs on cannot load.
    """
    found = []
    for path in ENGINE_LOOP_PATHS:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr not in ("call_func", "call_func_with_aggregation"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            waits = any(
                kw.arg == "wait_out" and getattr(kw.value, "value", None) is True
                for kw in node.keywords
            )
            found.append((path.name, node.lineno, node.args[0].value, waits))
    return found


def test_every_engine_rpc_is_classified_as_forward_or_not():
    calls = engine_rpc_calls()
    dispatched = {name for _, _, name, _ in calls}
    # Also checks the walker: a broken parse finds nothing, and an entry in
    # _FORWARD_FUNCS that no engine sends any more shows up here.
    assert dispatched >= AsyncIOProcManager._FORWARD_FUNCS

    known = AsyncIOProcManager._FORWARD_FUNCS | NOT_FORWARD
    unclassified = [
        f"{name} at {filename}:{lineno}"
        for filename, lineno, name, _ in calls
        if name not in known
    ]
    assert not unclassified, (
        "new engine RPC(s) not classified:\n  "
        + "\n  ".join(unclassified)
        + "\nIf one runs a model forward, add it to "
        "AsyncIOProcManager._FORWARD_FUNCS or the profiler window will not "
        "count it. Otherwise add it to NOT_FORWARD here."
    )


def test_every_forward_dispatch_waits_for_its_output():
    """The hook fires on the `wait_out=True` branch, where the forward is over.

    A fire-and-forget forward would return before the worker ran it, so it
    advances nothing -- the window would silently overrun.
    """
    fire_and_forget = [
        f"{name} at {filename}:{lineno}"
        for filename, lineno, name, waits in engine_rpc_calls()
        if name in AsyncIOProcManager._FORWARD_FUNCS and not waits
    ]
    assert not fire_and_forget, (
        "forward dispatch(es) without wait_out=True:\n  "
        + "\n  ".join(fire_and_forget)
        + "\nThe profiler window will not count these."
    )
