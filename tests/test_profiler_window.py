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
    from atom.model_engine.llm_engine import LLMEngine

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
    """Records the RPC names the handler sends, in order.

    `fail_on` is an RPC that raises, as a worker-side profiler failure does.
    """

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def call_func(self, func_name, *args, wait_out=False):
        self.calls.append(func_name)
        if func_name == self.fail_on:
            raise RuntimeError("worker said no")
        # Return what the real runners return: start_profiler answers with a
        # bare True, stop_profiler with the trace info.
        if func_name == "start_profiler":
            return True
        return {"trace_dir": "/tmp/traces", "elapsed": 0.0}


def make_handler(delay=0, max_iters=0, scheduler=None, fail_on=None):
    runner_mgr = FakeRunnerMgr(fail_on)
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


def broadcast(handlers, cmd, **payload):
    """Run one utility command on every handler, returning a result each.

    Dispatched through `_execute_utility_command` so that a command missing
    from `_UTILITY_HANDLERS`, and so unreachable from an engine, fails here.
    """
    results = []
    for handler, _ in handlers:
        handler._execute_utility_command(cmd, {"cmd": cmd, **payload})
        results.append(handler.output_queue.get_nowait()[1]["result"])
    return results


def start_window(handlers, body=None, token="a-run"):
    """The two rounds `LLMEngine.start_profile` drives, against *handlers*.

    What a server runs, so every window test starts its run through this.
    """
    payload = {} if body is None else dict(body)
    # "cmd" belongs to the command, not the window: `broadcast` puts it back.
    payload.pop("cmd", None)
    reserved = broadcast(handlers, "reserve_profile", token=token, **payload)
    if any("error" in result for result in reserved):
        broadcast(handlers, "release_profile", token=token)
        return reserved
    return broadcast(handlers, "commit_profile", token=token)


def rpc_steps(delay=0, max_iters=0, steps=20, body=None):
    """Run a window and report which step each RPC landed on.

    Step 0 is the `/start_profile` request itself, before any forward. A name
    absent from the result was never sent, so an equality check on the whole
    mapping also pins the cases that must never auto-stop.

    *delay* / *max_iters* are the launch flags; *body* is the request payload,
    which may carry a window of its own.
    """
    handler, mgr = make_handler(delay, max_iters)
    start_window([(handler, mgr)], body)
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
    one = [(handler, mgr)]

    first = start_window(one)[0]
    # The endpoint runs `"error" in result` and `"message" in result` on this,
    # so every branch has to answer with a dict rather than the RPC's own
    # return value, which is a bare True.
    assert isinstance(first, dict)
    assert "error" not in first and first["message"]
    # Asking for a window of its own as well, because a refused request must
    # not retarget the recording already in flight: the reply says 409 while
    # the trace silently runs to the refused caller's length.
    second = start_window(one, {"max_iters": 100}, token="another-run")[0]
    assert "error" in second and second["conflict"] is True

    run_steps(handler, delay + 10)
    assert mgr.calls == [
        "start_profiler",
        "stop_profiler",
    ], "the refused request re-armed the profiler or moved the live window"

    if delay:
        # The endpoint forwards this, so the wording lives in the handler only.
        assert first["armed_after_iters"] == delay
        assert f"{delay} engine steps" in first["message"]


def test_windows_reset_between_requests():
    handler, mgr = make_handler(delay=1, max_iters=2)
    one = [(handler, mgr)]
    one_window = ["start_profiler", "stop_profiler"]

    start_window(one)
    run_steps(handler, 3)
    assert mgr.calls == one_window

    # The second window has to re-arm the delay and restart the recorded
    # count, or its stop lands early instead of on the third step again.
    start_window(one)
    run_steps(handler, 2)
    assert mgr.calls == one_window + ["start_profiler"]
    handler.profiler_step()
    assert mgr.calls == one_window * 2

    # An explicit stop cancels a pending delay, so nothing opens behind it.
    start_window(one)
    broadcast(one, "stop_profile")
    run_steps(handler, 10)
    assert mgr.calls == one_window * 2 + ["stop_profiler"]


def test_a_failed_auto_start_leaves_the_engine_idle_and_says_so():
    """The delay is spent by the time the start runs, so a raise there ends
    the window with nobody to tell: the caller was answered when it armed.
    """
    handler, mgr = make_handler(delay=2, max_iters=5, fail_on="start_profiler")
    one = [(handler, mgr)]

    start_window(one)
    run_steps(handler, 2)

    assert mgr.calls == ["start_profiler"], "the failed start is the only RPC"
    assert not handler._profiler_active and not handler._profiler_pending
    assert "auto-start failed" in broadcast(one, "stop_profile")[0]["error"]

    # Idle rather than wedged: the next request is accepted and records.
    mgr.fail_on = None
    assert "error" not in start_window(one)[0]
    run_steps(handler, 2 + 5)
    assert mgr.calls[-2:] == ["start_profiler", "stop_profiler"]
    assert "error" not in broadcast(one, "stop_profile")[0], "a stale error"


# ── Per-request window ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body, expected",
    [
        # Both fields overridden: neither launch flag is consulted.
        ({"delay_iters": 0, "max_iters": 3}, {"start_profiler": 0, "stop_profiler": 3}),
        # One field each way -- the omitted one still comes from its flag.
        ({"max_iters": 1}, {"start_profiler": 4, "stop_profiler": 5}),
        ({"delay_iters": 1}, {"start_profiler": 1, "stop_profiler": 9}),
        # An explicit 0 is a value, not an omission: it has to beat the flag's
        # 8 and record until /stop_profile rather than fall back to it.
        ({"delay_iters": 0, "max_iters": 0}, {"start_profiler": 0}),
    ],
)
def test_request_body_overrides_the_launch_flags(body, expected):
    assert rpc_steps(delay=4, max_iters=8, body=body) == expected


def test_a_bodyless_request_keeps_the_launch_flags():
    """What the endpoint sends for a bodyless POST: both keys present and
    None, rather than absent.
    """
    payload = {"cmd": "start_profile", "delay_iters": None, "max_iters": None}
    assert rpc_steps(delay=2, max_iters=3, body=payload) == {
        "start_profiler": 2,
        "stop_profiler": 5,
    }
    assert rpc_steps(delay=2, max_iters=3, body=payload) == rpc_steps(2, 3)


def test_a_request_window_does_not_outlive_its_run():
    handler, mgr = make_handler(delay=0, max_iters=5)
    one = [(handler, mgr)]
    one_window = ["start_profiler", "stop_profiler"]

    start_window(one, {"max_iters": 1})
    run_steps(handler, 1)
    assert mgr.calls == one_window

    # The next request omits the field, so it falls back to the flag's 5,
    # not to the 1 the previous caller asked for.
    start_window(one)
    run_steps(handler, 4)
    assert mgr.calls == one_window + ["start_profiler"]
    handler.profiler_step()
    assert mgr.calls == one_window * 2
    assert (handler.profiler_delay_iters, handler.profiler_max_iters) == (
        0,
        5,
    ), "the launch flags are defaults and must stay untouched"


# ── Atomic multi-engine start ─────────────────────────────────────────────


def test_a_busy_engine_refuses_and_no_idle_engine_is_started():
    """The disagg case: two engines whose windows close at different times.

    A `/start_profile` in between finds one engine idle and the other still
    recording, and a trace the idle one began here is one the refused client
    will never stop.
    """
    decode, decode_mgr = make_handler(max_iters=3)
    prefill, prefill_mgr = make_handler(max_iters=9)
    pd = [(decode, decode_mgr), (prefill, prefill_mgr)]

    start_window(pd)
    run_steps(decode, 3)
    run_steps(prefill, 1)
    assert decode_mgr.calls == ["start_profiler", "stop_profiler"], "decode is done"
    assert prefill_mgr.calls == ["start_profiler"], "prefill is still recording"

    idle_before = list(decode_mgr.calls)
    refused = start_window(pd, {"max_iters": 100}, token="second-run")

    assert refused[0] == {"reserved": True}
    assert "error" in refused[1] and refused[1]["conflict"] is True
    assert decode_mgr.calls == idle_before, "the idle engine opened a second trace"
    assert prefill._profiler_active, "the busy engine's recording was disturbed"
    assert decode._profiler_reservation is None, "the refusal left a claim behind"

    # The 409's advice, and the happy path: reserve is inert, commit starts.
    broadcast(pd, "stop_profile")
    stopped = [list(mgr.calls) for _, mgr in pd]
    assert broadcast(pd, "reserve_profile", token="retry") == [
        {"reserved": True},
        {"reserved": True},
    ]
    assert [list(mgr.calls) for _, mgr in pd] == stopped, "reserve acted on a worker"

    broadcast(pd, "commit_profile", token="retry")
    assert [mgr.calls[-1] for _, mgr in pd] == ["start_profiler", "start_profiler"]


def test_only_the_token_holder_can_release_or_commit():
    """Defensive: `start_profile()` blocks, so FastAPI serialises the two
    requests today. The token holds if that endpoint stops blocking.
    """
    one = [make_handler()]
    mgr = one[0][1]

    # No "conflict" flag on a stray commit, so the endpoint answers 500.
    stray = broadcast(one, "commit_profile", token="never-reserved")[0]
    assert "error" in stray and "conflict" not in stray
    assert mgr.calls == []

    assert broadcast(one, "reserve_profile", token="winner") == [{"reserved": True}]
    assert broadcast(one, "reserve_profile", token="loser")[0]["conflict"] is True
    assert broadcast(one, "release_profile", token="loser") == [{"released": False}]

    assert broadcast(one, "commit_profile", token="winner") == [
        {"message": "Profiling started"}
    ]
    assert mgr.calls == ["start_profiler"]


def test_the_one_shot_start_profile_command_is_the_two_rounds_in_one():
    """Kept for a caller that holds one engine and sends the one command.

    An unregistered utility command is silently ignored, so an out-of-tree
    caller would get a no-op rather than an error if this drifted.
    """
    one_shot = [make_handler(delay=1, max_iters=2)]
    two_rounds = [make_handler(delay=1, max_iters=2)]

    assert broadcast(one_shot, "start_profile") == start_window(two_rounds)

    for handler, mgr in one_shot + two_rounds:
        run_steps(handler, 3)
        assert mgr.calls == ["start_profiler", "stop_profiler"]


def test_stopping_clears_a_reservation_left_by_a_dead_caller():
    one = [make_handler()]
    mgr = one[0][1]

    broadcast(one, "reserve_profile", token="abandoned")
    broadcast(one, "stop_profile")

    assert broadcast(one, "reserve_profile", token="fresh") == [{"reserved": True}]
    assert broadcast(one, "commit_profile", token="fresh")[0]["message"]
    assert mgr.calls == ["stop_profiler", "start_profiler"]


class FakeCoreMgr:
    """Records the rounds `LLMEngine` broadcasts.

    `refuse` is the index of an engine that reports a conflict on reserve;
    `timeout` makes the reserve round raise, as the real broadcast does when
    an engine stops answering.
    """

    def __init__(self, engines=2, refuse=None, timeout=False):
        self.engines = engines
        self.refuse = refuse
        self.timeout = timeout
        self.rounds = []

    def broadcast_utility_command_sync(self, cmd, timeout=300.0, **kwargs):
        self.rounds.append((cmd, kwargs))
        if cmd == "reserve_profile":
            if self.timeout:
                raise TimeoutError("engine never answered")
            results = [{"reserved": True} for _ in range(self.engines)]
            if self.refuse is not None:
                results[self.refuse] = {"error": "busy", "conflict": True}
        elif cmd == "commit_profile":
            results = [{"message": "Profiling started"} for _ in range(self.engines)]
        else:
            results = [{"released": True} for _ in range(self.engines)]
        return [{"cmd": cmd, "result": result} for result in results]


def engine_with(core_mgr):
    """An `LLMEngine` with only a core manager: `__init__` would load a
    tokenizer and spawn engine processes.
    """
    engine = LLMEngine.__new__(LLMEngine)
    engine.core_mgr = core_mgr
    return engine


def test_the_engine_reserves_everywhere_before_it_commits_anywhere():
    core_mgr = FakeCoreMgr()
    engine = engine_with(core_mgr)

    results = engine.start_profile(delay_iters=1, max_iters=2)

    assert [cmd for cmd, _ in core_mgr.rounds] == ["reserve_profile", "commit_profile"]
    # One token for both rounds, and the window travels with the reservation.
    reserve_kwargs, commit_kwargs = (kwargs for _, kwargs in core_mgr.rounds)
    assert reserve_kwargs["delay_iters"] == 1 and reserve_kwargs["max_iters"] == 2
    assert commit_kwargs == {"token": reserve_kwargs["token"]}
    assert all("error" not in r for r in results)

    engine.start_profile()
    tokens = {kwargs["token"] for _, kwargs in core_mgr.rounds}
    assert len(tokens) == 2, "a reused token lets one run commit another's reservation"


def test_one_refusal_releases_the_others_and_commits_nothing():
    core_mgr = FakeCoreMgr(refuse=1)

    results = engine_with(core_mgr).start_profile()

    assert [cmd for cmd, _ in core_mgr.rounds] == [
        "reserve_profile",
        "release_profile",
    ], "an engine was committed after a sibling refused"
    # The refusal round comes back: the endpoint needs every answer to pick
    # 409 over 500 and to say how many refused.
    assert [("error" in r) for r in results] == [False, True]


def test_a_reserve_timeout_releases_before_it_propagates():
    """Otherwise the engines that did answer hold a reservation for good,
    and every later /start_profile is refused until a restart.
    """
    core_mgr = FakeCoreMgr(timeout=True)

    with pytest.raises(TimeoutError):
        engine_with(core_mgr).start_profile()

    assert [cmd for cmd, _ in core_mgr.rounds] == [
        "reserve_profile",
        "release_profile",
    ]
    reserve_kwargs, release_kwargs = (kwargs for _, kwargs in core_mgr.rounds)
    assert release_kwargs["token"] == reserve_kwargs["token"]


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
    handler, mgr = make_handler(delay=1, max_iters=2, scheduler=scheduler)

    start_window([(handler, mgr)])
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
