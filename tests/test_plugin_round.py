# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plugin-round lease: begin/end, and the hot-plug it refuses.

A benchmark/evaluation harness brackets its measurement with
``POST /plugin/round`` (``begin`` then ``end``); while the round is active the
engine refuses a ``POST /plugin/hotplug`` with ``round-in-progress`` instead of
swapping the plan under the run.  CPU only, no AITER and no GPU: the lease
rules and refusal envelopes live in ``atom/model_engine/plugin_hotplug.py``
and are exercised directly, the handlers and the manager against a fake
engine, and the HTTP route -- which imports FastAPI -- as source, the way the
hot-plug wiring is already checked.
"""

import ast
import contextlib
import queue
import re
import sys
import types
from pathlib import Path

import pytest

from atom.model_engine import plugin_hotplug as rounds
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.engine_utility import (
    FIRE_AND_FORGET_UTILITY_CMDS,
    EngineUtilityHandler,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if not (REPO_ROOT / "atom").is_dir():
    # Running the file from outside the checkout (a CPU box running this one
    # test on its own): find the tree the imported atom package came from.
    REPO_ROOT = next(
        Path(entry)
        for entry in sys.path
        if (Path(entry) / "atom" / "config.py").is_file()
    )


# -- fixtures ----------------------------------------------------------------


class FakeEngine:
    """The slice of EngineCore a utility handler touches, plus what it changed.

    ``paused``/``resumed`` count the generation transitions so a refused
    hot-plug can be shown to have made none, and ``config`` carries a
    supported topology with hot-plug on (``plugin_hotplug.topology_refusal``
    must let the hot-plug through when no round is active).
    """

    def __init__(self):
        self.config = types.SimpleNamespace(
            plugin_hotplug=True,
            pipeline_parallel_size=1,
            enable_dp_attention=False,
            enable_rapidserve=False,
            parallel_config=types.SimpleNamespace(data_parallel_size=1),
        )
        self.plugin_round_id = None
        self.plugin_round_started_at = None
        self._generation_paused = False
        self._hotplug_forward_in_flight = False
        self.scheduler = None
        self.paused = 0
        self.resumed = 0

    def pause_generation(self, deadline=None):
        self.paused += 1
        self._generation_paused = True

    def resume_generation(self):
        self.resumed += 1
        self._generation_paused = False

    def has_pending_kv_work(self):
        return False


class FakeRunner:
    """The model-runner side: record the hot-plug call, report one plan."""

    def __init__(self):
        self.calls = []
        self.report = {
            "rank": 0,
            "status": "ok",
            "plan_id": "sha256:abc",
            "rows": [8],
            "sizes": [2],
            "capture_seconds": 0.1,
        }

    def call_func(self, name, payload=None, wait_out=False):
        self.calls.append((name, payload))
        return dict(self.report)


class Wired:
    """A real handler over a fake engine, and a real manager over that handler."""

    def __init__(self, engine_count=1):
        self.engine = FakeEngine()
        self.runner = FakeRunner()
        self.replies = queue.Queue()
        self.handler = EngineUtilityHandler(self.runner, self.replies, label="test")
        self.handler.engine = self.engine
        # A manager whose synchronous broadcast runs the utility command on the
        # one handler, so a round begun through the manager is the engine state
        # a hot-plug then sees.  A real CoreManager spawns processes; nothing
        # here should.
        self.manager = CoreManager.__new__(CoreManager)
        self.manager._init_shared_state(
            _MANAGER_CONFIG, label="test", local_engine_count=engine_count
        )
        self.manager.control_sockets = [None] * engine_count
        self.broadcast_calls = []

        def broadcast(cmd, timeout=None, **kwargs):
            self.broadcast_calls.append(cmd)
            handler_name = self.handler._UTILITY_HANDLERS[cmd]
            return [getattr(self.handler, handler_name)(dict(kwargs))]

        self.manager.broadcast_utility_command_sync = broadcast

    def close(self):
        self.manager.ctx.term()


_MANAGER_CONFIG = types.SimpleNamespace(
    dp_load_balance="round_robin",
    kv_events_config=types.SimpleNamespace(enable=False),
    tp_world_size=1,
    prefill_context_parallel_size=1,
    keep_finished_outputs=False,
    asyncio_mode=False,
)


@pytest.fixture
def wired():
    connected = Wired()
    try:
        yield connected
    finally:
        connected.close()


def _last_reply(replies):
    kind, reply = replies.get_nowait()
    assert kind == "UTILITY_RESPONSE"
    assert replies.empty(), "one answer per command"
    return reply


# -- payload parsing ---------------------------------------------------------


def test_a_round_body_parses():
    assert rounds.parse_plugin_round_payload(
        {"action": "begin", "round_id": "run-1"}
    ) == rounds.PluginRoundRequest(
        action="begin",
        round_id="run-1",
        timeout=rounds.DEFAULT_PLUGIN_ROUND_TIMEOUT_S,
    )
    request = rounds.parse_plugin_round_payload(
        {"action": "end", "round_id": "run-1", "timeout": 2.5}
    )
    assert request.action == "end"
    assert request.timeout == 2.5


def test_unknown_round_body_keys_are_ignored():
    # The caller and the server are deployed separately: a stray field must
    # not refuse a state change the caller can make correctly.
    request = rounds.parse_plugin_round_payload(
        {"action": "begin", "round_id": "run-1", "comment": "nightly"}
    )
    assert request.round_id == "run-1"


@pytest.mark.parametrize(
    "body",
    [
        {},
        [1, 2],
        {"action": "pause", "round_id": "run-1"},
        {"action": "begin"},
        {"action": "begin", "round_id": ""},
        {"action": "begin", "round_id": "   "},
        {"action": "begin", "round_id": 7},
        {"action": "begin", "round_id": "run-1", "timeout": 0},
        {"action": "begin", "round_id": "run-1", "timeout": -1.0},
        {"action": "begin", "round_id": "run-1", "timeout": True},
        {"action": "begin", "round_id": "run-1", "timeout": "1"},
    ],
)
def test_a_bad_round_body_is_a_400_with_its_reason(body):
    with pytest.raises(rounds.RoundPayloadError) as refused:
        rounds.parse_plugin_round_payload(body)
    assert refused.value.reason == "payload-schema"
    assert rounds.refusal_status(refused.value) == 400
    assert rounds.refusal_reason(refused.value) == "payload-schema"


# -- the lease itself --------------------------------------------------------


def _empty_engine():
    return types.SimpleNamespace(plugin_round_id=None, plugin_round_started_at=None)


def test_begin_takes_the_lease_and_end_releases_it():
    engine = _empty_engine()
    began = rounds.take_plugin_round(engine, "run-1", started_at=100.0)
    assert began == {"status": "ok", "round_id": "run-1", "started_at": 100.0}
    assert engine.plugin_round_id == "run-1"
    assert rounds.active_plugin_round(engine) == rounds.PluginRound("run-1", 100.0)

    ended = rounds.release_plugin_round(engine, "run-1", ended_at=160.0)
    assert ended["round_id"] == "run-1"
    assert ended["started_at"] == 100.0
    assert ended["ended_at"] == 160.0
    assert ended["duration_s"] == 60.0
    assert engine.plugin_round_id is None
    assert rounds.active_plugin_round(engine) is None


def test_a_second_begin_is_refused_and_leaves_the_first_round_alone():
    engine = _empty_engine()
    rounds.take_plugin_round(engine, "run-1", started_at=100.0)
    with pytest.raises(rounds.PluginRefusal) as refused:
        rounds.take_plugin_round(engine, "run-2")
    assert refused.value.reason == rounds.ROUND_IN_PROGRESS_REASON
    assert rounds.active_plugin_round(engine).round_id == "run-1"
    assert engine.plugin_round_started_at == 100.0


def test_end_requires_the_active_round_id():
    engine = _empty_engine()
    rounds.take_plugin_round(engine, "run-1", started_at=100.0)
    with pytest.raises(rounds.PluginRefusal) as refused:
        rounds.release_plugin_round(engine, "run-2")
    assert refused.value.reason == rounds.ROUND_MISMATCH_REASON
    assert rounds.active_plugin_round(engine).round_id == "run-1"


def test_end_without_a_round_is_refused():
    engine = _empty_engine()
    with pytest.raises(rounds.PluginRefusal) as refused:
        rounds.release_plugin_round(engine, "run-1")
    assert refused.value.reason == rounds.NO_ACTIVE_ROUND_REASON


@pytest.mark.parametrize("bogus", ["", "   ", None, 3])
def test_begin_validates_the_round_id_and_sets_nothing(bogus):
    engine = _empty_engine()
    with pytest.raises(rounds.RoundPayloadError):
        rounds.take_plugin_round(engine, bogus)
    assert rounds.active_plugin_round(engine) is None


def test_the_hotplug_refusal_names_the_active_round():
    engine = _empty_engine()
    rounds.take_plugin_round(engine, "run-1")
    refusal = rounds.round_in_progress_reply(rounds.HOTPLUG_CMD, engine)
    assert refusal["cmd"] == rounds.HOTPLUG_CMD
    assert refusal["reason"] == rounds.ROUND_IN_PROGRESS_REASON
    assert refusal["active_round"]["round_id"] == "run-1"
    assert "run-1" in refusal["error"]
    assert rounds.round_in_progress_reply(rounds.HOTPLUG_CMD, _empty_engine()) is None

    error = rounds.PluginRefusal(refusal["reason"], refusal["error"])
    assert rounds.refusal_status(error) == 409
    assert rounds.refusal_reason(error) == rounds.ROUND_IN_PROGRESS_REASON


def test_a_utility_refusal_reason_must_agree_across_engines():
    in_progress = rounds.ROUND_IN_PROGRESS_REASON
    assert rounds.command_refusal_reason([]) is None
    assert rounds.command_refusal_reason([{"status": "ok"}]) is None
    assert rounds.command_refusal_reason([{"error": "boom"}]) is None
    assert (
        rounds.command_refusal_reason(
            [
                {"error": "busy", "reason": in_progress},
                {"error": "busy", "reason": in_progress},
            ]
        )
        == in_progress
    )
    # Disagreeing engines, or one that named no code, stay a generic 409.
    assert (
        rounds.command_refusal_reason(
            [{"error": "busy", "reason": "a"}, {"error": "busy", "reason": "b"}]
        )
        is None
    )


def test_round_replies_are_merged_into_the_answer():
    began = rounds.aggregate_round_replies(
        "begin",
        "run-1",
        [
            {"round_id": "run-1", "started_at": 100.5},
            {"round_id": "run-1", "started_at": 100.0},
        ],
    )
    assert began == {
        "status": "ok",
        "action": "begin",
        "round_id": "run-1",
        "started_at": 100.0,
        "engines": 2,
    }
    ended = rounds.aggregate_round_replies(
        "end",
        "run-1",
        [
            {"round_id": "run-1", "started_at": 100.0, "ended_at": 160.0},
            {"round_id": "run-1", "started_at": 100.5, "ended_at": 161.0},
        ],
    )
    assert ended["started_at"] == 100.0
    assert ended["ended_at"] == 161.0
    assert ended["duration_s"] == 61.0
    with pytest.raises(rounds.PluginRefusal) as refused:
        rounds.aggregate_round_replies(
            "begin", "run-1", [{"round_id": "run-1"}, {"round_id": "run-2"}]
        )
    assert refused.value.reason == rounds.ROUND_STATE_MISMATCH_REASON


# -- the utility handlers ----------------------------------------------------


def test_the_round_handlers_are_registered():
    assert (
        EngineUtilityHandler._UTILITY_HANDLERS[rounds.BEGIN_ROUND_CMD]
        == "_handle_begin_plugin_round"
    )
    assert (
        EngineUtilityHandler._UTILITY_HANDLERS[rounds.END_ROUND_CMD]
        == "_handle_end_plugin_round"
    )
    # Neither may be fire-and-forget: their senders wait for the answer.
    assert rounds.BEGIN_ROUND_CMD not in FIRE_AND_FORGET_UTILITY_CMDS
    assert rounds.END_ROUND_CMD not in FIRE_AND_FORGET_UTILITY_CMDS


def test_the_begin_handler_takes_the_round(wired):
    reply = wired.handler._handle_begin_plugin_round({"round_id": "run-1"})
    assert reply["status"] == "ok"
    assert reply["round_id"] == "run-1"
    assert isinstance(reply["started_at"], float)
    assert wired.engine.plugin_round_id == "run-1"
    assert _last_reply(wired.replies) == reply


def test_a_second_begin_from_the_handler_is_a_409(wired):
    first = wired.handler._handle_begin_plugin_round({"round_id": "run-1"})
    second = wired.handler._handle_begin_plugin_round({"round_id": "run-2"})
    assert second["reason"] == rounds.ROUND_IN_PROGRESS_REASON
    assert rounds.refusal_status(rounds.PluginRefusal(second["reason"], "x")) == 409
    # Nothing moved: the first round still holds the plan with its start time.
    assert wired.engine.plugin_round_id == "run-1"
    assert wired.engine.plugin_round_started_at == first["started_at"]


def test_the_end_handler_clears_only_the_active_round(wired):
    wired.handler._handle_begin_plugin_round({"round_id": "run-1"})
    mismatch = wired.handler._handle_end_plugin_round({"round_id": "run-2"})
    assert mismatch["reason"] == rounds.ROUND_MISMATCH_REASON
    assert wired.engine.plugin_round_id == "run-1"

    ended = wired.handler._handle_end_plugin_round({"round_id": "run-1"})
    assert ended["status"] == "ok"
    assert ended["round_id"] == "run-1"
    assert ended["duration_s"] >= 0.0
    assert wired.engine.plugin_round_id is None
    assert wired.engine.plugin_round_started_at is None


def test_the_end_handler_without_a_round_is_a_409(wired):
    reply = wired.handler._handle_end_plugin_round({"round_id": "run-1"})
    assert reply["reason"] == rounds.NO_ACTIVE_ROUND_REASON


def test_a_hotplug_during_a_round_is_refused_with_no_state_change(wired):
    began = wired.handler._handle_begin_plugin_round({"round_id": "run-1"})
    reply = wired.handler._handle_hotplug_plugin({"payload": {"capture": False}})

    assert reply["reason"] == rounds.ROUND_IN_PROGRESS_REASON
    assert reply["active_round"]["round_id"] == "run-1"
    # The refusal comes before the pause: no generation transition, no
    # install and no capture happened, and the plan is the round's.
    assert wired.engine.paused == 0
    assert wired.engine.resumed == 0
    assert wired.runner.calls == []
    assert wired.engine.plugin_round_id == "run-1"
    assert wired.engine.plugin_round_started_at == began["started_at"]
    # The round guard is also first when the payload itself is unusable, so a
    # malformed hot-plug cannot be mistaken for a 400 while a round holds.
    assert (
        wired.handler._handle_hotplug_plugin({})["reason"]
        == rounds.ROUND_IN_PROGRESS_REASON
    )
    assert wired.runner.calls == []


def test_the_round_ending_releases_the_hotplug(wired):
    wired.handler._handle_begin_plugin_round({"round_id": "run-1"})
    wired.handler._handle_end_plugin_round({"round_id": "run-1"})

    reply = wired.handler._handle_hotplug_plugin({"payload": {"capture": False}})
    assert "error" not in reply
    assert reply["result"]["plan_id"] == "sha256:abc"
    assert wired.engine.paused == 1
    assert wired.engine.resumed == 1
    assert wired.runner.calls == [(rounds.HOTPLUG_CMD, {"capture": False})]


# -- the engine-core manager -------------------------------------------------


def test_the_manager_begins_and_ends_the_round(wired):
    began = wired.manager.begin_plugin_round("run-1", timeout=5.0)
    assert began["round_id"] == "run-1"
    assert began["action"] == "begin"
    assert began["engines"] == 1
    assert wired.engine.plugin_round_id == "run-1"
    # The cheap status view reads the cache; the engine holds the lease.
    assert wired.manager.plugin_round_status() == {
        "status": "ok",
        "active_round": {
            "round_id": "run-1",
            "started_at": began["started_at"],
        },
        "engines": 1,
    }

    ended = wired.manager.end_plugin_round("run-1", timeout=5.0)
    assert ended["round_id"] == "run-1"
    assert ended["action"] == "end"
    assert ended["duration_s"] >= 0.0
    assert wired.engine.plugin_round_id is None
    assert wired.manager.plugin_round_status()["active_round"] is None
    assert wired.broadcast_calls == [rounds.BEGIN_ROUND_CMD, rounds.END_ROUND_CMD]


def test_a_manager_round_gates_a_hotplug_and_end_releases_it(wired):
    wired.manager.begin_plugin_round("run-1")
    refused = wired.handler._handle_hotplug_plugin({"payload": {"capture": False}})
    assert refused["reason"] == rounds.ROUND_IN_PROGRESS_REASON
    assert wired.runner.calls == []

    wired.manager.end_plugin_round("run-1")
    allowed = wired.handler._handle_hotplug_plugin({"payload": {"capture": False}})
    assert "error" not in allowed
    assert wired.runner.calls == [(rounds.HOTPLUG_CMD, {"capture": False})]


def test_a_duplicate_begin_does_not_end_the_active_round(wired):
    wired.manager.begin_plugin_round("run-1")
    calls_after_begin = list(wired.broadcast_calls)

    with pytest.raises(rounds.PluginRefusal) as refused:
        wired.manager.begin_plugin_round("run-1")

    assert refused.value.reason == rounds.ROUND_IN_PROGRESS_REASON
    # The manager refused before broadcasting, so no rollback END was sent:
    # the active round is still held by the engine and its cache.
    assert wired.broadcast_calls == calls_after_begin
    assert wired.engine.plugin_round_id == "run-1"
    assert wired.manager.plugin_round_status()["active_round"]["round_id"] == "run-1"
    wired.manager.end_plugin_round("run-1")
    assert wired.engine.plugin_round_id is None


def test_a_conflicting_begin_is_rolled_back_and_keeps_its_reason(wired):
    calls = []

    def refuse(cmd, timeout=None, **kwargs):
        calls.append(cmd)
        raise rounds.PluginRefusal(
            rounds.ROUND_IN_PROGRESS_REASON, "another round is active"
        )

    wired.manager.broadcast_utility_command_sync = refuse
    with pytest.raises(rounds.PluginRefusal) as refused:
        wired.manager.begin_plugin_round("run-1")
    assert refused.value.reason == rounds.ROUND_IN_PROGRESS_REASON
    # Best-effort undo of the same id: an engine holding another round refuses
    # the mismatch, so a lost race never clears the winner's round.
    assert calls == [rounds.BEGIN_ROUND_CMD, rounds.END_ROUND_CMD]
    assert wired.manager.plugin_round_status()["active_round"] is None


def test_an_end_of_another_round_keeps_the_refusal_reason(wired):
    def refuse(cmd, timeout=None, **kwargs):
        raise rounds.PluginRefusal(
            rounds.ROUND_MISMATCH_REASON, "this engine runs another round"
        )

    wired.manager.broadcast_utility_command_sync = refuse
    with pytest.raises(rounds.PluginRefusal) as refused:
        wired.manager.end_plugin_round("run-2")
    assert refused.value.reason == rounds.ROUND_MISMATCH_REASON
    assert rounds.refusal_status(refused.value) == 409


def test_a_hotplug_refused_during_a_round_maps_to_409(wired):
    def refuse(cmd, timeout=None, **kwargs):
        raise rounds.PluginRefusal(
            rounds.ROUND_IN_PROGRESS_REASON, "a plugin round is active"
        )

    wired.manager.broadcast_utility_command_sync = refuse
    with pytest.raises(rounds.PluginRefusal) as refused:
        wired.manager.hotplug_plugin({"capture": False})
    # The API answers refusal_status/refusal_reason; neither may turn the
    # engine's 409 into a generic 500 or a 400.
    assert rounds.refusal_status(refused.value) == 409
    assert rounds.refusal_reason(refused.value) == rounds.ROUND_IN_PROGRESS_REASON


def test_a_coded_engine_refusal_survives_the_manager_round_trip(wired):
    """The reason travels engineer-reply -> _broadcast_and_wait -> 409.

    The tests above replace ``broadcast_utility_command_sync``, so they cannot
    show that the real round trip keeps ``reason`` instead of flattening the
    refusal to a plain RuntimeError; this one drives the real router path with
    a fake engine answer.
    """
    replies = queue.Queue()
    # Drop the fixture's one-handler shortcut: this test is about the real
    # synchronous broadcast path, so it must resolve on the class again.
    del wired.manager.broadcast_utility_command_sync
    wired.manager._rpc_router = types.SimpleNamespace(
        register=lambda request_id: contextlib.nullcontext(replies)
    )
    wired.manager._utility_phases = lambda cmd, kwargs: [range(1)]

    def broadcast(
        cmd,
        _dispatch_token=None,
        _ranks=None,
        request_id=None,
        timeout=None,
        deadline=None,
        **kwargs,
    ):
        replies.put(
            (
                0,
                {
                    "cmd": cmd,
                    "error": "refused: a plugin round is active",
                    "reason": rounds.ROUND_IN_PROGRESS_REASON,
                },
            )
        )

    wired.manager.broadcast_utility_command = broadcast
    with pytest.raises(rounds.PluginRefusal) as refused:
        wired.manager.hotplug_plugin({"capture": False})
    assert refused.value.reason == rounds.ROUND_IN_PROGRESS_REASON
    assert rounds.refusal_status(refused.value) == 409
    assert rounds.refusal_reason(refused.value) == rounds.ROUND_IN_PROGRESS_REASON


def test_the_status_view_is_advisory_without_an_engine(wired):
    wired.manager.control_sockets = []
    assert wired.manager.plugin_round_status() == {
        "status": "ok",
        "active_round": None,
        "engines": 0,
    }


# -- wiring: engine state, handler ordering, manager, HTTP routes ------------

TEXT = {
    path: (REPO_ROOT / path).read_text(encoding="utf-8")
    for path in (
        "atom/model_engine/engine_core.py",
        "atom/model_engine/engine_utility.py",
        "atom/model_engine/engine_core_mgr.py",
        "atom/entrypoints/openai/api_server.py",
    )
}


def _flat(source):
    """Whitespace-insensitive source text, for call-shape assertions."""
    return re.sub(r"\s+", " ", source)


def _classes(path):
    tree = ast.parse(TEXT[path])
    return {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }


def _methods(path, owner):
    return {
        node.name
        for node in _classes(path)[owner].body
        if isinstance(node, ast.FunctionDef)
    }


def test_the_engine_core_stores_the_round_and_nothing_else():
    tree = ast.parse(TEXT["atom/model_engine/engine_core.py"])
    core = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineCore"
    )
    init = next(
        node
        for node in core.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    # The whole round is two attributes; the scheduler and the generation loop
    # must never observe it (a blocked busy loop is the failure this guards).
    round_state = {"plugin_round_id", "plugin_round_started_at"}
    accesses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in round_state
    ]
    assert accesses, "EngineCore must store the active plugin round"
    inside_init = {id(node) for node in ast.walk(init)}
    assert all(
        id(node) in inside_init for node in accesses
    ), "round state is read outside EngineCore.__init__; it must be state only"

    assigned = set()
    for node in ast.walk(init):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    assert round_state <= assigned, "both round attributes must start as None"


def test_the_hotplug_handler_checks_the_round_before_pausing():
    flat = _flat(TEXT["atom/model_engine/engine_utility.py"])
    guard = flat.index("round_in_progress_reply(HOTPLUG_CMD, engine)")
    pause = flat.index("engine.pause_generation(deadline=deadline)")
    assert guard < pause, "a refusal must not pause anything"


def test_the_manager_owns_the_round_commands_and_their_reasons():
    manager_methods = _methods("atom/model_engine/engine_core_mgr.py", "CoreManager")
    assert {"begin_plugin_round", "end_plugin_round", "plugin_round_status"} <= (
        manager_methods
    )
    flat = _flat(TEXT["atom/model_engine/engine_core_mgr.py"])
    assert "BEGIN_ROUND_CMD, timeout=timeout, round_id=round_id" in flat
    assert "END_ROUND_CMD, timeout=timeout, round_id=round_id" in flat
    assert 'aggregate_round_replies("begin", round_id, replies)' in flat
    assert 'aggregate_round_replies("end", round_id, replies)' in flat
    # The coded refusal must be raised as one, not flattened to RuntimeError.
    assert "command_refusal_reason(failed)" in flat
    assert "raise PluginRefusal(reason, message)" in flat


def _engine_routes(path):
    tree = ast.parse(TEXT[path])
    routes = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "_engine_route"
            ):
                continue
            keywords = {kw.arg: kw.value for kw in decorator.keywords}
            route_path = decorator.args[0] if decorator.args else None
            routes.append((node, route_path, keywords))
    return routes


def _route_for(path, constant):
    match = [
        (node, keywords)
        for node, route_path, keywords in _engine_routes(path)
        if isinstance(route_path, ast.Name) and route_path.id == constant
    ]
    assert len(match) == 1, f"{constant} must be registered exactly once"
    return match[0]


def test_the_api_server_serves_the_internal_round_route():
    node, keywords = _route_for(
        "atom/entrypoints/openai/api_server.py", "PLUGIN_ROUND_ENDPOINT"
    )
    assert [method.value for method in keywords["methods"].elts] == ["POST"]
    assert keywords["include_in_schema"].value is False
    flat = _flat(
        ast.get_source_segment(TEXT["atom/entrypoints/openai/api_server.py"], node)
    )
    assert "parse_plugin_round_payload(body" in flat
    assert "begin_plugin_round" in flat and "end_plugin_round" in flat
    assert "round_request.round_id, round_request.timeout" in flat
    assert "refusal_status(exc)" in flat
    assert "JSONResponse(result)" in flat
    assert "asyncio.to_thread" in flat
    # A malformed body is a validated 400 with a reason, never a crash.
    assert 'RoundPayloadError("payload-schema", "body is not valid JSON")' in flat


def test_the_api_server_serves_the_internal_status_route():
    node, keywords = _route_for(
        "atom/entrypoints/openai/api_server.py", "PLUGIN_STATUS_ENDPOINT"
    )
    assert [method.value for method in keywords["methods"].elts] == ["GET"]
    assert keywords["include_in_schema"].value is False
    flat = _flat(
        ast.get_source_segment(TEXT["atom/entrypoints/openai/api_server.py"], node)
    )
    assert "plugin_round_status" in flat
    assert "hotplug_enabled(config)" in flat


def test_the_round_endpoints_are_not_in_the_openai_documentation():
    flat = _flat(TEXT["atom/entrypoints/openai/api_server.py"])
    assert 'PLUGIN_ROUND_ENDPOINT, methods=["POST"], include_in_schema=False' in flat
    assert 'PLUGIN_STATUS_ENDPOINT, methods=["GET"], include_in_schema=False' in flat
