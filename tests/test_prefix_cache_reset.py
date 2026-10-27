# SPDX-License-Identifier: MIT
import ast
import logging
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.prefix_cache_reset import (
    PrefixCacheBusyError,
    PrefixCacheResetGate,
)


def manager():
    mgr = object.__new__(CoreManager)
    mgr.pp_size = 1
    mgr._prefix_cache_reset_gate = PrefixCacheResetGate()
    mgr.broadcast_utility_command_sync = Mock(
        return_value=[{"result": {"status": "ok"}}, {"result": {"status": "ok"}}]
    )
    return mgr


def test_reset_requires_every_rank_in_both_phases():
    mgr = manager()
    assert mgr.reset_prefix_cache() == {
        "success": True,
        "ranks": 2,
        "statistics_reset": False,
    }
    assert [
        c.kwargs["dry_run"] for c in mgr.broadcast_utility_command_sync.call_args_list
    ] == [True, False]


@pytest.mark.parametrize(
    "failure", [TimeoutError("missing rank"), RuntimeError("rank failed")]
)
def test_partial_reset_never_reports_success_and_releases_gate(failure):
    mgr = manager()
    mgr.broadcast_utility_command_sync.side_effect = [
        [{"result": {"status": "ok"}}],
        failure,
    ]
    with pytest.raises(type(failure)):
        mgr.reset_prefix_cache()
    with mgr._prefix_cache_reset_gate.resetting():
        pass


def test_busy_preflight_does_not_clear_any_rank():
    mgr = manager()
    mgr.broadcast_utility_command_sync.return_value = [
        {"result": {"status": "ok"}},
        {"result": {"status": "busy", "reason": "queued"}},
    ]
    with pytest.raises(PrefixCacheBusyError):
        mgr.reset_prefix_cache()
    assert mgr.broadcast_utility_command_sync.call_count == 1


def test_submission_remains_busy_until_terminal_scheduler_output():
    mgr = manager()
    gate = mgr._prefix_cache_reset_gate
    with gate.submitting([SimpleNamespace(id="req")]):
        pass
    with pytest.raises(PrefixCacheBusyError):
        mgr.reset_prefix_cache()
    mgr.broadcast_utility_command_sync.assert_not_called()
    gate.finished("req")
    assert mgr.reset_prefix_cache()["success"]


def test_new_admission_cannot_cross_reset_fence():
    gate = PrefixCacheResetGate()
    started = threading.Event()
    admitted = threading.Event()

    def submit():
        started.set()
        with gate.submitting([SimpleNamespace(id="new")]):
            admitted.set()

    with gate.resetting():
        t = threading.Thread(target=submit)
        t.start()
        assert started.wait(1)
        assert not admitted.wait(0.05)
    t.join(1)
    assert admitted.is_set()


@pytest.mark.parametrize(
    "busy,external,dry,expected",
    [
        (False, False, False, "ok"),
        (False, False, True, "ok"),
        (True, False, False, "busy"),
        (False, True, False, "unsupported"),
    ],
)
def test_scheduler_reset_guards(busy, external, dry, expected):
    bm = SimpleNamespace(clear_cache=Mock(), state_offload=None)
    scheduler = SimpleNamespace(
        block_manager=bm,
        kv_connector=object() if external else None,
        has_requests=lambda: busy,
    )
    out = queue.Queue()
    handler = EngineUtilityHandler(Mock(), out, scheduler=scheduler)
    handler._execute_utility_command(
        "reset_prefix_cache", {"dry_run": dry, "request_id": "reset1"}
    )
    _, reply = out.get_nowait()
    assert reply["request_id"] == "reset1"
    assert reply["result"]["status"] == expected
    assert bm.clear_cache.call_count == int(expected == "ok" and not dry)
    handler.runner_mgr.assert_not_called()


def test_clear_failure_is_reported_without_crashing_engine():
    bm = SimpleNamespace(
        clear_cache=Mock(side_effect=RuntimeError("bad index")), state_offload=None
    )
    out = queue.Queue()
    h = EngineUtilityHandler(
        Mock(),
        out,
        scheduler=SimpleNamespace(
            block_manager=bm, kv_connector=None, has_requests=lambda: False
        ),
    )
    h._execute_utility_command("reset_prefix_cache", {})
    assert out.get_nowait()[1]["error"] == "bad index"


@pytest.mark.parametrize(
    "error,status",
    [
        (None, 200),
        (PrefixCacheBusyError("busy"), 409),
        (NotImplementedError("offload"), 501),
        (TimeoutError("rank missing"), 504),
        (RuntimeError("partial failure"), 500),
    ],
)
def test_http_contract(error, status):
    # Exercise the actual route body without importing GPU startup dependencies.
    source = Path(__file__).parents[1] / "atom/entrypoints/openai/api_server.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "reset_prefix_cache"
    )
    app = FastAPI()
    reset = Mock(return_value={"success": True}, side_effect=error)
    ns = {
        "app": app,
        "engine": SimpleNamespace(reset_prefix_cache=reset),
        "HTTPException": HTTPException,
        "PrefixCacheBusyError": PrefixCacheBusyError,
        "logger": logging.getLogger("test"),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)  # noqa: S102 - isolated route test
    with TestClient(app) as client:
        assert client.post("/reset_prefix_cache").status_code == status
        ns["engine"] = None
        assert client.post("/reset_prefix_cache").status_code == 503


def test_real_prefix_cache_cold_hit_reset_cold(seq_factory):
    from test_kv_events import _admit, _bm_with_events

    bm = _bm_with_events()
    prompt = list(range(12))
    first = seq_factory(prompt)
    assert bm.can_allocate_memoized(first) == 0
    _admit(bm, first)
    bm.deallocate(first)
    probe = seq_factory(prompt)
    assert bm.can_allocate_memoized(probe) > 0
    output = queue.Queue()
    handler = EngineUtilityHandler(
        Mock(),
        output,
        scheduler=SimpleNamespace(
            block_manager=bm, kv_connector=None, has_requests=lambda: False
        ),
    )
    handler._execute_utility_command("reset_prefix_cache", {})
    assert output.get_nowait()[1]["result"]["status"] == "ok"
    assert bm.kv.num_indexed == 0
    assert bm.can_allocate_memoized(probe) == 0
    assert bm.can_allocate_memoized(seq_factory(prompt)) == 0
