# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The branches behind `ATOM_HIGH_PRIORITY_COMPUTE_STREAM`.

The TBO branch is what keeps V4.1's comm-ahead ordering (its comm stream is
already High), so it has to stay a branch rather than become a refactor
casualty. The decision and both stream builders live in `compute_stream` and
import directly; a fake `torch.cuda` is all they need to run without a device.

The RapidServe pools are runner methods and a CPU runner cannot import
`ModelRunner`: aiter resolves the chip architecture through `rocminfo` on the
way in. They are small and self-contained, so these tests compile them out of
the module source and run in the pre-checkin suite rather than only where a
device happens to be, as `test_model_runner_staging_fence` does for the
staging-fence contracts.
"""

import ast
import logging
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.model_engine import compute_stream as cs

FLAG = "ATOM_HIGH_PRIORITY_COMPUTE_STREAM"
MODEL_RUNNER = Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
# The pool builders left on the runner. The two that construct the compute
# stream itself are imported above instead of compiled from source.
POOL_BUILDERS = (
    "_new_pool_fallback_stream",
    "create_prefill_stream_pool",
    "create_decode_stream_pool",
)


def _config(**overrides):
    return SimpleNamespace(
        **{"enable_tbo": False, "enable_tbo_decode": False, **overrides}
    )


class _FakeStream:
    """Stand-in for torch.cuda.Stream that records device and priority."""

    def __init__(self, device=None, priority=0):
        self.device = device
        self.priority = priority

    @staticmethod
    def priority_range():
        return 0, -1


class _FakeEvent:
    """The decode pool records two events; this keeps the tests off a device."""


@pytest.fixture
def cuda(monkeypatch):
    """Put the imported builders on a fake device; return what they ordered."""
    calls = []
    monkeypatch.setattr(cs.torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(
        cs.torch.cuda, "synchronize", lambda device=None: calls.append("synchronize")
    )
    monkeypatch.setattr(
        cs.torch.cuda, "set_stream", lambda stream: calls.append(("set_stream", stream))
    )
    return calls


def _pool_builders():
    """The runner's pool builders, compiled from the module source."""
    module = ast.parse(MODEL_RUNNER.read_text())
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "RapidServeModelRunner"
    )
    methods = [
        next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for name in POOL_BUILDERS
    ]
    namespace = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(Stream=_FakeStream, Event=_FakeEvent)
        ),
        "logger": logging.getLogger(__name__),
        # The fallback defers to the imported builder, so the pools cannot
        # drift from the stream the runner thread actually entered.
        "new_compute_priority_stream": cs.new_compute_priority_stream,
    }
    exec(  # noqa: S102 - runs the production pool builders extracted above
        compile(ast.Module(body=methods, type_ignores=[]), str(MODEL_RUNNER), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def runner(cuda):
    """A bare runner carrying only what the pool builders read."""
    runner = SimpleNamespace(
        device="cuda:3",
        config=_config(),
        calls=cuda,
        compute_stream=None,
        _CU_POOL_FRACTIONS=[0.5],
    )
    for name, built in _pool_builders().items():
        if isinstance(built, types.FunctionType):
            setattr(runner, name, types.MethodType(built, runner))
    return runner


def test_flag_off_keeps_the_default_stream(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert cs.high_priority_compute_stream_enabled(_config()) is False


def test_flag_on_without_tbo_switches(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    assert cs.high_priority_compute_stream_enabled(_config()) is True


@pytest.mark.parametrize("knob", ["enable_tbo", "enable_tbo_decode"])
def test_tbo_wins_over_the_flag(monkeypatch, caplog, knob):
    monkeypatch.setenv(FLAG, "1")
    with caplog.at_level("WARNING", logger="atom"):
        assert cs.high_priority_compute_stream_enabled(_config(**{knob: True})) is False
    assert FLAG in caplog.text


def test_flag_on_switches_this_thread(cuda, monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    stream = cs.enter_compute_stream(_config(), "cuda:3")

    _, high = _FakeStream.priority_range()
    assert stream.priority == high
    assert stream.device == "cuda:3"
    # The default stream has to drain before the switch: a torch stream does
    # not sync with it, so anything still queued there would overlap.
    assert cuda == ["synchronize", ("set_stream", stream)]


def test_flag_off_builds_no_stream(cuda, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)

    assert cs.enter_compute_stream(_config(), "cuda:3") is None
    assert cuda == []


def _init_calls():
    """`(line, callee name)` for every call in `ModelRunner.__init__`."""
    module = ast.parse(MODEL_RUNNER.read_text())
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    init = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    calls = []
    for node in ast.walk(init):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name:
            calls.append((node.lineno, name))
    return calls


def test_the_thread_switches_streams_only_once_the_weights_are_loaded():
    """The loader hands each `weight_loader` to a thread pool, and those
    threads launch on the default stream, which a non-blocking stream is not
    ordered against. Switching before the load leaves the init `zero_`s and
    `process_weights_after_loading` racing the copies, for silently wrong
    weights. `enter_compute_stream` drains the device before it switches, so
    the ordering is the call's position in `__init__`.

    Matched by name shape rather than exact names: the loading entry points
    have already been restructured once under this branch, and a rename that
    left this test matching nothing would retire it silently.
    """
    calls = _init_calls()
    switch = [line for line, name in calls if name == "enter_compute_stream"]
    assert len(switch) == 1, "one switch, or this test checks the wrong one"
    loads = {
        name: line
        for line, name in calls
        if "load" in name.lower() or name == "get_model"
    }
    assert loads, "no model load found in __init__; have they been renamed?"

    assert switch[0] > max(loads.values()), (
        f"enter_compute_stream at line {switch[0]} runs before a weight load "
        f"({loads}); the loader threads' copies are unordered against it"
    )


@pytest.mark.parametrize("flag", ["0", "1"])
def test_rapidserve_prefill_and_decode_pools_share_a_priority(
    runner, monkeypatch, flag
):
    """One side of a disaggregated rank must not outrank the other, and the
    flag has to reach prefill as well as decode."""
    monkeypatch.setenv(FLAG, flag)
    runner.compute_stream = cs.enter_compute_stream(runner.config, runner.device)

    runner.create_prefill_stream_pool()
    runner.create_decode_stream_pool()

    prefill = runner._prefill_streams[None]
    decode = runner._decode_streams[None]
    expected = _FakeStream.priority_range()[1] if flag == "1" else 0
    assert (prefill.priority, decode.priority) == (expected, expected)
    assert (prefill.device, decode.device) == (runner.device, runner.device)
