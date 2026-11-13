# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The branches behind `ATOM_HIGH_PRIORITY_COMPUTE_STREAM`.

The TBO branch is what keeps V4.1's comm-ahead ordering (its comm stream is
already High), so it has to stay a branch rather than become a refactor
casualty. That decision lives in `compute_stream` and imports directly.

The streams themselves are built by `ModelRunner`, which a CPU runner cannot
import: aiter resolves the chip architecture through `rocminfo` on the way in.
The builders are small and self-contained, so these tests compile them out of
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

from atom.model_engine.compute_stream import high_priority_compute_stream_enabled

FLAG = "ATOM_HIGH_PRIORITY_COMPUTE_STREAM"
MODEL_RUNNER = Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
# The builders, by the class each belongs to: one that moves between the two
# fails here rather than leaving the wrong one under test.
STREAM_BUILDERS = {
    "ModelRunner": ("_new_compute_priority_stream", "_enter_compute_stream"),
    "RapidServeModelRunner": (
        "_new_pool_fallback_stream",
        "create_prefill_stream_pool",
        "create_decode_stream_pool",
    ),
}


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


def _stream_builders(calls):
    """The runner's stream builders, compiled from the module source."""
    module = ast.parse(MODEL_RUNNER.read_text())
    methods = []
    for class_name, names in STREAM_BUILDERS.items():
        owner = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        for name in names:
            methods.append(
                next(
                    node
                    for node in owner.body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                )
            )
    namespace = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=_FakeStream,
                Event=_FakeEvent,
                synchronize=lambda device=None: calls.append("synchronize"),
                set_stream=lambda stream: calls.append(("set_stream", stream)),
            )
        ),
        "logger": logging.getLogger(__name__),
        "high_priority_compute_stream_enabled": high_priority_compute_stream_enabled,
    }
    exec(  # noqa: S102 - runs the production stream builders extracted above
        compile(ast.Module(body=methods, type_ignores=[]), str(MODEL_RUNNER), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def runner():
    """A bare runner carrying only what the stream builders read."""
    calls = []
    runner = SimpleNamespace(
        device="cuda:3",
        config=_config(),
        calls=calls,
        compute_stream=None,
        _CU_POOL_FRACTIONS=[0.5],
    )
    for name, built in _stream_builders(calls).items():
        if isinstance(built, types.FunctionType):
            setattr(runner, name, types.MethodType(built, runner))
    return runner


def test_flag_off_keeps_the_default_stream(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert high_priority_compute_stream_enabled(_config()) is False


def test_flag_on_without_tbo_switches(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    assert high_priority_compute_stream_enabled(_config()) is True


@pytest.mark.parametrize("knob", ["enable_tbo", "enable_tbo_decode"])
def test_tbo_wins_over_the_flag(monkeypatch, caplog, knob):
    monkeypatch.setenv(FLAG, "1")
    with caplog.at_level("WARNING", logger="atom"):
        assert high_priority_compute_stream_enabled(_config(**{knob: True})) is False
    assert FLAG in caplog.text


def test_flag_on_switches_this_thread(runner, monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    stream = runner._enter_compute_stream()

    _, high = _FakeStream.priority_range()
    assert stream.priority == high
    assert stream.device == runner.device
    # The default stream has to drain before the switch: a torch stream does
    # not sync with it, so anything still queued there would overlap.
    assert runner.calls == ["synchronize", ("set_stream", stream)]


def test_flag_off_builds_no_stream(runner, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)

    assert runner._enter_compute_stream() is None
    assert runner.calls == []


@pytest.mark.parametrize("flag", ["0", "1"])
def test_rapidserve_prefill_and_decode_pools_share_a_priority(
    runner, monkeypatch, flag
):
    """One side of a disaggregated rank must not outrank the other, and the
    flag has to reach prefill as well as decode."""
    monkeypatch.setenv(FLAG, flag)
    runner.compute_stream = runner._enter_compute_stream()

    runner.create_prefill_stream_pool()
    runner.create_decode_stream_pool()

    prefill = runner._prefill_streams[None]
    decode = runner._decode_streams[None]
    expected = _FakeStream.priority_range()[1] if flag == "1" else 0
    assert (prefill.priority, decode.priority) == (expected, expected)
    assert (prefill.device, decode.device) == (runner.device, runner.device)
