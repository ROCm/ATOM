# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The branches behind `ATOM_HIGH_PRIORITY_COMPUTE_STREAM`.

The TBO branch is what keeps V4.1's comm-ahead ordering (its comm stream is
already High), so it has to stay a branch rather than become a refactor
casualty. The decision lives in `compute_stream` so these cases can be checked
on a machine with no GPU build; importing the runner resolves the chip
architecture through aiter, so only the cases below that build a stream are
gated on a device.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.model_engine.compute_stream import high_priority_compute_stream_enabled

FLAG = "ATOM_HIGH_PRIORITY_COMPUTE_STREAM"

needs_cuda_build = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="importing ModelRunner resolves the chip architecture through aiter",
)


def _config(**overrides):
    return SimpleNamespace(
        **{"enable_tbo": False, "enable_tbo_decode": False, **overrides}
    )


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


class _FakeStream:
    """Stand-in for torch.cuda.Stream that records device and priority."""

    def __init__(self, device=None, priority=0):
        self.device = device
        self.priority = priority

    @staticmethod
    def priority_range():
        return 0, -1


class _FakeEvent:
    """The pools record events; this keeps the tests off the device."""


def _bare_runner(monkeypatch, cls, **config):
    """An instance of `cls` with torch.cuda stubbed and nothing else built."""
    calls = []
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device=None: calls.append("synchronize")
    )
    monkeypatch.setattr(
        torch.cuda, "set_stream", lambda stream: calls.append(("set_stream", stream))
    )

    runner = cls.__new__(cls)
    runner.device = torch.device("cuda", 3)
    runner.config = _config(**config)
    runner.calls = calls
    return runner


@pytest.fixture
def runner(monkeypatch):
    from atom.model_engine.model_runner import ModelRunner

    return _bare_runner(monkeypatch, ModelRunner)


@pytest.fixture
def rapidserve_runner(monkeypatch):
    """Builds a RapidServe runner that has entered its compute stream, as an
    engine core does before it builds its stream pools."""
    from atom.model_engine.model_runner import RapidServeModelRunner

    def build():
        runner = _bare_runner(
            monkeypatch, RapidServeModelRunner, disagg_constrained=False
        )
        runner.compute_stream = runner._enter_compute_stream()
        return runner

    return build


@needs_cuda_build
def test_flag_on_switches_this_thread(runner, monkeypatch):
    monkeypatch.setenv(FLAG, "1")

    stream = runner._enter_compute_stream()

    _, high = _FakeStream.priority_range()
    assert stream.priority == high
    assert stream.device == runner.device
    # The default stream has to drain before the switch: a torch stream does
    # not sync with it, so anything still queued there would overlap.
    assert runner.calls == ["synchronize", ("set_stream", stream)]


@needs_cuda_build
def test_decode_pool_matches_the_compute_stream(runner, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    compute_stream = runner._enter_compute_stream()

    fallback = runner._new_compute_priority_stream()

    assert fallback.priority == compute_stream.priority
    assert fallback.device == runner.device


@needs_cuda_build
@pytest.mark.parametrize("flag", ["0", "1"])
def test_rapidserve_prefill_and_decode_pools_share_a_priority(
    rapidserve_runner, monkeypatch, flag
):
    """One side of a disaggregated rank must not outrank the other, and the
    flag has to reach prefill as well as decode."""
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    runner = rapidserve_runner()

    runner.create_prefill_stream_pool()
    runner.create_decode_stream_pool()

    prefill = runner._prefill_streams[None]
    decode = runner._decode_streams[None]
    expected = _FakeStream.priority_range()[1] if flag == "1" else 0
    assert (prefill.priority, decode.priority) == (expected, expected)
    assert (prefill.device, decode.device) == (runner.device, runner.device)
