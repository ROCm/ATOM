# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_GRAPH_SIDE_STREAMS=0: a graph window holding Engram's lookup records
no fork. `EngramStaging.start` looks the rows up on the compute stream after the
cursor advance and `join` has nothing to close; otherwise (and in eager
forwards) the lookup keeps its side stream."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.utils import forward_context

try:
    from atom.model_ops.engram.device import staging as staging_mod
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    pytest.skip(f"requires the device Engram imports: {_e}", allow_module_level=True)


@pytest.fixture
def staging(monkeypatch):
    compute = Mock(name="compute")
    entered = []

    @contextmanager
    def guard(stream):
        entered.append(stream)
        yield

    monkeypatch.setattr(staging_mod.torch.cuda, "current_stream", lambda *_: compute)
    monkeypatch.setattr(staging_mod.torch.cuda, "stream", guard)
    s = staging_mod.EngramStaging.__new__(staging_mod.EngramStaging)
    s.host = SimpleNamespace(device="cpu")
    s.stream = Mock(name="side")
    s.step = SimpleNamespace(served_by_round=False)
    s._served = s._inline = False
    order = []
    s._snapshot = Mock(side_effect=lambda w: order.append("snapshot"))
    s._advance_cursor = Mock(side_effect=lambda w: order.append("cursor"))
    s._lookup = Mock(side_effect=lambda w, *a: order.append("lookup"))
    return SimpleNamespace(staging=s, compute=compute, entered=entered, order=order)


def _single(monkeypatch, value):
    monkeypatch.setattr(forward_context, "graph_records_single_stream", lambda: value)


def test_a_single_stream_recording_looks_up_inline(staging, monkeypatch):
    _single(monkeypatch, True)
    s = staging.staging
    s.start(8)
    assert staging.order == ["snapshot", "cursor", "lookup"]
    s._lookup.assert_called_once_with(8, staging.compute)
    s.stream.wait_stream.assert_not_called()
    assert staging.entered == []  # no side-stream guard
    s.join()
    staging.compute.wait_stream.assert_not_called()


def test_otherwise_the_lookup_forks_and_join_closes_it(staging, monkeypatch):
    _single(monkeypatch, False)
    s = staging.staging
    s.start(8)
    assert staging.order == ["snapshot", "cursor", "lookup"]
    s.stream.wait_stream.assert_called_once_with(staging.compute)
    assert staging.entered == [s.stream]
    s.join()
    staging.compute.wait_stream.assert_called_once_with(s.stream)


def test_a_forked_step_after_an_inline_one_joins_again(staging, monkeypatch):
    s = staging.staging
    _single(monkeypatch, True)
    s.start(8)
    _single(monkeypatch, False)
    s.start(8)
    s.join()
    staging.compute.wait_stream.assert_called_once_with(s.stream)


def test_the_lookup_marks_each_layer_on_its_stream():
    s = staging_mod.EngramStaging.__new__(staging_mod.EngramStaging)
    s.host = SimpleNamespace(
        layer_ids=[2, 5],
        _tp_group=None,
        local_heads=1,
        head_start=0,
        total_heads=1,
        buffers={
            2: SimpleNamespace(gpu=torch.zeros(4, 3)),
            5: SimpleNamespace(gpu=torch.zeros(4, 3)),
        },
    )
    s.local_width = 3
    s.lookup = torch.zeros(2 * 4 * 3)
    s.uva = SimpleNamespace(row_ids=torch.zeros(2, 4, dtype=torch.int64))
    s.gather = Mock()
    s.collective = None
    s.stream = Mock(name="side")
    s.done = {2: Mock(), 5: Mock()}
    compute = Mock(name="compute")
    s._lookup(4, compute)
    for event in s.done.values():
        event.record.assert_called_once_with(compute)
    for event in s.done.values():
        event.reset_mock()
    s._lookup(4)  # the side stream by default
    for event in s.done.values():
        event.record.assert_called_once_with(s.stream)
