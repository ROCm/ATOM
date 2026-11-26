# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_GRAPH_SIDE_STREAMS=0: a CUDA graph records no side-stream fork, so HIP
binds no parallel branch of it to a hardware queue; eager forwards keep theirs.

One predicate, `graph_records_single_stream`, answers for every fork over the
whole graph window: the capture loop's warmup forward as well as its capture,
and any other stream capture. Unset, the switch follows GPU_MAX_HW_QUEUES."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.utils import envs, forward_context, region_graph


@pytest.fixture
def switch(monkeypatch):
    """Set ATOM_GRAPH_SIDE_STREAMS (None: unset) over a default queue pool."""
    monkeypatch.delenv("GPU_MAX_HW_QUEUES", raising=False)

    def set_to(value):
        if value is None:
            monkeypatch.delenv("ATOM_GRAPH_SIDE_STREAMS", raising=False)
        else:
            monkeypatch.setenv("ATOM_GRAPH_SIDE_STREAMS", value)

    return set_to


@pytest.fixture
def window(monkeypatch):
    """Place the forward in or out of the capture loop (`in_hipgraph`) and a
    stream capture, independently."""
    state = SimpleNamespace(in_hipgraph=False, capturing=False)
    main = Mock(name="main")
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(in_hipgraph=state.in_hipgraph, main_stream=main),
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state.capturing
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())

    def place(*, in_hipgraph: bool, capturing: bool):
        state.in_hipgraph, state.capturing = in_hipgraph, capturing

    place.main = main
    return place


@pytest.mark.parametrize(
    ("queues", "value", "side_streams"),
    [
        (None, None, True),
        ("4", None, True),
        ("5", None, False),
        ("8", None, False),
        ("abc", None, True),
        ("5", "1", True),
        ("4", "0", False),
        (None, "0", False),
    ],
)
def test_unset_the_switch_follows_the_queue_pool(
    monkeypatch, queues, value, side_streams
):
    """HIP's pool holds four queues by default; a fifth is the first to share
    the launch queue's command processor pipe. An explicit value wins."""
    for name, v in (("GPU_MAX_HW_QUEUES", queues), ("ATOM_GRAPH_SIDE_STREAMS", value)):
        if v is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, v)
    assert envs.ATOM_GRAPH_SIDE_STREAMS is side_streams


@pytest.mark.parametrize(
    ("in_hipgraph", "capturing", "in_window"),
    [
        (True, False, True),  # the capture loop's warmup forward
        (True, True, True),  # its capture
        (False, True, True),  # a plugin frontend's own capture
        (False, False, False),  # eager
    ],
)
@pytest.mark.parametrize("value", [None, "1", "0"])
def test_one_predicate_covers_the_whole_graph_window(
    switch, window, value, in_hipgraph, capturing, in_window
):
    switch(value)
    window(in_hipgraph=in_hipgraph, capturing=capturing)
    single = forward_context.graph_records_single_stream()
    assert single == (in_window and value == "0")


@pytest.mark.parametrize("value", [None, "0"])
def test_side_stream_forks_in_the_window_only_where_allowed(switch, window, value):
    switch(value)
    window(in_hipgraph=True, capturing=False)
    side = Mock(name="side")
    with forward_context.side_stream(side) as (issuing, joining):
        pass
    if value == "0":
        assert (issuing, joining) == (window.main, None)
        side.wait_stream.assert_not_called()
    else:
        assert (issuing, joining) == (side, window.main)
        side.wait_stream.assert_called_once_with(window.main)


@pytest.mark.parametrize("value", [None, "0"])
def test_region_forks_follow_side_stream(switch, window, monkeypatch, value):
    """A whole-forward capture's region forks where `side_stream` does; a
    token graph's region runs eagerly between segments, so its fork stays."""
    switch(value)
    window(in_hipgraph=True, capturing=True)
    side = Mock(name="side")
    monkeypatch.setattr(region_graph, "_recorder", lambda: None)
    assert region_graph.forks(side) is (value != "0")
    monkeypatch.setattr(region_graph, "_recorder", lambda: object())
    assert region_graph.forks(side)


@pytest.fixture
def gate(monkeypatch, atom_modules, window):
    """`_dual_stream_is_active` on a decode step, placed in or out of a graph
    window."""
    dispatch = sys.modules["atom.model_ops.module_dispatch_ops"]
    monkeypatch.setattr(dispatch, "plugin_graph_is_piecewise", lambda: False)
    monkeypatch.setattr(
        sys.modules["atom.utils.tbo.ubatching"], "tbo_active", lambda: False
    )
    monkeypatch.setattr(
        dispatch,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(is_prefill=False)),
    )
    moe = SimpleNamespace(
        _use_dual_stream=True,
        _dual_stream_token_threshold=1024,
        _dual_stream_decode_token_threshold=1024,
    )

    def active(*, in_hipgraph: bool, capturing: bool) -> bool:
        window(in_hipgraph=in_hipgraph, capturing=capturing)
        return dispatch._dual_stream_is_active(moe, 4)

    return active


@pytest.mark.parametrize("value", [None, "1"])
def test_graphs_record_the_shared_expert_fork_by_default(gate, switch, value):
    switch(value)
    assert gate(in_hipgraph=True, capturing=False)
    assert gate(in_hipgraph=True, capturing=True)
    assert gate(in_hipgraph=False, capturing=False)


def test_off_the_whole_window_keeps_the_shared_expert_inline(gate, switch):
    """The warmup forward stays inline as its capture does."""
    switch("0")
    assert not gate(in_hipgraph=True, capturing=False), "warmup"
    assert not gate(in_hipgraph=True, capturing=True), "capture"
    assert not gate(in_hipgraph=False, capturing=True), "a plugin's capture"
    assert gate(in_hipgraph=False, capturing=False), "eager keeps its side stream"


@pytest.mark.parametrize(("value", "forks"), [(None, True), ("1", True), ("0", False)])
def test_v4_compressors_fork_under_a_graph_only_when_allowed(
    monkeypatch, atom_modules, switch, window, value, forks
):
    v4 = atom_modules.deepseek_v4
    switch(value)
    monkeypatch.setattr(v4, "tbo_active", lambda: False)
    attn = SimpleNamespace(_use_async_compress=True)
    for in_hipgraph in (True, False):
        window(in_hipgraph=in_hipgraph, capturing=False)
        monkeypatch.setattr(
            v4,
            "get_forward_context",
            lambda h=in_hipgraph: SimpleNamespace(in_hipgraph=h),
        )
        got = v4.DeepseekV4Attention._compressors_async(attn)
        assert bool(got) == (forks and in_hipgraph)
