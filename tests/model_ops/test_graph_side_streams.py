# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_GRAPH_SIDE_STREAMS=0: a CUDA graph records no side-stream fork, so HIP
binds no parallel branch of it to a hardware queue; eager forwards keep theirs."""

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def gate(monkeypatch, atom_modules):
    """`_dual_stream_is_active` on a decode step, with capture state settable."""
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
    capturing = {"now": False}
    monkeypatch.setattr(
        dispatch.torch.cuda, "is_current_stream_capturing", lambda: capturing["now"]
    )
    moe = SimpleNamespace(
        _use_dual_stream=True,
        _dual_stream_token_threshold=1024,
        _dual_stream_decode_token_threshold=1024,
    )

    def active(*, recording: bool) -> bool:
        capturing["now"] = recording
        return dispatch._dual_stream_is_active(moe, 4)

    return active


@pytest.mark.parametrize("value", [None, "1"])
def test_graphs_record_the_shared_expert_fork_by_default(gate, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ATOM_GRAPH_SIDE_STREAMS", raising=False)
    else:
        monkeypatch.setenv("ATOM_GRAPH_SIDE_STREAMS", value)
    assert gate(recording=True)
    assert gate(recording=False)


def test_off_a_recording_keeps_the_shared_expert_inline(gate, monkeypatch):
    monkeypatch.setenv("ATOM_GRAPH_SIDE_STREAMS", "0")
    assert not gate(recording=True)
    assert gate(recording=False), "eager forwards keep their side stream"


@pytest.mark.parametrize(("value", "forks"), [(None, True), ("1", True), ("0", False)])
def test_v4_compressors_fork_under_a_graph_only_when_allowed(
    monkeypatch, atom_modules, value, forks
):
    v4 = atom_modules.deepseek_v4
    if value is None:
        monkeypatch.delenv("ATOM_GRAPH_SIDE_STREAMS", raising=False)
    else:
        monkeypatch.setenv("ATOM_GRAPH_SIDE_STREAMS", value)
    monkeypatch.setattr(v4, "tbo_active", lambda: False)
    attn = SimpleNamespace(_use_async_compress=True)
    for in_hipgraph in (True, False):
        fc = SimpleNamespace(in_hipgraph=in_hipgraph)
        monkeypatch.setattr(v4, "get_forward_context", lambda fc=fc: fc)
        got = v4.DeepseekV4Attention._compressors_async(attn)
        assert bool(got) == (forks and in_hipgraph)
