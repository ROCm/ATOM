# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Under DP every prefill recording pass starts behind a CPU-group barrier.

Each pass runs collectives for real. Without the barrier a rank that reaches
the prefill recordings late (a cold start's JIT) leaves its peers spinning in a
device-side collective while it is still capturing, and startup can hang.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def model_runner():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


def _runner(model_runner, monkeypatch, dp, events):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        model_runner,
        "get_dp_group",
        lambda: SimpleNamespace(barrier=lambda: events.append("barrier")),
    )
    builder = SimpleNamespace(
        build_for_token_graph_capture=lambda n: ("tokens", n),
        build_for_req_graph_capture=lambda bs, finishing: ("reqs", bs, finishing),
        prefill_req_graph_sizes=[2, 1],
    )
    runner = SimpleNamespace(
        rank=1,
        config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp)),
        attn_metadata_builder=builder,
        running_tokens_per_dp_rank=np.zeros(dp, dtype=np.int64),
        warmup_max_tokens=lambda: 1024,
    )
    runner._build_capture = lambda build, *args: (
        None,
        SimpleNamespace(scheduled_tokens=8, step=build(*args)),
    )
    runner._run_capture_prefill = lambda attn_metadata, context, mode: (
        events.append(("pass", context.step, mode))
    )
    runner._capture_barrier = lambda: model_runner.ModelRunner._capture_barrier(runner)
    return runner


def _check_every_pass_follows_a_barrier(events):
    passes = [i for i, e in enumerate(events) if e != "barrier"]
    assert passes
    for i in passes:
        assert i > 0 and events[i - 1] == "barrier", events


def test_token_graph_passes_wait_for_every_dp_rank(model_runner, monkeypatch):
    monkeypatch.setenv("ATOM_TOKEN_GRAPH_SIZES", "256,512")
    events = []
    runner = _runner(model_runner, monkeypatch, 2, events)
    model_runner.ModelRunner._record_tr_graphs(runner)
    _check_every_pass_follows_a_barrier(events)
    assert [e[1:] for e in events if e != "barrier"] == [
        (("tokens", 512), None),
        (("tokens", 512), "record"),
        (("tokens", 256), None),
        (("tokens", 256), "record"),
    ]


def test_request_graph_passes_wait_for_every_dp_rank(model_runner, monkeypatch):
    events = []
    runner = _runner(model_runner, monkeypatch, 2, events)
    model_runner.ModelRunner._record_prefill_req_graphs(runner)
    _check_every_pass_follows_a_barrier(events)
    # Two sizes, finishing and not, a warmup and a recording each.
    assert events.count("barrier") == 8


def test_single_dp_rank_never_waits(model_runner, monkeypatch):
    monkeypatch.setenv("ATOM_TOKEN_GRAPH_SIZES", "256")
    events = []
    runner = _runner(model_runner, monkeypatch, 1, events)
    model_runner.ModelRunner._record_tr_graphs(runner)
    model_runner.ModelRunner._record_prefill_req_graphs(runner)
    assert "barrier" not in events
    # One token bucket and two request buckets (finishing only), two passes each.
    assert len(events) == 6
