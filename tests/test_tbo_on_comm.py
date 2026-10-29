# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tbo_on_comm`: a collective a TBO ubatch runs on the comm stream must hold
what it returns for the compute stream that reads it (why:
test_tbo_mark_used_on_compute.py). CPU only: the TBO hooks and
`record_stream` are monkeypatched."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ub = pytest.importorskip("atom.utils.tbo.ubatching")


@pytest.fixture
def tbo(monkeypatch):
    """A TBO ubatch whose stream switches and `record_stream` calls are logged."""
    log = []
    monkeypatch.setattr(ub, "tbo_active", lambda: True)
    monkeypatch.setattr(
        ub, "tbo_yield_and_switch_from_compute_to_comm", lambda: log.append("comm")
    )
    monkeypatch.setattr(ub, "tbo_switch_to_compute_sync", lambda: log.append("compute"))
    ctx = SimpleNamespace(compute_stream="compute stream")
    monkeypatch.setattr(ub, "_get_current_tbo_context", lambda: ctx)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda t, s: log.append((id(t), s))
    )
    return log


def test_outside_tbo_it_only_calls(monkeypatch):
    monkeypatch.setattr(ub, "tbo_active", lambda: False)
    t = torch.zeros(2)
    assert ub.tbo_on_comm(lambda x: x, t) is t


def test_every_tensor_returned_is_held_for_the_compute_stream(tbo):
    a, b = torch.zeros(2), torch.ones(3)

    def collective(x):
        tbo.append("collective")
        return a, None, b, x

    out = ub.tbo_on_comm(collective, 7)
    assert out[0] is a and out[2] is b and out[3] == 7
    held = [(id(a), "compute stream"), (id(b), "compute stream")]
    assert tbo == ["comm", "collective", "compute", *held]
    tbo.clear()
    assert ub.tbo_on_comm(lambda: a) is a
    assert tbo == ["comm", "compute", (id(a), "compute stream")]
    # FusedMoE's DP gather returns its rows with their counts.
    tbo.clear()
    ub.tbo_on_comm(lambda: ((a, b), 3, [3, 5]))
    assert tbo == ["comm", "compute", *held]


def test_both_compressor_projections_are_held(tbo, monkeypatch):
    v4 = pytest.importorskip("atom.models.deepseek_v4")
    monkeypatch.setattr(v4, "get_pcp_world_size", lambda: 8)
    gathered = []

    def gather(ts, ws):
        gathered.extend(t * ws for t in ts)
        return list(gathered)

    monkeypatch.setattr(v4, "pcp_allgather_rerange_many", gather)
    compressors = [SimpleNamespace(wkv_gate=lambda x, k=k: x + k) for k in (1, 2)]
    x = torch.zeros(4, 3)
    out = v4.DeepseekV4Attention._pcp_gather_compressor_inputs(x, compressors)
    assert [id(t) for t in out] == [id(t) for t in gathered]
    assert tbo == ["comm", "compute", *((id(t), "compute stream") for t in out)]


def test_tp_all_reduce_output_is_held(tbo, monkeypatch):
    """Pure TP+TBO's overlapped all-reduce allocates its output on the comm
    stream, like the PCP gathers."""
    mdo = pytest.importorskip("atom.model_ops.module_dispatch_ops")
    monkeypatch.setenv("ATOM_TBO_TP_AR_MODE", "overlap")
    monkeypatch.setattr(ub, "tbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(ub, "tbo_get_comm_stream", lambda: "comm stream")
    reduced = torch.ones(3)

    def all_reduce(x, stream):
        tbo.append(("all_reduce", stream))
        return reduced

    monkeypatch.setattr(
        ub, "tbo_get_ubatch_tp_comm", lambda i: SimpleNamespace(all_reduce=all_reduce)
    )
    assert mdo.tbo_all_reduce(torch.zeros(3)) is reduced
    held = (id(reduced), "compute stream")
    assert tbo == ["comm", ("all_reduce", "comm stream"), "compute", held]


def test_side_stream_compressors_hold_their_gathered_inputs(monkeypatch):
    """Under a graph the compressors run on side streams, while a PCP prefill
    gathers their inputs on the main stream, which may reuse that memory once
    `_launch_compressors` returns."""
    v4 = pytest.importorskip("atom.models.deepseek_v4")
    log = []
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda t, s: log.append((id(t), s))
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.delenv("ATOM_V4_BYPASS_ATTN", raising=False)
    combined = [torch.zeros(2), torch.ones(2)]
    streams = [SimpleNamespace(wait_stream=lambda s: None) for _ in combined]
    read = []
    pairs = [(lambda x, combined, **kw: read.append(combined), s) for s in streams]
    attn = SimpleNamespace(
        compress_ratio=4,
        _compressors_async=lambda: True,
        _compressor_streams=lambda: pairs,
        _pcp_gather_compressor_inputs=lambda x, cs: combined,
    )
    fc = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(
            compress_plans={4: None},
            state_slot_in=None,
            state_slot_out=None,
            block_tables=None,
        ),
        main_stream=None,
    )
    monkeypatch.setattr(v4, "get_forward_context", lambda: fc)
    monkeypatch.setattr(v4, "_pcp_active", lambda: True)
    v4.DeepseekV4Attention._launch_compressors(attn, torch.zeros(2))
    assert read == combined
    assert log == [(id(c), s) for c, s in zip(combined, streams)]


def test_tbo_comm_phases_go_through_tbo_on_comm():
    """A comm phase written out by hand -- yield, collective, switch back --
    skips the hold: the PCP+TBO extend K was overwritten that way. Mori's
    dispatch / combine return views of its own buffers, which the caching
    allocator never hands out, so its hand-written phases stay."""
    v4 = pytest.importorskip("atom.models.deepseek_v4")
    moe = pytest.importorskip("atom.model_ops.moe")
    mdo = pytest.importorskip("atom.model_ops.module_dispatch_ops")
    for src in (
        inspect.getsource(v4),
        inspect.getsource(moe.FusedMoE._run_experts),
        inspect.getsource(moe.FusedMoE.forward_impl_graph),
        inspect.getsource(mdo.tbo_all_reduce),
    ):
        assert "tbo_yield_and_switch_from_compute_to_comm(" not in src
