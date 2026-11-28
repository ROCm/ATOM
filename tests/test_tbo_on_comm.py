# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`tbo_on_comm`: by default a collective a TBO ubatch runs on the comm stream
holds what it returns for the compute stream that reads it (why:
test_tbo_mark_used_on_compute.py); the DP gather / scatter opt out, and the TP
all-reduce, which reduces into compute-stream memory, has nothing to hold.
CPU only: the TBO hooks and `record_stream` are monkeypatched."""

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
        ub, "_tbo_yield_and_switch_from_compute_to_comm", lambda: log.append("comm")
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
    # Nested: rows returned with their counts.
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
    compressors = [SimpleNamespace(project=lambda x, k=k: x + k) for k in (1, 2)]
    x = torch.zeros(4, 3)
    out = v4.DeepseekV4Attention._pcp_gather_compressor_inputs(x, compressors)
    assert [id(t) for t in out] == [id(t) for t in gathered]
    assert tbo == ["comm", "compute", *((id(t), "compute stream") for t in out)]


def test_without_hold_nothing_is_held(tbo):
    a = torch.zeros(2)
    assert ub.tbo_on_comm(lambda x: x, a, hold=False) is a
    assert tbo == ["comm", "compute"]


@pytest.fixture
def tp_all_reduce(tbo, monkeypatch):
    """`tbo_all_reduce` in an overlapping ubatch, its communicator's calls and
    every `torch.empty_like` logged into `tbo`."""
    mdo = pytest.importorskip("atom.model_ops.module_dispatch_ops")
    monkeypatch.setenv("ATOM_TBO_TP_AR_MODE", "overlap")
    monkeypatch.setattr(ub, "tbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(ub, "tbo_get_comm_stream", lambda: "comm stream")
    empty_like = torch.empty_like

    def logged_empty_like(t, **kwargs):
        out = empty_like(t, **kwargs)
        tbo.append(("alloc", id(out)))
        return out

    def all_reduce(x, out_tensor, stream):
        tbo.append(("all_reduce", id(out_tensor), stream))
        out_tensor.copy_(4 * x)

    comm = SimpleNamespace(all_reduce=all_reduce, disabled=False)
    monkeypatch.setattr(ub, "tbo_get_ubatch_tp_comm", lambda i: comm)
    monkeypatch.setattr(torch, "empty_like", logged_empty_like)
    return SimpleNamespace(op=mdo.tbo_all_reduce, comm=comm)


def test_tp_all_reduce_reduces_into_compute_stream_memory(tbo, tp_all_reduce):
    """Allocated on the comm stream and not held, the output could be given to
    the partner's next hand-off while this ubatch still reads it; held, it
    faulted V4-Pro TBO (HSA_STATUS_ERROR_OUT_OF_RESOURCES). Allocated on the
    compute stream before the hand-off instead, it needs no hold."""
    out = tp_all_reduce.op(torch.ones(3))
    assert out.tolist() == [4.0, 4.0, 4.0]
    assert tbo == [
        ("alloc", id(out)),
        "comm",
        ("all_reduce", id(out), "comm stream"),
        "compute",
    ]


def test_tp_all_reduce_refuses_a_disabled_communicator(tbo, tp_all_reduce):
    # A disabled pynccl communicator returns without writing `out_tensor`.
    tp_all_reduce.comm.disabled = True
    with pytest.raises(RuntimeError, match="ATOM_TBO_TP_AR_MODE=inline"):
        tp_all_reduce.op(torch.ones(3))
    assert tbo == []


def test_only_tp_and_dp_collectives_opt_out_of_the_hold():
    """The PCP gathers (MoE split pass, V4 compressor inputs, extend K) keep
    the hold: without it the PCP+TBO extend K was overwritten."""
    import pathlib

    import atom

    root = pathlib.Path(atom.__file__).parent
    opt_outs = {
        str(p.relative_to(root)): p.read_text().count("hold=False")
        for p in root.rglob("*.py")
        if "hold=False" in p.read_text()
    }
    assert opt_outs == {
        "model_ops/module_dispatch_ops.py": 1,  # TP all-reduce: nothing to hold
        "model_ops/moe.py": 3,  # DP gather, reduce_scatterv, unpadded RS
        "utils/tbo/ubatching.py": 1,  # the docstring
    }, opt_outs


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
    pairs = [(lambda projected, **kw: read.append(projected), s) for s in streams]
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


def test_comm_phase_primitives_are_private():
    """A comm phase written out by hand -- yield, collective, switch back --
    skips the hold: the PCP+TBO extend K was overwritten that way. So the
    switch into a comm phase and the hold are not exported, and only MoRI
    reaches the switch: its dispatch / combine return views of its own
    buffers, which the caching allocator never hands out."""
    import pathlib

    import atom
    import atom.utils.tbo as tbo_pkg

    for name in ("yield_and_switch_from_compute_to_comm", "mark_used_on_compute"):
        assert not hasattr(tbo_pkg, f"tbo_{name}"), name
        assert not hasattr(tbo_pkg, f"_tbo_{name}"), name
        assert not hasattr(ub, f"tbo_{name}"), name
    root = pathlib.Path(atom.__file__).parent
    users = {
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if "_tbo_yield_and_switch_from_compute_to_comm" in p.read_text()
    }
    assert users == {
        "utils/tbo/ubatching.py",
        "model_ops/fused_moe/mori_prepare_finalize.py",
    }, users


def test_without_pcp_the_projection_runs_on_the_side_stream(monkeypatch):
    """The compressor GEMM is part of what the side streams overlap with the
    Q/KV projections."""
    v4 = pytest.importorskip("atom.models.deepseek_v4")
    log = []

    class _Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, s):
            pass

        def __enter__(self):
            log.append(("enter", self.name))

        def __exit__(self, *exc):
            log.append(("exit", self.name))

    class _Compressor:
        def __init__(self, name):
            self.name = name

        def project(self, x):
            log.append(("project", self.name))
            return x + 1

        def __call__(self, projected, **kw):
            log.append(("compress", self.name))

    monkeypatch.setattr(torch.cuda, "stream", lambda s: s)
    monkeypatch.delenv("ATOM_V4_BYPASS_ATTN", raising=False)
    pairs = [(_Compressor(n), _Stream(n)) for n in ("main", "indexer")]
    attn = SimpleNamespace(
        compress_ratio=4,
        _compressors_async=lambda: True,
        _compressor_streams=lambda: pairs,
        _pcp_gather_compressor_inputs=lambda x, cs: pytest.fail(),
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
    monkeypatch.setattr(v4, "_pcp_active", lambda: False)
    v4.DeepseekV4Attention._launch_compressors(attn, torch.zeros(2))
    assert log == [
        (step, n)
        for n in ("main", "indexer")
        for step in ("enter", "project", "compress", "exit")
    ]
