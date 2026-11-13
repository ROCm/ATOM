# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hybrid / wide step's loop: the (ar, res) hand-off, one begin_step a step per half,
the aux taps, the pipeline pick (fake halves on CPU tensors; the kernels are GPU tests).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.models.minimax_m3.mono import hybrid
from atom.models.minimax_m3.mono.config import HIDDEN

N = 3
N_LAYERS = 3 + 4  # the dense prefix + four MoE layers


class _Front:
    needs_rows = False

    def __init__(self):
        self.begun = 0
        self.seen = []

    def prepare(self, n):
        return True

    def begin_step(self):
        self.begun += 1

    def __call__(self, i, lw, layer, positions, ar, res, want_entering, **_):
        self.seen.append((i, lw.layer_id, bool(want_entering)))
        entering = ar + res
        return entering + 1, entering * 2, entering  # res_mid, x_moe, entering


class _Back:
    def __init__(self):
        self.begun = 0

    def prepare(self, n):
        return True

    def begin_step(self):
        self.begun += 1

    def __call__(self, i, lw, layer, x_moe, fwd, n, ar_out):
        return x_moe + 3


def _runner(aux_taps=()):
    layers = [lambda p, h, r, capture_aux=False: (h + 1, h if r is None else r + 1)] * 3
    layers += [object()] * 4
    model = SimpleNamespace(
        layers=layers,
        aux_hidden_state_layers=(),
        get_input_embeddings=lambda ids: torch.ones(ids.numel(), HIDDEN),
        norm=lambda ar, res: (ar + res, ar + res + 0.5),
    )
    written = {}
    aux = SimpleNamespace(layer_ids=list(aux_taps)) if aux_taps else None
    return (
        SimpleNamespace(
            model=model,
            sparse=[SimpleNamespace(layer_id=3 + i) for i in range(4)],
            dev=torch.device("cpu"),
            parts=None,
            aux=aux,
            rank=0,
            _write_aux=lambda lid, v: written.setdefault(lid, v.clone()),
        ),
        written,
    )


def _expected(n_moe=4):
    """(the final norm's output, its residual, the stream entering each MoE layer): by hand."""
    ar, res = 4.0, 3.0  # the fake prefix: three layers of (h + 1, r + 1) from h = 1
    entering = []
    for _ in range(n_moe):
        entering.append(ar + res)
        res, x_moe = entering[-1] + 1, entering[-1] * 2
        ar = x_moe + 3
    return ar + res, ar + res + 0.5, entering


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(hybrid, "tensor_model_parallel_all_reduce", lambda t: t)
    monkeypatch.setattr(hybrid, "get_forward_context", lambda: SimpleNamespace())
    front, back = _Front(), _Back()
    monkeypatch.setattr(
        hybrid, "_half", lambda r, p, side, kind: front if side == "front" else back
    )
    return front, back


def test_pair_handoff_and_one_begin_step(pipe):
    front, back = pipe
    runner, _ = _runner()
    dec = hybrid.HybridDecode(runner, "hybrid")
    out = dec.forward(
        torch.zeros(N, dtype=torch.int64), torch.zeros(N, dtype=torch.int64)
    )
    assert torch.equal(out, torch.full((N, HIDDEN), _expected()[0]))
    assert front.begun == back.begun == 1
    assert [s[:2] for s in front.seen] == [(i, 3 + i) for i in range(4)]
    assert not any(
        s[2] for s in front.seen
    )  # nothing taps: no entering stream asked for


def test_dspark_taps_write_the_stream_entering_each_layer(pipe):
    front, _ = pipe
    runner, written = _runner(aux_taps=(2, 5, 6))
    hybrid.HybridDecode(runner, "wide").forward(
        torch.zeros(N, dtype=torch.int64), torch.zeros(N, dtype=torch.int64)
    )
    _, final_residual, entering = _expected()
    # tap L = the stream entering layer L + 1: layers 3 and 6 are asked for theirs
    assert [s[2] for s in front.seen] == [True, False, False, True]
    assert sorted(written) == [2, 5, 6]
    assert torch.all(written[2] == entering[0]) and torch.all(written[5] == entering[3])
    assert torch.all(
        written[6] == final_residual
    )  # the last layer's: the final norm's residual


def test_pipelines_are_front_back_pairs():
    assert set(hybrid.PIPELINES) == {"hybrid", "wide"}
    assert hybrid.PIPELINES["wide"] == ("full", "fused")


def test_mono_front_feeds_the_kernel_its_own_buffers(monkeypatch):
    """MonoFront: layer i's ``ar`` lands in ars[i % 2], hidden alternates by parity,
    x_moe is shared, and the kernel gets the step's ``rows``."""
    import sys
    import types

    calls = []

    class Front64:
        def __init__(self, runner, ars):
            self.ars = ars

        def run_layer(self, i, lw, fwd, positions, res, rows, hidden, x_moe):
            calls.append(
                (i, self.ars[i % 2][:1].clone(), res.data_ptr(), hidden.data_ptr())
            )
            hidden.copy_(self.ars[i % 2][: hidden.shape[0]] + res)
            x_moe.fill_(i + 1.0)

    mod = types.ModuleType("atom.models.minimax_m3.mono.front64")
    mod.Front64 = Front64
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    runner, _ = _runner()
    runner.sparse = [SimpleNamespace(layer_id=3 + i, index_topk=True) for i in range(3)]
    parts = hybrid._Parts(runner)
    front = hybrid.MonoFront(runner, parts)
    ar, res = torch.full((N, HIDDEN), 1.0), torch.full((N, HIDDEN), 2.0)
    out = []
    for i in range(3):
        res, x_moe, entering = front(
            i,
            runner.sparse[i],
            None,
            None,
            ar,
            res,
            i == 1,
            fwd=None,
            n=N,
            rows=("bt", "sl"),
        )
        out.append((res.clone(), x_moe.data_ptr(), entering))
        ar = parts.ars[(i + 1) % 2][:N]  # the back half writes the next layer's ar here
        ar.fill_(10.0 * (i + 1))
    assert torch.all(out[0][0] == 3.0) and out[0][2] is None  # ar 1 + res 2
    assert torch.all(out[1][0] == 10.0 + 3.0) and torch.equal(
        out[1][2], torch.full((N, HIDDEN), 13.0)
    )
    assert out[0][1] == out[1][1] == parts.x_moe.data_ptr()  # one x_moe buffer
    assert (
        calls[0][3] != calls[1][3] and calls[0][3] == calls[2][3]
    )  # hidden alternates
    assert calls[1][2] == calls[0][3]  # layer 1 reads layer 0's hidden as its res


def test_mono_front_is_refused_when_a_layer_reuses_a_selection(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("atom.models.minimax_m3.mono.front64")
    mod.Front64 = lambda runner, ars: None
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    runner, _ = _runner()
    runner.sparse = [
        SimpleNamespace(layer_id=3, index_topk=True),
        SimpleNamespace(layer_id=4, index_topk=False),
    ]
    with pytest.raises(RuntimeError):
        hybrid.MonoFront(runner, hybrid._Parts(runner))


def test_full_layer_writes_the_next_layers_ar_in_place(monkeypatch):
    """MonoFull: one launch a layer; the kernel's ar_out is the NEXT layer's input buffer, so the
    fused back half has nothing to copy."""
    import sys
    import types

    launched = []

    class Front64:
        def __init__(self, runner, ars):
            self.runner, self.ars = runner, ars

        def run_layer(self, *a):
            raise AssertionError("the front-only launch")

    mod = types.ModuleType("atom.models.minimax_m3.mono.front64")
    mod.Front64 = Front64
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(
        hybrid,
        "launch_full_layer",
        lambda front, i, lw, fwd, pos, res, rows, hidden, out: launched.append(
            (i, hidden, out)
        ),
    )
    runner, _ = _runner()
    runner.shared_mxfp8 = True
    runner.sparse = [SimpleNamespace(layer_id=3 + i, index_topk=True) for i in range(2)]
    parts = hybrid._Parts(runner)
    full = hybrid.MonoFull(runner, parts)
    ar, res = torch.ones(N, HIDDEN), torch.ones(N, HIDDEN)
    hidden, x_moe, entering = full(
        0, runner.sparse[0], None, None, ar, res, False, fwd=None, n=N, rows=None
    )
    assert x_moe is None and entering is None
    full.front.run_layer(
        0, runner.sparse[0], None, None, res, None, hidden, parts.ars[1][:N]
    )
    assert launched[0][1].data_ptr() == parts.hiddens[0].data_ptr()
    assert launched[0][2].data_ptr() == parts.ars[1].data_ptr()
    assert (
        hybrid.FusedBack()(0, None, None, None, None, N, parts.ars[1][:N]).data_ptr()
        == parts.ars[1].data_ptr()
    )
    runner.shared_mxfp8 = False
    with pytest.raises(RuntimeError):
        hybrid.MonoFull(runner, parts)


def test_the_kernel_symbol_must_spell_the_measured_front_options(monkeypatch):
    """A front / whole-layer kernel built without the per-size options (n_split 4 above 32 rows, the group
    pipeline, 13 copy rounds) is another kernel: ``MonoFront.prepare`` refuses it instead of timing it.
    """
    import sys
    import types

    r = SimpleNamespace(attn_mxfp8=True, nvfp4=True)
    assert hybrid.symbol_tokens(r, 64, True) == {"s64", "mb1", "pp1", "cb13", "ns4"}
    assert hybrid.symbol_tokens(r, 32, False) == {
        "s32",
        "fr1",
        "pp1",
        "cb13",
    }  # ns8 is the default: not spelled
    assert hybrid.symbol_tokens(
        SimpleNamespace(attn_mxfp8=False, nvfp4=False), 64, False
    ) == {"s64", "fr1", "cb13"}

    class Front64:
        def __init__(self, runner, ars):
            self.runner, self.builds = runner, {64: object()}

        def prepare(self, n):
            return True

    mod = types.ModuleType("atom.models.minimax_m3.mono.front64")
    mod.Front64 = Front64
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    runner, _ = _runner()
    runner.attn_mxfp8, runner.nvfp4 = True, True
    runner.sparse = [SimpleNamespace(layer_id=3, index_topk=True)]
    front = hybrid.MonoFront(runner, hybrid._Parts(runner))
    good = (
        "atom::minimax_m3_mono_layer_tp4_s64_ib0_lb1_ih1_io0_tl0_it1_fr1_ns4_pp1_cb13"
    )
    old = "atom::minimax_m3_mono_layer_tp4_s64_ib0_lb1_ih1_io0_tl0_it1_fr1"  # front64 before its options
    monkeypatch.setattr(hybrid, "kernel_names", lambda launcher: [good])
    assert front.prepare(64)
    monkeypatch.setattr(hybrid, "kernel_names", lambda launcher: [old])
    assert not front.prepare(64)
