# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DSpark aux taps under the hi runtime's sharded residual: ``HiLayers._key`` gives the tap layers, and only them, a
program of their own (``window_layer_blocks(full_r_a=True)``: R_A whole) at the widths where the hi runtime's
``layer.residual_layout`` says "sharded"; replicated, every key is the plain one. The keys are pure Python
(``hi_compose``); the hi layer needs FlyDSL (its test skips without it)."""

import inspect
import sys
import types

import pytest
from aiter_stub import atom_without_aiter

from atom.models.deepseek_v41.mono import hi_compose
from atom.mono.runtime.consensus import MonoUnsupported

AUX = (37, 38, 39)
WIDTHS = (1, 2, 6, 8, 12, 16, 24)


def _hi_layer_module(layout):
    """A stand-in for the hi runtime's ``deepseek_v41.layer``: ``residual_layout`` as ``layout(rows)`` says."""

    def residual_layout(tp, split=False, knobs=None, rows=None):
        return layout(rows)

    return types.SimpleNamespace(residual_layout=residual_layout)


def _layers(L):
    m = object.__new__(hi_compose.HiPrograms)  # (only what ``_key`` reads)
    m.L, m.tp, m.chain, m.aux_ids, m.band_of = L, 4, False, AUX, {}
    m.plan = {
        i: types.SimpleNamespace(program_key=("consumer", 4 if i % 2 else 128))
        for i in range(40)
    }
    return m


def _keys(m):
    return {(i, s): m._key(i, s) for i in m.plan for s in WIDTHS}


def _base(m):
    return {
        (i, s): (m.plan[i].program_key, s, False, False) for i in m.plan for s in WIDTHS
    }


def test_replicated_keys_are_plain():
    m = _layers(_hi_layer_module(lambda rows: "replicated"))
    assert _keys(m) == _base(m)


@pytest.mark.parametrize(
    "name, layout",
    [
        ("sharded", lambda rows: "sharded"),
        (
            "auto",
            lambda rows: (
                "sharded" if rows is not None and 6 <= rows <= 8 else "replicated"
            ),
        ),
    ],
)
def test_keys_differ_only_for_the_taps(name, layout):
    m = _layers(_hi_layer_module(layout))
    got, base = _keys(m), _base(m)
    differ = {k for k in got if got[k] != base[k]}
    assert differ == {(i, s) for i in AUX for s in WIDTHS if layout(s) == "sharded"}
    assert differ  # (some width sharded)
    for i, s in differ:
        assert got[i, s] == base[i, s] + ("full_r_a",)
        assert m.full_r_a(i, s)
    # a tap layer's program is not shared with a non-tap layer of the same kind; non-tap layers share as before
    assert got[37, 6] != got[35, 6] and got[35, 6] == got[33, 6]


def test_program_takes_the_tap():
    assert "full_r_a" in inspect.signature(hi_compose.HiPrograms._program).parameters


def test_hi_layer_takes_full_r_a():
    """With FlyDSL installed: the hi runtime's ``window_layer_blocks`` takes ``full_r_a`` and ``residual_layout`` leaves 1
    row replicated under "auto"."""
    pytest.importorskip("flydsl")
    pytest.importorskip(
        "aiter"
    )  # (the hi device code re-exports atom.mono.device's primitives)
    from atom.models.deepseek_v41.mono.hi.v41 import layer as L

    assert "full_r_a" in inspect.signature(L.window_layer_blocks).parameters
    # (RESIDUAL "auto", the schedules' default; the same at both served TP sizes)
    for tp in (2, 4):
        assert L.residual_layout(tp, rows=6) == "sharded"
        assert L.residual_layout(tp, rows=1) == "replicated"
    m = _layers(L)
    assert m._key(37, 6)[-1] == "full_r_a"
    assert m._key(36, 6)[-1] is False and m._key(37, 1)[-1] is False


def test_first_served_step_writes_the_taps(monkeypatch):
    """The end-step's taps on the first served step: ATOM's aux buffers are bound before the end-step launch takes
    its tap list (``_taps``), not later by ``CTX`` (``_ctx_taps``), which reads them right after it. Bound only by
    ``CTX``, the first served step (and a graph captured on it) wrote no taps and ``CTX`` read stale ones. The bind
    takes the extract the hi kernels compute (``DeepseekV41DSpark._target_layer_input``) and refuses any other.
    """
    import torch

    with (
        atom_without_aiter() as load
    ):  # (Triton stubbed with AITER where it is not installed)
        hook = load("atom.models.deepseek_v41.mono.hi_runner")
    # (the draft module needs AITER and FlyDSL: only the extract it names)
    draft = types.SimpleNamespace(_target_layer_input=lambda inputs, block: None)
    monkeypatch.setitem(
        sys.modules,
        "atom.models.deepseek_v41.dspark",
        types.SimpleNamespace(DeepseekV41DSpark=draft),
    )
    m = object.__new__(hook.HiLayers)
    m.aux, m.D = None, types.SimpleNamespace(TAPS=len(AUX))
    m.h_in0 = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    m.r_a = {i: torch.zeros(1, 4, 8, dtype=torch.bfloat16) for i in AUX}
    bufs = [torch.zeros(4, 8, dtype=torch.bfloat16) for _ in AUX]
    cfg = types.SimpleNamespace(dspark_target_layer_ids=list(AUX))
    m.cx_drafter = types.SimpleNamespace(_aux_buffers=bufs, aux_extract=draft._target_layer_input,
                                         speculative_config=types.SimpleNamespace(draft_model_hf_config=cfg))  # fmt: skip
    taps = m._taps()
    assert [b for _, b in taps] == bufs and [r for r, _ in taps] == [
        m.r_a[i] for i in AUX
    ]
    m.aux, m.cx_drafter.aux_extract = None, lambda inputs, block: None
    with pytest.raises(MonoUnsupported, match="aux extract"):
        m._taps()
