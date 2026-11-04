# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``ATOM_MONO_MLA_PREP=1``: an MLA layer's attention seam as the tail of the
preceding layer's K2 (``kernels.stages.mla_prep``, P0), host side.

GPU-free: the flags, the bind-time refusals, which K2 builds carry the stage,
and that a folded layer takes the stage's outputs while replaying the
production seam's allocations. Whether the stage is right needs a GPU
(``ATOM_MONO_MLA_PREP_CHECK``).
"""

from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono.config import Dims
from atom.utils import envs

ROWS, HIDDEN = 4, 8


@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    from atom.models.kimi_k3.mono import runner

    return runner


def test_flags_default_off(monkeypatch):
    for name in ("ATOM_MONO_MLA_PREP", "ATOM_MONO_MLA_PREP_LAYERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ATOM_MONO_MLA_PREP_CHECK", raising=False)
    assert envs.ATOM_MONO_MLA_PREP == "0"
    assert envs.ATOM_MONO_MLA_PREP_LAYERS == ""
    assert envs.ATOM_MONO_MLA_PREP_CHECK is False


def _runner(runner_module, prep=None):
    r = object.__new__(runner_module.K3MonoDecodeRunner)
    r.rank = 0
    if prep is not None:
        r.mla_prep = {3: prep}
        r.mla_prep_prev = {2: 3}
    return r


def _prep(blocks=1):
    seam = SimpleNamespace(eps=1e-5, out_eps=2e-5, nw=None, qk=None, out_nw=None)
    return SimpleNamespace(mode=1, blocks=blocks, seam=seam)


def test_unbound_runner_builds_no_prep(runner_module):
    r = _runner(runner_module)
    assert r._prep_key(2, 1) == {}


def test_prep_key_only_on_the_preceding_layer_and_inside_the_width(
    runner_module,
):
    from atom.models.kimi_k3.mono.kernels.stages import mla_prep as k2pr

    r = _runner(runner_module, _prep(blocks=1))
    assert r._prep_key(2, 1) == {
        "mla_prep": 1,
        "prep_nblocks": 1,
        "prep_eps": 1e-5,
        "prep_out_eps": 2e-5,
    }
    assert r._prep_key(2, k2pr.MAX_ROWS)["mla_prep"] == 1
    assert r._prep_key(2, k2pr.MAX_ROWS + 1) == {}
    assert r._prep_key(3, 1) == {}


def test_sub_mode_3_is_refused(runner_module, monkeypatch):
    from atom.mono.runtime.consensus import MonoUnsupported

    r = _runner(runner_module)
    r.plans = []
    monkeypatch.setenv("ATOM_MONO_MLA_PREP", "3")
    with pytest.raises(MonoUnsupported, match="sub-modes 1"):
        r._bind_mla_prep()
    monkeypatch.setenv("ATOM_MONO_MLA_PREP", "0")
    r._bind_mla_prep()
    assert r.mla_prep == {} and r.mla_prep_prev == {}


def test_a_folded_layer_takes_the_stage_outputs(runner_module):
    """With K2(M-1)'s outputs pending, the layer skips its seam, takes ``x8`` /
    ``xs`` as ``normed`` and ``pref`` as its prefix, and replays the seam's
    allocations (its ``pref`` held for the forward loop)."""
    r = _runner(runner_module, _prep())
    layer = SimpleNamespace(
        self_attention_attn_res=lambda *a: pytest.fail("the seam ran")
    )
    r.inner = SimpleNamespace(layers={3: layer})
    r.mla_prep_check = False
    pend = {
        "pref": torch.zeros(ROWS, HIDDEN),
        "x8": torch.zeros(ROWS, HIDDEN),
        "xs": torch.ones(ROWS, 1),
    }
    r._prep_pending = {3: pend}
    plan = SimpleNamespace(layer_id=3)
    prefix = torch.ones(ROWS, HIDDEN)
    blocks = torch.zeros(ROWS, 4, HIDDEN)
    add = torch.full((ROWS, HIDDEN), 2.0)
    pend["attn"] = None
    got, normed, new_prefix, mirror, attn = r._prepped_seam(
        plan, prefix, blocks, add, None
    )
    assert attn is None
    assert got is layer and new_prefix is pend["pref"]
    assert normed[0] is pend["x8"] and normed[1] is pend["xs"]
    y, y_scale = mirror
    assert y.shape == (ROWS, HIDDEN) and y.dtype == torch.float8_e4m3fn
    assert y_scale.shape == (ROWS, 1)
    assert r._pref_hold_next.shape == prefix.shape
    assert r._prep_pending == {}


def test_an_unfolded_layer_runs_its_seam(runner_module):
    r = _runner(runner_module)
    calls = {}

    def seam(plan, prefix, blocks, add, add2):
        calls["seam"] = True
        return "layer", "normed", "prefix", 1, -1

    r._seam = seam
    plan = SimpleNamespace(layer_id=3)
    out = r._prepped_seam(plan, None, None, None, None)
    assert out == ("layer", "normed", "prefix", None, None) and calls["seam"]


def test_bind_refuses_before_reading_the_seam():
    from atom.models.kimi_k3.mono import weights
    from atom.mono.runtime.consensus import MonoUnsupported

    sa = SimpleNamespace(fuse_input_norm_quant=True)
    off = SimpleNamespace(enabled=False, out_norm=None)
    layer = SimpleNamespace(self_attn=sa, self_attention_attn_res=off)
    with pytest.raises(MonoUnsupported, match="attention residuals off"):
        weights.bind_mla_prep(layer, 3, Dims(8), 1)
    with pytest.raises(MonoUnsupported, match="sub-modes"):
        weights.bind_mla_prep(layer, 3, Dims(8), 4)
