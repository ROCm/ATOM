# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contracts of ATOM_KIMI_MONO_MLA_ATTN_RES: the switch, and which
mix the gfx1250 runner's attention-side seam takes for a full-attention layer
(production's unless the switch is on and the layer runs the MLA front) and
for a KDA layer (the mono mix, as before)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="the K3 runner imports aiter")


def test_the_switch_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_ATTN_RES", raising=False)
    assert envs.ATOM_KIMI_MONO_MLA_ATTN_RES is False
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_ATTN_RES", "1")
    assert envs.ATOM_KIMI_MONO_MLA_ATTN_RES is True


@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="the gfx1250 runner imports FlyDSL")
    from atom.models.kimi_k3.mono import runner_gfx1250

    return runner_gfx1250


LAYER = 3  # mixes one block, closes none


def _seam(runner_module, monkeypatch, fronts=(LAYER,), bf16=False, bound=True):
    """The runner's seam of layer LAYER on recording mixes: its return, the
    calls and the layer."""
    calls = []

    def site(prefix, blocks, add, add2):
        calls.append(("site", prefix, tuple(blocks.shape), add, add2))
        return "site_normed", "site_prefix"

    def mono(of, prefix, blocks, add, add2):
        calls.append(("mono", of, prefix, tuple(blocks.shape), add, add2))
        return "mono_normed", "mono_prefix"

    monkeypatch.setattr(runner_module, "_mono_attn_res", mono)
    runner = object.__new__(runner_module.K3Gfx1250MonoDecodeRunner)
    layer = SimpleNamespace(self_attention_attn_res=site)
    runner.inner = SimpleNamespace(layers={LAYER: layer})
    if bound:
        runner.mla_fronts = dict.fromkeys(fronts)
    out = runner._seam(
        SimpleNamespace(layer_id=LAYER),
        "prefix",
        torch.zeros(2, 3, 4),
        "add",
        None,
        bf16,
    )
    return out, calls, layer


def test_switched_off_the_layer_mixes_on_productions_site(runner_module, monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_MLA_ATTN_RES", raising=False)
    out, calls, layer = _seam(runner_module, monkeypatch)
    assert calls == [("site", "prefix", (2, 1, 4), "add", None)]
    assert out == (layer, "site_normed", "site_prefix", 1, -1)


def test_switched_on_a_front_layer_mixes_on_the_mono_kernel(runner_module, monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_ATTN_RES", "1")
    out, calls, layer = _seam(runner_module, monkeypatch)
    assert calls == [
        ("mono", layer.self_attention_attn_res, "prefix", (2, 1, 4), "add", None)
    ]
    assert out == (layer, "mono_normed", "mono_prefix", 1, -1)


@pytest.mark.parametrize("bound", [True, False])
def test_switched_on_a_layer_without_a_front_keeps_production(
    runner_module, monkeypatch, bound
):
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_ATTN_RES", "1")
    out, calls, layer = _seam(runner_module, monkeypatch, fronts=(7,), bound=bound)
    assert [name for name, *_ in calls] == ["site"]
    assert out == (layer, "site_normed", "site_prefix", 1, -1)


@pytest.mark.parametrize("switch", ["0", "1"])
def test_a_kda_layers_bf16_seam_is_the_mono_mix_either_way(
    runner_module, monkeypatch, switch
):
    monkeypatch.setenv("ATOM_KIMI_MONO_MLA_ATTN_RES", switch)
    out, calls, layer = _seam(runner_module, monkeypatch, fronts=(), bf16=True)
    assert [name for name, *_ in calls] == ["mono"]
    assert out == (layer, "mono_normed", "mono_prefix", 1, -1)
