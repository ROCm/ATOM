# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2 on Kimi-K3's full-attention (MLA) layers.

K2 is ``o_proj`` plus everything after it, none of which is KDA's. A
full-attention layer keeps its production attention up to ``o_proj``'s input
and hands the rest of the layer to K2. These tests pin the plain-Python parts:
which layers bind and what a step does with them. Whether K2 is right on those layers (per-layer error, step time,
GSM8K) needs a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono.config import Dims, layer_block
from atom.mono.runtime.consensus import MonoUnsupported

LAYERS = 93  # Kimi-K3
# the full-attention layers (config.json linear_attn_config.kda_layers is 1-based
# and lists the others)
MLA_LAYERS = [i for i in range(LAYERS) if i % 4 == 3] + [92]


def test_no_full_attention_layer_closes_a_block():
    """K2 restarts the prefix only on a block-write layer, and the MLA path does
    not bank a block, so every MLA layer must be a plain one with a block behind
    it."""
    assert len(MLA_LAYERS) == 24
    for i in MLA_LAYERS:
        before, write_idx = layer_block(i)
        assert write_idx == -1, i
        assert before >= 1, i


def test_k2_tags_stay_clear_of_k1_s():
    pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    from atom.models.kimi_k3.mono.runner import K2_TAG0

    k1_tags = {i + 1 for i in range(LAYERS)}  # ``Mailbox``: tag = layer + 1
    k2_tags = {K2_TAG0 + i for i in range(LAYERS)}
    assert not k1_tags & k2_tags
    assert max(k2_tags) < 2**31  # a mailbox word is an Int32


# ----------------------------------------------------------------- bind_model
def _layer(*, linear, ffn=True):
    layer = SimpleNamespace(is_linear_attn=linear)
    if ffn:
        layer.block_sparse_moe = object()
    return layer


def _model(n: int, full: tuple[int, ...] = (3,)):
    """Layer 0 dense, ``full`` full-attention, the rest KDA."""
    layers = [
        _layer(linear=True, ffn=False) if i == 0 else _layer(linear=i not in full)
        for i in range(n)
    ]
    return SimpleNamespace(layers=layers)


@pytest.fixture
def stub_binds(monkeypatch):
    from atom.models.kimi_k3.mono import weights

    monkeypatch.setattr(
        weights, "bind_kda", lambda layer, lid, dims: SimpleNamespace(layer_id=lid)
    )
    monkeypatch.setattr(
        weights,
        "bind_moe",
        lambda layer, lid, dims, rank: SimpleNamespace(
            layer_id=lid, write_idx=layer_block(lid)[1]
        ),
    )
    return weights


def test_full_attention_layers_get_k2_alone(stub_binds):
    plans = stub_binds.bind_model(_model(6), None, 0)
    assert [p.mono for p in plans] == [False, True, True, False, True, True]
    assert [p.ffn_only for p in plans] == [False, False, False, True, False, False]
    assert plans[3].kda is None and plans[3].moe.layer_id == 3
    # layer 0 is dense: it never gets K2
    assert (plans[0].kda, plans[0].moe) == (None, None)


def test_a_full_attention_layer_that_closes_a_block_keeps_production_k2(
    stub_binds, caplog
):
    """One layer's refusal must not take the KDA layers' mono down with it."""
    with caplog.at_level("WARNING", logger="atom"):
        plans = stub_binds.bind_model(_model(13, full=(3, 12)), None, 0)
    assert (plans[12].kda, plans[12].moe) == (None, None)
    assert plans[3].ffn_only and plans[1].mono
    assert "layer 12 keeps its production K2" in caplog.text
    assert "closes a block" in caplog.text


def test_the_block_check_runs_before_bind_moe_allocates(stub_binds, monkeypatch):
    seen = []
    monkeypatch.setattr(
        stub_binds,
        "bind_moe",
        lambda layer, lid, dims, rank: seen.append(lid)
        or SimpleNamespace(layer_id=lid, write_idx=layer_block(lid)[1]),
    )
    stub_binds.bind_model(_model(13, full=(3, 12)), None, 0)
    assert 12 not in seen and 3 in seen


def test_a_failing_bind_moe_on_a_full_attention_layer_falls_back_alone(
    stub_binds, monkeypatch
):
    real = stub_binds.bind_moe

    def flaky(layer, lid, dims, rank):
        if lid == 3:
            raise MonoUnsupported("o_proj is quantized")
        return real(layer, lid, dims, rank)

    monkeypatch.setattr(stub_binds, "bind_moe", flaky)
    plans = stub_binds.bind_model(_model(8, full=(3, 7)), None, 0)
    assert (plans[3].kda, plans[3].moe) == (None, None)
    assert plans[7].ffn_only and plans[1].mono and plans[4].mono


def test_no_kda_layer_still_refuses(stub_binds):
    model = SimpleNamespace(
        layers=[_layer(linear=True, ffn=False), _layer(linear=False)]
    )
    with pytest.raises(MonoUnsupported, match="no layer takes the kernels"):
        stub_binds.bind_model(model, None, 0)


# ------------------------------------------------------------------- runner
@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    from atom.models.kimi_k3.mono import runner

    return runner


ROWS, HIDDEN = 4, 8


def _k2_layer(calls, core):
    def seam(prefix, blocks, add, add2):
        calls["seam"] = (prefix, blocks, add, add2)
        return "normed", calls["new_prefix"]

    def forward_pre_o(positions, normed):
        calls["attn"] = (positions, normed)
        return core

    return SimpleNamespace(
        self_attention_attn_res=seam,
        self_attn=SimpleNamespace(forward_pre_o=forward_pre_o),
    )


def _ffn_only_runner(runner_module, calls, core):
    from atom.mono.runtime.aux_taps import AuxTaps

    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.dims = Dims(8)
    runner.aux = AuxTaps((), ())
    runner.inner = SimpleNamespace(layers={3: _k2_layer(calls, core)})
    runner._k1 = runner._fallback = lambda *a, **k: pytest.fail("not this layer's path")

    def k2(build, plan, core_in, prefix, blocks):
        calls["k2"] = (build, plan, core_in, prefix, blocks)
        return "k2 out"

    runner._k2 = k2
    return runner


def _step(runner_module, core):
    from atom.models.kimi_k3.mono.weights import LayerPlan

    calls = {"new_prefix": torch.zeros(ROWS, HIDDEN)}
    runner = _ffn_only_runner(runner_module, calls, core)
    plan = LayerPlan(3, None, object())
    prefix, blocks = torch.ones(ROWS, HIDDEN), torch.zeros(ROWS, 4, HIDDEN)
    add, add2 = torch.full((ROWS, HIDDEN), 2.0), None
    args = ("build", plan, "positions", prefix, blocks, add, add2, None)
    return runner, calls, args


def test_a_full_attention_layer_runs_its_attention_then_k2(runner_module):
    core = torch.zeros(ROWS, Dims(8).proj, dtype=torch.bfloat16)
    runner, calls, args = _step(runner_module, core)
    out = runner._layer(*args)

    _, plan, _, prefix, blocks, add, add2, _ = args
    seam_prefix, seam_blocks, seam_add, seam_add2 = calls["seam"]
    before, _ = layer_block(3)
    assert seam_prefix is prefix and seam_add is add and seam_add2 is add2
    assert seam_blocks.shape == (ROWS, before, HIDDEN)
    assert calls["attn"] == ("positions", "normed")
    # K2 gets the seam's prefix (it updates it in place), not the one it was handed
    assert calls["k2"][0] == "build" and calls["k2"][1] is plan
    assert calls["k2"][2] is core
    assert calls["k2"][3] is calls["new_prefix"]
    assert calls["k2"][4] is blocks
    assert out == (calls["new_prefix"], "k2 out", None)


def test_the_next_layer_is_handed_the_same_triple_a_kda_layer_returns(runner_module):
    core = torch.zeros(ROWS, Dims(8).proj, dtype=torch.bfloat16)
    runner, calls, args = _step(runner_module, core)
    prefix, ffn, shared = runner._layer(*args)
    assert prefix is calls["new_prefix"] and ffn == "k2 out" and shared is None


@pytest.mark.parametrize(
    "core",
    [
        torch.zeros(ROWS, Dims(8).proj, dtype=torch.float16),
        torch.zeros(ROWS, Dims(8).proj + 8, dtype=torch.bfloat16),
        torch.zeros(ROWS - 1, Dims(8).proj, dtype=torch.bfloat16),
        torch.zeros(Dims(8).proj, ROWS, dtype=torch.bfloat16).t(),
    ],
)
def test_a_core_k2_cannot_read_is_refused(runner_module, core):
    runner, _, args = _step(runner_module, core)
    with pytest.raises(MonoUnsupported, match="attention core"):
        runner._layer(*args)


def test_a_tile_padded_core_is_read_for_its_first_rows(runner_module):
    core = torch.zeros(ROWS + 4, Dims(8).proj, dtype=torch.bfloat16)
    runner, calls, args = _step(runner_module, core)
    runner._layer(*args)
    assert calls["k2"][2] is core


def _plan(layer_id, *, kda, moe):
    return SimpleNamespace(layer_id=layer_id, kda=kda, moe=moe, mono=kda and moe)


def _stub_moe(blocks=1):
    seam = SimpleNamespace(eps=1e-6, out_eps=1e-6)
    return SimpleNamespace(blocks=blocks, write_idx=-1, seam=seam, ln_eps=1e-6)


def _stub_kda(blocks=1):
    seam = SimpleNamespace(eps=1e-6, out_eps=1e-6)
    return SimpleNamespace(
        blocks=blocks,
        write_idx=-1,
        seam=seam,
        onorm_eps=1e-6,
        lower_bound=-5.0,
    )


def test_k2_builds_cover_full_attention_layers_and_k1_builds_do_not(runner_module):
    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.debug = False
    runner.state_len = 3
    runner.plans = [
        _plan(1, kda=_stub_kda(), moe=_stub_moe()),
        _plan(3, kda=None, moe=_stub_moe()),
        _plan(5, kda=None, moe=None),
    ]
    assert sorted(runner._k2_keys(8)) == [1, 3]
    assert sorted(runner._k1_keys(8, 8)) == [1]
    # same blocks, same eps: the full-attention layer reuses the KDA layer's build
    keys = runner._k2_keys(8)
    assert keys[1] == keys[3]


# ----------------------------------------------------------- forward_pre_o
def test_forward_pre_o_is_forward_without_o_proj():
    try:
        from atom.models.kimi_k3.model import KimiFullAttention
    except ImportError as error:  # the model module imports the GPU stack
        pytest.skip(f"atom.models.kimi_k3.model needs the GPU stack: {error}")

    core = torch.ones(2, 4)
    fake = SimpleNamespace(_gated_core=lambda positions, hidden: (core, None))
    assert KimiFullAttention.forward_pre_o(fake, None, None) is core

    quantized = SimpleNamespace(_gated_core=lambda positions, hidden: (core, core))
    with pytest.raises(AssertionError, match="quantized"):
        KimiFullAttention.forward_pre_o(quantized, None, None)
