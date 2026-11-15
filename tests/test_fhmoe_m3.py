# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 FHMoE (ATOM_M3_FHMOE): the refusal table, the activation pick, the
dummy routed row, the row limit. CPU only; the kernel is the aiter op test's."""

import importlib
from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter")
fh = importlib.import_module("atom.model_ops.fused_moe.fhmoe_m3")

OK = {
    "gfx": "gfx950",
    "tp_size": 4,
    "hidden": 6144,
    "inter": 3072,
    "experts": 128,
    "top_k": 4,
    "n_shared": 1,
    "activation": aiter.ActivationType.Swiglu,
    "shared_ok": True,
    "in_use": {"expert parallel": False, "data parallel": False},
    "gu_interleave": False,
    "has_bias": False,
    "lossy_shared_flag": False,
    "triton": False,
}


def test_the_tuned_shape_is_served():
    assert fh.refusal(**OK) is None


@pytest.mark.parametrize(
    ("change", "word"),
    [
        ({"gfx": "gfx942"}, "gfx950"),
        ({"tp_size": 8, "inter": 3072}, "tuned for"),
        ({"hidden": 4096}, "tuned for"),
        ({"experts": 256}, "tuned for"),
        ({"top_k": 8}, "tuned for"),
        ({"n_shared": 0}, "tuned for"),
        ({"activation": aiter.ActivationType.Silu}, "Swiglu"),
        ({"shared_ok": False}, "MXFP8"),
        ({"in_use": {"expert parallel": True}}, "expert parallel"),
        ({"in_use": {"data parallel": True}}, "data parallel"),
        ({"gu_interleave": True}, "interleaved"),
        ({"has_bias": True}, "biases"),
        ({"lossy_shared_flag": True}, "lossy"),
        ({"triton": True}, "Triton"),
    ],
)
def test_every_unsupported_config_is_refused_with_its_reason(change, word):
    assert word in fh.refusal(**{**OK, **change})


def test_the_refusal_is_logged_once(caplog):
    fh._refused_logged = False
    with caplog.at_level("WARNING", logger="atom"):
        fh.log_refusal("first")
        fh.log_refusal("second")
    assert [r.getMessage() for r in caplog.records if "refused" in r.getMessage()] == [
        "ATOM_M3_FHMOE=1 refused, keeping the unfused shared expert: first"
    ]


@pytest.mark.parametrize(
    ("act", "rows", "pick"),
    [
        ("auto", 1, "a8"),
        ("auto", 4, "a8"),
        ("auto", 5, "a8"),
        ("auto", 64, "a8"),
        ("auto", 255, "a8"),
        ("auto", 256, "a4"),
        ("auto", 2048, "a4"),
        ("a8", 64, "a8"),
        ("a4", 1, "a4"),
    ],
)
def test_activation_pick(monkeypatch, act, rows, pick):
    monkeypatch.setenv("ATOM_M3_FHMOE_ACT", act)
    assert fh.pick_act(rows) == pick


def test_an_unknown_activation_mode_is_an_error(monkeypatch):
    monkeypatch.setenv("ATOM_M3_FHMOE_ACT", "bf16")
    with pytest.raises(ValueError, match="ATOM_M3_FHMOE_ACT"):
        fh.pick_act(8)


def _layer(rows=fh._ROUTED + 1):
    p = torch.nn.Parameter
    return SimpleNamespace(
        w13_weight=p(torch.full((rows, 4, 2), 7, dtype=torch.uint8), False),
        w2_weight=p(torch.full((rows, 2, 4), 7, dtype=torch.uint8), False),
        w13_weight_scale=p(torch.zeros((rows, 4, 1), dtype=torch.uint8), False),
        w2_weight_scale=p(torch.zeros((rows, 2, 1), dtype=torch.uint8), False),
    )


def test_the_dummy_routed_row_is_zero_weights_and_unit_scales():
    layer = _layer()
    fh.FhmoeM3(layer).fill_dummy_row()
    for w in (layer.w13_weight, layer.w2_weight):
        assert (w[: fh._ROUTED] == 7).all() and not w[fh._ROUTED :].any()
    for s in (layer.w13_weight_scale, layer.w2_weight_scale):
        assert not s[: fh._ROUTED].any() and (s[fh._ROUTED :] == 0x7F).all()


def test_steps_above_the_tuned_rows_are_left_to_the_production_path():
    x = torch.empty(fh.MAX_ROWS + 1, 8)
    assert fh.FhmoeM3(_layer()).run(x, None, None, None) is None
