# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 draft's fused paths apply only to a native MXFP8 1x32
linear; anything else (weight dequantized to bf16, other weight dtype, no
scales, tile-misaligned offsets) takes the unfused path."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dspark = pytest.importorskip("atom.models.minimax_m3.dspark")


def _lin(**kw):
    d = {
        "native_a8_group_rows": 1,
        "runs_bf16": False,
        "weight": torch.empty(16, 32, dtype=torch.float8_e4m3fn),
        "weight_scale": torch.empty(16, 1, dtype=torch.uint8),
    }
    d.update(kw)
    return SimpleNamespace(**d)


def test_native_mxfp8_accepted():
    assert dspark._is_native_mxfp8(_lin())


@pytest.mark.parametrize(
    "kw",
    [
        {"native_a8_group_rows": None},
        {"runs_bf16": True},
        {"weight": torch.empty(16, 32, dtype=torch.bfloat16)},
        {"weight_scale": None},
    ],
)
def test_other_linears_rejected(kw):
    assert not dspark._is_native_mxfp8(_lin(**kw))


def test_context_kv_weight_falls_back():
    attn = SimpleNamespace(
        qkv_proj=_lin(runs_bf16=True, bias=None), q_size=2048, kv_size=128
    )
    model = SimpleNamespace(
        _ctx_kv=None, context_layers=[SimpleNamespace(self_attn=attn)]
    )
    assert dspark.MiniMaxM3DSpark._context_kv_weight(model) is None
    assert model._ctx_kv == ()
