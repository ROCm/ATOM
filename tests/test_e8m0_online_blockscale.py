# SPDX-License-Identifier: MIT
"""Online per_1x128 FP8 with E8M0 block scales (QuantizationConfig.blockscale_e8m0_scale)."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

aiter = pytest.importorskip("aiter")
QuantType, dtypes = aiter.QuantType, aiter.dtypes

linear_mod = importlib.import_module("atom.model_ops.linear")
LayerQuantConfig = importlib.import_module("atom.quant_spec").LayerQuantConfig
quantize = importlib.import_module(
    "atom.quantization.quark.utils"
).quantize_weight_to_fp8_128x128_blockscale
kimi_k3 = importlib.import_module("atom.models.kimi_k3")


def test_e8m0_scale_is_the_fp32_scale_rounded_up_to_a_power_of_two():
    w = torch.randn(300, 520) * 0.02  # neither dim a multiple of 128
    _, s32 = quantize(w, dtypes.fp8)
    q, se = quantize(w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0)

    assert se.dtype == dtypes.fp8_e8m0 and tuple(se.shape) == (3, 5)
    ratio = se.float() / s32
    assert torch.all((ratio >= 1) & (ratio < 2))
    assert q.float().abs().max() <= torch.finfo(dtypes.fp8).max  # no clipping

    exact = torch.zeros(128, 128)
    exact[0, 0] = torch.finfo(dtypes.fp8).max * 2.0**-7
    assert (
        quantize(exact, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0)[1].float().item()
        == 2.0**-7
    )


def _bf16_layer(monkeypatch, e8m0, output_size=2112, source=None, *, load=False):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    monkeypatch.setattr(
        linear_mod,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x128, quant_dtype=dtypes.fp8, is_dynamic=True
    )
    quant_config = SimpleNamespace(
        online_quant=True,
        blockscale_e8m0_scale=e8m0,
        get_layer_quant_config=lambda *_a, use_online_quant=False, **_k: (
            target if use_online_quant else LayerQuantConfig()
        ),
    )
    layer = linear_mod.ColumnParallelLinear(
        256, output_size, quant_config=quant_config, prefix="layers.3.qkv_a"
    )
    if source is None:
        source = torch.randn(output_size, 256, dtype=torch.bfloat16) * 0.02
    layer.weight.data = source
    if load:
        # The load path: online quantize, then pad and preshuffle.
        layer.process_weights_after_loading()
    else:
        layer.online_quantize_weight()
    return layer


def _same_bytes(a, b):
    return torch.equal(a.detach().view(torch.uint8), b.detach().view(torch.uint8))


def _updater(**kwargs):
    from atom.rollout.weight_updater import WeightUpdaterMixin

    class _Updater(WeightUpdaterMixin):
        label = "test"
        device = torch.device("cpu")
        rank = 0
        world_size = 1

    updater = _Updater()
    for name, value in kwargs.items():
        setattr(updater, name, value)
    return updater


@pytest.mark.parametrize("output_size", [2112, 2048])
def test_rollout_sync_reproduces_the_loaded_e8m0_weight(monkeypatch, output_size):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    source_a = torch.randn(output_size, 256, dtype=torch.bfloat16) * 0.02
    source_b = torch.randn(output_size, 256, dtype=torch.bfloat16) * 0.02
    layer = _bf16_layer(monkeypatch, True, output_size, source_a, load=True)
    ref = _bf16_layer(monkeypatch, True, output_size, source_b, load=True)
    before = layer.weight.detach().view(torch.uint8).clone()

    _updater()._requantize_fp8_weight(layer, "weight", layer.weight, source_b)

    assert not torch.equal(layer.weight.detach().view(torch.uint8), before)
    assert _same_bytes(layer.weight, ref.weight)
    assert _same_bytes(layer.weight_scale, ref.weight_scale)
    assert layer.weight_scale.dtype == dtypes.fp8_e8m0
    if output_size % 128 != 0:
        assert layer.is_output_padded
        assert tuple(layer.weight.shape) == (2176, 256)

    if not torch.cuda.is_available():
        return
    layer.cuda()
    ref.cuda()
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    y = layer(x)
    y_ref = ref(x)
    assert y.shape == (4, output_size)
    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)


def test_rollout_sync_shards_logical_n_before_padding(monkeypatch):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    stale = torch.randn(2112, 256, dtype=torch.bfloat16) * 0.02
    local = torch.randn(2112, 256, dtype=torch.bfloat16) * 0.02
    other = torch.randn(2112, 256, dtype=torch.bfloat16) * 0.02
    layer = _bf16_layer(monkeypatch, True, 2112, stale, load=True)
    ref = _bf16_layer(monkeypatch, True, 2112, local, load=True)

    # Rank 0 of a column-parallel pair. The stored N is padded (2176); the
    # full training weight is 2 * 2112, which is not 2 * 2176.
    _updater(world_size=2, rank=0)._requantize_fp8_weight(
        layer, "weight", layer.weight, torch.cat([local, other], dim=0)
    )

    assert _same_bytes(layer.weight, ref.weight)
    assert _same_bytes(layer.weight_scale, ref.weight_scale)


@pytest.mark.parametrize("e8m0", [True, False])
def test_online_target_decides_the_scale_format(monkeypatch, e8m0):
    layer = _bf16_layer(monkeypatch, e8m0)

    assert layer.quant_type == QuantType.per_1x128
    assert layer.weight_scale.dtype == (dtypes.fp8_e8m0 if e8m0 else torch.float32)
    assert layer.blockscale_e8m0_scale is e8m0
    assert layer.scale_dtype == layer.weight_scale.dtype


def test_online_e8m0_weight_is_padded_to_128_rows(monkeypatch):
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    layer = _bf16_layer(monkeypatch, e8m0=True)

    layer.is_output_padded = layer._maybe_pad_a8w8_preshuffle_output()

    assert tuple(layer.weight.shape) == (2176, 256)
    assert torch.all(layer.weight.data[2112:].float() == 0)
    assert tuple(layer.weight_scale.shape) == (17, 2)  # ceil(2112/128) rows already
    assert layer._output_size_before_padding == 2112
    assert layer.supports_out() is False


def test_fp32_blockscale_weight_is_not_padded():
    module = SimpleNamespace(
        weight=nn.Parameter(
            torch.zeros(2112, 256, dtype=torch.uint8), requires_grad=False
        ),
        weight_scale=torch.ones(17, 2, dtype=torch.float32),
        quant_type=QuantType.per_1x128,
        params_dtype=dtypes.fp8,
    )
    assert linear_mod.LinearBase._maybe_pad_a8w8_preshuffle_output(module) is False


@pytest.mark.parametrize(
    "quant_type,switch,expected",
    [
        (QuantType.per_1x128, True, True),
        (QuantType.per_1x128, False, False),
        (QuantType.per_Token, True, False),
    ],
)
def test_k3_fused_producers_stand_down_for_an_e8m0_consumer(
    quant_type, switch, expected
):
    cfg = SimpleNamespace(blockscale_e8m0_scale=switch)
    assert kimi_k3._is_e8m0_blockscale(cfg, quant_type) is expected
