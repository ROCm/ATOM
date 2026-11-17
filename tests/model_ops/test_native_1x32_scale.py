# SPDX-License-Identifier: MIT
"""A preshuffled native 1x32 FP8 weight's scale takes aiter's GEMM layout."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale

from atom.model_ops.linear import LinearBase
from atom.rollout.weight_updater import WeightUpdaterMixin

N, K, u8 = 128, 256, torch.uint8
_g = torch.Generator().manual_seed(0)
OLD, NEW = torch.randint(0, 255, (2, N, K // 32), dtype=u8, generator=_g)


def _want(raw):
    return shuffle_mxfp8_bpreshuffle_scale(raw.view(dtypes.fp8_e8m0)).view(u8)


@pytest.fixture
def layer():
    layer = SimpleNamespace(
        weight=nn.Parameter(torch.randn(N, K).to(dtypes.fp8), requires_grad=False),
        weight_scale=nn.Parameter(
            OLD.clone().view(dtypes.fp8_e8m0), requires_grad=False
        ),
        quant_type=QuantType.per_1x32,
        params_dtype=dtypes.fp8,
        needs_preshuffled_weight=True,  # shuffled on every arch
        native_a8_group_rows=1,
        weight_scale_row_group=1,
        bias=None,
        _parameters={},
    )
    LinearBase.process_weights_after_loading(layer)
    return layer


def test_load(layer):
    assert layer.weight.is_shuffled
    assert torch.equal(layer.weight_scale.view(u8), _want(OLD))


def test_sync(layer):
    weight, ptr = layer.weight.view(u8).clone(), layer.weight_scale.data_ptr()
    layer.weight_scale.data.copy_(NEW.view(dtypes.fp8_e8m0))
    WeightUpdaterMixin()._post_process_fp8_weight(layer, layer.weight_scale)
    assert torch.equal(layer.weight_scale.view(u8), _want(NEW))
    assert layer.weight_scale.data_ptr() == ptr
    assert torch.equal(layer.weight.view(u8), weight)
    WeightUpdaterMixin()._post_process_fp8_weight(layer, layer.weight)
    assert torch.equal(layer.weight_scale.view(u8), _want(NEW))


@pytest.mark.parametrize("start", [0, 16, 32, 48, 96])
def test_row_view(layer, start):
    if start % 32:
        with pytest.raises(ValueError, match="32-row boundaries"):
            LinearBase.make_row_view(layer, start, 32)
    else:
        view = LinearBase.make_row_view(layer, start, 32)
        want = _want(OLD[start : start + 32])
        assert torch.equal(view.weight_scale.view(u8), want)
