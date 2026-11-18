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


def _layer(rows, scale):
    layer = SimpleNamespace(
        weight=nn.Parameter(
            torch.randn(len(scale) * rows, scale.shape[1] * 32).to(dtypes.fp8),
            requires_grad=False,
        ),
        weight_scale=nn.Parameter(
            scale.clone().view(dtypes.fp8_e8m0), requires_grad=False
        ),
        quant_type=QuantType.per_1x32,
        params_dtype=dtypes.fp8,
        needs_preshuffled_weight=True,  # shuffled on every arch
        native_a8_group_rows=rows,
        weight_scale_row_group=rows,
        bias=None,
        _parameters={},
    )
    LinearBase.process_weights_after_loading(layer)
    return layer


@pytest.fixture
def layer():
    return _layer(1, OLD)


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


def test_sync_32x32():
    # A native 32x32 scale stays row-major at load and after a sync. Its
    # (16, 32) shape is one the old 16x16 `shuffle_weights` would permute.
    old, new = torch.randint(0, 255, (2, 16, 32), dtype=u8, generator=_g)
    layer = _layer(32, old)
    assert layer.weight.is_shuffled
    assert torch.equal(layer.weight_scale.view(u8), old)
    layer.weight_scale.data.copy_(new.view(dtypes.fp8_e8m0))
    WeightUpdaterMixin()._post_process_fp8_weight(layer, layer.weight_scale)
    assert torch.equal(layer.weight_scale.view(u8), new)


@pytest.mark.parametrize("start", [0, 16, 32, 48, 96])
def test_row_view(layer, start):
    if start % 32:
        with pytest.raises(ValueError, match="32-row boundaries"):
            LinearBase.make_row_view(layer, start, 32)
    else:
        view = LinearBase.make_row_view(layer, start, 32)
        want = _want(OLD[start : start + 32])
        assert torch.equal(view.weight_scale.view(u8), want)
