# SPDX-License-Identifier: MIT
"""A preshuffled native 1x32 FP8 weight keeps its E8M0 scale in the layout
aiter's GEMM reads (``shuffle_mxfp8_bpreshuffle_scale``, n32k4 on gfx1250):
after the load, after a weight sync, and through a row view."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the scale layout under test is AITER's")

from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale

from atom.model_ops.linear import LinearBase
from atom.rollout.weight_updater import WeightUpdaterMixin

N, K = 128, 256


def _raw_scale(seed=0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 255, (N, K // 32), dtype=torch.uint8, generator=g)
    return raw.view(dtypes.fp8_e8m0)


def _loaded_layer():
    """The attributes the load, the sync and `make_row_view` read, loaded."""
    layer = SimpleNamespace(
        weight=nn.Parameter(torch.zeros(N, K, dtype=dtypes.fp8), requires_grad=False),
        weight_scale=nn.Parameter(_raw_scale(), requires_grad=False),
        quant_type=QuantType.per_1x32,
        params_dtype=dtypes.fp8,
        # Shuffled on every arch, so the test does not depend on the GPU.
        needs_preshuffled_weight=True,
        native_a8_group_rows=1,
        weight_scale_row_group=1,
        bias=None,
        _parameters={},
        prefix="test",
    )
    LinearBase.process_weights_after_loading(layer)
    return layer


def _bytes(t):
    return t.data.view(torch.uint8)


def _expected(raw):
    return _bytes(shuffle_mxfp8_bpreshuffle_scale(raw.contiguous()))


class _Updater(WeightUpdaterMixin):
    label = "test"
    device = torch.device("cpu")
    rank = 0
    world_size = 1


def test_load_lays_the_scale_out_as_aiter_does():
    layer = _loaded_layer()
    assert layer.weight.is_shuffled
    assert torch.equal(_bytes(layer.weight_scale), _expected(_raw_scale()))


def test_sync_lays_a_new_scale_out_like_the_load_in_place():
    layer = _loaded_layer()
    weight, ptr = _bytes(layer.weight).clone(), layer.weight_scale.data_ptr()
    # What the update loop does with a synced E8M0 scale: copy, then post-process.
    layer.weight_scale.data.copy_(_raw_scale(seed=1))
    _Updater()._post_process_fp8_weight(layer, layer.weight_scale)

    assert torch.equal(_bytes(layer.weight_scale), _expected(_raw_scale(seed=1)))
    assert layer.weight_scale.data_ptr() == ptr  # a captured graph keeps reading it
    assert torch.equal(_bytes(layer.weight), weight)  # the weight is not reshuffled


def test_sync_of_the_weight_leaves_the_scale_alone():
    layer = _loaded_layer()
    scale = _bytes(layer.weight_scale).clone()
    _Updater()._post_process_fp8_weight(layer, layer.weight)
    assert torch.equal(_bytes(layer.weight_scale), scale)


@pytest.mark.parametrize("start", [16, 48])
def test_row_view_refuses_a_cut_inside_a_32_row_scale_tile(start):
    with pytest.raises(ValueError, match="32-row boundaries"):
        LinearBase.make_row_view(_loaded_layer(), start, 64)


@pytest.mark.parametrize("start, length", [(0, 32), (32, 64), (96, 32)])
def test_row_view_of_32_aligned_rows_keeps_their_scale(start, length):
    view = LinearBase.make_row_view(_loaded_layer(), start, length)
    want = _expected(_raw_scale()[start : start + length])
    assert torch.equal(_bytes(view.weight_scale), want)
