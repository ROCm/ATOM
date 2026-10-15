# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Synthetic Quark-MXFP8 linear layout checks; no model files or GPU."""

from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter")
if not hasattr(aiter, "QuantType"):
    pytest.skip("full AITER Python API is unavailable", allow_module_level=True)
from aiter import QuantType, dtypes

from atom.models.minimax_m3.mono.config import HIDDEN, INTER, O_K, IndexHeads
from atom.models.minimax_m3.mono.kernels.common import row_major_scale_index
from atom.models.minimax_m3.mono.weights import _mxfp8, attention_mxfp8
from atom.mono.runtime.consensus import MonoUnsupported


def _linear(rows: int, cols: int):
    weight = torch.empty(rows, cols, dtype=dtypes.fp8, device="meta")
    return SimpleNamespace(
        quant_type=QuantType.per_1x32,
        params_dtype=dtypes.fp8,
        native_a8_group_rows=1,
        weight=weight,
        weight_scale=torch.empty(
            rows, cols // 32, dtype=dtypes.fp8_e8m0, device="meta"
        ),
        is_output_padded=False,
    )


@pytest.mark.parametrize(
    "rows,cols",
    [
        (IndexHeads(4, 0).rows, HIDDEN),
        (HIDDEN, O_K),
        (2 * INTER, HIDDEN),
        (HIDDEN, INTER),
    ],
    ids=["qkv", "o", "gate_up", "down"],
)
def test_exact_quark_mxfp8_linear_layout(rows, cols):
    linear = _linear(rows, cols)
    weight, scale = _mxfp8(linear, rows, cols, "linear")
    assert tuple(weight.shape) == (rows, cols)
    assert scale.numel() == rows * cols // 32
    assert weight.untyped_storage()._cdata == linear.weight.untyped_storage()._cdata
    assert (
        scale.untyped_storage()._cdata
        == linear.weight_scale.untyped_storage()._cdata
    )


def test_shuffled_shared_weight_is_rejected():
    linear = _linear(2 * INTER, HIDDEN)
    linear.weight.is_shuffled = True
    with pytest.raises(MonoUnsupported, match="weight shuffled"):
        _mxfp8(linear, 2 * INTER, HIDDEN, "shared")


def test_noncontiguous_mxfp8_weight_is_rejected():
    linear = _linear(2 * INTER, HIDDEN)
    linear.weight = torch.empty(
        HIDDEN, 2 * INTER, dtype=dtypes.fp8, device="meta"
    ).t()
    with pytest.raises(MonoUnsupported, match="weight"):
        _mxfp8(linear, 2 * INTER, HIDDEN, "shared")


def test_attention_mode_comes_from_both_projection_metadata():
    native = SimpleNamespace(
        self_attn=SimpleNamespace(
            qkv_proj=_linear(1, 32),
            o_proj=_linear(1, 32),
        )
    )
    assert attention_mxfp8(native, 3)
    ptpc = SimpleNamespace(
        quant_type=QuantType.per_Token,
        params_dtype=dtypes.fp8,
        native_a8_group_rows=None,
    )
    native.self_attn.o_proj = ptpc
    with pytest.raises(MonoUnsupported, match="mixed attention quantization"):
        attention_mxfp8(native, 3)
    native.self_attn.qkv_proj = ptpc
    assert not attention_mxfp8(native, 3)


def test_row_major_scale_index():
    cols = HIDDEN // 32
    for row_group in (0, 1, 127):
        for chunk in range(HIDDEN // 128):
            for lane in range(64):
                row = row_group * 16 + lane % 16
                col = chunk * 4 + lane // 16
                index = row_major_scale_index(row, col, cols)
                assert index == row * cols + col
                assert 0 <= index < (row_group + 1) * 16 * cols
