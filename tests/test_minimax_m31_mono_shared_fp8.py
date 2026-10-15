# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Synthetic Quark-MXFP8 shared-expert layout checks; no model files or GPU."""

from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter")
if not hasattr(aiter, "QuantType"):
    pytest.skip("full AITER Python API is unavailable", allow_module_level=True)
from aiter import QuantType, dtypes

from atom.models.minimax_m3.mono.config import HIDDEN, INTER
from atom.models.minimax_m3.mono.weights import _mxfp8
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
    [(2 * INTER, HIDDEN), (HIDDEN, INTER)],
    ids=["gate_up", "down"],
)
def test_exact_quark_mxfp8_shared_linear_layout(rows, cols):
    weight, scale = _mxfp8(_linear(rows, cols), rows, cols, "shared")
    assert tuple(weight.shape) == (rows, cols)
    assert scale.numel() == rows * cols // 32


def test_shuffled_shared_weight_is_rejected():
    linear = _linear(2 * INTER, HIDDEN)
    linear.weight.is_shuffled = True
    with pytest.raises(MonoUnsupported, match="weight shuffled"):
        _mxfp8(linear, 2 * INTER, HIDDEN, "shared")
