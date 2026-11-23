# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``weights._attn_gemm``: a native MXFP8 1x32 attention linear (E8M0 scales per 32-block)
and a ptpc-FP8 one (f32 scale a row) are both accepted, each with its own flag; a
scale of the wrong shape or dtype is refused (CPU, no kernel)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
from aiter import QuantType, dtypes

from atom.models.minimax_m3.mono import weights
from atom.mono.runtime.consensus import MonoUnsupported

ROWS, COLS = 32, 256


def _linear(quant_type, scale):
    return SimpleNamespace(
        quant_type=quant_type,
        weight=torch.zeros(ROWS, COLS, dtype=torch.uint8).view(dtypes.fp8),
        weight_scale=scale,
        params_dtype=dtypes.fp8,
        native_a8_group_rows=1 if quant_type == QuantType.per_1x32 else None,
        is_output_padded=False,
    )


@pytest.fixture(autouse=True)
def _preshuffled(monkeypatch):
    monkeypatch.setattr(weights, "weight_is_stored_preshuffled", lambda *a, **k: True)


def test_native_mxfp8_is_accepted_with_its_flag():
    s = torch.zeros(ROWS, COLS // 32, dtype=torch.uint8)
    w, sc, mx = weights._attn_gemm(_linear(QuantType.per_1x32, s), ROWS, COLS, "x")
    assert mx and sc.shape == (ROWS, COLS // 32) and w.shape == (ROWS, COLS)


def test_ptpc_is_accepted_without_it():
    s = torch.zeros(ROWS, 1, dtype=torch.float32)
    _, sc, mx = weights._attn_gemm(_linear(QuantType.per_Token, s), ROWS, COLS, "x")
    assert not mx and sc.numel() == ROWS


@pytest.mark.parametrize(
    "scale",
    [
        torch.zeros(ROWS, COLS // 32 - 1, dtype=torch.uint8),
        torch.zeros(ROWS, COLS // 32, dtype=torch.float32),
        None,
    ],
)
def test_a_wrong_mxfp8_scale_is_refused(scale):
    with pytest.raises(MonoUnsupported):
        weights._attn_gemm(_linear(QuantType.per_1x32, scale), ROWS, COLS, "x")
