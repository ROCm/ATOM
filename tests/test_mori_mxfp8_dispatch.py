# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MORI_FP8_DISPATCH: one branch decides the MXFP8 wire format.

The staging scale geometry MoRI is built with (scale_dim, scale_type_size) has
to match what prepare() sends -- hidden/32 one-byte e8m0 scales -- and the
format is only sound where aiter's fused_moe takes fp8 rows as already
quantized, which the runtime check asks aiter itself.
"""

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch
from aiter import ActivationType, QuantType, dtypes

import atom.model_ops.fused_moe.mori_prepare_finalize as mpf
from atom.model_ops.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)


@pytest.fixture
def mxfp8_env(monkeypatch):
    monkeypatch.setattr(mpf, "_MXFP8_DISPATCH", True)
    monkeypatch.setattr(mpf, "_FP4_DISPATCH", False)


def test_mxfp8_format_geometry(mxfp8_env):
    quant_config = mxfp4_w4a16_moe_quant_config(w1_scale=None, w2_scale=None)
    fmt = mpf.resolve_mori_dispatch(torch.bfloat16, 7168, quant_config)
    assert fmt.dtype == dtypes.fp8
    assert fmt.quant_type == QuantType.per_1x32
    assert (fmt.scale_dim, fmt.scale_type_size) == (224, 1)
    assert fmt.is_mxfp8 and not fmt.is_fp4


def test_mxfp8_refuses_non_mxfp4_experts(mxfp8_env):
    with pytest.raises(ValueError, match="MXFP4"):
        mpf.resolve_mori_dispatch(
            torch.bfloat16, 7168, fp8_w8a8_moe_quant_config(None, None)
        )


def test_fp4_and_mxfp8_are_exclusive(mxfp8_env, monkeypatch):
    monkeypatch.setattr(mpf, "_FP4_DISPATCH", True)
    with pytest.raises(ValueError, match="at most one"):
        mpf.resolve_mori_dispatch(torch.bfloat16, 7168, None)


@pytest.mark.parametrize(
    "gate_mode, bound, ok",
    [
        ("interleave", "0", True),
        ("interleave", "256", False),
        ("separated", "0", False),
    ],
)
def test_consumable_check_follows_aiter(monkeypatch, gate_mode, bound, ok):
    monkeypatch.setattr("aiter.fused_moe.get_gfx", lambda: "gfx950", raising=False)
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", bound)
    mpf.check_mxfp8_dispatch_consumable.cache_clear()
    args = (QuantType.per_1x32, dtypes.fp4x2, ActivationType.Silu, gate_mode, 0)
    if ok:
        mpf.check_mxfp8_dispatch_consumable(*args)
    else:
        with pytest.raises(RuntimeError, match="ATOM_MORI_FP8_DISPATCH"):
            mpf.check_mxfp8_dispatch_consumable(*args)
    mpf.check_mxfp8_dispatch_consumable.cache_clear()
