# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect

import pytest
import torch
from aiter import QuantType, dtypes

if not torch.cuda.is_available():
    pytest.skip("MXFP4 layout integration requires ROCm", allow_module_level=True)

from atom.model_ops.fp4_layout import (
    Fp4BackendKind,
    MXScaleLayout,
    decode_fp4_backend_layout_code,
    resolve_fp4_backend_spec,
)
from atom.model_ops.layernorm import _aiter_rms_quant_fake
from atom.model_ops.linear import gemm_a4w4_quant, gemm_a4w4_quant_fake
from atom.models.deepseek_v2 import (
    _fuse_rmsnorm_fp4_quant_fake,
)
from atom.models.kimi_k3 import _should_fuse_routed_norm_quant


@pytest.mark.parametrize(
    "arch,triton,nonshuffle,kind,layout",
    [
        ("gfx1250", False, False, Fp4BackendKind.DEFAULT_AITER, MXScaleLayout.OPUS_F4),
        (
            "gfx950",
            False,
            False,
            Fp4BackendKind.DEFAULT_AITER,
            MXScaleLayout.AITER_E8M0,
        ),
        (
            "gfx1250",
            True,
            False,
            Fp4BackendKind.TRITON_PRESHUFFLE,
            MXScaleLayout.AITER_E8M0,
        ),
        (
            "gfx1250",
            True,
            True,
            Fp4BackendKind.TRITON_NONSHUFFLE,
            MXScaleLayout.ROW_MAJOR,
        ),
    ],
)
def test_fp4_backend_layout_contract(arch, triton, nonshuffle, kind, layout):
    spec = resolve_fp4_backend_spec(arch, dtypes.fp4x2, triton, nonshuffle)
    assert spec is not None
    assert (spec.kind, spec.activation_scale_layout, spec.weight_scale_layout) == (
        kind,
        layout,
        layout,
    )
    assert decode_fp4_backend_layout_code(spec.backend_layout_code) == (
        kind,
        layout,
        layout,
    )


def test_non_fp4_has_no_fp4_backend_contract():
    assert resolve_fp4_backend_spec("gfx1250", dtypes.fp8, True, False) is None


@pytest.mark.parametrize(
    "layout,shape",
    [
        (MXScaleLayout.ROW_MAJOR, (5, 8)),
        (MXScaleLayout.AITER_E8M0, (256, 8)),
        (MXScaleLayout.OPUS_F4, (32, 8)),
    ],
)
def test_fused_rmsnorm_fake_allocates_consumer_layout(layout, shape):
    x = torch.empty((5, 256), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((256,), device="cuda", dtype=torch.bfloat16)
    _, scale, _ = _aiter_rms_quant_fake(
        x,
        weight,
        1e-6,
        QuantType.per_1x32.value,
        False,
        value_dtype=dtypes.fp4x2,
        mxfp4_scale_layout=layout,
    )
    assert tuple(scale.shape) == shape


@pytest.mark.parametrize(
    "layout,shape",
    [
        (MXScaleLayout.ROW_MAJOR, (5, 8)),
        (MXScaleLayout.AITER_E8M0, (256, 8)),
        (MXScaleLayout.OPUS_F4, (32, 8)),
    ],
)
def test_legacy_fused_rmsnorm_fake_uses_explicit_layout(layout, shape):
    x = torch.empty((5, 256), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((256,), device="cuda", dtype=torch.bfloat16)
    _, scale, *_ = _fuse_rmsnorm_fp4_quant_fake(
        x,
        weight,
        1e-6,
        shuffle=False,
        scale_shuffle_padding=False,
        mxfp4_scale_layout=layout,
    )
    assert tuple(scale.shape) == shape


def test_gemm_custom_op_carries_one_backend_layout_scalar():
    assert gemm_a4w4_quant is not None
    assert tuple(inspect.signature(gemm_a4w4_quant_fake).parameters)[-1] == (
        "backend_layout_code"
    )
    assert "backend_layout_code" in str(torch.ops.aiter.gemm_a4w4_quant.default._schema)


@pytest.mark.parametrize(
    "quant_type,dtype,expected",
    [
        (QuantType.per_1x32, dtypes.fp4x2, True),
        (QuantType.per_1x32, dtypes.fp8, False),
        (QuantType.per_1x128, dtypes.fp8, True),
        (QuantType.per_Token, dtypes.fp8, True),
        (QuantType.per_Tensor, dtypes.fp8, False),
    ],
)
def test_kimi_routed_norm_fusion_contract(quant_type, dtype, expected):
    assert _should_fuse_routed_norm_quant(True, quant_type, dtype) is expected
    assert not _should_fuse_routed_norm_quant(False, quant_type, dtype)
