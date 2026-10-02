# SPDX-License-Identifier: MIT
"""Kimi-K3 SiTUv2 + ptpc FP8 shape guards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

try:
    from aiter import QuantType, dtypes

    import atom.models.kimi_k3 as kimi_k3
    from atom.model_ops.kimi_k3.activations import (
        situ_and_mul,
        situ_and_mul_quant,
    )

    _IMPORT_ERR = None
except ImportError as _e:  # aiter/triton absent on a CPU-only runner
    _IMPORT_ERR = str(_e)

needs_aiter = pytest.mark.skipif(
    _IMPORT_ERR is not None,
    reason=f"Kimi-K3 model ops require AITER: {_IMPORT_ERR}",
)
needs_gpu = pytest.mark.skipif(
    _IMPORT_ERR is not None or not torch.cuda.is_available(),
    reason=f"SiTUv2 quant kernels require a GPU: {_IMPORT_ERR}",
)


class _LinearStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


@needs_aiter
@pytest.mark.parametrize(
    ("intermediate_size", "expected_fused"),
    [(6144, True), (33792, False)],
)
def test_kimi_mlp_fuses_only_aiter_supported_situ_shapes(
    monkeypatch, intermediate_size, expected_fused
):
    monkeypatch.setattr(kimi_k3, "MergedColumnParallelLinear", _LinearStub)
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", _LinearStub)
    monkeypatch.setattr(
        kimi_k3,
        "_effective_layer_quant",
        lambda *args: (QuantType.per_Token, dtypes.fp8),
    )
    config = SimpleNamespace(
        hidden_size=7168,
        intermediate_size=33792,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    mlp = kimi_k3.KimiMLP(
        config,
        intermediate_size=intermediate_size,
        quant_config=object(),
        prefix="model.layers.0.mlp",
    )

    assert mlp._fuse_act_quant is expected_fused
    assert mlp.act_fn.fused_quant is expected_fused


@needs_gpu
def test_dense_situ_quant_falls_back_for_kimi_k3_dimension():
    torch.manual_seed(0)
    x = torch.randn(2, 2 * 33792, dtype=torch.bfloat16, device="cuda")

    quantized, scale = situ_and_mul_quant(x, beta=4.0, linear_beta=25.0)
    reference = situ_and_mul(x, beta=4.0, linear_beta=25.0).float()
    dequantized = quantized.float() * scale

    torch.testing.assert_close(dequantized, reference, rtol=0.05, atol=0.5)


@needs_gpu
@pytest.mark.parametrize("intermediate_size", [6144, 33792])
def test_situ_quant_accepts_empty_batches(intermediate_size):
    x = torch.empty(
        0,
        2 * intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    quantized, scale = situ_and_mul_quant(x, beta=4.0, linear_beta=25.0)

    assert quantized.shape == (0, intermediate_size)
    assert scale.shape == (0, 1)
