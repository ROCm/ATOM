# SPDX-License-Identifier: MIT
"""Fused Q/K/V quantization's descales must work directly with FP8 FMHA."""

import math

import pytest
import torch
from aiter.ops.flydsl import flydsl_flash_attn_fp8_func
from aiter.ops.quant import per_tensor_quant_hip

from atom.model_ops.triton_fused_qkv_quant import fused_qkv_per_tensor_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("zero", [False, True])
def test_fused_quant_fmha_handoff(zero):
    q = torch.randn((96, 12, 192), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((512, 12, 192), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((512, 12, 256), device="cuda", dtype=torch.bfloat16)[..., 128:]
    if zero:
        for x in (q, k, v):
            x.zero_()
    q8, k8, v8, qs, ks, vs, _, _ = fused_qkv_per_tensor_quant(
        q, k, v, q_scale_factor=1.2345
    )
    args = {
        "cu_seqlens_q": torch.tensor([0, 32, 64, 96], device="cuda", dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor(
            [0, 256, 256, 512], device="cuda", dtype=torch.int32
        ),
        "max_seqlen_q": 32,
        "max_seqlen_kv": 256,
        "cross_seqlen": True,
        "causal": False,
        "return_lse": True,
    }
    actual, lse = flydsl_flash_attn_fp8_func(
        q8, k8, v8, q_descale=qs, k_descale=ks, v_descale=vs, **args
    )
    assert torch.isfinite(actual).all()
    assert (actual[32:64] == 0).all()
    if zero:
        assert (actual == 0).all()
        # Two sequences attend 256 equal zero logits; the middle has no KV.
        expected = torch.full_like(lse, math.log(256))
        expected[:, 32:64] = -math.inf
        torch.testing.assert_close(lse, expected, rtol=1e-6, atol=1e-6)
    else:
        quantized = [
            per_tensor_quant_hip(
                x.contiguous().view(-1, 128), quant_dtype=torch.float8_e4m3fn
            )
            for x in (q, k, v)
        ]
        (qr, qrs), (kr, krs), (vr, vrs) = quantized
        expected, ref_lse = flydsl_flash_attn_fp8_func(
            qr.view(q.shape),
            kr.view(k.shape),
            vr.view(v.shape),
            q_descale=qrs * 1.2345,
            k_descale=krs,
            v_descale=vrs,
            **args,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(lse, ref_lse, rtol=0, atol=0)
