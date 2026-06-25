# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical test for the dense shared-expert path with SwiGLU-OAI activation.

Regression: ``Mxfp4MoEMethod._apply_shared_experts_dense`` hard-asserted the
SiLU activation path, so MiniMax-M3 (``ActivationType.Swiglu`` *with* fused
shared experts) crashed with::

    AssertionError: dense shared-expert GEMM only supports the SiLU activation path

MiniMax-M3 does not interleave gate/up weights, so the dense GEMM output is
split ``[gate | up]`` — exactly what ``swiglu_oai_split`` consumes. The dense
shared expert must therefore replicate ``MiniMaxM3MLP.forward``:
gate_up GEMM -> swiglu_oai_split -> down GEMM, with the SwiGLU-OAI math
``gate * sigmoid(alpha*gate) * (up + beta)`` (alpha=1.702, beta=1.0), not SiLU.

These tests run the *real* fixed code path (gemm_a16wfp4 + the activation
branch). The fix only changes the *activation*, so the reference reuses the
*same* ``gemm_a16wfp4`` for both matmuls and differs only in the activation
math (computed independently in plain torch). This isolates the activation and
avoids conflating it with the kernel's mxfp4/bf16 GEMM precision. The tests
prove:
  * the SwiGLU branch matches the SwiGLU-OAI reference, and
  * it is genuinely different from the SiLU reference (i.e. the fix changed
    behaviour, it is not silently equivalent), and
  * the SiLU branch (DeepSeek) is unchanged.
"""

import types

import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an AMD GPU"
)

SCALE_GROUP_SIZE = 32

# e2m1 (fp4) decode table: sign | 2-bit exp | 1-bit mantissa.
_MXFP4_TABLE = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _mxfp4_to_f32(packed: torch.Tensor) -> torch.Tensor:
    """Decode a uint8 tensor of two packed e2m1 nibbles to f32 (last dim x2)."""
    x = packed.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    table = torch.tensor(_MXFP4_TABLE, dtype=torch.float32, device=packed.device)
    return table[x.long()]


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    return 2.0 ** (scale.to(torch.float32) - 127)


def _make_weight(n: int, k: int, *, seed: int):
    """Random fp4-packed weight (n, k//2) + e8m0 scales (n, k//32).

    The bf16 dequantization is not returned: the reference reuses the kernel's
    own GEMM, so we never need a from-scratch dequant matmul.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    low = torch.randint(0, 16, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    high = torch.randint(0, 16, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    packed = low | (high << 4)
    # e8m0 scales near 1.0 (bias 127) keep the dequant range sane.
    scales = torch.randint(
        125, 130, (n, k // SCALE_GROUP_SIZE), dtype=torch.uint8, device="cuda", generator=g
    )
    return packed, scales


def _ref_swiglu_oai(gate_up, alpha, beta, limit):
    n = gate_up.shape[-1] // 2
    gate = gate_up[:, :n].to(torch.float32)
    up = gate_up[:, n:].to(torch.float32)
    if limit is not None:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(gate_up.dtype)


def _ref_silu(gate_up, limit):
    n = gate_up.shape[-1] // 2
    gate = gate_up[:, :n].to(torch.float32)
    up = gate_up[:, n:].to(torch.float32)
    if limit > 0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    return (gate * torch.sigmoid(gate) * up).to(gate_up.dtype)


def _build_method_and_layer(hidden, inter, *, alpha, beta, limit):
    from atom.config import LayerQuantConfig
    from atom.model_ops.moe import Mxfp4MoEMethod, MoEActivationQuant
    from aiter import QuantType
    from unittest.mock import MagicMock

    qc = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=torch.float4_e2m1fn_x2,
        quant_method="quark",
    )
    method = Mxfp4MoEMethod(qc, MagicMock())
    # Exercise the a16w4 (bf16 activation) path so we feed plain bf16 inputs.
    method.act_quant = MoEActivationQuant.BF16

    w13, s13 = _make_weight(2 * inter, hidden, seed=1)  # (2I, H)
    w2, s2 = _make_weight(hidden, inter, seed=2)  # (H, I)

    layer = types.SimpleNamespace(
        num_fused_shared_experts=1,
        shared_w13_weight=w13.unsqueeze(0),
        shared_w13_weight_scale=s13.unsqueeze(0),
        shared_w2_weight=w2.unsqueeze(0),
        shared_w2_weight_scale=s2.unsqueeze(0),
        shared_w13_bias=None,
        shared_w2_bias=None,
        swiglu_limit=limit,
        swiglu_alpha=alpha,
        swiglu_beta=beta,
    )
    return method, layer, (w13, s13), (w2, s2)


def _kernel_gemm(act, packed_scale):
    """Reuse the exact same GEMM the dense path uses, so the only difference
    between the dense path and the reference is the activation."""
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    weight, scale = packed_scale
    return gemm_a16wfp4(act, weight, scale, dtype=torch.bfloat16)


@cuda_only
def test_swiglu_shared_expert_matches_reference():
    from aiter import ActivationType
    from atom.model_ops.moe import Mxfp4MoEMethod

    if not _fp4_available():
        pytest.skip("MXFP4 not supported on this architecture")

    hidden, inter, M = 256, 256, 64
    alpha, beta, limit = 1.702, 1.0, 7.0
    method, layer, w13, w2 = _build_method_and_layer(
        hidden, inter, alpha=alpha, beta=beta, limit=limit
    )

    torch.manual_seed(0)
    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.5

    out = Mxfp4MoEMethod._apply_shared_experts_dense(
        method, layer, x, ActivationType.Swiglu
    )

    # Reference reuses the SAME kernel GEMM; only the activation is computed
    # independently (plain torch), isolating the fix from GEMM precision.
    gate_up = _kernel_gemm(x, w13)
    inter_ref = _ref_swiglu_oai(gate_up, alpha, beta, limit)
    out_ref = _kernel_gemm(inter_ref, w2)

    # SiLU on the same gate_up must be clearly different (proves the branch
    # matters and the fix is not silently equivalent to the old code).
    inter_silu = _ref_silu(gate_up, limit)
    out_silu = _kernel_gemm(inter_silu, w2)

    err_swiglu = (out.float() - out_ref.float()).abs().mean().item()
    err_vs_silu = (out_ref.float() - out_silu.float()).abs().mean().item()

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)
    assert err_vs_silu > 10 * max(err_swiglu, 1e-6), (
        f"swiglu vs silu too close to distinguish "
        f"(err_swiglu={err_swiglu}, err_vs_silu={err_vs_silu})"
    )


@cuda_only
def test_silu_shared_expert_unchanged():
    from aiter import ActivationType
    from atom.model_ops.moe import Mxfp4MoEMethod

    if not _fp4_available():
        pytest.skip("MXFP4 not supported on this architecture")

    hidden, inter, M = 256, 256, 64
    limit = 7.0
    method, layer, w13, w2 = _build_method_and_layer(
        hidden, inter, alpha=1.702, beta=1.0, limit=limit
    )

    torch.manual_seed(0)
    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.5

    out = Mxfp4MoEMethod._apply_shared_experts_dense(
        method, layer, x, ActivationType.Silu
    )

    gate_up = _kernel_gemm(x, w13)
    inter_ref = _ref_silu(gate_up, limit)
    out_ref = _kernel_gemm(inter_ref, w2)

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=1e-2, atol=1e-2)


def _fp4_available():
    try:
        import aiter.ops.triton.utils._triton.arch_info as arch_info

        return arch_info.is_fp4_avail()
    except Exception:
        return torch.cuda.is_available()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
