# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference for the MXFP8 GEMMs: the contract every variant meets.

Contract (``out = x @ W^T``):

- ``x`` is a bf16 / fp16 ``[M, K]`` activation with ``K % 256 == 0``. Every
  variant quantizes it per 1x32 block along K with the rule of
  ``quantize_mxfp8``: ``e = clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0,
  254)`` (E8M0), ``code = e4m3fn(x * 2^(127 - e))``, round to nearest even.
  The quantizer multiplies by the reciprocal: ``2^(e - 127)`` is subnormal for
  ``e = 0`` and flushes to zero on CDNA.
- ``W`` is given as E4M3 codes ``[N, K]`` and E8M0 exponents ``[N, K // 32]``
  made by the same rule.
- Both operands are dequantized exactly, the product accumulates in FP32, and
  the output is rounded once to ``out_dtype``. ``mxfp8_gemm_reference`` takes
  operands already quantized (a test's own codes, the GEMM entries that take
  codes); with ``accumulate=torch.float64`` it is the exact product, rounded
  once (the decode-window tests at their tile edges).
- Tolerance: a variant differs from this reference only by accumulation order
  and the output rounding, so every output must be finite with cosine
  similarity >= ``COS_MIN`` and relative L2 error <= ``REL_L2_MAX``. Repeated
  launches on the same inputs are bitwise identical.

The fused gate_up row (``mxfp8_gateup_swigluq``, ``W`` = ``[gate; up]`` with
``2I`` rows) returns the down GEMM's MXFP8 activation instead of BF16: the GEMM
output is rounded to BF16, SwiGLU-OAI (``swiglu_oai``) runs in FP32 on those
values, ``H`` is rounded to BF16 and quantized with ``quantize_mxfp8``. A
different accumulation order can move a value of ``H`` across an E4M3 rounding
boundary, so its dequantized output meets the looser ``GATEUP_COS_MIN`` /
``GATEUP_REL_L2_MAX``.
"""

import torch

E4M3_MAX = 448.0
BLOCK = 32
COS_MIN = 0.99999
REL_L2_MAX = 1e-3  # measured 1.8e-4 .. 2.4e-4 on the ten GEMM cases
GATEUP_COS_MIN = 0.99999  # measured 0.9999994 at M 4096 / 4352
GATEUP_REL_L2_MAX = 5e-3  # measured 9.9e-4 .. 1.0e-3


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, K]`` -> E4M3 codes ``[R, K]`` and E8M0 exponents ``[R, K // 32]``,
    row-major (no preshuffle)."""
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (amax / E4M3_MAX).log2().ceil().add(127).clamp(0, 254)
    codes = (blocks * torch.exp2(127 - exponent).unsqueeze(-1)).reshape(rows, k)
    return codes.to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def dequantize_mxfp8(
    codes: torch.Tensor, exponents: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """E4M3 codes and E8M0 exponents -> FP32 (or ``dtype``) values (exact)."""
    rows, k = codes.shape
    scale = torch.exp2(exponents.to(dtype) - 127).unsqueeze(-1)
    return (codes.to(dtype).reshape(rows, k // BLOCK, BLOCK) * scale).reshape(rows, k)


def mxfp8_gemm_reference(
    codes: torch.Tensor,
    exponents: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_exponents: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    accumulate: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The contract's ``A @ W^T`` for a quantized activation (row-major E4M3 codes
    ``[M, K]`` and E8M0 exponents ``[M, K // 32]``) and a row-major MXFP8 weight:
    the product of the exact values in ``accumulate``, rounded once."""
    a = dequantize_mxfp8(codes, exponents, accumulate)
    w = dequantize_mxfp8(weight_codes, weight_exponents, accumulate)
    return (a @ w.T).to(out_dtype)


def mxfp8_linear_reference(
    x: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_exponents: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """The contract's ``x @ W^T`` for a bf16 / fp16 activation and a row-major
    (not preshuffled) MXFP8 weight."""
    return mxfp8_gemm_reference(
        *quantize_mxfp8(x), weight_codes, weight_exponents, out_dtype
    )


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Cosine similarity and relative L2 error of ``got`` against ``ref``."""
    g, r = got.float().flatten(), ref.float().flatten()
    return {
        "cos": torch.nn.functional.cosine_similarity(g, r, dim=0).item(),
        "rel_l2": ((g - r).norm() / r.norm()).item(),
    }


def swiglu_oai(
    gate_up: torch.Tensor, alpha: float, beta: float, limit: float
) -> torch.Tensor:
    """SwiGLU-OAI on the split layout ``[*, 2I]`` (gate = first half, up = second
    half), in FP32: ``gate = min(gate, limit)``, ``up = clamp(up, -limit, limit)``,
    ``gate * sigmoid(alpha * gate) * (up + beta)``."""
    i = gate_up.shape[-1] // 2
    gate = gate_up[..., :i].float().clamp(max=limit)
    up = gate_up[..., i:].float().clamp(-limit, limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


def mxfp8_gateup_swiglu_reference(
    x: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_exponents: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract of the fused gate_up row: E4M3 codes ``[M, I]`` and E8M0
    exponents ``[M, I // 32]`` (row-major) of ``H = bf16(swiglu_oai(bf16(x @ W^T)))``."""
    gate_up = mxfp8_linear_reference(x, weight_codes, weight_exponents)
    return quantize_mxfp8(swiglu_oai(gate_up, alpha, beta, limit).to(torch.bfloat16))


def swiglu_bf16_reference(gu,alpha=1.702,beta=1.0,limit=7.0,form='fp32'):
    """Model activation on materialized BF16 gate/up values, returning BF16."""
    gate,up=gu.chunk(2,-1)
    if form=='bf16_chain':
        gate=gate.clamp(max=limit);up=up.clamp(-limit,limit)
        return gate*torch.sigmoid(alpha*gate)*(up+beta)
    assert form=='fp32'
    gate=gate.float().clamp(max=limit);up=up.float().clamp(-limit,limit)
    return (gate*torch.sigmoid(alpha*gate)*(up+beta)).bfloat16()


def mxfp8_gateup_swiglu_bf16_reference(a,a_scale,w,w_scale,alpha=1.702,beta=1.0,limit=7.0,form='fp32'):
    """FP64 GEMM rounded to BF16, caller's SwiGLU form, BF16 output before down quantization.

    Operands use row-major reference layouts. The target form has FP32 activation
    math and one BF16 rounding; bf16_chain rounds every model intermediate.
    Tests require per-row cosine >=0.9999 versus this FP64-dot reference and
    raw BF16 equality versus activation of the incumbent materialized GEMM.
    """
    gu=mxfp8_gemm_reference(a,a_scale,w,w_scale,accumulate=torch.float64)
    return swiglu_bf16_reference(gu,alpha,beta,limit,form)
