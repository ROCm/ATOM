# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of the dense decoder block's calls: compositions of the per-kernel references.

- ``mxfp8_swiglu_oai_mlp``: ``a`` = the dequantized gate_up input codes; ``h = bf16(a @ Wgu^T)`` (fp32 accumulate);
  ``act`` = SwiGLU-OAI of ``h`` in the caller's form (``bf16_chain``: each op rounded to bf16, the uncompiled aten
  chain; ``fp32``: one rounding); ``out = bf16(quant(act) @ Wd^T)`` (MXFP8 quant per 32 columns, e8m0 scales).
- ``gemma_rmsnorm_mxfp8_linear``: ``bf16(quant(bf16(gemma_rmsnorm(x))) @ W^T)``.
- ``DenseDecoderRanks4``: its kernels' own contracts (``bf16_concat_linear_n1536_k30720``, ``vocab_gather_h6144`` +
  ``vocab_gather_gemma_mxfp8_h6144``, ``allreduce_gemma_bf16_mxfp8_h6144``); it only picks each consumer's code
  layout and the fc's 2048-row pieces. ``forward`` and ``store_context_kv`` are the sequences of these calls (and
  ``norm_rope_fp8_insert_d128_rot64``'s multi-layer insert) they replace, bit for bit.

Tolerance: cosine ~1 against these references (the GEMMs' fp32 K order; the fused 1-8-row gate_up rounds each dot
product to bf16 like the GEMM); the tests compare every call with the per-kernel chain bit for bit.
"""

import torch

from amd_kernels.gemm.mxfp8_common.reference import dequantize_mxfp8, quantize_mxfp8
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144.reference import gemma_rmsnorm_reference


def _swiglu(h, alpha, beta, limit, form):
    i = h.shape[-1] // 2
    if form == "bf16_chain":
        gate = torch.clamp(h[..., :i], max=limit)
        up = torch.clamp(h[..., i:], min=-limit, max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up + beta)
    gate = torch.clamp(h[..., :i].float(), max=limit)
    up = torch.clamp(h[..., i:].float(), min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).bfloat16()


def mlp_reference(y, gate_up_codes, gate_up_exp, down_codes, down_exp, alpha, beta, limit, form="bf16_chain"):
    """``y`` bf16 [m, 6144] (the gate_up input before its quant), row-major MXFP8 weights -> bf16 [m, 6144]."""
    a = dequantize_mxfp8(*quantize_mxfp8(y))
    h = (a @ dequantize_mxfp8(gate_up_codes, gate_up_exp).T).bfloat16()
    act = _swiglu(h, alpha, beta, limit, form)
    d = dequantize_mxfp8(*quantize_mxfp8(act))
    return (d @ dequantize_mxfp8(down_codes, down_exp).T).bfloat16()


def norm_linear_reference(x, norm_weight, eps, codes, exp):
    a = dequantize_mxfp8(*quantize_mxfp8(gemma_rmsnorm_reference(x, norm_weight, eps)))
    return (a @ dequantize_mxfp8(codes, exp).T).bfloat16()
