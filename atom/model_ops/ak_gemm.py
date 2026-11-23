# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""amd-kernels (inferact/amd-kernels) MXFP8 GEMMs for the native MXFP8 1x32 linears of MiniMax-M3.1 at TP4.

``ATOM_AK_MXFP8_GEMM=1`` (default off; needs the amd-kernels checkout on PYTHONPATH): a preshuffled MXFP8 1x32 linear
of one of the shapes below, called without a producer's ``x_scale``, runs ``gemm.mxfp8_common.mxfp8_linear`` (its own
activation quantizer + GEMM) at the row counts where it beat ``quantize_fp8`` + AITER's GEMM on one MI355X
(graph replay, cold weights), else today's path. The weight stays as ATOM stores it ((16, 16) preshuffled codes);
the scales get a copy in AITER's shuffled layout at load (a permutation, ~1.3 MB per sparse layer). Outputs equal
today's bit for bit at most row counts, else within rel L2 5e-5 (the split-K order of some decode tiles).
"""

import functools
import os

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

ENABLED = os.environ.get("ATOM_AK_MXFP8_GEMM", "0") == "1"
MAX_TOKENS = int(os.environ.get("ATOM_AK_MXFP8_GEMM_MAX_TOKENS", "32768"))  # the variants compiled at load

# (N, K) -> rows where amd-kernels wins (akSurvey/res/gemm1.jsonl: qkv, o_proj, shared gate_up / down, dense
# gate_up / down at TP4)
_ROWS = {
    (2560, 6144): lambda m: m <= 16 or m >= 192,
    (6144, 2048): lambda m: True,
    (1536, 6144): lambda m: m <= 384 or m >= 6144,
    (6144, 768): lambda m: m <= 256,
    (6144, 6144): lambda m: True,
    (6144, 3072): lambda m: True,
}
# p1: SP prefill o_proj_full (N6144 K8192, gemm/mxfp8_n6144_k8192), only with ATOM_P1_SP_OPROJ_AK=1 (the shape
# exists only when ATOM_M3_SP_PREFILL builds o_proj_full)
if os.environ.get("ATOM_P1_SP_OPROJ_AK", "0") == "1":
    _ROWS[(6144, 8192)] = lambda m: True


def shape_supported(n: int, k: int) -> bool:
    return (n, k) in _ROWS


@functools.cache
def _gemm():
    from amd_kernels.gemm.mxfp8_common import gemm

    return gemm


def prepare(weight: torch.Tensor, weight_scale: torch.Tensor, max_num_tokens: int) -> torch.Tensor:
    """At load, after ATOM's (16, 16) weight shuffle: the scales in AITER's shuffled layout (a new tensor; ATOM's
    compact scales stay for today's path) and every amd-kernels variant of this shape compiled up to
    ``max_num_tokens`` rows (graph capture must not compile)."""
    from aiter.ops.shuffle import shuffle_scale_a16w4

    s = shuffle_scale_a16w4(weight_scale.view(torch.float8_e8m0fnu), 1, False)
    _gemm().warmup_mxfp8_gemm_weight(weight, s, max_num_tokens, torch.bfloat16)
    return s


def _fake(x, weight, weight_scale, ak_weight_scale):
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16, device=x.device)


@torch_compile_guard(gen_fake=_fake, mutates_args=[])
def ak_mxfp8_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, ak_weight_scale: torch.Tensor
) -> torch.Tensor:
    n, k = weight.shape
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if _ROWS[(n, k)](m):
        y = _gemm().mxfp8_linear(x2, weight, ak_weight_scale)
    else:
        from atom.model_ops.blockscale import native_quant_linear

        y = native_quant_linear(x2, weight, weight_scale, weight_group_rows=1, weight_preshuffled=True,
                                dtype=torch.bfloat16)
    return y.view(*lead, n)


def ak_mxfp8_gemm_preshuffled(
    x: torch.Tensor, x_scale: torch.Tensor, weight: torch.Tensor, ak_weight_scale: torch.Tensor
) -> torch.Tensor:
    """p1 (eager SP prefill only): row-major MXFP8 codes ``x`` [M, K] whose e8m0 scales ``x_scale`` are already in
    AITER's shuffled layout (>= M rows, 32-row blocks) -> ``mxfp8_gemm`` without a re-quant. Same kernels and
    bytes as ``mxfp8_linear`` on the bf16 rows those codes were quantized from (its prefill rows quantize with
    ``mxfp8_e4m3_quantize_aiter`` and call ``mxfp8_gemm`` with preshuffled scales)."""
    m, k = x.shape
    s = x_scale[: (m + 31) // 32 * 32]
    return _gemm().mxfp8_gemm(x, s, weight, ak_weight_scale, torch.bfloat16, activation_scale_preshuffled=True)
