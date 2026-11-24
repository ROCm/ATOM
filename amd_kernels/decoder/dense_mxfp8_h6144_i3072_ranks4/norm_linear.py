# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Plain Gemma RMSNorm -> MXFP8 codes in the consumer GEMM's layout -> the MXFP8 GEMM, one call (the context K/V
projection of a draft: ``hidden_norm`` + the K/V rows of every layer's qkv)."""

import torch

from amd_kernels.warm_up import entry

from .mlp import _gemm_codes

H = 6144
# the norm's tiers (row-major / tiled codes, the four-wave form above 512 rows) and the GEMM rows of the decode and
# prefill regimes (the linear integration's own warm-up compiles the GEMM's table up to its token budget)
WARMUP_ROWS = tuple(range(1, 257)) + (288, 320, 384, 448, 512, 513, 640, 768, 1024, 1025, 1536, 2048, 2049, 3072,
                                      4096, 6144, 8192, 12288, 16384, 24576, 32768)


def _norm_linear(x: torch.Tensor, norm_weight: torch.Tensor, eps: float, w: torch.Tensor,
                 ws: torch.Tensor) -> torch.Tensor:
    """[m, N] bf16 = MXFP8 GEMM(quant(gemma_rmsnorm(x)), W) for ``x`` [m, 6144] bf16 and a preshuffled MXFP8 weight
    [N, 6144]: the norm writes the codes the GEMM's table row reads (tiled or row-major). The quant reads the bf16
    norm output (``gemma_rmsnorm_bf16_mxfp8_h6144``'s rule). The first call compiles the norm's and the GEMM's
    variants."""
    from amd_kernels.gemm.mxfp8_common import gemm as G
    from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8

    m = x.shape[0]
    n, k = w.shape
    assert k == H
    if m == 0:
        return torch.empty((0, n), dtype=torch.bfloat16, device=x.device)
    tile = G.prequant_code_tile_rows(m, n, k, torch.bfloat16)
    q, s = gemma_rmsnorm_mxfp8(x, norm_weight, eps, tile)
    return _gemm_codes(q, s, w, ws, m)


def _inputs(mode, n, x, norm_weight, eps, w, ws):
    return (torch.zeros((n, H), dtype=torch.bfloat16, device=x.device), norm_weight, eps, w, ws), {}


gemma_rmsnorm_mxfp8_linear = entry("decoder/dense_mxfp8_h6144_i3072_ranks4:norm_linear",
                                   mode=lambda x, norm_weight, eps, w, ws: tuple(w.shape),
                                   sizes=lambda mode: WARMUP_ROWS, inputs=_inputs)(_norm_linear)
