# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The router GEMM with the MXFP4 + MXFP8 input quant workgroups in its launch (``router_gemm_nblocks`` ``quant``),
then ``router_topk_sorted``'s top-4: the chains, sum order, tie order and quantizers of the separate launches. The
quant workgroups write disjoint outputs and need no synchronization."""
import struct

import torch

from .router_gemm import nblocks_config, _run_compiled
from .router_gemm_nblocks import _get_kernel as _get_router
from .router_topk_sorted import _get_kernel as _get_topk

H, E, K = 6144, 128, 4


# (first row, last row, quant 1 appended / 2 prepended): router + top-k + sort chain -0.042..-0.053 us at 58-64 rows
QUANT_TIERS = ((58, 64, 1), (65, 128, 2), (193, 256, 1))


def router_quant_placement(m: int) -> int:
    """0 (the quant stays in the top-k launch), 1 appended, 2 prepended."""
    for lo, hi, placement in QUANT_TIERS:
        if lo <= m <= hi:
            return placement
    return 0


def router_dual_quant_for(m: int) -> bool:
    """Whether ``m`` takes this launch pair (a ``QUANT_TIERS`` row)."""
    return router_quant_placement(m) != 0


def router_gemm_topk_mxfp4_mxfp8_quant(x, router_weight, correction_bias, routed_scaling_factor):
    """-> ((MXFP4 codes, scales), (MXFP8 codes, scales), topk weights, topk ids, router partials [m, split, 128])."""
    m = x.shape[0]
    assert router_dual_quant_for(m)
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] == H
    if (router_weight.shape != (E, H) or router_weight.dtype != torch.float32
            or not router_weight.is_contiguous()):
        raise ValueError("dual input quant requires the split-K router range")
    assert correction_bias.dtype == torch.float32 and correction_bias.is_contiguous()
    config = nblocks_config(m)
    split = config[0]
    token_major = split == 48 or 65 <= m <= 128
    partials = torch.empty((split, m, E), dtype=torch.float32, device=x.device)
    ids = torch.empty((m, K), dtype=torch.int32, device=x.device)
    weights = torch.empty((m, K), dtype=torch.float32, device=x.device)
    q = torch.empty((m, H // 2), dtype=torch.uint8, device=x.device)
    s = torch.empty((m, H // 32), dtype=torch.uint8, device=x.device)
    q8 = torch.empty((m, H), dtype=torch.uint8, device=x.device)
    s8 = torch.empty((m, H // 32), dtype=torch.uint8, device=x.device)
    stream = torch.cuda.current_stream(x.device)
    _run_compiled(_get_router(*config, router_quant_placement(m), False, token_major),
                  x.data_ptr(), router_weight.data_ptr(), partials.data_ptr(), m, x.stride(0),
                  q.data_ptr(), s.data_ptr(), q8.data_ptr(), s8.data_ptr(), stream)
    _run_compiled((_get_topk(E, K, split, True) if token_major else _get_topk(E, K, split)), partials.data_ptr(), correction_bias.data_ptr(),
                  weights.data_ptr(), ids.data_ptr(), partials.numel() * 4, m, split * E if token_major else E, E if token_major else m * E, K,
                  struct.unpack("<i", struct.pack("<f", routed_scaling_factor))[0], stream)
    return (q, s), (q8, s8), weights, ids, partials.view(m, split, E) if token_major else partials.permute(1, 0, 2)
