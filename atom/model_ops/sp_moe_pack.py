# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lossless packing of prequantized MoE inputs and local routing metadata.

Each rank contributes four contiguous byte segments. A single all-gather
retains rank-major segment order; unpacking places each field in global token
order. Integer loads/stores preserve FP4, E8M0 and FP32 routing bits exactly.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_routes_kernel(W, I, P, NR: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    weights = tl.load(W + x, x < NR, other=0)
    ids = tl.load(I + x, x < NR, other=0)
    tl.store(P + x, weights, x < NR)
    tl.store(P + NR + x, ids, x < NR)


@triton.jit
def _unpack_kernel(
    P,
    Q,
    S,
    W,
    I,
    NQ: tl.constexpr,
    NS: tl.constexpr,
    NR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = tl.program_id(1)
    total = NQ + NS + 2 * NR
    value = tl.load(P + rank * total + x, x < total, other=0)
    tl.store(Q + rank * NQ + x, value, x < NQ)
    tl.store(S + rank * NS + x - NQ, value, (x >= NQ) & (x < NQ + NS))
    tl.store(W + rank * NR + x - NQ - NS, value, (x >= NQ + NS) & (x < NQ + NS + NR))
    tl.store(I + rank * NR + x - NQ - NS - NR, value, (x >= NQ + NS + NR) & (x < total))


def quantize_pack_moe_inputs(x, weights, ids):
    """Use the existing HIP quantizer to write directly into packed segments.

    Only routing metadata needs a separate packing kernel. The FP4 and scale
    calculations are the same entry point and parameters as get_hip_quant's
    per-1x32, non-shuffled quantizer.
    """
    from aiter import dtypes
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    tokens, hidden = x.shape
    assert hidden % 256 == 0 and x.dtype == torch.bfloat16
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert weights.shape == ids.shape and weights.shape[0] == tokens
    assert weights.is_contiguous() and ids.is_contiguous()
    nq, ns, nr = tokens * hidden // 2, tokens * hidden // 32, weights.numel() * 4
    storage = torch.empty(nq + ns + 2 * nr, dtype=torch.uint8, device=x.device)
    quantized = storage[:nq].view(dtypes.fp4x2).view(tokens, hidden // 2)
    scale = storage[nq : nq + ns].view(dtypes.fp8_e8m0).view(tokens, hidden // 32)
    dynamic_per_group_scaled_quant(quantized, x, scale, 32, shuffle_scale=False)
    _pack_routes_kernel[(triton.cdiv(weights.numel(), 1024),)](
        weights.view(torch.int32),
        ids,
        storage[nq + ns :].view(torch.int32),
        weights.numel(),
        1024,
    )
    return storage.view(torch.bfloat16).view(tokens, -1), quantized, scale


def unpack_moe_inputs(packed, quantized, scale, weights, ids, world):
    """Recover four contiguous global fields from a gathered packed buffer."""
    prototypes = (quantized, scale, weights, ids)
    outputs = tuple(x.new_empty((x.shape[0] * world, x.shape[1])) for x in prototypes)
    nq, ns, nr = (x.numel() * x.element_size() // 4 for x in prototypes[:3])
    total = (nq + ns + 2 * nr) * world
    assert (
        packed.is_contiguous() and packed.numel() * packed.element_size() == total * 4
    )
    _unpack_kernel[(triton.cdiv(total // world, 1024), world)](
        packed.view(torch.int32),
        *(x.view(torch.int32) for x in outputs),
        nq,
        ns,
        nr,
        1024,
    )
    return outputs
