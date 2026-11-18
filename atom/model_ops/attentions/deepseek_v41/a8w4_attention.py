# SPDX-License-Identifier: MIT
"""A8W4 attention over NVFP4 CSA and MXFP8 SWA source planes."""

import torch
from aiter.ops.mla_sparse_prefill import get_mla_sparse_prefill_v41_backend

from atom.utils import envs


def supports_a8w4_attention(query, softmax_scale):
    """Query before Q preparation; AITER owns all native backend eligibility.

    Describe the contiguous quantized planes produced below, not the BF16
    input. AITER's public V4.1 capability API is a required dependency.
    """
    mode = envs.ATOM_DSV41_A8W4_ATTN
    if mode == "off":
        return False
    return (
        get_mla_sparse_prefill_v41_backend(
            query.device,
            (query.shape[:-2].numel(), *query.shape[-2:]),
            unified_kv_dtype=torch.uint8,
            softmax_scale=softmax_scale,
            q_dtype=torch.float8_e4m3fn,
            q_scale_dtype=torch.float8_e8m0fnu,
            unified_kv_scale_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            kv_scale_dtype=torch.float8_e8m0fnu,
            layout="contiguous",
        )
        is not None
    )


def configured_a8w4_attention(config, device):
    """Check the target path for offline calibration compatibility.

    A one-token query tests whether this deployment can use A8W4 at all.
    Layer preparation still checks each live shape, including backend limits.
    This startup-only probe uses the same capability query as attention.
    """
    if config.kv_cache_dtype != "fp4" or envs.ATOM_DSV41_A8W4_ATTN == "off":
        return False
    model = config.hf_config
    tp = 1 if config.enable_dp_attention else config.tensor_parallel_size
    device = (
        torch.device("cuda", device)
        if isinstance(device, int)
        else torch.device(device)
    )
    query = torch.empty(
        (1, model.num_attention_heads // tp, model.head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    return supports_a8w4_attention(query, model.head_dim**-0.5)
