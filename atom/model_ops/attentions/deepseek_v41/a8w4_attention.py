# SPDX-License-Identifier: MIT
"""A8W4 attention over NVFP4 CSA and MXFP8 SWA source planes."""

import torch

from atom.utils import envs
from atom.utils.decorators import mark_trace

try:
    from aiter.ops.mla_sparse_prefill import (
        get_mla_sparse_prefill_v41_backend,
        mla_sparse_prefill_v41,
    )
except ImportError:
    get_mla_sparse_prefill_v41_backend = None
    mla_sparse_prefill_v41 = None


def supports_a8w4_attention(query, softmax_scale):
    """Query before Q preparation; AITER owns all native backend eligibility.

    Describe the contiguous quantized planes produced below, not the BF16
    input. Older AITER builds without the query retain the BF16 adapter.
    """
    mode = envs.ATOM_DSV41_A8W4_ATTN
    if mode not in ("auto", "off"):
        raise ValueError("ATOM_DSV41_A8W4_ATTN must be auto or off")
    if mode == "off" or get_mla_sparse_prefill_v41_backend is None:
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


@mark_trace
def sparse_attn_v41_a8w4_paged_2src(
    query,
    pool,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    *,
    out,
    num_kv_splits=None,
):
    """Both backends consume the same planes and per-source int32 row IDs.

    Prefill's SWA source holds batch-local history followed by current KV;
    decode reads the ring directly. No cache row is dequantized here. AITER
    owns the backend's split heuristic and partial-workspace layout, as it
    does for the A8W8 wrapper.
    """
    values, scales = query
    data, scale = pool
    window, window_scale = kv
    return mla_sparse_prefill_v41(
        values,
        scales,
        data,
        scale.view(torch.float8_e4m3fn),
        kv_indices_prefix,
        kv_indptr_prefix,
        window.view(torch.float8_e4m3fn),
        window_scale.view(torch.float8_e8m0fnu),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=out,
        num_kv_splits=num_kv_splits,
    )
