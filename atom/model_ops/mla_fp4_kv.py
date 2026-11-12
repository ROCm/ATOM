# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Packed FP4 MLA KV cache: the ``--kv_cache_dtype`` names and row geometry.

The formats store E2M1 values packed two per byte in a ``torch.uint8`` cache
and differ in their scales and rope (see ``aiter.ops.triton.utils.fp4_kv``):

- ``mxfp4``: one E8M0 scale per 32 values (OCP MX).
- ``mxfp4_b16``: one E8M0 scale per 16 values.
- ``mxfp4_fp8rope``: the latent as ``mxfp4``, rope as FP8 E4M3.
- ``nvfp4``: one E4M3 scale per 16 values.

A row is the packed bytes of one token, so it is narrower than
``kv_lora_rank + qk_rope_head_dim``: 306 bytes for MXFP4, 324 for MXFP4 with
16-value blocks and for NVFP4, 336 with FP8 rope at DeepSeek's 512 + 64. The
query stays FP8.
"""

import torch
from aiter import dtypes
from aiter.ops.triton.utils.fp4_kv import fp4_kv_row_bytes

FP4_KV_CACHE_DTYPES = ("mxfp4", "mxfp4_b16", "mxfp4_fp8rope", "nvfp4")

# The FP4 decode reads 128-token shuffled blocks: its scale shuffle tiles 32
# lanes by 4 token groups.
FP4_KV_BLOCK_SIZE = 128


def is_fp4_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in FP4_KV_CACHE_DTYPES


def mla_kv_cache_torch_dtype(kv_cache_dtype: str) -> torch.dtype:
    """Element dtype of the MLA cache tensor."""
    if is_fp4_kv_cache(kv_cache_dtype):
        return torch.uint8
    return dtypes.d_dtypes[kv_cache_dtype]


def mla_query_torch_dtype(kv_cache_dtype: str) -> torch.dtype:
    """Dtype the decode query is written in: FP8 against an FP4 cache."""
    if is_fp4_kv_cache(kv_cache_dtype):
        return dtypes.fp8
    return dtypes.d_dtypes[kv_cache_dtype]


def mla_kv_cache_row_width(
    entry_dim: int, qk_rope_head_dim: int, kv_cache_dtype: str
) -> int:
    """Elements per token row of the cache tensor: ``entry_dim`` values, or its
    packed byte count for an FP4 cache."""
    if is_fp4_kv_cache(kv_cache_dtype):
        return fp4_kv_row_bytes(
            entry_dim - qk_rope_head_dim, qk_rope_head_dim, kv_cache_dtype
        )
    return entry_dim
