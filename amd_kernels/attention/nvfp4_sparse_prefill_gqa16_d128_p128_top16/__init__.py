# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse prefill attention: 16 query heads per KV head, head dim 128, 128-token
pages, 16 selected pages per token (FlyDSL, gfx950)."""

from .host import (
    nvfp4_sparse_prefill,
    nvfp4_sparse_prefill_bf16,
    nvfp4_sparse_prefill_fp8,
    nvfp4_sparse_prefill_fp16,
    supports_nvfp4_sparse_prefill,
)

__all__ = [
    "nvfp4_sparse_prefill",
    "nvfp4_sparse_prefill_bf16",
    "nvfp4_sparse_prefill_fp16",
    "nvfp4_sparse_prefill_fp8",
    "supports_nvfp4_sparse_prefill",
]
