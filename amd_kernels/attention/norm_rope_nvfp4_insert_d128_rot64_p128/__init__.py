# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""q/k/index norm + RoPE with the NVFP4 K/V and index-K inserts in one launch, head
dim 128, 128-token pages (FlyDSL, gfx950)."""

from .host import insert_blocker, norm_rope_nvfp4_insert, splitk_reduce_norm_rope_nvfp4_insert, warmup_prefill_kernels
from .qkv import mxfp8_gemm_qkv_norm_rope_nvfp4_insert, qkv_norm_rope_nvfp4_insert

__all__ = [
    "insert_blocker",
    "mxfp8_gemm_qkv_norm_rope_nvfp4_insert",
    "norm_rope_nvfp4_insert",
    "qkv_norm_rope_nvfp4_insert",
    "splitk_reduce_norm_rope_nvfp4_insert",
    "warmup_prefill_kernels",
]
