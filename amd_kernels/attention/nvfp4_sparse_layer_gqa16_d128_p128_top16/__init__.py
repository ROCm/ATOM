# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One NVFP4 sparse attention layer (16 query heads per KV head, head dim 128, 128-token pages, top-16): its insert,
decode and prefill entries at the served compute dtypes, the decode indexer and the warm-up of every variant."""

from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    mxfp8_gemm_qkv_norm_rope_nvfp4_insert,
    qkv_norm_rope_nvfp4_insert,
)

from .layer import (
    DECODE_COMPUTE,
    PREFILL_COMPUTE,
    PREFILL_LAZY,
    QUANT_OUT_MIN_ROWS,
    index_sparse_decode,
    init,
    quant_out_buffers,
    sparse_decode,
    sparse_prefill,
)

__all__ = [
    "DECODE_COMPUTE",
    "PREFILL_COMPUTE",
    "PREFILL_LAZY",
    "QUANT_OUT_MIN_ROWS",
    "index_sparse_decode",
    "init",
    "mxfp8_gemm_qkv_norm_rope_nvfp4_insert",
    "qkv_norm_rope_nvfp4_insert",
    "quant_out_buffers",
    "sparse_decode",
    "sparse_prefill",
]
