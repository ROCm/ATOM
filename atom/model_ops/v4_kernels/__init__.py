# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 attention backend Triton kernels.

These kernels replace the per-seq Python state-write logic in
`atom/models/deepseek_v4.py` (PR-A: kill .item() / unlock CUDAGraph). All
take batched tensors (positions, slot_per_token, cu_seqlens_q) — nothing is
derived from device data via `.item()`.
"""

from atom.model_ops.v4_kernels.compress_plan import (
    CompressPlan,
    make_compress_plans,
)
from atom.model_ops.v4_kernels.csa_translate_pack import (
    csa_translate_pack,
    csa_translate_pack_reference,
)
from atom.model_ops.v4_kernels.fused_compress import (
    fused_compress_attn,
    fused_compress_attn_reference,
)
from atom.model_ops.v4_kernels.indexer_weights import scale_indexer_weights
from atom.model_ops.v4_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_decode_reference,
)
from atom.model_ops.v4_kernels.paged_prefill import (
    sparse_attn_v4_paged_prefill,
    sparse_attn_v4_paged_prefill_reference,
)
from atom.model_ops.v4_kernels.inverse_rope import inverse_rope_inplace
from atom.model_ops.v4_kernels.state_writes import update_compressor_states, swa_write

__all__ = [
    "update_compressor_states",
    "swa_write",
    "fused_compress_attn",
    "fused_compress_attn_reference",
    "sparse_attn_v4_paged_decode",
    "sparse_attn_v4_paged_decode_reference",
    "sparse_attn_v4_paged_prefill",
    "sparse_attn_v4_paged_prefill_reference",
    "csa_translate_pack",
    "csa_translate_pack_reference",
    "CompressPlan",
    "make_compress_plans",
    "inverse_rope_inplace",
    "scale_indexer_weights",
]
