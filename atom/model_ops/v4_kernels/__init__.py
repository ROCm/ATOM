# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 attention backend Triton kernels.

These kernels replace the per-seq Python state-write logic in
`atom/models/deepseek_v4.py` (PR-A: kill .item() / unlock CUDAGraph). All
take batched tensors (positions, slot_per_token, cu_seqlens_q) — nothing is
derived from device data via `.item()`.
"""

import logging
from typing import Any

from aiter.jit.utils.chip_info import get_gfx

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
from atom.model_ops.v4_kernels.indexer_weights import (
    scale_indexer_weights,
)
from atom.model_ops.v4_kernels.inverse_rope import inverse_rope_inplace
from atom.model_ops.v4_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_decode_reference,
)
from atom.model_ops.v4_kernels.paged_decode_indices import (
    build_v4_paged_decode_indptr,
    build_v4_paged_decode_indptr_reference,
    hca_compress_paged_offsets,
    write_v4_paged_decode_indices,
    write_v4_paged_decode_indices_reference,
)
from atom.model_ops.v4_kernels.paged_prefill import (
    sparse_attn_v4_paged_prefill,
    sparse_attn_v4_paged_prefill_reference,
)
from atom.model_ops.v4_kernels.paged_prefill_indices import (
    write_v4_paged_prefill_indices,
    write_v4_paged_prefill_indices_reference,
)
from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import (
    QKNormRopeOut,
    qk_norm_rope_maybe_quant,
    qk_norm_rope_maybe_quant_fp8_2buff,
    qk_norm_rope_maybe_quant_reference,
)
from atom.model_ops.v4_kernels.state_writes import (
    swa_write,
    swa_write_2buff_prepacked,
    update_compressor_states,
)

__all__ = [
    "FP4_MQA_BLOCK_K",
    "FP4_MQA_PARALLEL_UNIT_NUM",
    "CompressPlan",
    "QKNormRopeOut",
    "build_v4_paged_decode_indptr",
    "build_v4_paged_decode_indptr_reference",
    "csa_translate_pack",
    "csa_translate_pack_reference",
    "fp4_indexer_enabled",
    "fused_compress_attn",
    "fused_compress_attn_reference",
    "hca_compress_paged_offsets",
    "inverse_rope_inplace",
    "make_compress_plans",
    "qk_norm_rope_maybe_quant",
    "qk_norm_rope_maybe_quant_fp8_2buff",
    "qk_norm_rope_maybe_quant_reference",
    "scale_indexer_weights",
    "sparse_attn_v4_paged_decode",
    "sparse_attn_v4_paged_decode_reference",
    "sparse_attn_v4_paged_prefill",
    "sparse_attn_v4_paged_prefill_reference",
    "swa_write",
    "swa_write_2buff_prepacked",
    "update_compressor_states",
    "write_v4_paged_decode_indices",
    "write_v4_paged_decode_indices_reference",
    "write_v4_paged_prefill_indices",
    "write_v4_paged_prefill_indices_reference",
]

logger = logging.getLogger("atom")

# FP4 indexer persistent-grid schedule params, shared by the decode
# (`pa_mqa_logits_fp4`) and prefill (`pa_mqa_logits_fp4_prefill`) kernels.
# The attention metadata builder precomputes each path's cta_info with these
# and the scorer passes the matching block_k, so layout and grid agree. They
# live here (rather than in either caller) because both the builder and the
# model-side scorer must use the SAME values. Mirrors the kernel defaults.
FP4_MQA_PARALLEL_UNIT_NUM = 512
FP4_MQA_BLOCK_K = 256


def fp4_indexer_enabled(index_cache_dtype: Any, *, warn: bool = False) -> bool:
    """Is the FP4 CSA indexer active? Single source of truth for the predicate.

    Two call sites must reach the SAME verdict, and neither can be dropped:

    * `DeepseekV4AttentionMetadataBuilder.__init__` — authoritative. Picks the
      cache-pool layout and re-asserts the flag onto each Indexer in
      `build_kv_cache_tensor`.
    * `Indexer.__init__` — must already be correct BEFORE that re-assert.
      `model_runner._maybe_warmup()` traces the graphed `_attn_pre`/`forward_pre`
      piece before `allocate_kv_cache()` -> `build_kv_cache_tensor()` runs, so an
      Indexer that defaulted to False would bake the FP8 branch (`q_scale=None`)
      into the graph while the eager `indexer_score_topk` later took the FP4
      branch. It is also the ONLY setter under the vLLM / SGLang plugins, which
      never call `build_kv_cache_tensor`.

    Keeping the predicate in one place is what stops the two from drifting —
    a divergence is silent at startup and only surfaces as a graph/eager dtype
    mismatch. Lives here rather than in either caller for the same reason as
    `FP4_MQA_*` above.

    gfx942 keeps the FP8 indexer because its FP4 path is unsupported. Pass
    `warn=True` from the builder only — it runs once, while `Indexer.__init__`
    runs per CSA layer and would repeat the message.
    """
    if index_cache_dtype != "fp4":
        return False
    gfx = get_gfx()
    if gfx == "gfx942":
        if warn:
            logger.warning(
                "The DeepSeek-V4 FP4 indexer is unsupported on %r. Falling "
                "back to the FP8 indexer.",
                gfx,
            )
        return False
    return True
