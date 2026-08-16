# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Pure MLA KV / index-cache layout helpers (no AITER import).

Kept free of GPU / AITER deps so non-GPU unit tests can lock the compact
IndexShare layout without pulling in the full attention stack.
"""

_INDEX_CACHE_SCALE_BYTES = 4
_INDEX_CACHE_ALIGNMENT_BYTES = 16


def _mla_kv_cache_dim(hf_config) -> int:
    """Return the packed MLA latent width (compressed KV plus RoPE lane)."""
    return int(hf_config.kv_lora_rank) + int(hf_config.qk_rope_head_dim)


def _aligned_index_cache_dim(index_head_dim: int) -> int:
    """Return bytes per packed FP8 index-key row.

    AITER stores ``index_head_dim`` FP8 key bytes followed by one inline FP32
    scale (4 bytes) in a byte tensor. The row is padded to 16 bytes because the
    sparse gather/Inductor path requires aligned row strides. Changing the
    scale dtype requires a coordinated AITER kernel ABI change.
    """
    packed_bytes = int(index_head_dim) + _INDEX_CACHE_SCALE_BYTES
    return (
        (packed_bytes + _INDEX_CACHE_ALIGNMENT_BYTES - 1)
        // _INDEX_CACHE_ALIGNMENT_BYTES
        * _INDEX_CACHE_ALIGNMENT_BYTES
    )


def _global_index_cache_layer_ids(
    indexer_types,
    num_hidden_layers: int,
    num_draft_layers: int,
) -> tuple[int, ...]:
    """Return global layers that own an index-key cache slice.

    GLM-5.2 ``shared`` layers reuse a preceding full layer's temporary top-k
    positions and do not construct an indexer, so their index-key cache slices
    are dead. Other sparse MLA models have no ``indexer_types`` schedule and
    retain the existing one-slice-per-layer layout.
    """
    target_layer_ids = range(num_hidden_layers)
    if indexer_types is not None:
        target_layer_ids = (
            layer_id
            for layer_id in target_layer_ids
            # MTP layers are not included in indexer_types. Only the GLM
            # "shared" value means no indexer module/cache owner; DeepSeek's
            # index_topk_pattern "S" has different semantics and keeps a cache.
            if layer_id >= len(indexer_types) or indexer_types[layer_id] != "shared"
        )
    return tuple(target_layer_ids) + tuple(
        range(num_hidden_layers, num_hidden_layers + num_draft_layers)
    )
