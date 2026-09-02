# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA cache/workspace geometry, derived from the HF config alone.

Pure integer arithmetic on config fields: no torch, no AITER. That is
deliberate -- `aiter_mla` imports AITER at module scope, so anything living
there can only be exercised on a machine with the kernels, and these are
exactly the contracts worth pinning on a plain CI runner. Same reasoning as
the lazy `atom/model_ops/__init__.py` resolution.
"""


def mla_kv_entry_dim(hf_config) -> int:
    """Width of one MLA KV cache entry.

    Normally ``kv_lora_rank + qk_rope_head_dim``. A NoPE model (GLM-5.3-Flash,
    ``qk_rope_head_dim == 0``) materializes the rope block at a padded width and
    holds it at zero so the standard 576-wide MLA kernels apply unchanged; it
    declares that padded width as ``mla_kv_entry_dim``. Sizing the cache from
    the raw config instead would allocate 512-wide rows under a 576-wide write.
    """
    declared = getattr(hf_config, "mla_kv_entry_dim", None)
    if declared:
        return int(declared)
    return hf_config.kv_lora_rank + hf_config.qk_rope_head_dim


def mla_qk_head_dim(hf_config) -> int:
    """Per-head q/k width the MLA kernels and their workspaces are built for.

    The rope block's width has to come from `mla_kv_entry_dim`, not from the
    raw config. A NoPE model leaves ``qk_rope_head_dim`` at its true 0 so the
    INDEXER stays NoPE, and widens the block to a zero pad on the MLA side
    only, so the raw sum understates the MLA width by exactly the pad.

    Getting this wrong is not a size warning, it is a compile error one kernel
    deep: `gather_kv_b_proj` takes the rope width from the destination buffer
    (``qk_nope_pe_dim = k_prefix.shape[-1]``) and the nope width from the
    kv_b_proj weight, so a buffer sized 256 against a 256-wide nope half makes
    ``KV_PeDim = 0`` and Triton rejects the resulting ``tl.arange(0, 0)``.
    """
    rope_dim = mla_kv_entry_dim(hf_config) - hf_config.kv_lora_rank
    return hf_config.qk_nope_head_dim + rope_dim


def aligned_index_cache_dim(hf_config) -> int:
    """Indexer key plus fp32 scale, padded to a 16-byte row."""
    index_dim = hf_config.index_head_dim + 4
    return ((index_dim + 15) // 16) * 16
