# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which MiniMax-M3 layers are sparse and which of them reuse the previous
layer's top-k. Pure config reads, so the startup checks in ``atom.config`` use
the same rule as the model without importing it (and AITER)."""

from transformers import PretrainedConfig


def sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, enabled in enumerate(freq) if enabled != 0}


def sparse_attention_layer_ordinals(config: PretrainedConfig) -> dict[int, int]:
    return {
        layer_id: ordinal
        for ordinal, layer_id in enumerate(sorted(sparse_attention_layer_ids(config)))
    }


def should_skip_index_topk(config: PretrainedConfig, layer_id: int) -> tuple[bool, int]:
    """(whether ``layer_id`` reuses the previous sparse layer's top-k, its
    sparse-layer ordinal or -1)."""
    sparse_ordinal = sparse_attention_layer_ordinals(config).get(layer_id, -1)
    if sparse_ordinal < 0:
        return False, sparse_ordinal
    if not getattr(config, "use_index_cache", False):
        return False, sparse_ordinal

    index_topk_freq = int(getattr(config, "index_topk_freq", 1) or 1)
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    if index_topk_pattern is not None:
        if 0 <= sparse_ordinal < len(index_topk_pattern):
            return index_topk_pattern[sparse_ordinal] == "S", sparse_ordinal
        return False, sparse_ordinal

    if index_topk_freq <= 0:
        raise ValueError("index_topk_freq must be a positive integer")
    if index_topk_freq == 1:
        return False, sparse_ordinal

    # MiniMax-M3 schedules sharing by sparse-layer ordinal, not absolute layer id.
    offset = int(getattr(config, "index_skip_topk_offset", 0))
    return max(sparse_ordinal - offset, 0) % index_topk_freq != 0, sparse_ordinal


def reuses_index_topk(config: PretrainedConfig) -> bool:
    """Whether any sparse layer reuses the previous layer's top-k."""
    return any(
        should_skip_index_topk(config, layer_id)[0]
        for layer_id in sparse_attention_layer_ids(config)
    )
