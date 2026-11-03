# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which layers of a DSA model reuse an earlier layer's index top-k.

The one classification the model (which layers run their indexer), the MLA
layers (which ones own the work plan built from a selection) and the runner
(which pipeline stage hands its selection to the next) all follow.
"""

from transformers import PretrainedConfig


def skips_index_topk(config: PretrainedConfig, layer_id: int) -> bool:
    """Whether layer `layer_id` reuses a prior layer's top-k instead of
    selecting its own."""
    if not getattr(config, "use_index_cache", False):
        # IndexShare (e.g. GLM-5.2): index_topk_freq > 1 shares the indexer across
        # layers, so enable the cache even if the config omits the flag; otherwise
        # there is nothing to skip.
        if int(getattr(config, "index_topk_freq", 1)) > 1:
            config.use_index_cache = True
        else:
            return False

    # GLM-5.2 MTP layer (index >= num_hidden_layers): the MTP block ships its
    # OWN indexer weights and computes its own top-k for the drafted position,
    # so do not skip it. `index_share_for_mtp_iteration` only concerns sharing
    # across MULTIPLE MTP draft steps (num_speculative_tokens>1); it does NOT
    # mean the MTP reuses the target model's index. Matches vLLM upstream and
    # the ATOM sglang plugin, which both run the MTP indexer independently.
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_id >= num_hidden_layers:
        return False

    # GLM-5.2 IndexShare: per-layer schedule, "shared" reuses the prior "full"
    # layer's topk. Authoritative when present; else fall back to pattern/freq.
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        return (
            0 <= layer_id < len(indexer_types) and indexer_types[layer_id] == "shared"
        )

    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    if index_topk_pattern is not None:
        return (
            0 <= layer_id < len(index_topk_pattern)
            and index_topk_pattern[layer_id] == "S"
        )

    index_topk_freq = int(getattr(config, "index_topk_freq", 1))
    if index_topk_freq <= 0:
        raise ValueError("index_topk_freq must be a positive integer")
    # offset defaults to 1 = prior `layer_id - 1` behavior for DeepSeek configs.
    offset = int(getattr(config, "index_skip_topk_offset", 1))
    return max(layer_id - offset, 0) % index_topk_freq != 0


def any_layer_skips_index_topk(config: PretrainedConfig) -> bool:
    """Whether some layer of the model reuses another's top-k."""
    return any(
        skips_index_topk(config, layer_id)
        for layer_id in range(int(getattr(config, "num_hidden_layers", 0)))
    )
