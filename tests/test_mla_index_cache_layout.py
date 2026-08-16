# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-safe tests for compact MLA index-cache layout helpers.

These helpers live in ``mla_cache_layout`` so the non-GPU Pre Checkin gate can
exercise IndexShare compaction without importing AITER.
"""

from types import SimpleNamespace

from atom.model_ops.attentions.mla_cache_layout import (
    _aligned_index_cache_dim,
    _global_index_cache_layer_ids,
    _mla_kv_cache_dim,
)
from atom.models.utils import get_pp_indices


def test_global_index_cache_layout_excludes_shared_and_keeps_mtp():
    assert _global_index_cache_layer_ids(
        ("full", "shared", "shared", "full"), 4, 2
    ) == (0, 3, 4, 5)


def test_global_index_cache_layout_without_schedule_is_unchanged():
    assert _global_index_cache_layer_ids(None, 4, 1) == (0, 1, 2, 3, 4)


def test_cache_dimensions_are_derived_and_index_rows_are_aligned():
    hf_config = SimpleNamespace(kv_lora_rank=480, qk_rope_head_dim=32)

    assert _mla_kv_cache_dim(hf_config) == 512
    assert _aligned_index_cache_dim(111) == 128
    assert _aligned_index_cache_dim(124) == 128
    assert _aligned_index_cache_dim(125) == 144


def test_local_total_layers_adds_mtp_only_on_drafter_stage():
    """Mirror ModelRunner._get_local_total_num_layers without importing it.

    ModelRunner pulls AITER at import time; the PP/MTP accounting itself is
    just get_pp_indices + optional draft depth on the last stage.
    """
    num_hidden = 6
    num_draft = 2

    start, end = get_pp_indices(num_hidden, 0, 2)
    assert end - start == 3

    start, end = get_pp_indices(num_hidden, 1, 2)
    assert (end - start) + num_draft == 5
