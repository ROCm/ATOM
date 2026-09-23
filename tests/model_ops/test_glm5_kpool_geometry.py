# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts shared by GLM-5.3's indexer and metadata builder."""

import pytest
import torch

from atom.model_ops.glm5_next.geometry import (
    effective_kpool_size,
    get_query_request_indices,
    pooled_path_enabled,
    speculative_kpool_history_size,
    speculative_pool_scratch_width,
    speculative_verify_enabled,
    topk_output_width,
)


def test_pooled_width_includes_tail_and_alignment(monkeypatch):
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "1")

    assert pooled_path_enabled(4)
    assert effective_kpool_size(4) == 4
    assert topk_output_width(2048, 4) == 2176


def test_off_switch_restores_token_granular_geometry(monkeypatch):
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "0")

    assert not pooled_path_enabled(4)
    assert effective_kpool_size(4) == 1
    assert topk_output_width(2048, 4) == 2048


def test_nonpooled_model_ignores_switch(monkeypatch):
    monkeypatch.setenv("ATOM_GLM5_KPOOL", "1")

    assert not pooled_path_enabled(1)
    assert effective_kpool_size(1) == 1
    assert topk_output_width(2048, 1) == 2048


def test_query_rows_map_across_an_empty_request():
    cu_seqlens_q = torch.tensor([0, 3, 3, 7, 9], dtype=torch.int32)

    rows = get_query_request_indices(cu_seqlens_q, 9)

    assert rows.tolist() == [0, 0, 0, 2, 2, 2, 2, 3, 3]


def test_query_rows_map_for_one_request():
    cu_seqlens_q = torch.tensor([0, 4], dtype=torch.int32)

    assert get_query_request_indices(cu_seqlens_q, 4).tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("pool_size", "num_speculative_tokens", "expected"),
    [
        (4, None, 4),
        (4, 1, 8),
        (4, 2, 16),
        (4, 3, 16),
        (4, 6, 32),
    ],
)
def test_speculative_history_covers_live_rows(
    pool_size,
    num_speculative_tokens,
    expected,
):
    size = speculative_kpool_history_size(pool_size, num_speculative_tokens)

    assert size == expected
    if num_speculative_tokens is not None:
        required = pool_size + 2 * (num_speculative_tokens + 1)
        assert size >= required
        assert size & (size - 1) == 0
        assert size // 2 < required


@pytest.mark.parametrize(
    ("is_prefill", "num_spec_decodes", "max_seqlen_q", "expected"),
    [
        (True, 1, 4, False),
        (False, 1, 1, True),
        (False, 0, 4, True),
        (False, 0, 1, False),
    ],
)
def test_speculative_verify_dispatch(
    is_prefill,
    num_spec_decodes,
    max_seqlen_q,
    expected,
):
    assert (
        speculative_verify_enabled(
            is_prefill=is_prefill,
            num_spec_decodes=num_spec_decodes,
            max_seqlen_q=max_seqlen_q,
        )
        is expected
    )


def test_speculative_scratch_uses_live_batch_length():
    assert speculative_pool_scratch_width(2051, 4) == 513
    assert speculative_pool_scratch_width(128 * 1024, 4) == 32 * 1024
