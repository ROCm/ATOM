# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared persistent-grid parameters for the DeepSeek-V4 FP4 indexer."""

FP4_MQA_PARALLEL_UNIT_NUM = 512
FP4_MQA_BLOCK_K = 256


def fp4_mqa_prefill_parallel_unit_num(
    num_rows: int,
    max_seq_len: int,
    *,
    block_k: int = FP4_MQA_BLOCK_K,
    min_parallel_unit_num: int = FP4_MQA_PARALLEL_UNIT_NUM,
) -> int:
    """Select the gfx950 FP4 prefill persistent-grid size.

    ``compute_prefill_schedule`` launches exactly ``parallel_unit_num`` CTAs,
    not merely up to that many. Too small a grid folds multiple K chunks onto
    one CTA; too large a grid launches dummy CTAs after all useful row/chunk
    pairs have been assigned. C48 measurements show that short Q benefits from
    more K parallelism, while a full 16K-token prefill already supplies enough
    row parallelism unless its compressed context is longer than 64 chunks.

    Cap the selected grid by the maximum number of useful (row, K-chunk) pairs
    so short contexts do not pay for empty CTAs. The returned value is always
    at least ``num_rows``, which is required for correctness by the scheduler.
    """
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}")
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
    if block_k <= 0:
        raise ValueError(f"block_k must be positive, got {block_k}")
    if min_parallel_unit_num <= 0:
        raise ValueError(
            "min_parallel_unit_num must be positive, got "
            f"{min_parallel_unit_num}"
        )

    chunks_per_row = max(1, (max_seq_len + block_k - 1) // block_k)
    if num_rows <= 1536:
        target_splits_per_row = 5
    elif num_rows >= 16384 and chunks_per_row <= 64:
        target_splits_per_row = 1
    else:
        target_splits_per_row = 2

    target_grid = max(min_parallel_unit_num, num_rows * target_splits_per_row)
    useful_grid_cap = num_rows * chunks_per_row
    return min(target_grid, useful_grid_cap)
