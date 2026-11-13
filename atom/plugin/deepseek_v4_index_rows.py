# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4's index rows under a plugin bridge, which publishes the step's
tensors (`indexer_meta`, the per-token decode fields) rather than `IndexRows`.

A bridge's caller slices that metadata to a call's real rows, so the rows are
cut here, at the call, from the rows it is handed: FP8 only (no bridge binds
an FP4 index cache), and only host arithmetic -- the tensors are the bridge's.
"""

from atom.model_ops.indexer import ConcatKeys, IndexRows, concat_rows, fp8_rows


def v4_paged_rows(md, rows: int, width: int) -> IndexRows:
    """A decode or verify step's first `rows` rows: each off its own table
    row (`block_tables_per_token`) to its own visibility
    (`csa_n_committed_per_token`), into a `width`-column plane."""
    return fp8_rows(
        md.csa_n_committed_per_token[:rows],
        width,
        None,
        md.block_tables_per_token[:rows],
    )


def v4_concat_rows(meta: dict, block_tables) -> IndexRows:
    """Prefill rows against their requests' committed keys gathered end to
    end, as the bridge laid them out (`cu_committed_gpu`): row r over the
    `visible_gpu[r]` of them from `cu_starts_gpu[r]`."""
    keys = ConcatKeys(
        block_tables=block_tables,
        cu_committed=meta["cu_committed_gpu"],
        total=meta["total_committed"],
        row_starts=meta["cu_starts_gpu"],
    )
    return concat_rows(meta["visible_gpu"], keys)
