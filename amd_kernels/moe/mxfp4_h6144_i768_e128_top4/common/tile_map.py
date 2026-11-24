# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project

"""GEMM1's expert-major tile order. Sorted rows come in expert order, so an expert's valid m-blocks are one run
``[lo, hi)`` of ``sorted_expert_ids``; its (m, n) tiles are listed together, n-major, so consecutive blocks share the
W13 slab::

    tile_map[NB_N*lo + n*(hi - lo) + (m - lo)] = m << N_BITS | n  (n < NB_N)
    tile_map[valid_blocks*NB_N .. grid)        = -1               (idle blocks)
    tile_map[grid]                              = valid_blocks*NB_N

``tile_map_grid`` keeps the table's last ``CLAIM_COUNTERS`` entries past every valid one, so they always come out -1:
the persistent GEMM1s use them as per-XCD claim counters (the others see idle blocks). The routing sorts
(``sort_count_all``, ``sort_tile_map``, ``sort_scales_all``) and ``scale_sort_tile_map`` write it.
"""

_MAX_BLOCKS = (
    4096  # >= max_sorted / BM: 1409 at 32768 tokens (BM 128) and at 65536 (BM 256)
)
CLAIM_COUNTERS = 8  # one per XCD (gemm1_prefill_h6144_i3072_persistent)


def tile_map_grid(num_m_blocks: int, I: int) -> int:  # noqa: E741
    """GEMM1's grid for the table: every (m, n) tile of the allocation rounded up to 8 XCDs, plus the
    ``CLAIM_COUNTERS`` spare entries."""
    return (num_m_blocks * (I // 128) + 7) // 8 * 8 + CLAIM_COUNTERS
