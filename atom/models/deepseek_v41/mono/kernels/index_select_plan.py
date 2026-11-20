# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""LDS selection segments sized for a standalone indexer."""


def row_segment(bound_max):
    # Leave room for a short speculative suffix instead of adding a tiny leaf
    # at the common 8K/80K boundaries. Widths remain whole per-thread quads.
    return 18432 if bound_max > 16384 else 10240


def segment_for(segment, kind):
    return min(segment, 8192) if kind == "blk" else segment
