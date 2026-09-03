# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small, CPU-testable geometry contracts for GLM-5.3's pooled indexer."""

from atom.utils import envs


def pooled_path_enabled(index_kpool: int) -> bool:
    """Whether the pooled indexer path is in force."""
    return index_kpool > 1 and envs.ATOM_GLM5_KPOOL


def effective_kpool_size(index_kpool: int) -> int:
    """Configured pool size, or one when the pooled path is disabled."""
    return index_kpool if pooled_path_enabled(index_kpool) else 1


def topk_output_width(topk: int, index_kpool: int) -> int:
    """Physical row width shared by the indexer producer and MLA metadata."""
    kpool = effective_kpool_size(index_kpool)
    if kpool <= 1:
        return topk
    return ((topk + kpool - 1 + 127) // 128) * 128
