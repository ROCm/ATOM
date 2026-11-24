# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Top-16 of the index block scores (FlyDSL, gfx950): the ids of decode and prefill rows, and of prefill rows with the
attend's page table."""

from .decode_indices_kernel import index_topk_decode_indices
from .host import (
    index_topk_prefill,
    index_topk_prefill_indices,
    supports_index_topk,
)

__all__ = [
    "index_topk_decode_indices",
    "index_topk_prefill",
    "index_topk_prefill_indices",
    "supports_index_topk",
]
