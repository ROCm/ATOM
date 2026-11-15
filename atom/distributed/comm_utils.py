# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Helpers shared by the collectives of every parallel dimension."""

import torch


def gather_rows(gather, rows):
    """`gather(list of tensors) -> list of tensors` over `rows`, one-byte
    dtypes carried as uint8: aiter's grouped `all_gatherv` copies bytes
    whatever the dtype, but its PyNCCL fallback has no fp8 / e8m0 / fp4x2."""
    gathered = gather(
        [t.view(torch.uint8) if t.element_size() == 1 else t for t in rows]
    )
    return tuple(g.view(t.dtype) for g, t in zip(gathered, rows))
