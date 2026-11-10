# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared-expert ID reassignment for LEGACY_AITER gathered rows."""

import torch


def copy_shared_expert_ids_(
    topk_ids: torch.Tensor, shared_template: torch.Tensor, top_k: int
) -> None:
    """Copy the gathered batch's shared IDs, keeping invalid routed rows invalid.

    Only shared columns in the gathered output are written. Routed columns
    and the prefilled template stay unchanged, including on graph replay.
    """
    rows, width = topk_ids.shape
    num_shared = width - top_k
    assert shared_template.shape == (rows, num_shared)
    n_out = rows * num_shared
    if n_out == 0:
        return
    if not topk_ids.is_cuda:
        shared_ids = topk_ids[:, top_k:]
        shared_ids.copy_(shared_template)
        shared_ids.masked_fill_(topk_ids[:, :1] < 0, -1)
        return

    from atom.model_ops.fused_moe.gathered_shared_ids_triton import (
        _copy_shared_ids_kernel,
    )

    _copy_shared_ids_kernel[((n_out + 255) // 256,)](
        topk_ids,
        shared_template,
        n_out,
        *topk_ids.stride(),
        *shared_template.stride(),
        top_k,
        num_shared,
        BLOCK=256,
    )
