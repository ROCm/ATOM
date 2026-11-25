# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 block-alignment padding utilities.

Provides ``get_fp8_align_size`` (recipe → alignment) and ``pad_to_block``
(pad a tensor along a dimension).  Analogous to Megatron-Core's
``get_fp8_align_size`` / ``get_padding`` but operating at the GEMM level.
"""

import torch


def get_fp8_align_size(scaling_type: str, block_size: int = 128) -> int:
    """Return the alignment size required for FP8 GEMM given a scaling recipe.

    Args:
        scaling_type: One of ``"delayed"``, ``"dynamic"``, ``"per_token"``,
            ``"blockwise"``, ``"blockwise2d"``, ``"mxfp8"``, ``"none"``.
        block_size: The block size configured for blockwise/mxfp8 recipes.

    Returns:
        The number of elements that tensor dimensions must be divisible by.
        Returns 1 when no block alignment is needed.
    """
    if scaling_type in ("delayed", "dynamic", "per_token", "none"):
        return 1
    if scaling_type in ("blockwise", "blockwise2d"):
        return block_size
    if scaling_type == "mxfp8":
        return 32
    if scaling_type == "mxfp4":
        return 32
    raise ValueError(f"Unknown scaling_type={scaling_type!r}")


def pad_to_block(
    tensor: torch.Tensor,
    align_size: int,
    dim: int = -1,
) -> tuple[torch.Tensor, int]:
    """Pad *tensor* along *dim* so its size is a multiple of *align_size*.

    Args:
        tensor: Input tensor of any shape.
        align_size: Target alignment (must be >= 1).
        dim: Dimension to pad (supports negative indexing).

    Returns:
        ``(padded_tensor, orig_size)`` where *orig_size* is the original
        size along *dim*.  If no padding is needed, *padded_tensor* **is**
        the original tensor (zero-copy).  Callers unpad via
        ``tensor.narrow(dim, 0, orig_size)`` or equivalent slice.
    """
    if align_size < 1:
        raise ValueError(f"align_size must be >= 1, got {align_size}")

    dim = dim % tensor.ndim
    orig_size = tensor.size(dim)
    remainder = orig_size % align_size
    if remainder == 0:
        return tensor, orig_size

    pad_amount = align_size - remainder
    # F.pad pads from the last dim backwards; build the pad spec accordingly.
    pad_spec = [0] * (2 * (tensor.ndim - 1 - dim)) + [0, pad_amount]
    return torch.nn.functional.pad(tensor, pad_spec), orig_size
