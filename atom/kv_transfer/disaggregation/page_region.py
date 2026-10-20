# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build a `PageRegion` from the tensor that owns its bytes.

Kept out of `types`, which stays a torch-free contract: this is where a
backend's allocation becomes both halves of a PAGE unit at once.
"""

from __future__ import annotations

import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion, PageRegion


def page_region(
    tensor: torch.Tensor,
    *,
    semantic_role: str,
    unit_bytes: int | None = None,
    total_bytes: int | None = None,
) -> PageRegion:
    """The region of ``tensor``'s PAGE blocks and a byte view aliasing them.

    The view is a zero-copy ``uint8 [num_units, 1, unit_bytes]`` alias: the
    last two dimensions are opaque copy geometry, and LMCache stores the same
    bytes in the same order whatever shape a block is given.

    ``unit_bytes`` defaults to ``tensor``'s row stride and ``total_bytes`` to
    all of it. An allocation that holds more than its PAGE blocks (DSV4's
    planes also hold SLOT rows after them) passes both.
    """
    if not tensor.is_contiguous():
        raise ValueError(f"PAGE tensor {semantic_role!r} must be contiguous")
    available = tensor.numel() * tensor.element_size()
    if unit_bytes is None:
        unit_bytes = tensor.stride(0) * tensor.element_size()
    if total_bytes is None:
        total_bytes = available
    if unit_bytes <= 0 or total_bytes % unit_bytes or total_bytes > available:
        raise ValueError(
            f"PAGE tensor {semantic_role!r} has {available} bytes; cannot "
            f"publish {total_bytes} bytes in units of {unit_bytes}"
        )
    # `view`, never `reshape`: a copy would publish bytes nobody writes.
    byte_view = tensor.view(torch.uint8).view(-1)[:total_bytes]
    return PageRegion(
        region=KVTransferRegion(
            base_addr=tensor.data_ptr(),
            total_bytes=total_bytes,
            unit_bytes=unit_bytes,
            semantic_role=semantic_role,
        ),
        view=byte_view.view(-1, 1, unit_bytes),
    )


__all__ = ["page_region"]
