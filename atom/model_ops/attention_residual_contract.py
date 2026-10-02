# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared state transition for attention-residual block boundaries."""

from __future__ import annotations

import torch


def resolve_attn_res_prefix(
    prefix_sum: torch.Tensor | None,
    add_hidden: torch.Tensor | None,
    add_hidden2: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Resolve the block-close sentinel before an AttnRes-compatible stage."""

    if prefix_sum is None:
        prefix_sum, add_hidden, add_hidden2 = add_hidden, add_hidden2, None
    assert prefix_sum is not None
    return prefix_sum, add_hidden, add_hidden2
