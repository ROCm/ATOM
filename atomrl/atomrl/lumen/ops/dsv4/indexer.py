# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""AIter-backed DSV4 DSA indexer."""

from __future__ import annotations

from collections.abc import Callable

import torch


def get_indexer_impl() -> str:
    """Return the fixed indexer backend name for compatibility."""
    return "aiter"


def get_batched_indexer_fwd() -> Callable[..., torch.Tensor]:
    from atomrl.lumen.kernels.dsv4.indexer.aiter_dsv4_indexer import (
        batched_indexer_fwd,
    )

    return batched_indexer_fwd


def get_batched_indexer_bwd() -> (
    Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    from atomrl.lumen.kernels.dsv4.indexer.aiter_dsv4_indexer import (
        batched_indexer_bwd,
    )

    return batched_indexer_bwd
