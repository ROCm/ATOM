# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Triton sparse MLA for DSV4 training."""

from __future__ import annotations

from collections.abc import Callable

import torch


def get_sparse_mla_backend() -> str:
    """Return the fixed sparse MLA backend name for compatibility."""
    return "triton"


def get_sparse_attn_fn() -> Callable[..., torch.Tensor]:
    from atomrl.lumen.kernels.dsv4.sparse_mla.triton_sparse_mla import (
        sparse_attn_triton,
    )

    return sparse_attn_triton
