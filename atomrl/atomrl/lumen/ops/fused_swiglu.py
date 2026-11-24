# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU activation dispatch.

Dispatches to AITER's Triton fused SiLU-and-mul kernels which compute
``silu(y1) * y2`` (forward) and the full backward in single kernel
launches each, eliminating the ~6-8 separate elementwise launches that
Megatron's ``@jit_fuser`` swiglu/swiglu_back produce.

Enable globally via ``LUMEN_FUSED_SWIGLU=1`` (installed by megatron_patches).
"""

import logging

import torch
from aiter.ops.triton.fusions.activation import (
    fused_silu_mul as _AITER_SWIGLU_FWD,
)
from aiter.ops.triton.fusions.activation import (
    silu_and_mul_backward as _AITER_SWIGLU_BWD,
)

logger = logging.getLogger(__name__)


def fused_swiglu(y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU forward via the AITER main API."""
    return _AITER_SWIGLU_FWD(y)


def fused_swiglu_backward(grad_output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU backward via the AITER main API."""
    return _AITER_SWIGLU_BWD(grad_output, y)
