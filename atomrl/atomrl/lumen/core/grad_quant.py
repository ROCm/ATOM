# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Backward-compatibility shim for gradient quantization.

The implementation now lives in
:class:`~atomrl.lumen.quantize.ScalingManager`.  This module re-exports
the public API so that existing call sites (attention, rmsnorm, etc.) continue
to work without changes.
"""

import torch

# Kept in sync with atomrl.lumen.quantize.scaling_manager.GRAD_QUANT_TYPES, which is the
# source of truth; duplicated as a literal here to keep this shim import-light.
GRAD_QUANT_TYPES = (None, "fp8", "mxfp8", "mxfp4", "fp4")


def quantize_grad_tensor(
    tensor: torch.Tensor,
    grad_quant_type: str | None,
    fp8_dtype: torch.dtype | None = None,
    block_size: int = 32,
) -> torch.Tensor:
    """Quantize *tensor* to a low-precision format and dequantize back.

    Delegates to :meth:`ScalingManager.quantize_grad_tensor`.
    """
    if grad_quant_type is None:
        return tensor
    from atomrl.lumen.quantize.scaling_manager import ScalingManager

    return ScalingManager.quantize_grad_tensor(
        tensor,
        grad_quant_type,
        fp8_dtype=fp8_dtype,
        block_size=block_size,
    )
