# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Re-export of the FP8 per-block weight quantizer.

This module was a byte-identical duplicate of
``atomrl.engine.inference.fp8_weight_quantizer``. Both the training and inference
weight-sync paths already imported the *inference* copy, so the duplicate could only
ever drift out of sync. Kept as a shim so existing imports keep working.
"""

from atomrl.engine.inference.fp8_weight_quantizer import (  # noqa: F401
    _per_block_cast_to_fp8,
    _scale_name_for_weight,
    _should_quantize,
    quantize_weights_fp8_per_block,
)

__all__ = ["quantize_weights_fp8_per_block"]
