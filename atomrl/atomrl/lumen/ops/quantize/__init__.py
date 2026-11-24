# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Quantization ops — functional API and autograd-aware quantized linear.

``linear`` is loaded lazily to keep the functional ops import lightweight.
"""

from atomrl.lumen.ops.quantize.ops import (
    convert_from_mxfp4,
    convert_from_mxfp4_2d,
    convert_from_mxfp8,
    convert_to_mxfp4,
    convert_to_mxfp4_2d,
    convert_to_mxfp4_dual_axis,
    convert_to_mxfp8,
    dequant_fp8_tensorwise_impl,
    dequant_hadamard_quant_mxfp4,
    dequant_transpose_mxfp4,
    dual_layout_quant_mxfp4,
    hadamard_quant_mxfp4,
    hadamard_transform,
    is_cdna4,
    quant_fp8_blockwise_impl,
    quant_fp8_blockwise_segment_m_impl,
    quant_fp8_tensorwise_impl,
    transpose_packed_fp4,
)

__all__ = [
    # Quantized linear (autograd)
    "QuantizedLinearFunction",
    "convert_from_mxfp4",
    "convert_from_mxfp4_2d",
    # Pure quant/dequant ops
    "convert_from_mxfp8",
    # MXFP4 quant/dequant ops
    "convert_to_mxfp4",
    "convert_to_mxfp4_2d",
    "convert_to_mxfp4_dual_axis",
    "convert_to_mxfp8",
    "dequant_fp8_tensorwise_impl",
    "dequant_hadamard_quant_mxfp4",
    "dequant_transpose_mxfp4",
    "dual_layout_quant_mxfp4",
    "hadamard_quant_mxfp4",
    "hadamard_transform",
    "is_cdna4",
    "quant_fp8_blockwise_impl",
    "quant_fp8_blockwise_segment_m_impl",
    "quant_fp8_tensorwise_impl",
    "quantized_linear",
    "transpose_packed_fp4",
]

_LINEAR_EXPORTS = ("QuantizedLinearFunction", "quantized_linear")


def __getattr__(name):
    if name not in _LINEAR_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from atomrl.lumen.ops.quantize.linear import (
        QuantizedLinearFunction,
        quantized_linear,
    )

    return {
        "QuantizedLinearFunction": QuantizedLinearFunction,
        "quantized_linear": quantized_linear,
    }[name]
