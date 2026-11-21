"""Lumen HIP C++ extension kernels.

The ``_fused_quant_transpose`` module is compiled from
``fused_quant_transpose.cu`` when ROCm / hipcc is available.  It exposes a
single function:

    static_quant_transpose_amax(out_row, out_col, amax_out, input, scale)

which fuses FP8 quantization + transpose + amax into one HIP kernel,
reading the BF16/FP16 input exactly once.
"""
