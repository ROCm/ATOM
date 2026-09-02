"""Triton kernels for Qwen3.8-Flash-Next QSA, vendored from AITER.

Source: ROCm/aiter PR #4882 ("[Triton/Gluon] [QSA] Add paged sparse attention
kernels"), branch `haic0:qsa-paged-sparse-attention`, files
`aiter/ops/triton/_triton_kernels/attention/qsa_*.py`. MIT licensed,
Copyright (C) 2026 Advanced Micro Devices, Inc.

Vendored rather than imported because that PR is not merged and is absent from
the AITER build ATOM runs against. The PR also ships gfx950 Gluon
specializations, but they are shape-locked to a geometry this checkpoint does
not have (head_dim 128 / GQA group 5 / 8 index heads, versus 256 / 12 / 4), so
only the portable Triton path is useful here and only it is vendored.
"""
