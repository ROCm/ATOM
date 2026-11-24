# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Paired gate/up MXFP8 GEMM + SwiGLU-OAI at 1-8 rows (dense MLP, TP4)."""

from .host import mxfp8_gateup_swiglu_bf16, warmup_mxfp8_gateup_swiglu_bf16

__all__ = ["mxfp8_gateup_swiglu_bf16", "warmup_mxfp8_gateup_swiglu_bf16"]
