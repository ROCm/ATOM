# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Arch-capability helpers for portable-path routing."""

from aiter.jit.utils.chip_info import get_gfx_runtime


def aiter_hip_kernels_supported() -> bool:
    """aiter's hand-written HIP kernels (fused norm, activation, cache, ...)
    emit gfx9-only packed/FP8-cvt instructions (v_pk_mul_f32, fp8 cvt) that are
    absent on RDNA (gfx11/gfx12). On unsupported arches, callers route to the
    portable torch-native / Triton implementations instead."""
    return get_gfx_runtime().startswith("gfx9")
