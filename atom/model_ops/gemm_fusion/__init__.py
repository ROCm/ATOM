# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ulysses SP attention QKV GEMM fused with the SP exchange through AITER
gemm_fusion plans (``--gemm-fusion-profile``)."""


def initialize_gemm_fusion(model, config):
    """After weight processing, before warmup and capture. Returns the
    runtime, or None when ``--gemm-fusion-profile`` is unset."""
    if config.gemm_fusion_profile is None:
        return None
    from .runtime import initialize

    return initialize(model, config)
