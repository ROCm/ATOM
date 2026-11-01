# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""LDS scale layout retained by the legacy TP2 reference kernels."""

from atom.models.deepseek_v41.mono.gfx950.kernels.moe import DOWN_R_MAX, down_scale_words


def wsl_words(d):
    return max(256, DOWN_R_MAX * down_scale_words(d))
