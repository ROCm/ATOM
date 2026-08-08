# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Model architecture configs for diffusion pipelines."""

from atom.diffusion.configs.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3DiTArchConfig,
)

__all__ = [
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3DiTArchConfig",
]
