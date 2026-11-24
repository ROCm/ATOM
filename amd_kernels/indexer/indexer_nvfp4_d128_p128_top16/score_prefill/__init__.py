# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Prefill index scorer over NVFP4 index pages (FlyDSL, gfx950)."""

from .host import (
    nvfp4_index_score_prefill,
    supports_nvfp4_index_score_prefill,
)

__all__ = [
    "nvfp4_index_score_prefill",
    "supports_nvfp4_index_score_prefill",
]
