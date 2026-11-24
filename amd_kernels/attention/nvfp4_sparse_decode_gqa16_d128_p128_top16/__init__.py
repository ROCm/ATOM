# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse decode attention on gfx950 (FlyDSL): 16 query heads per KV head, head dim 128, 128-token pages,
top-16 blocks per row."""

from .host import nvfp4_sparse_decode

__all__ = ["nvfp4_sparse_decode"]
