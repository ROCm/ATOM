# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Architecture defaults for the DeepSeek-V4 CSA indexer cache."""

from typing import Final

_FP4_UNSUPPORTED_GFX: Final = frozenset({"gfx942"})


def v4_fp4_indexer_supported(gfx: str) -> bool:
    """Return whether the V4 FP4 indexer is supported on ``gfx``."""

    return gfx not in _FP4_UNSUPPORTED_GFX


def resolve_v4_index_cache_dtype(
    index_cache_dtype: str | None,
    *,
    gfx: str | None,
    allow_fp4_default: bool = True,
) -> str:
    """Resolve an optional V4 index-cache override for the current GPU.

    Explicit ``bf16``/``fp8``/``fp4`` requests are preserved. When the caller
    leaves the setting unset, native single-node V4 uses FP4 everywhere except
    gfx942, whose default remains FP8. Integrations that have not implemented
    the FP4 cache layout can set ``allow_fp4_default=False`` to retain FP8.
    """

    if index_cache_dtype is not None:
        return index_cache_dtype
    if not allow_fp4_default:
        return "fp8"
    if gfx is None:
        raise ValueError("gfx is required when resolving the V4 FP4 default")
    if v4_fp4_indexer_supported(gfx):
        return "fp4"
    return "fp8"
