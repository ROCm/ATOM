# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV-cache dtype helpers shared by SGLang model adapters."""

from __future__ import annotations


def is_fp8_kv_cache_dtype(kv_cache_dtype: object) -> bool:
    """Recognize CLI aliases, ATOM's ``fp8`` canonical form, and torch dtypes."""
    cache_dtype = getattr(kv_cache_dtype, "cache_dtype", kv_cache_dtype)
    normalized = str(cache_dtype).lower().removeprefix("torch.")
    return normalized.startswith(("fp8", "float8"))
