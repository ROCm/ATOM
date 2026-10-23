# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono's GDN state view, attached through the existing installer boundary."""

from atom.models.qwen4_exp_mono.config import gdn_cache_layout_supported


class MonoCacheBuilder:
    """Delegate native metadata and pools; adapt only the GDN binding view.

    Mono's recurrence reads a logical KV matrix with contiguous K. The native
    builder allocates the same state storage and owns its slot lifetime. This
    adapter changes only the returned tensor's strides, without copying state
    or modifying the native builder's prefill, capture or checkpoint methods.
    Native fallback kernels already honor those strides.
    """

    def __init__(self, native):
        self._native = native

    def __getattr__(self, name):
        return getattr(self._native, name)

    def build_kv_cache_tensor(self, module):
        cache = self._native.build_kv_cache_tensor(module)
        if (
            cache is not None
            and hasattr(module, "base_linear_attention")
            and gdn_cache_layout_supported(module.impl, self._native.replayssm)
            and cache.v_cache.stride(-1) == 1
        ):
            # Native FlyDSL decode may already have installed this view.
            cache.v_cache = cache.v_cache.transpose(-1, -2)
        return cache
