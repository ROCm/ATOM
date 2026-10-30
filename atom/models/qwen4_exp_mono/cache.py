# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono's GDN state view, attached through the existing installer boundary."""

from atom.models.qwen4_exp_mono.config import (
    gdn_cache_layout_supported,
    uses_mtp_target,
)


def metadata_builder_cls(native_cls):
    """Adapt cache binding before warmup through the existing backend factory.

    The physical GDN layout must be selected before prefill writes state, so it belongs to
    cache construction rather than the lazy runner's first decode.
    """
    from atom.models.qwen4_exp_mono.config import config_refusal

    class QwenMonoMetadataBuilder(native_cls):
        def build_kv_cache_tensor(self, module):
            cache = super().build_kv_cache_tensor(module)
            if (
                cache is not None
                and config_refusal(self.model_runner.config) is None
                and not uses_mtp_target(self.model_runner.config)
                and hasattr(module, "base_linear_attention")
                and gdn_cache_layout_supported(module.impl, self.replayssm)
                and cache.v_cache.stride(-1) == 1
            ):
                cache.v_cache = cache.v_cache.transpose(-1, -2)
            return cache

    return QwenMonoMetadataBuilder
