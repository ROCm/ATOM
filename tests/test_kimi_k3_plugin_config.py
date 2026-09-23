# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from atom.config import (
    _MULTIMODAL_MODEL_TYPES,
    _PLUGIN_SUPPORTED_MULTIMODAL_MODELS,
)


def test_kimi_k3_is_plugin_supported_multimodal():
    # Keep the full HF vision config when ATOM prepares Kimi-K3 for a plugin.
    assert "kimi_k3" in _MULTIMODAL_MODEL_TYPES
    assert "kimi_k3" in _PLUGIN_SUPPORTED_MULTIMODAL_MODELS
