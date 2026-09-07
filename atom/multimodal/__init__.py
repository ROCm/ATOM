# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multimodal input handling: the registry and its contract.

The engine talks to this package through :func:`build_multimodal_inputs` and
:func:`get_mrope_input_positions`. Nothing model-specific lives here — each
model's preprocessing sits beside the model itself, in
``atom/models/<family>_vl.py``, and is reached through the registry.
"""

from atom.multimodal.protocol import MultiModalInputBuilder
from atom.multimodal.registry import (
    build_multimodal_inputs,
    expand_media_placeholders,
    get_mrope_input_positions,
)

__all__ = [
    "MultiModalInputBuilder",
    "build_multimodal_inputs",
    "expand_media_placeholders",
    "get_mrope_input_positions",
]
