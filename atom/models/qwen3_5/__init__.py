# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5: the original implementation (``model``) and the fused per-layer decode
path (``mono``).

Code that patches module globals (the vLLM plugin swaps ``Qwen3_5GatedDeltaNet``)
must patch ``atom.models.qwen3_5.model``: the classes resolve their globals there,
not on this package.

The model's names resolve on first access, so importing a GPU-free submodule
(``mono.layout``, ``mono.config``) does not import the model and AITER.
"""

import importlib


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.model"), name)
