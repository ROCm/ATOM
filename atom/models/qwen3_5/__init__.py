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

# the package's submodules: ``from atom.models.qwen3_5 import mono`` asks for the attribute
# first, which must load the submodule alone (not the model, and with it AITER)
_SUBMODULES = ("mono", "model")


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    return getattr(importlib.import_module(f"{__name__}.model"), name)
