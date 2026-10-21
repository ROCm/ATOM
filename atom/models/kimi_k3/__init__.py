# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3: the original implementation (``model``) and the fused per-layer
decode path (``mono``). Every name this module used to hold still resolves here,
so ``from atom.models.kimi_k3 import KimiK3ForCausalLM`` is unchanged.

Code that *patches* module globals (the SGLang plugin swaps ``Attention``, the
vLLM plugin swaps ``KimiKDAAttention`` / ``KimiLinearModel``) must patch
``atom.models.kimi_k3.model``: the classes resolve their globals there, and an
assignment on this package would bind a name nothing reads.

Resolution is deferred to first access, so importing a GPU-free submodule
(``mono.config``, ``mono.layout``) does not drag in the model and AITER.
"""

import importlib


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__name__}.model"), name)


def __dir__():
    return sorted(dir(importlib.import_module(f"{__name__}.model")))
