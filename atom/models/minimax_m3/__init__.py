# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3: the original implementation (``model``) and the fused per-layer
decode path (``mono``). ``MiniMaxM3SparseForCausalLM`` stays the single entry point.

Code that patches module globals (the SGLang plugin swaps ``Attention``) must
patch ``atom.models.minimax_m3.model``: the classes resolve their globals
there, not on this package.

The names below resolve on first access, so importing a GPU-free submodule
(``mono.layout``, ``mono.config``) does not import the model and AITER.
"""

import importlib

__all__ = [
    "MiniMaxM3Attention",
    "MiniMaxM3DecoderLayer",
    "MiniMaxM3MLP",
    "MiniMaxM3MoE",
    "MiniMaxM3Model",
    "MiniMaxM3SparseAttention",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3SparseForConditionalGenerationTextOnly",
    "make_minimax_m3_expert_params_mapping",
]


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module(f"{__name__}.model"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
