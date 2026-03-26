# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Centralized environment variable definitions for ATOM.

All ATOM-specific environment variables are defined in the
``environment_variables`` dict below.  Access them via attribute syntax::

    from atom.utils import envs
    if envs.ATOM_PROFILER_MORE:
        ...

Values are evaluated lazily on first access via ``__getattr__``.  To add a
new variable, append an entry to ``environment_variables`` with a lambda that
reads ``os.getenv`` and returns the typed value.

Third-party / dependency env vars (NCCL, torch, HuggingFace, AITER, FLA) are
documented at the bottom of this file but NOT managed here.
"""

import os
from typing import Any, Callable

environment_variables: dict[str, Callable[[], Any]] = {
    # --- Data Parallelism ---
    "ATOM_DP_RANK": lambda: int(os.getenv("ATOM_DP_RANK", "0")),
    "ATOM_DP_RANK_LOCAL": lambda: int(os.getenv("ATOM_DP_RANK_LOCAL", "0")),
    "ATOM_DP_SIZE": lambda: int(os.getenv("ATOM_DP_SIZE", "1")),
    "ATOM_DP_MASTER_IP": lambda: os.getenv("ATOM_DP_MASTER_IP", "127.0.0.1"),
    "ATOM_DP_MASTER_PORT": lambda: int(os.getenv("ATOM_DP_MASTER_PORT", "29500")),
    # --- Compilation & Execution ---
    "ATOM_USE_TRITON_GEMM": lambda: os.getenv("ATOM_USE_TRITON_GEMM", "0") == "1",
    "ATOM_USE_TRITON_MXFP4_BMM": lambda: os.getenv("ATOM_USE_TRITON_MXFP4_BMM", "0")
    == "1",
    # --- Kernel Fusion Toggles ---
    # QK-norm-rope-cache-quant fusion for Qwen3-MoE; disabled by default.
    # Enable for Qwen3-MoE to get better performance.
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION": lambda: os.getenv(
        "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION", "0"
    )
    == "1",
    "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION": lambda: os.getenv(
        "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION", "1"
    )
    == "1",
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION": lambda: os.getenv(
        "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION", "1"
    )
    == "1",
    "ATOM_ENABLE_DS_QKNORM_FUSION": lambda: os.getenv(
        "ATOM_ENABLE_DS_QKNORM_FUSION", "1"
    )
    == "1",
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": lambda: os.getenv(
        "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION", "1"
    )
    == "1",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT": lambda: os.getenv(
        "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT", "1"
    )
    == "1",
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT": lambda: os.getenv(
        "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT", "1"
    )
    == "1",
    # --- Profiling & Logging ---
    "ATOM_TORCH_PROFILER_DIR": lambda: os.getenv("ATOM_TORCH_PROFILER_DIR", None),
    "ATOM_PROFILER_MORE": lambda: os.getenv("ATOM_PROFILER_MORE", "0") == "1",
    "ATOM_LOG_MORE": lambda: int(os.getenv("ATOM_LOG_MORE", "0")) != 0,
    # --- Model Loading ---
    "ATOM_DISABLE_MMAP": lambda: os.getenv("ATOM_DISABLE_MMAP", "false").lower()
    == "true",
    # --- Plugin Mode ---
    "ATOM_DISABLE_VLLM_PLUGIN": lambda: os.getenv(
        "ATOM_DISABLE_VLLM_PLUGIN", "0"
    ).lower()
    == "1",
    "ATOM_DISABLE_VLLM_PLUGIN_ATTENTION": lambda: os.getenv(
        "ATOM_DISABLE_VLLM_PLUGIN_ATTENTION", "0"
    ).lower()
    == "1",
    "ATOM_USE_CUSTOM_ALL_GATHER": lambda: os.getenv(
        "ATOM_USE_CUSTOM_ALL_GATHER", "1"
    ).lower()
    == "1",
    # --- MoE (DeepSeek-style shared experts) ---
    # Dual-stream MoE only when num_tokens <= threshold; 0 disables dual-stream registration.
    "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": lambda: int(
        os.getenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "1024")
    ),
}


def is_set(name: str) -> bool:
    """Return True if the env var *name* is explicitly set (even if empty)."""
    val = os.getenv(name)
    return val is not None and val != ""


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Third-party / dependency env vars (documented only, NOT managed here)
# ---------------------------------------------------------------------------
# MASTER_ADDR, MASTER_PORT        — PyTorch distributed; set in model_runner.py
# AITER_LOG_LEVEL                 — AITER library log verbosity
# AITER_QUICK_REDUCE_QUANTIZATION — AITER; set conditionally in model_runner.py
# TORCHINDUCTOR_CACHE_DIR         — PyTorch Inductor; set in compiler_inferface.py
# TRITON_CACHE_DIR                — Triton compiler; set in compiler_inferface.py
# HF_TOKEN                        — HuggingFace Hub auth token
# HF_HUB_ENABLE_HF_TRANSFER      — HuggingFace fast transfers
# NCCL_DEBUG, NCCL_TIMEOUT        — NCCL diagnostics
# FLA_COMPILER_MODE, FLA_CI_ENV,
#   FLA_GDN_FIX_BT, FLA_USE_CUDA_GRAPH,
#   FLA_TRIL_PRECISION             — FLA ops library
# VLLM_PP_LAYER_PARTITION         — vLLM legacy (still active in models/utils.py)
# VLLM_USE_MODELSCOPE             — vLLM legacy (benchmarks)
