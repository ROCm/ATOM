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
    "ATOM_USE_TRITON_MLA": lambda: os.getenv("ATOM_USE_TRITON_MLA", "0") == "1",
    "ATOM_USE_TRITON_MOE": lambda: os.getenv("ATOM_USE_TRITON_MOE", "0") == "1",
    # Replace AITER HIP-backed kernels with AITER Triton equivalents at every
    # call site that has one. Off by default; turn on to validate Triton paths
    # on architectures where HIP/CK kernels are unavailable (e.g. gfx1250).
    "ATOM_REPLACE_HIP_WITH_TRITON": lambda: os.getenv(
        "ATOM_REPLACE_HIP_WITH_TRITON", "0"
    )
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
    # --- Per-layer Tensor Dump (FusedMoE) ---
    # Set to a directory to dump FusedMoE forward inputs/outputs as torch
    # tensors. When unset (default), dumping is fully disabled and has zero
    # overhead. Output layout:
    #   <dir>/rank<rank>/<phase>/iter<idx>/<layer_name>.{hidden_in,router_logits,hidden_out}.pt
    # where phase is "prefill" or "decode".
    "ATOM_DUMP_MOE_DIR": lambda: os.getenv("ATOM_DUMP_MOE_DIR", None),
    # Comma-separated substrings; only layers whose layer_name contains any of
    # them will be dumped. Empty string means dump all FusedMoE layers.
    "ATOM_DUMP_MOE_LAYERS": lambda: os.getenv("ATOM_DUMP_MOE_LAYERS", ""),
    # Max number of forward iterations to dump per (rank, phase). Use a small
    # value (e.g. 1-2) to avoid filling the disk. Set to -1 for unlimited.
    "ATOM_DUMP_MOE_MAX_ITERS": lambda: int(os.getenv("ATOM_DUMP_MOE_MAX_ITERS", "1")),
    # When ATOM_DUMP_MOE_DIR is set, weights of every matched FusedMoE layer
    # are also dumped (once per (rank, layer)) under
    #   <dir>/rank<rank>/weights/<layer_name>/<param>.pt
    # MoE weights can be huge (per-expert), so always pair with
    # ATOM_DUMP_MOE_LAYERS to restrict scope. Set this to "1" to disable the
    # weight dump and keep only the activation dump.
    "ATOM_DUMP_MOE_SKIP_WEIGHTS": lambda: os.getenv(
        "ATOM_DUMP_MOE_SKIP_WEIGHTS", "0"
    )
    == "1",
    # RTL (rocm-trace-lite) GPU kernel tracing — set to output directory to enable.
    # When set, the server launch is wrapped with `rtl trace` to collect per-kernel
    # GPU timestamps for both prefill and decode phases.
    "ATOM_RTL_TRACE_DIR": lambda: os.getenv("ATOM_RTL_TRACE_DIR", None),
    # --- Model Loading ---
    "ATOM_DISABLE_MMAP": lambda: os.getenv("ATOM_DISABLE_MMAP", "false").lower()
    == "true",
    # Use a thread pool for weight loading instead of main-process sequential I/O.
    # Set to 0 to disable if the thread pool causes hangs (e.g. on MI455).
    "ATOM_LOADER_USE_THREADPOOL": lambda: os.getenv("ATOM_LOADER_USE_THREADPOOL", "1")
    == "1",
    # --- Attention Backend ---
    # Use unified_attention (flash-style) for MHA paged/prefill attention instead
    # of pa_decode_gluon. Set to 1 to enable the unified_attention path.
    "ATOM_USE_UNIFIED_ATTN": lambda: os.getenv("ATOM_USE_UNIFIED_ATTN", "0") == "1",
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
    # --- MTP (relaxed mtp for quantized mtp) ---
    "ATOM_ENABLE_RELAXED_MTP": lambda: os.getenv("ATOM_ENABLE_RELAXED_MTP", "0").lower()
    == "1",
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
