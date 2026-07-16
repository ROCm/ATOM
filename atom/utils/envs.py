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
    # Prefix for process titles set via set_process_title (shown in ps/top/rocm-smi)
    "ATOM_PROCESS_NAME_PREFIX": lambda: os.getenv("ATOM_PROCESS_NAME_PREFIX", "ATOM"),
    # --- Compilation & Execution ---
    "ATOM_USE_TRITON_GEMM": lambda: os.getenv("ATOM_USE_TRITON_GEMM", "0") == "1",
    "ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE": lambda: (
        os.getenv("ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE", "0") == "1"
    ),
    "ATOM_USE_TRITON_MXFP4_BMM": lambda: (
        os.getenv("ATOM_USE_TRITON_MXFP4_BMM", "0") == "1"
    ),
    "ATOM_USE_TRITON_MLA": lambda: os.getenv("ATOM_USE_TRITON_MLA", "0") == "1",
    # Use the block_size=64 *shuffled* KV-cache Triton/Gluon MLA kernels
    # (aiter.ops.triton.attention.mla.mla_decode_fwd + the shuffled cat/cache
    # write kernels) instead of the SGLang-style page_size=1 decode path.
    # Requires ATOM_USE_TRITON_MLA=1 (selects TritonMLABackend).
    "ATOM_USE_TRITON_MLA_SHUFFLE_KV": lambda: (
        os.getenv("ATOM_USE_TRITON_MLA_SHUFFLE_KV", "0") == "1"
    ),
    "ATOM_USE_TRITON_MOE": lambda: os.getenv("ATOM_USE_TRITON_MOE", "0") == "1",
    "ATOM_USE_TRITON_MOE_DECODE": lambda: os.getenv("ATOM_USE_TRITON_MOE_DECODE", "0")
    == "1",
    "ATOM_MLA_PAGE_SIZE": lambda: int(os.getenv("ATOM_MLA_PAGE_SIZE", "1")),
    # --- Kernel Fusion Toggles ---
    # fused_compress_attn: switch between Triton (default historical) and a
    # flydsl drop-in for V4-Pro Compressor (Main BF16 + Indexer FP8) paths.
    # "auto" picks flydsl when shape matches the supported configs (D ∈
    # {128, 512}, RD=64, OVERLAP=1, RATIO=4); "always" forces it (errors on
    # unsupported); "never" pins Triton. flydsl pure-GPU time beats Triton
    # across the full range on V4-Pro (1.1x small N → 2-3x at N≥4096).
    "ATOM_FUSED_COMPRESS_USE_FLYDSL": lambda: os.getenv(
        "ATOM_FUSED_COMPRESS_USE_FLYDSL", "auto"
    ).lower(),
    # QK-norm-rope-cache-quant fusion for Qwen3-MoE; disabled by default.
    # Enable for Qwen3-MoE to get better performance.
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION", "0") == "1"
    ),
    "ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION", "1") == "1"
    ),
    "ATOM_ENABLE_DS_QKNORM_QUANT_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_QKNORM_QUANT_FUSION", "1") == "1"
    ),
    "ATOM_ENABLE_DS_QKNORM_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_QKNORM_FUSION", "1") == "1"
    ),
    "ATOM_ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION", "1") == "1"
    ),
    # DSA sparse-indexer prefill: KV-dimension chunk size (in tokens) for
    # `fp8_mqa_logits`. The dense logits buffer is [prefill_tokens, total_kv];
    # total_kv = sum of all co-scheduled prefill contexts and is NOT bounded by
    # max_num_batched_tokens, so a concurrency burst of long-context requests
    # can drive a single allocation to tens of GiB (see GLM-5.2 OOM #1376).
    # Target peak (in MiB) for the indexer's dense fp32 logits buffer during
    # prefill. The indexer chunks along the query (Q) dimension so the buffer
    # stays ~[q_chunk, total_kv] with q_chunk = budget_bytes // (total_kv * 4);
    # q_chunk shrinks automatically as total_kv (the unbounded KV dimension)
    # grows. Under chunked prefill num_rows is already capped by
    # max_num_batched_tokens, so a fixed row count would not adapt to total_kv
    # (see GLM-5.2 OOM #1376) — a memory budget does. Each chunk still scores
    # the full KV, so every row's top-k is exact (no cross-chunk merge). Set to
    # 0 to disable chunking (always single-shot).
    "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB": lambda: int(
        os.getenv("ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", "2048")
    ),
    # GLM-5.2 (glm_moe_dsa): enable the fused indexer qk-rope + fp8-quant + kv-cache
    # kernel (indexer_qk_rope_quant_and_cache), same path DeepSeek-V3.2 uses. GLM's
    # indexer dims (index_head_dim=128, qk_rope_head_dim=64, per_1x128, neox rope) are
    # identical to V3.2, so the fusion is math-equivalent to the unfused path. Set to
    # "0" to fall back to the per-op Python path if a regression is suspected.
    "ATOM_ENABLE_GLM_FUSED_INDEXER": lambda: (
        os.getenv("ATOM_ENABLE_GLM_FUSED_INDEXER", "1") == "1"
    ),
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION", "1") == "1"
    ),
    # Replicate the vocab embedding on every TP rank (full table per rank, purely
    # local lookup) instead of TP-sharding it — eliminates the post-embedding
    # all-reduce. Applies to BOTH the main/target model and the speculative draft
    # (EAGLE3 head + MTP draft steps), so the collective is dropped on every embed
    # on the critical path. Only used where the embedding is independent of the
    # still TP-sharded lm_head (EAGLE3 draft; GLM-5.2 target+MTP, whose
    # tie_word_embeddings=False), so the lookup is bit-identical to the sharded
    # masked-embedding + all-reduce path. Trades (tp-1)/tp of the embedding's
    # memory per rank for one fewer collective per embed. Default on; set "0" to
    # fall back to the sharded VocabParallelEmbedding.
    "ATOM_REPLICATE_VOCAB_EMBED": lambda: (
        os.getenv("ATOM_REPLICATE_VOCAB_EMBED", "1") == "1"
    ),
    "ATOM_ENABLE_GDN_DECODE_LOSSY_FAST": lambda: (
        os.getenv("ATOM_ENABLE_GDN_DECODE_LOSSY_FAST", "0").lower() == "1"
    ),
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT": lambda: (
        os.getenv("ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT", "1") == "1"
    ),
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT": lambda: (
        os.getenv("ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT", "1") == "1"
    ),
    # --- Profiling & Logging ---
    "ATOM_TORCH_PROFILER_DIR": lambda: os.getenv("ATOM_TORCH_PROFILER_DIR", None),
    "ATOM_PROFILER_MORE": lambda: os.getenv("ATOM_PROFILER_MORE", "0") == "1",
    "ATOM_PROFILER_TIMEOUT": lambda: float(os.getenv("ATOM_PROFILER_TIMEOUT", "300")),
    "ATOM_LOG_MORE": lambda: int(os.getenv("ATOM_LOG_MORE", "0")) != 0,
    # RTL (rocm-trace-lite) GPU kernel tracing — set to output directory to enable.
    # When set, the server launch is wrapped with `rtl trace` to collect per-kernel
    # GPU timestamps for both prefill and decode phases.
    "ATOM_RTL_TRACE_DIR": lambda: os.getenv("ATOM_RTL_TRACE_DIR", None),
    # --- Model Loading ---
    "ATOM_DISABLE_MMAP": lambda: (
        os.getenv("ATOM_DISABLE_MMAP", "false").lower() == "true"
    ),
    # Use a thread pool for weight loading instead of main-process sequential I/O.
    # Set to 0 to disable if the thread pool causes hangs (e.g. on gfx1250).
    "ATOM_LOADER_USE_THREADPOOL": lambda: (
        os.getenv("ATOM_LOADER_USE_THREADPOOL", "1") == "1"
    ),
    # --- Attention Backend ---
    # Use unified_attention (flash-style) for MHA paged/prefill attention instead
    # of pa_decode_gluon. Set to 1 to enable the unified_attention path.
    "ATOM_USE_UNIFIED_ATTN": lambda: os.getenv("ATOM_USE_UNIFIED_ATTN", "0") == "1",
    # Force Triton attention fallbacks where available. Set to 1 to bypass
    # optional ASM/OPUS fast paths during debugging.
    "ATOM_FORCE_ATTN_TRITON": lambda: (os.getenv("ATOM_FORCE_ATTN_TRITON", "0") == "1"),
    # Use gluon pa decode for some models
    "ATOM_USE_GLUON_PA_DECODE": lambda: (
        os.getenv("ATOM_USE_GLUON_PA_DECODE", "0") == "1"
    ),
    # --- Plugin Mode ---
    "ATOM_DISABLE_VLLM_PLUGIN": lambda: (
        os.getenv("ATOM_DISABLE_VLLM_PLUGIN", "0").lower() == "1"
    ),
    "ATOM_USE_CUSTOM_ALL_GATHER": lambda: (
        os.getenv("ATOM_USE_CUSTOM_ALL_GATHER", "1").lower() == "1"
    ),
    "ATOM_USE_FLYDSL_GDR": lambda: os.getenv("ATOM_USE_FLYDSL_GDR", "0").lower() == "1",
    # --- MoE (DeepSeek-style shared experts) ---
    # Dual-stream MoE only when num_tokens <= threshold; 0 disables dual-stream registration.
    "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": lambda: int(
        os.getenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "1024")
    ),
    # Gate/Up interleave mode for MoE weight preshuffle and kernel gate_mode.
    # "0" (default) = SEPARATED layout; "1" = INTERLEAVE layout.
    "ATOM_MOE_GU_ITLV": lambda: os.getenv("ATOM_MOE_GU_ITLV", "0") == "1",
    # --- MTP (relaxed mtp for quantized mtp) ---
    "ATOM_ENABLE_RELAXED_MTP": lambda: (
        os.getenv("ATOM_ENABLE_RELAXED_MTP", "0").lower() == "1"
    ),
    # --- Atomesh ---
    # Build atomesh when installing ATOM from source.
    "ATOM_MESH_BUILD": lambda: os.getenv("ATOM_MESH_BUILD", "0") == "1",
    # Route the OpenAI-compatible server entrypoint through Atomesh.
    "USE_ATOMESH_ENTRYPOINTS": lambda: (
        os.getenv("USE_ATOMESH_ENTRYPOINTS", "0") == "1"
    ),
    # --- Gradient Control ---
    # Enable gradient tracking on model parameters.  Default "0" (disabled)
    # is correct for inference; set to "1" only for training / fine-tuning.
    "ATOM_REQUIRES_GRAD": lambda: os.getenv("ATOM_REQUIRES_GRAD", "0") == "1",
    # --- Bpreshuffle for weight ---
    # Preshuffle weight.  Default "1" (enabled)
    "ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE": lambda: (
        os.getenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1") == "1"
    ),
    "ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM": lambda: (
        os.getenv("ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM", "0") == "1"
    ),
    # --- V4 Attention Backend Refactor (PR-A: kill .item(), unlock CUDAGraph) ---
    # `legacy` (default) keeps the per-seq Python dispatch loop with .item()
    # syncs in deepseek_v4.py. `new` routes through V4AttentionBackend with
    # batched Triton kernels (no GPU→CPU sync, CUDAGraph-capturable).
    # During Phase 1/2 migration, individual sites can be flipped to `new`
    # for byte-equal A/B verification via dump-bisect.
    "ATOM_V4_BACKEND": lambda: os.getenv("ATOM_V4_BACKEND", "legacy"),
    # Comma-separated layer ids to route through the new backend (others stay
    # legacy). Empty means: respect ATOM_V4_BACKEND for all layers. Used for
    # layer-by-layer bisect during migration. Example: "0,3,15,30".
    "ATOM_V4_BACKEND_LAYERS": lambda: os.getenv("ATOM_V4_BACKEND_LAYERS", ""),
    # --- Debug Dump (atom/utils/debug_helper/) ---
    # All disabled (empty / no-op) by default. Set to enable instrumentation
    # for forward / weight / sampler bisecting; safe to leave wired in
    # production paths.
    #
    # Forward hidden_state dump per Block.
    "ATOM_FWD_DUMP_DIR": lambda: os.getenv("ATOM_FWD_DUMP_DIR", ""),
    "ATOM_FWD_DUMP_LAYERS": lambda: os.getenv("ATOM_FWD_DUMP_LAYERS", ""),
    # Override for non-DeepSeek models (e.g. "DecoderLayer" for Llama).
    "ATOM_FWD_DUMP_BLOCK_CLASS": lambda: os.getenv(
        "ATOM_FWD_DUMP_BLOCK_CLASS", "Block"
    ),
    "ATOM_FWD_DUMP_LAYER_ATTR": lambda: os.getenv(
        "ATOM_FWD_DUMP_LAYER_ATTR", "layer_id"
    ),
    "ATOM_FWD_DUMP_ONE_SHOT": lambda: os.getenv("ATOM_FWD_DUMP_ONE_SHOT", "1") == "1",
    # Per-rank weight dump + sys.exit(0) — for byte-equal weight comparison.
    "ATOM_WEIGHT_DUMP_DIR": lambda: os.getenv("ATOM_WEIGHT_DUMP_DIR", ""),
    "ATOM_WEIGHT_DUMP_LAYERS": lambda: os.getenv("ATOM_WEIGHT_DUMP_LAYERS", "0"),
    "ATOM_WEIGHT_DUMP_EXIT": lambda: os.getenv("ATOM_WEIGHT_DUMP_EXIT", "1") == "1",
    # Sampler top-K logits log — int K, 0/empty disables.
    "ATOM_DEBUG_TOPK": lambda: int(os.getenv("ATOM_DEBUG_TOPK", "0") or "0"),
    "ATOM_DEBUG_TOPK_PATH": lambda: os.getenv("ATOM_DEBUG_TOPK_PATH", ""),
    # KV cache event publisher (see atom/distributed/kv_events.py).
    "ATOM_KV_EVENTS_ENABLE": lambda: os.getenv("ATOM_KV_EVENTS_ENABLE", "0") == "1",
    "ATOM_KV_EVENTS_PUBLISHER": lambda: os.getenv("ATOM_KV_EVENTS_PUBLISHER", "zmq"),
    "ATOM_KV_EVENTS_ENDPOINT": lambda: os.getenv(
        "ATOM_KV_EVENTS_ENDPOINT", "tcp://127.0.0.1:5557"
    ),
    "ATOM_KV_EVENTS_TOPIC": lambda: os.getenv("ATOM_KV_EVENTS_TOPIC", ""),
    "ATOM_KV_EVENTS_HWM": lambda: int(os.getenv("ATOM_KV_EVENTS_HWM", "0") or "0"),
    "ATOM_KV_EVENTS_BUFFER_STEPS": lambda: int(
        os.getenv("ATOM_KV_EVENTS_BUFFER_STEPS", "10000") or "10000"
    ),
    # Force-skip the draft-model forward in eagle/MTP propose() and return
    # sentinel draft token ids (int max) so rejection_sampler rejects all
    # speculative tokens. Used to reproduce 100% rejection behavior — the
    # worst case for ring-buffer aliasing in compressor state caches.
    # Default: False (run the draft model normally).
    "ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL": lambda: (
        os.getenv("ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL", "0") == "1"
    ),
    # --- PrefillDelayer (cross-DP prefill alignment) ---
    # Master switch; default on. Set "0" to disable construction.
    # The delayer is a prefill COALESCER: it holds back prefill admission under
    # DP-attention until the accumulated prefill fills a worthwhile forward, so
    # fragmented short-input prefills / small partial tail chunks batch into one
    # forward instead of firing many tiny ones.
    "ATOM_ENABLE_PREFILL_DELAYER": lambda: (
        os.getenv("ATOM_ENABLE_PREFILL_DELAYER", "1") == "1"
    ),
    # Fill target: release prefill once accumulated pending tokens reach
    # target_fill * max_num_batched_tokens (averaged across prefillable ranks).
    # In (0, 1]; higher batches harder (fewer, larger prefills) at some TTFT
    # cost. Default 0.7.
    "ATOM_PREFILL_DELAYER_TARGET_FILL": lambda: float(
        os.getenv("ATOM_PREFILL_DELAYER_TARGET_FILL", "0.7")
    ),
    # TTFT bound: max consecutive scheduler ticks a held prefill waits before
    # force-release (deterministic across ranks; replaces the old wall-clock +
    # pass-count pair).
    "ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS", "30")
    ),
    # Tight bound (ticks) for a held mid-chunked-prefill: a partial holds already
    # allocated KV, so it force-releases sooner than a fresh prefill.
    "ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS", "8")
    ),
    # Consecutive non-growing ticks after which the coalescer gives up waiting
    # (burst ended, more won't come) and releases.
    "ATOM_PREFILL_DELAYER_STALL_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_STALL_TICKS", "3")
    ),
    # KV high watermark: at/above this KV usage a prefillable rank force-releases
    # (can't accumulate a bigger batch anyway).
    "ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK": lambda: float(
        os.getenv("ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK", "0.9")
    ),
    # Optional KV-usage low watermark: below it a prefillable rank force-releases
    # (GPU starving — feed it). Empty string => None => disabled.
    "ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK": lambda: (
        None
        if os.getenv("ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK", "") == ""
        else float(os.getenv("ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK"))
    ),
    # TTFT SLA guard: if any rank's oldest schedulable waiting prefill has queued
    # (since arrival) >= this many ms, force-release regardless of the fill
    # target. Bounds worst-case TTFT. Empty string => None => disabled (set this
    # to your TTFT budget in ms to activate; a small value under heavy backlog
    # will fire every tick and defeat coalescing, so size it to the SLA).
    "ATOM_PREFILL_DELAYER_MAX_QUEUE_MS": lambda: (
        None
        if os.getenv("ATOM_PREFILL_DELAYER_MAX_QUEUE_MS", "") == ""
        else float(os.getenv("ATOM_PREFILL_DELAYER_MAX_QUEUE_MS"))
    ),
    # --- TBO prefill ubatch splitting ---
    # Split prefill ubatches at the exact token midpoint (vLLM-DBO style),
    # cutting through a request if needed for perfectly balanced 50/50 ubatches.
    # Default on; set "0" to fall back to the request-boundary balanced split.
    "ATOM_TBO_PREFILL_TOKEN_SPLIT": lambda: (
        os.getenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", "1") == "1"
    ),
    # Minimum prefill tokens (per rank) required to TBO-split.
    "ATOM_TBO_PREFILL_MIN_TOKENS": lambda: int(
        os.getenv("ATOM_TBO_PREFILL_MIN_TOKENS", "8192")
    ),
    # --- NUMA binding ---
    # Master switch: pin each GPU worker to its GPU-local NUMA node's CPU cores
    # and preferred memory. Default off so baseline/pinned A/B stays clean.
    "ATOM_NUMA_BIND": lambda: os.getenv("ATOM_NUMA_BIND", "0") == "1",
    # Auto-detect the GPU->NUMA-node mapping (amdsmi first, sysfs fallback).
    # Default on, so `ATOM_NUMA_BIND=1` alone is zero-config.
    "ATOM_AUTO_NUMA_BIND": lambda: os.getenv("ATOM_AUTO_NUMA_BIND", "1") == "1",
    # Explicit per-global-rank node ids (comma separated), overriding auto, e.g.
    # ATOM_NUMA_NODE="0,0,0,0,1,1,1,1". A single value applies to all ranks.
    "ATOM_NUMA_NODE": lambda: os.getenv("ATOM_NUMA_NODE", ""),
    # Raise instead of warn when binding fails.
    "ATOM_CRASH_ON_NUMA_BIND_FAILURE": lambda: (
        os.getenv("ATOM_CRASH_ON_NUMA_BIND_FAILURE", "0") == "1"
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
