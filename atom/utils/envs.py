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
from collections.abc import Callable
from typing import Any

environment_variables: dict[str, Callable[[], Any]] = {
    # --- Data Parallelism ---
    "ATOM_DP_RANK": lambda: int(os.getenv("ATOM_DP_RANK", "0")),
    "ATOM_DP_RANK_LOCAL": lambda: int(os.getenv("ATOM_DP_RANK_LOCAL", "0")),
    "ATOM_DP_SIZE": lambda: int(os.getenv("ATOM_DP_SIZE", "1")),
    "ATOM_DP_SIZE_LOCAL": lambda: int(os.getenv("ATOM_DP_SIZE_LOCAL", "1")),
    "ATOM_DP_MASTER_IP": lambda: os.getenv("ATOM_DP_MASTER_IP", "127.0.0.1"),
    "ATOM_DP_MASTER_PORT": lambda: int(os.getenv("ATOM_DP_MASTER_PORT", "29500")),
    # Rendezvous base port; set per role when prefill/decode share a node.
    "ATOM_DP_BASE_PORT": lambda: int(os.getenv("ATOM_DP_BASE_PORT", "0")),
    # Token-equivalent cost of one in-flight request for the "least_tokens" DP
    # load-balance strategy. The per-rank load score is
    #   sum(prompt_tokens) + ATOM_DP_LB_REQ_EQUIV * num_in_flight_requests
    # so a larger value biases routing toward request-count balance (decode
    # pressure) and a smaller value toward prompt-token balance (prefill
    # pressure). See engine_core_mgr.CoreManager._select_dp_rank_locked.
    "ATOM_DP_LB_REQ_EQUIV": lambda: int(os.getenv("ATOM_DP_LB_REQ_EQUIV", "512")),
    # Place a new agent session on the lightest DP rank, then keep every later
    # request on that immutable cache owner. Existing sessions never spill;
    # child correlation ids are independently load-placed rather than
    # inheriting their parent's owner.
    "ATOM_DP_SESSION_AFFINITY": lambda: os.getenv(
        "ATOM_DP_SESSION_AFFINITY", "0"
    ).lower()
    in {"1", "true", "yes", "on"},
    # Prefix for process titles set via set_process_title (shown in ps/top/rocm-smi)
    "ATOM_PROCESS_NAME_PREFIX": lambda: os.getenv("ATOM_PROCESS_NAME_PREFIX", "ATOM"),
    # SGLang's GLM-5.2 and DeepSeek V4 prefill CP paths still force
    # attention TP size to 1.
    # ATOM remaps the SGLang world into internal TP x PCP groups.
    # 0 means unset.
    "ATOM_SGLANG_PCP_SIZE": lambda: int(os.getenv("ATOM_SGLANG_PCP_SIZE", "0") or "0"),
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
    # Use mori dispatch_combine_v2 (FlyDSL/cco, gfx1250 wave32) instead of the
    # production mori v1 (mori.ops.EpDispatchCombineOp) for the EP+DP MoE
    # all2all. v1 is authored for gfx942/950 and does not run on gfx1250; v2 is
    # the gfx1250-capable path. Only takes effect when the mori all2all path is
    # active (dp_size>1 + expert-parallel + mori installed).
    "ATOM_MORI_V2": lambda: os.getenv("ATOM_MORI_V2", "0") == "1",
    # gemm2-fused EP combine: the a8w4 grouped gemm2 epilogue P2P-writes its
    # weighted per-(token,k) results straight into the peers' combine staging, so
    # combine only barriers + sums. Requires the a8w4 (fp8 act + mxfp4 weight)
    # path; ignored unless ATOM_MORI_V2 is on.
    # This also selects the transport: 1 binds aiter's MegaMoEGfx1250, which owns
    # the fused dispatch/combine pair (its dispatch kernel is picked by aiter's
    # own MEGA_DISPATCH=flydsl|mori), 0 binds mori's v2 op-layer running plain
    # gather, i.e. the untouched upstream baseline.
    "ATOM_MORI_V2_FUSED": lambda: os.getenv("ATOM_MORI_V2_FUSED", "0") == "1",
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
    # QK-norm-rope-cache-quant fusion for Qwen3 dense and MoE; disabled by default.
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
    # Set to 0 to stop a refused state-cache hit from placing a checkpoint of
    # its own, leaving the prompt-end anchor as the only placement. Overrides
    # --state-checkpoint-demand so the policy can be flipped without touching a
    # launch script. The rung's write traffic may cost more in evictions than
    # its reuse is worth — see `BlockManager._record_checkpoint_demand`.
    "ATOM_STATE_CHECKPOINT_DEMAND": lambda: (
        os.getenv("ATOM_STATE_CHECKPOINT_DEMAND", "1") == "1"
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
    # GLM-5.3 pooled sparse indexer. Disabling it is an exact A/B only while
    # sequence length <= index_topk; the model refuses longer requests.
    "ATOM_GLM5_KPOOL": lambda: os.getenv("ATOM_GLM5_KPOOL", "1") == "1",
    # Bring-up/debug controls for the GLM-5.3 text path.
    "ATOM_GLM5_FORCE_DENSE_MLA": lambda: (
        os.getenv("ATOM_GLM5_FORCE_DENSE_MLA", "0") == "1"
    ),
    "ATOM_GLM5_DISABLE_FUSED_MHC": lambda: (
        os.getenv("ATOM_GLM5_DISABLE_FUSED_MHC", "0") == "1"
    ),
    # Kimi-K3 DSpark draft: fuse the per-layer context-row KV write
    # (K3DSparkMLAAttention.write_context_kv) into one Triton kernel --
    # RMSNorm(kv_c) + rope(k_pe) + concat + paged-cache store, versus today's
    # four launches plus a throwaway `empty_like` for the rope's query side.
    # That chain runs once per draft layer (5) per drafting step over every
    # scheduled target token, and each op re-reads the 576-wide latent from HBM
    # only to hand it to the next, so the win is launch count and round trips,
    # not FLOPs. Falls back per call when the cache layout or the rope is not
    # the plain one the kernel understands (see
    # MLAAttention.write_context_kv_latent). Measured on Kimi-K3 (MI355X, TP8,
    # fp8 KV): one 4.65us kernel replaces a 14us three-kernel chain, saving 39us
    # per drafting step at B=1 and ~36us at B=64 -- 0.1% of the step, since the
    # draft runs under a cudagraph and the launches this removes cost host time,
    # not device time. Against the per-op chain the stored latent differs on
    # ~1 element in 5M, where it is the fused kernel that matches an fp64
    # reference: aiter's RMSNorm rounds x*rstd to bf16 before applying w.
    # Set to "0" to force the per-op chain if a regression is suspected. That
    # chain is the eligibility fallback above, not debug code, so it stays
    # reachable (and runs the first write of every layer) either way.
    "ATOM_DSPARK_FUSED_CTX_KV": lambda: (
        os.getenv("ATOM_DSPARK_FUSED_CTX_KV", "1") == "1"
    ),
    "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION": lambda: (
        os.getenv("ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION", "1") == "1"
    ),
    # DSpark block sampling: replace the Markov head's
    #   bias = W1[x] @ W2.float().t() ; argmax(base_logits + bias)
    # with one fused Triton kernel (atom/model_ops/dspark_markov_sample.py).
    # The unfused spelling casts the whole [V, r] W2 table to fp32 INSIDE the
    # T-iteration loop (~252MB of traffic per iteration for a table that never
    # changes) and materializes two [B, V] fp32 tensors per iteration (42MB
    # each at B=64, V=163840) that only an argmax ever reads. The fused path
    # keeps W2 bf16, adds the base logits in the GEMM epilogue and reduces to
    # ids in registers, so W2 is read exactly once per block position and no
    # [B, V] intermediate exists. Covers both native DSpark block samplers,
    # Kimi-K3 (r=256) and DeepSeek-V4 (r=512); the op is shape-generic and
    # hands anything it cannot index back to the reference, but only K3 has
    # been run on hardware.
    # The bias GEMV moves from an fp32 matmul over an fp32 copy of W2 to bf16
    # MFMA with an fp32 accumulator. Every product is exact in fp32 either way,
    # so the result is equal to the reference up to accumulation order; whether
    # an argmax anywhere flips on a last-ulp tie is empirical. Measured on
    # Kimi-K3 (MI355X, TP8, fp8 KV, full GSM8K at 64 concurrency): acceptance
    # 87.08% against 87.06% unfused, accept-length distribution equal to within
    # 0.1pp, flexible-extract inside the run-to-run band (0.9477 / 0.9591 fused,
    # 0.9522 unfused). Saves 145 us per drafting step at B=1 and ~235 us at
    # B=64. Set to "0" to force the reference spelling if an acceptance-rate
    # regression is suspected -- the fastest way to rule this kernel in or out,
    # since the two paths are not bit-identical by construction. Read once when
    # the Markov head is built, so it must be set before the server starts.
    "ATOM_DSPARK_FUSED_MARKOV_SAMPLE": lambda: (
        os.getenv("ATOM_DSPARK_FUSED_MARKOV_SAMPLE", "1") == "1"
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
    # --- ReplaySSM (linear-attention state) ---
    # Cache the SSM *inputs* (k, u, g) instead of one full recurrent state per
    # speculative token.  The state pool stops scaling with the MTP window --
    # e.g. Kimi-K3 at mtp_k=2, 64 seqs, tp=8 drops from 10.3 GiB to ~3.5 GiB --
    # and the full-state write moves off the per-step path (rewritten only when
    # the record buffer fills).  Rollback becomes a cursor move.
    # See atom/model_ops/fla_ops/replayssm.py.
    "ATOM_ENABLE_REPLAYSSM": lambda: (
        None
        if os.getenv("ATOM_ENABLE_REPLAYSSM") is None
        else os.getenv("ATOM_ENABLE_REPLAYSSM", "0").lower() == "1"
    ),
    # Record-buffer depth L.  Must be >= 2*(mtp_k+1); raised automatically if
    # set too low.  Larger L means fewer checkpoint write-backs but a longer
    # rebuild each step; upstream measured 8-16 as the sweet spot.
    "ATOM_REPLAYSSM_CACHE_LEN": lambda: int(
        os.getenv("ATOM_REPLAYSSM_CACHE_LEN", "16")
    ),
    # "auto" | "serial" | "ut".  The UT-transform verify route only beats the
    # serial one at verify windows >= ~12 tokens (measured on gfx950), so
    # "auto" keeps practical MTP windows on the serial route.
    "ATOM_REPLAYSSM_ROUTE": lambda: os.getenv("ATOM_REPLAYSSM_ROUTE", "auto").lower(),
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT": lambda: (
        os.getenv("ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT", "1") == "1"
    ),
    "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT": lambda: (
        os.getenv("ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT", "1") == "1"
    ),
    # --- Profiling & Logging ---
    "ATOM_TORCH_PROFILER_DIR": lambda: os.getenv("ATOM_TORCH_PROFILER_DIR", None),
    # Move the startup heap (model, compiled graph, tokenizer, KV block pool)
    # into CPython's permanent generation once warmup is done, so collections
    # stop scanning it.  On by default; set 0 to keep the old behaviour.
    # See freeze_gc_heap in atom/utils/gc_utils.py.
    "ATOM_GC_FREEZE": lambda: os.getenv("ATOM_GC_FREEZE", "1") == "1",
    # Log every garbage collection: generation, duration, objects reclaimed.
    "ATOM_GC_DEBUG": lambda: os.getenv("ATOM_GC_DEBUG", "0") == "1",
    # "t0,t1,t2" for gc.set_threshold(); empty keeps CPython's default.
    # Read independently by the API server, each EngineCore and each
    # ModelRunner worker -- thresholds are per-interpreter.  A fallback for
    # ATOM_GC_FREEZE=0: freezing removes the cost of a pass, this only spaces
    # the passes out.  See tune_gc in atom/utils/gc_utils.py.
    "ATOM_GC_THRESHOLD": lambda: os.getenv("ATOM_GC_THRESHOLD", "").strip(),
    "ATOM_PROFILER_MORE": lambda: os.getenv("ATOM_PROFILER_MORE", "0") == "1",
    # When profiling is active, append detailed attention aggregates (sqsq, sqsk, sk)
    # to the prefill[]/decode[] trace labels emitted by ModelRunner.run_model.
    "ATOM_ENABLE_DETAILED_ANNOTATION": lambda: (
        os.getenv("ATOM_ENABLE_DETAILED_ANNOTATION", "0") == "1"
    ),
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
    # Worker threads for weight loading. >1 (default 16) enables the batched
    # parallel loader (per-fused-param CPU staging flushed with one H2D copy)
    # with that many threads; set to 1 to fall back to the original sequential
    # per-expert path.
    "ATOM_LOADER_NUM_THREADS": lambda: int(os.getenv("ATOM_LOADER_NUM_THREADS", "16")),
    # Warm the page cache with a background sequential reader instead of
    # leaving it to demand faults through the mmap. Measured on a local NVMe:
    # the fault-driven pattern sustains 3.2 GB/s where the device does 6.9 and
    # a single sequential reader alone reaches 6.06, so the gap is the access
    # pattern rather than queue depth. On DeepSeek-R1 MXFP4 (350 GiB, TP=4)
    # this takes a cold load from ~156s to ~68s and leaves the warm case
    # slightly better too, so it is on by default; turn it off for storage
    # where a second sequential reader competes with the loader instead of
    # feeding it.
    "ATOM_LOADER_PREFETCH": lambda: (
        os.getenv("ATOM_LOADER_PREFETCH", "true").lower() == "true"
    ),
    # Shards read concurrently by the prefetcher. Kept small: the device here
    # saturates at 2 streams, and every thread also competes with the loader.
    "ATOM_LOADER_PREFETCH_THREADS": lambda: int(
        os.getenv("ATOM_LOADER_PREFETCH_THREADS", "4")
    ),
    "ATOM_LOADER_PREFETCH_BLOCK_MB": lambda: int(
        os.getenv("ATOM_LOADER_PREFETCH_BLOCK_MB", "16")
    ),
    # Hint the kernel to read each shard ahead. Off by default because it never
    # paid for itself once measured on its own: the read loop runs far ahead of
    # the workers, so WILLNEED is issued for the whole checkpoint within
    # seconds, and asking for 350 GiB of read-ahead is bookkeeping the kernel
    # largely discards. Cold load 193.0s vs 194.2s -- no effect; warm load
    # 27.3s vs 42.8s -- 15.5s of pure overhead. Kept as a switch rather than
    # deleted so the behaviour can be restored on storage that behaves
    # differently. Ignored when prefetching, which supersedes it.
    "ATOM_LOADER_FADVISE": lambda: (
        os.getenv("ATOM_LOADER_FADVISE", "false").lower() == "true"
    ),
    # Fail loading when the checkpoint does not deliver every routed expert of
    # a fused MoE parameter. On by default: the alternative is a model that
    # loads happily with some expert slots left at their init values, which
    # only shows up much later as an accuracy drop. Set to false to downgrade
    # to a warning when bringing up a checkpoint that is known to be partial.
    "ATOM_LOADER_STRICT_COVERAGE": lambda: (
        os.getenv("ATOM_LOADER_STRICT_COVERAGE", "true").lower() == "true"
    ),
    # Quantize eligible modules as they load to reduce peak memory. Streaming
    # quantizes local TP shards, so results may differ slightly from offline.
    "ATOM_ONLINE_QUANT_STREAMING": lambda: (
        os.getenv("ATOM_ONLINE_QUANT_STREAMING", "0").lower() in ("1", "true")
    ),
    # Tail workers for H2D, quantization, and source release. More workers
    # increase overlap and in-flight memory; 0 runs inline.
    "ATOM_ONLINE_QUANT_STREAMING_THREADS": lambda: int(
        os.getenv("ATOM_ONLINE_QUANT_STREAMING_THREADS", "4")
    ),
    # Stage on the host and upload once per completed parameter. Disabling this
    # buffers loader calls on meta and forces a single-threaded checkpoint walk.
    "ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING": lambda: (
        os.getenv("ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING", "1").lower()
        in ("1", "true")
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
    # Pure-DP LM head strategy (only active when the model TP group is size 1 and
    # the DP group is size > 1, i.e. pure DP attention on decode; every other
    # case — TP>1, DP off, prefill, ragged/draft rows — auto-falls back to the
    # replicated path via `_can_use_dp_sharded_head`):
    #   "all2all"   -> shard vocab across the DP group + gather hidden, then a
    #                  single all-to-all delivers each rank its own rows x full
    #                  vocab (config ③, minimal comm). DEFAULT.
    #   "allgather" -> same shard/gather, but vocab all-gather + scatter rows
    #                  (config ②); kept for A/B and as a fallback.
    #   "default"   -> replicated full-vocab GEMM per DP rank (big GEMM, no comm);
    #                  explicit opt-out.
    "ATOM_DP_LM_HEAD_MODE": lambda: os.getenv(
        "ATOM_DP_LM_HEAD_MODE", "all2all"
    ).lower(),
    # Pure-DP draft greedy argmax: shard the draft lm_head vocab across the DP
    # group so each rank reads [H, V/dp] instead of the full [H, V], exchanging
    # only the packed [N, 2]. Active only for a unified (rectangular) pure-DP
    # draft step; everything else falls back to the replicated local argmax.
    "ATOM_DP_DRAFT_ARGMAX": lambda: os.getenv("ATOM_DP_DRAFT_ARGMAX", "1").lower()
    in ("1", "true"),
    # Row count (running_tokens) above which it falls back: the hidden gather
    # grows with rows, so the shard only pays at small M. ~256 is the V4-Pro
    # crossover.
    "ATOM_DP_DRAFT_ARGMAX_MAX_ROWS": lambda: int(
        os.getenv("ATOM_DP_DRAFT_ARGMAX_MAX_ROWS", "256")
    ),
    "ATOM_USE_FLYDSL_GDR": lambda: os.getenv("ATOM_USE_FLYDSL_GDR", "0").lower() == "1",
    # Capture each declared draft pass into a per-captured-size CUDAGraph as it is
    # warmed, so the draft replays instead of relaunching every kernel. 0 drafts
    # eagerly. On by default: DSpark tp1 acceptance is 65.25% captured vs 65.21%
    # eager. Named for the draft, not a flavor -- see `DraftGraph.will_capture`.
    "ATOM_DRAFT_CUDAGRAPH": lambda: (
        os.getenv("ATOM_DRAFT_CUDAGRAPH", "1").lower() == "1"
    ),
    # --- MoE (DeepSeek-style shared experts) ---
    # Dual-stream MoE only when num_tokens <= threshold; 0 disables dual-stream registration.
    "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD": lambda: int(
        os.getenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "1024")
    ),
    # Fuse into a per-rank replica without EPLB. EPLB always fuses it as routed.
    "ATOM_FUSE_SHARED_EXPERT": lambda: (
        os.getenv("ATOM_FUSE_SHARED_EXPERT", "1").lower() == "1"
    ),
    # Opt-in: MoE shared||routed fork inside a PIECEWISE-captured piece. Capture
    # holds it and GSM8K is unmoved; off until a throughput win is shown. Shared
    # dispatcher, so this moves V2/V3.2/K3 too. See docs.
    "ATOM_DUAL_STREAM_PIECEWISE": lambda: os.getenv("ATOM_DUAL_STREAM_PIECEWISE", "0")
    == "1",
    # Gate/Up interleave mode for MoE weight preshuffle and kernel gate_mode.
    # "0" (default) = SEPARATED layout; "1" = INTERLEAVE layout.
    "ATOM_MOE_GU_ITLV": lambda: os.getenv("ATOM_MOE_GU_ITLV", "0") == "1",
    # --- MoE all2all (MoRI) wire format ---
    "ATOM_MORI_FP4_DISPATCH": lambda: (os.getenv("ATOM_MORI_FP4_DISPATCH", "0") == "1"),
    # Combine-side codec. "none" (the MoRI default) sends bf16 back;
    # "fp8_blockwise" selects EpCombineIntraNodeKernel_*_fp8bwq_*.
    "ATOM_MORI_COMBINE_QUANT": lambda: os.getenv("ATOM_MORI_COMBINE_QUANT", "none"),
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
    # Run the DSpark draft model eager, bypassing torch.compile, while leaving
    # the target compiled. Rollback lever for the draft's compiled path; prefer
    # it over --level 0, which disables compilation for BOTH models. Note that
    # --enforce-eager does NOT disable it: support_torch_compile keys off
    # compilation_config.level only (see atom/utils/decorators.py:485).
    # Default: False (compile the draft).
    "ATOM_DSPARK_DISABLE_COMPILE": lambda: (
        os.getenv("ATOM_DSPARK_DISABLE_COMPILE", "0") == "1"
    ),
    # NOTE: DSpark runtime knobs (confidence_schedule, ragged,
    # ragged_graph_sizes, q_buckets, disable_sps_calib) are no longer env vars.
    # They are configured via --dspark-config (JSON dict) and carried in
    # config.dspark (see atom/config.py DSparkConfig). See
    # recipes/DSpark.md.
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
    # cost. Default 0.9.
    "ATOM_PREFILL_DELAYER_TARGET_FILL": lambda: float(
        os.getenv("ATOM_PREFILL_DELAYER_TARGET_FILL", "0.9")
    ),
    # TTFT bound: max consecutive scheduler ticks a held prefill waits before
    # force-release (deterministic across ranks; replaces the old wall-clock +
    # pass-count pair).
    "ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS", "200")
    ),
    # Tight bound (ticks) for a held mid-chunked-prefill: a partial holds already
    # allocated KV, so it force-releases sooner than a fresh prefill.
    "ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS", "100")
    ),
    # Consecutive non-growing ticks after which the coalescer gives up waiting
    # (burst ended, more won't come) and releases.
    "ATOM_PREFILL_DELAYER_STALL_TICKS": lambda: int(
        os.getenv("ATOM_PREFILL_DELAYER_STALL_TICKS", "10")
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
    # After a prefill forward, protect this many scheduler passes for decode
    # before allowing another prefill. Mirrors SGLang's
    # --prefill-decode-interval; 0 disables the hard interval.
    "ATOM_PREFILL_DECODE_INTERVAL": lambda: int(
        os.getenv("ATOM_PREFILL_DECODE_INTERVAL", "0")
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
    # --- PCP MoE comm mode ---
    # Fold the PCP (prefill-context-parallel) dim into the MoE tp/ep sharding.
    # Only meaningful when prefill_context_parallel_size > 1;
    # Default "1": all-gather hidden 1/W -> full before MoE and slice
    # full -> 1/W after, so MoE sees the complete token set (MoE itself is
    # untouched / PCP-agnostic). Costs one extra hidden all-gather per layer.
    # "0": MoE runs on each rank's 1/W token shard with no extra comm.
    "ATOM_PCP_MOE_MERGE": lambda: os.getenv("ATOM_PCP_MOE_MERGE", "1") == "1",
    # Pure-TP TBO all_reduce overlap mode (see module_dispatch_ops.tbo_all_reduce):
    #   "overlap" (default): move the AR onto the comm stream so it overlaps the
    #             partner ubatch's compute. Per-ubatch pynccl comms keep the
    #             cross-rank enqueue order consistent (hang-free).
    #   "inline": run the AR on the current stream, no overlap. Plan-A baseline.
    "ATOM_TBO_TP_AR_MODE": lambda: os.getenv("ATOM_TBO_TP_AR_MODE", "overlap"),
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
    # PP-boundary hidden_states/residual are TP-replicated; when on, each rank
    # sends only its 1/tp_size slice and the receiver all-gathers, cutting PP
    # link traffic by tp_size. Default on; set "0" for full-tensor sends.
    "ATOM_PP_SEND_ALLGATHER": lambda: os.getenv("ATOM_PP_SEND_ALLGATHER", "1") == "1",
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
# LMCACHE_EC_PIN_TIMEOUT_SEC      — LMCache library's own source-pin timeout;
#                                   read in kv_transfer/offload/_offload_common.py
#                                   (offload_save_abandon_timeout_s) to derive the
#                                   engine's save-abandon window from it, so the
#                                   two stay ordered. The scheduler never reads the
#                                   env itself -- it asks the connector, via
#                                   save_abandon_timeout_s. ATOM does not own the
#                                   knob, hence no default of its own here.
# OFFLOAD_MAX_PENDING_SAVES       — offload connector queue-depth bound;
#                                   defined/defaulted in
#                                   kv_transfer/offload/_offload_common.py and
#                                   documented in kv_transfer/offload/README.md.
#                                   The state tier shares it (scheduler.py)
#                                   rather than adding a second knob.
