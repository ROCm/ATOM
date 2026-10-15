# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Static DeepSeek-V4 shard dimensions (V4-Pro at TP8)."""

from __future__ import annotations

# Shared-KV MQA: one head_dim vector per token is both K and V, RoPE in its tail.
HIDDEN = 7168
Q_LORA = 1536
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448, the FP8-quantized part of the KV row
O_LORA = 1024
O_GROUPS = 2  # local; 16 global / 8 ranks
WINDOW = 128
# ATOM's V4 KV block size; a block holds BLOCK_TOKENS // ratio compressed entries.
BLOCK_TOKENS = 256

N_EXPERTS = 384
TOP_K = 6
MOE_SLOTS = 1 + TOP_K
SHARED_EXPERT = N_EXPERTS  # the shared expert sits last in the bank
INTER = 384  # local; 3072 global / 8 ranks
ROUTE_SCALE = 2.5
SWIGLU_LIMIT = 10.0

HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
HC_MIX = (2 + HC_MULT) * HC_MULT  # pre | post | comb, packed in that order

EPS = 1e-6
SCALE_BM = 128
FP8_MAX = 448.0
SOFTMAX_SCALE = HEAD_DIM**-0.5

SUPPORTED_SAMPLES = tuple(range(1, 9))
# the launch widths a plain decode step pads to (samples a launch)
LAUNCH_WIDTHS = (1, 2, 4, 8)
MAX_TOKENS = LAUNCH_WIDTHS[-1]
SUPPORTED_PEERS = (1, 2, 4, 8)
# The split-attention kernel requires heads % WAVES == 0 and heads <= 16.
SUPPORTED_HEADS = (8, 16)
# Launches per step, each with its own epoch tag (61 layers, up to 4 each with MTP).
MAX_LAYERS_PER_STEP = 256

# V4-Pro compress_ratios: [128, 128] + [4, 128] * 29 + [4] + [0] -- layers 0-1 HCA, then
# CSA iff even; no ratio-0 main layer, the trailing 0 is the MTP block.
COMPRESS_SWA = 0  # sliding window only (the MTP block)
COMPRESS_CSA = 4  # CSA, with the lightning indexer
COMPRESS_HCA = 128  # HCA, dense over compressed entries


def compress_ratios(n_layers: int = 61, n_mtp: int = 1) -> tuple[int, ...]:
    """V4-Pro's per-layer schedule, MTP entries appended.

    Checkpoints may ship more entries than layers (-0813 has 64), so validate with >=.
    """
    main = [COMPRESS_HCA, COMPRESS_HCA] + [
        COMPRESS_CSA if i % 2 == 0 else COMPRESS_HCA for i in range(2, n_layers)
    ]
    return tuple(main + [COMPRESS_SWA] * n_mtp)


COMPRESS_ROPE_THETA = 1.6e5
# Lightning indexer (CSA only). Replicated on every rank, as ATOM: each rank scores with all
# 64 heads, so the scores are bit-identical across ranks with no all-reduce.
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 1024
KEY_BLOCK = 64  # the split-attention key tile; the index list is padded to it

# The checkpoint the kernel is built for (DeepSeek-V4-Pro): every config.json value it
# bakes in, global (before the TP8 split). A checkpoint differing in any is refused.
V4_PRO_CONFIG = {
    "hidden_size": HIDDEN,
    "q_lora_rank": Q_LORA,
    "o_lora_rank": O_LORA,
    "head_dim": HEAD_DIM,
    "qk_rope_head_dim": ROPE_DIM,
    "num_attention_heads": 128,
    "o_groups": 16,
    "n_routed_experts": N_EXPERTS,
    "num_experts_per_tok": TOP_K,
    "moe_intermediate_size": 3072,
    "n_shared_experts": 1,
    "sliding_window": WINDOW,
    "routed_scaling_factor": ROUTE_SCALE,
    "swiglu_limit": SWIGLU_LIMIT,
    "rms_norm_eps": EPS,
    "scoring_func": "sqrtsoftplus",
    "norm_topk_prob": True,
    "index_n_heads": INDEX_HEADS,
    "index_head_dim": INDEX_HEAD_DIM,
    "index_topk": INDEX_TOPK,
    "hc_mult": HC_MULT,
    "hc_sinkhorn_iters": HC_SINKHORN_ITERS,
    "hc_eps": HC_EPS,
    "compress_rope_theta": COMPRESS_ROPE_THETA,
    "num_hidden_layers": 61,
}


def checkpoint_refusal(hf_config) -> str | None:
    """Why a checkpoint is not the one the kernel is built for, or None."""
    for key, want in V4_PRO_CONFIG.items():
        got = getattr(hf_config, key, None)
        if got != want:
            return f"{key} {got!r}, the kernel's is {want!r}"
    n = hf_config.num_hidden_layers
    ratios = tuple(getattr(hf_config, "compress_ratios", ()))[:n]
    if ratios != compress_ratios(n, n_mtp=0):
        return "compress_ratios differ from V4-Pro's layer schedule"
    return None


def validate_shard(
    samples: int,
    heads: int,
    rank: int,
    npes: int,
    window: int = WINDOW,
    compress_ratio: int = COMPRESS_SWA,
) -> None:
    """Validate the V4 shard contract before allocating GPU buffers."""
    if samples not in SUPPORTED_SAMPLES:
        raise ValueError(f"samples must be one of {SUPPORTED_SAMPLES}, got {samples}")
    if heads not in SUPPORTED_HEADS:
        raise ValueError(f"heads must be one of {SUPPORTED_HEADS}, got {heads}")
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if window <= 0 or window % 64:
        raise ValueError(f"window must be a positive multiple of 64, got {window}")
