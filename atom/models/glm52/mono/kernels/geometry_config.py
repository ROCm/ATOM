# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""GLM-5.2's own geometry, and the widths its kernel is built for.

The vocabulary these are expressed in -- ``LayerConfig``, ``MoeMode``, the weight
and cache layouts -- is model-free and lives in ``atom.mono.weights.config``;
what is GLM-5.2's alone is here. The shared names are re-exported, so a kernel
source taking both from one place need not care which is which.
"""

from __future__ import annotations

from dataclasses import replace

from atom.mono.weights.config import (  # noqa: F401  (re-exported)
    AttentionWeight,
    EPS,
    ExpertActivation,
    ExpertWeight,
    FP8_MAX,
    KvCacheLayout,
    LayerConfig,
    MAX_LAYERS_PER_STEP,
    MOE_FORMATS,
    MoeFormat,
    MoeMode,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    RouterWeightLayout,
    SCALE_BM,
    SUPPORTED_PEERS,
    SUPPORTED_SAMPLES,
    as_kv_cache_layout,
    as_layer_config,
    as_moe_mode,
    as_mxfp4_scale_layout,
    as_mxfp4_weight_layout,
    as_router_weight_layout,
    moe_format,
    register_layer_config,
)


GLM5_CONFIG = register_layer_config(
    LayerConfig(
    name="glm5",
    hidden=6144,
    q_lora=2048,
    kv_lora=512,
    pe_dim=64,
    nope_dim=192,
    v_dim=256,
    n_experts=256,
    top_k=8,
    inter=256,
    route_scale=2.5,
    local_heads=8,
    )
)


GLM5_REFERENCE_TP = 8


GLM5_TP_SIZES = (4, GLM5_REFERENCE_TP)


GLM5_GRAPH_BATCHES = tuple(range(1, 97))


GLM5_AGENTX_BATCHES = (1, 2)


GLM5_QUERY_LENGTHS = (1, 4, 5, 6)


GLM5_KERNEL_SAMPLES = (1, 2, 4, 5, 6, 8, 10, 12)


GLM5_GLOBAL_HEADS = GLM5_CONFIG.local_heads * GLM5_REFERENCE_TP


GLM5_GLOBAL_INTER = GLM5_CONFIG.inter * GLM5_REFERENCE_TP


def glm5_tp_config(tp_size: int) -> LayerConfig:
    """Return the one GLM-5 shard geometry for ``tp_size``."""

    if tp_size not in GLM5_TP_SIZES:
        raise ValueError(
            f"GLM-5 tensor parallel size must be one of {GLM5_TP_SIZES}, got {tp_size}"
        )
    return replace(
        GLM5_CONFIG,
        local_heads=GLM5_GLOBAL_HEADS // tp_size,
        inter=GLM5_GLOBAL_INTER // tp_size,
    )


def glm5_attention_heads(tp_size: int, dcp_size: int = 1) -> int:
    if dcp_size not in (1, 4) or tp_size % dcp_size:
        raise ValueError(
            f"unsupported GLM-5 TP/DCP geometry: tp={tp_size}, dcp={dcp_size}"
        )
    return GLM5_GLOBAL_HEADS // (tp_size // dcp_size)


def glm5_kernel_samples(samples: int, query_length: int) -> int:
    """Choose an LDS-safe request-aligned launch width."""

    if query_length not in GLM5_QUERY_LENGTHS or samples % query_length:
        raise ValueError(
            f"unsupported GLM-5 decode shape samples={samples}, query_length={query_length}"
        )
    for chunk in reversed(GLM5_KERNEL_SAMPLES):
        if chunk % query_length == 0 and samples % chunk == 0:
            return chunk
    raise ValueError(
        f"no GLM-5 kernel chunk for samples={samples}, query_length={query_length}"
    )


HIDDEN = GLM5_CONFIG.hidden


Q_LORA = GLM5_CONFIG.q_lora


KV_LORA = GLM5_CONFIG.kv_lora


PE_DIM = GLM5_CONFIG.pe_dim


NOPE_DIM = GLM5_CONFIG.nope_dim


V_DIM = GLM5_CONFIG.v_dim


QKV_A_ROWS = GLM5_CONFIG.qkv_a_rows


N_EXPERTS = GLM5_CONFIG.n_experts


EXPERT_TOP_K = GLM5_CONFIG.top_k


TOP_K = EXPERT_TOP_K


MOE_SLOTS = GLM5_CONFIG.moe_slots


SHARED_EXPERT = GLM5_CONFIG.shared_expert


INTER = GLM5_CONFIG.inter


ROUTE_SCALE = GLM5_CONFIG.route_scale


SOFTMAX_SCALE = GLM5_CONFIG.softmax_scale


LOCAL_HEADS = GLM5_CONFIG.local_heads


def validate_shard(
    samples: int,
    heads: int,
    rank: int,
    npes: int,
    sparse_attention_topk: int,
    model_config: LayerConfig | str = GLM5_CONFIG,
    supported_samples=SUPPORTED_SAMPLES,
    expected_heads: int | None = None,
) -> None:
    """Validate one model profile before allocating GPU buffers."""

    config = as_layer_config(model_config)
    if samples not in supported_samples:
        raise ValueError(f"samples must be one of {supported_samples}, got {samples}")
    expected_heads = config.local_heads if expected_heads is None else expected_heads
    if heads != expected_heads:
        raise ValueError(
            f"{config.name} requires {expected_heads} local heads, got {heads}"
        )
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if sparse_attention_topk <= 0 or sparse_attention_topk % 64:
        raise ValueError(
            "sparse_attention_topk must be a positive multiple of 64, "
            f"got {sparse_attention_topk}"
        )
