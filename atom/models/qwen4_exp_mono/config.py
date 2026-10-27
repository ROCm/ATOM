# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The exact Qwen3.8-Flash-Next deployment served by the mono runtime."""

from copy import copy

import torch

from atom.models.qwen4_exp_mono.shapes import (
    DECODE_ROWS,
    EXPECTED_LAYER_TYPES,
    EXPERTS,
    GDN_CONV_WIDTH,
    GDN_HEAD_DIM,
    GDN_KEY_HEADS,
    GDN_VALUE_HEADS,
    HC,
    HC_RANK,
    HIDDEN,
    INTERMEDIATE,
    LAYERS,
    QSA_COMPRESS_RATIO,
    QSA_HEAD_DIM,
    QSA_HEADS,
    QSA_INDEX_BUDGET,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_HEADS,
    QSA_INDEX_KV_HEADS,
    QSA_KV_HEADS,
    SUPPORTED_TP,
    TOPK,
)
from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.utils import envs

ENABLED_TP = SUPPORTED_TP


def enabled_widths(tp):
    return DECODE_ROWS if tp in ENABLED_TP else ()


def uses_native_mxfp4_moe(atom_config) -> bool:
    """Experimental TP1 Quark seam: Mono GDN/HC, native MXFP4 MoE/QSA."""
    quant = getattr(atom_config, "quant_config", None)
    hf_quant = getattr(quant, "hf_quant_config", None) or {}
    return (
        atom_config.tensor_parallel_size == 1
        and getattr(quant, "quant_method", None) == "quark"
        and not getattr(quant, "online_quant", False)
        and hf_quant.get("global_quant_config", {}).get("weight", {}).get("dtype")
        == "fp4"
    )


def _shape_checks(hf) -> list[tuple[bool, str]]:
    layer_types = tuple(
        "full_attention" if kind == "qwen_sparse_attention" else kind
        for kind in getattr(hf, "layer_types", ())
    )
    ple = tuple(getattr(hf, "ple_layer_ids", ()) or ())
    return [
        (getattr(hf, "num_hidden_layers", None) == LAYERS, "not 48 layers"),
        (getattr(hf, "hidden_size", None) == HIDDEN, "hidden size"),
        (getattr(hf, "hc_count", None) == HC, "HC count"),
        (getattr(hf, "hc_lowrank", None) == HC_RANK, "HC rank"),
        (getattr(hf, "num_experts", None) == EXPERTS, "expert count"),
        (getattr(hf, "num_experts_per_tok", None) == TOPK, "top-k"),
        (
            getattr(hf, "moe_intermediate_size", None) == INTERMEDIATE,
            "expert intermediate",
        ),
        (
            getattr(hf, "shared_expert_intermediate_size", None) == INTERMEDIATE,
            "shared expert intermediate",
        ),
        (
            layer_types == EXPECTED_LAYER_TYPES,
            "layer type table",
        ),
        (
            (
                getattr(hf, "linear_num_key_heads", None),
                getattr(hf, "linear_num_value_heads", None),
                getattr(hf, "linear_key_head_dim", None),
                getattr(hf, "linear_value_head_dim", None),
                getattr(hf, "linear_conv_kernel_dim", None),
            )
            == (
                GDN_KEY_HEADS,
                GDN_VALUE_HEADS,
                GDN_HEAD_DIM,
                GDN_HEAD_DIM,
                GDN_CONV_WIDTH,
            ),
            "GDN shape",
        ),
        (
            (
                getattr(hf, "num_attention_heads", None),
                getattr(hf, "num_key_value_heads", None),
                getattr(hf, "head_dim", None),
            )
            == (QSA_HEADS, QSA_KV_HEADS, QSA_HEAD_DIM),
            "QSA shape",
        ),
        (
            (
                getattr(hf, "indexer_n_heads", None),
                getattr(hf, "indexer_kv_heads", None),
                getattr(hf, "indexer_head_dim", None),
                getattr(hf, "indexer_budget", None),
                getattr(hf, "indexer_compress_ratio", None),
            )
            == (
                QSA_INDEX_HEADS,
                QSA_INDEX_KV_HEADS,
                QSA_INDEX_HEAD_DIM,
                QSA_INDEX_BUDGET,
                QSA_COMPRESS_RATIO,
            ),
            "QSA index shape",
        ),
        (ple == (2,), f"PLE layers {ple}"),
        (getattr(hf, "ple_conv_kernel_size", None) == 4, "PLE convolution"),
        (getattr(hf, "ngram_size", None) == 3, "PLE ngram size"),
        (getattr(hf, "rms_norm_eps", None) == 1e-6, "RMS epsilon"),
        (getattr(hf, "hidden_act", None) == "silu", "hidden activation"),
        (
            getattr(hf, "mamba_ssm_dtype", None) == "float32",
            "GDN state dtype",
        ),
        (
            getattr(hf, "norm_topk_prob", True) is True,
            "router renormalization",
        ),
        (
            getattr(hf, "output_gate_type", None) == "sigmoid",
            "QSA output gate",
        ),
    ]


def gdn_cache_layout_supported(attention, replayssm: bool) -> bool:
    """Whether Qwen mono may consume this GDN cache in contiguous-K layout."""

    return (
        envs.ATOM_MONO_ENABLE
        and not replayssm
        and attention.tp_size in ENABLED_TP
        and attention.num_k_heads == GDN_KEY_HEADS
        and attention.num_v_heads == GDN_VALUE_HEADS
        and attention.head_k_dim == GDN_HEAD_DIM
        and attention.head_v_dim == GDN_HEAD_DIM
    )


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot exercise the mono runtime, or ``None``."""
    from atom.models.qwen4_exp_mono.dims import Dims

    if not envs.ATOM_MONO_ENABLE:
        return "ATOM_MONO_ENABLE is off"
    tp = atom_config.tensor_parallel_size
    expert_owned = bool(atom_config.enable_expert_parallel)
    # Qwen owns its EP admission below; keep the engine's config immutable.
    admission = copy(atom_config)
    admission.enable_expert_parallel = False
    checks = common_refusals(admission, Dims, SUPPORTED_TP)
    checks += [
        (tp in ENABLED_TP, f"TP {tp} production gate"),
        (
            expert_owned == (tp == 8),
            (
                "TP8 requires MoE expert ownership"
                if tp == 8
                else "MoE expert ownership is only supported at TP8"
            ),
        ),
        (
            getattr(atom_config, "speculative_config", None) is None,
            "speculative decode",
        ),
        (
            getattr(atom_config, "kv_cache_dtype", None) == "bf16"
            or (
                uses_native_mxfp4_moe(atom_config)
                and getattr(atom_config, "kv_cache_dtype", None) == "fp8"
            ),
            "KV cache is not BF16",
        ),
        (atom_config.torch_dtype == torch.bfloat16, "activation dtype is not BF16"),
        *(_shape_checks(atom_config.hf_config)),
    ]
    return first_refusal(checks)
