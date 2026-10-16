# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The exact Qwen3.8-Flash-Next deployment served by the mono runtime."""

import torch

from atom.mono.runtime.deployment import common_refusals, first_refusal
from atom.utils import envs

# Widths with a stable host/state ABI. A width enters production routing only
# after its persistent kernels pass compile, resource, numerical and graph
# replay gates; keep that smaller set explicit while S=2/4/8 are built out.
SUPPORTED_WIDTHS = (1, 2, 4, 8)
ENABLED_WIDTHS = (1,)
SUPPORTED_TP = (1, 8)
# TP8 has passed the hybrid runtime and replay probes.  Keep production routing
# closed until fixed final hidden/logits/argmax thresholds are recorded; P5/P6
# continue to compile against SUPPORTED_TP through explicit probes.
ENABLED_TP = (1,)
ALIGN = 16
LAYERS = 48
HIDDEN = 2560
HC = 4
HC_RANK = 320
EXPERTS = 512
TOPK = 10
INTERMEDIATE = 640
SHARED_EXPERTS = 1
ROUTER_OUTPUTS = EXPERTS + SHARED_EXPERTS
FP8_BLOCK = 128

# QSA specialization used by every fourth layer.  Keep these beside the GDN
# dimensions: they are part of the deployment/build key, not values a kernel
# may discover from runtime metadata.
QSA_HEADS = 24
QSA_KV_HEADS = 2
QSA_HEAD_DIM = 256
QSA_Q_SIZE = QSA_HEADS * QSA_HEAD_DIM
QSA_KV_SIZE = QSA_KV_HEADS * QSA_HEAD_DIM
QSA_QKV_SIZE = 2 * QSA_Q_SIZE + 2 * QSA_KV_SIZE
QSA_INDEX_HEADS = 4
QSA_INDEX_KV_HEADS = 1
QSA_INDEX_HEAD_DIM = 128
QSA_INDEX_Q_SIZE = QSA_INDEX_HEADS * QSA_INDEX_HEAD_DIM
QSA_INDEX_KV_SIZE = QSA_INDEX_KV_HEADS * QSA_INDEX_HEAD_DIM
QSA_INDEX_QK_SIZE = QSA_INDEX_Q_SIZE + QSA_INDEX_KV_SIZE
QSA_INDEX_BUDGET = 2048
QSA_COMPRESS_RATIO = 4
QSA_MAX_MODEL_LEN = 8192
QSA_DIRECT_CONTEXT = 256
QSA_CONTEXT_BUCKETS = (QSA_DIRECT_CONTEXT, QSA_MAX_MODEL_LEN)
QSA_ROTARY_DIM = 64
QSA_MROPE_SECTION = (11, 11, 10)
QSA_MROPE_INTERLEAVED = True

# Gated DeltaNet specialization used by the 36 linear-attention layers.  The
# production FP8 checkpoint deliberately leaves every GDN projection in BF16;
# these widths therefore describe the packed BF16 projection that mono reads,
# not a generic Qwen configuration.
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_DIM = 128
GDN_CONV_WIDTH = 4
GDN_KEY_DIM = GDN_KEY_HEADS * GDN_HEAD_DIM
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_DIM
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VALUE_DIM
GDN_PROJ_DIM = GDN_CONV_DIM + GDN_VALUE_DIM + 2 * GDN_VALUE_HEADS
EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
    for layer in range(LAYERS)
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
        and envs.ATOM_QWEN_MONO_ENABLE
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
    if not envs.ATOM_QWEN_MONO_ENABLE:
        return "ATOM_QWEN_MONO_ENABLE is off"
    tp = atom_config.tensor_parallel_size
    expert_owned = bool(atom_config.enable_expert_parallel)
    checks = common_refusals(
        atom_config,
        Dims,
        SUPPORTED_TP,
        allow_expert_parallel=True,
    )
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
            getattr(atom_config, "kv_cache_dtype", None) == "bf16",
            "KV cache is not BF16",
        ),
        (atom_config.torch_dtype == torch.bfloat16, "activation dtype is not BF16"),
        *(_shape_checks(atom_config.hf_config)),
    ]
    return first_refusal(checks)
