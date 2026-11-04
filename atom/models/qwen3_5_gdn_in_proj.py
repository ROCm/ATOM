# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5 GDN input-projection merge decision.

This module is deliberately aiter-free so that tests can import the decision
logic on a CPU-only runner without skipping.
"""

from __future__ import annotations

import torch

from atom.quant_spec import LayerQuantConfig, QuantType
from atom.utils import envs

_QWEN3_5_PACKED_MODULES_MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
    ".gate.": (".gate.", 0),
    "shared_expert_gate": ("gate", 1),
}

_MERGED_IN_PROJ_MAPPING = {
    "in_proj_qkv": ("in_proj_qkvzba", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvzba", 3),
    "in_proj_b": ("in_proj_qkvzba", 4),
    "in_proj_a": ("in_proj_qkvzba", 5),
}


def _linear_attention_prefixes(atom_config, layers_prefix: str) -> list[str]:
    """`<layers_prefix>.<i>.linear_attn` for every GDN layer."""
    hf_config = atom_config.hf_config
    config = getattr(hf_config, "text_config", None) or hf_config
    layer_types = getattr(config, "layer_types", None) or []
    return [
        f"{layers_prefix}.{i}.linear_attn"
        for i, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]


def gdn_in_proj_is_merged(atom_config, layers_prefix: str = "model.layers") -> bool:
    """True when the GDN input projection is one ``in_proj_qkvzba`` layer.

    BF16 checkpoints merge unconditionally, as they always have. A quantized
    checkpoint merges only when, in every GDN layer, ``in_proj_qkvz``,
    ``in_proj_ba`` and the merged ``in_proj_qkvzba`` name all carry the same
    per-token spec: per-token activation scales are one value per row and
    per-channel weight scales one per column, so concatenating along the output
    dimension just concatenates the scales. A block-scale checkpoint must not
    merge -- ``in_proj_ba`` contributes 32 columns per rank, which is not a whole
    number of 128-column scale blocks.

    ``layers_prefix`` is the decoder stack's module prefix as the layers are
    built (``model.layers``, or ``language_model.model.layers`` under a multimodal
    wrapper): exact-name quantization excludes only match the full name.
    """
    quant_config = atom_config.quant_config
    if quant_config is None:
        return False
    if quant_config.global_quant_config.quant_dtype == torch.bfloat16:
        return True
    if not envs.ATOM_QWEN35_GDN_MERGE_QKVZBA:
        return False
    prefixes = _linear_attention_prefixes(atom_config, layers_prefix)
    if not prefixes:
        return False
    for prefix in prefixes:
        qkvz = quant_config.get_layer_quant_config(f"{prefix}.in_proj_qkvz")
        ba = quant_config.get_layer_quant_config(f"{prefix}.in_proj_ba")
        merged = quant_config.get_layer_quant_config(f"{prefix}.in_proj_qkvzba")
        if not (qkvz == ba == merged):
            return False
        if qkvz.quant_type != QuantType.per_Token or qkvz.weight_block_size:
            return False
    return True


def _apply_merged_in_proj_mapping(
    mapping: dict, atom_config, layers_prefix: str = "model.layers"
) -> dict:
    if not gdn_in_proj_is_merged(atom_config, layers_prefix):
        return mapping
    mapping.pop("in_proj_qkvz", None)
    mapping.pop("in_proj_ba", None)
    mapping["in_proj_qkvzba"] = ("in_proj_qkvzba", None)
    mapping.update(_MERGED_IN_PROJ_MAPPING)
    return mapping
