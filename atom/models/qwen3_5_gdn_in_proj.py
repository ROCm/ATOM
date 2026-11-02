# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether Qwen3.5's GDN input projection is one merged `in_proj_qkvzba` layer.

The native model (`atom/models/qwen3_5.py`) and the SGLang / vLLM wrappers all
build the layer and the checkpoint mapping from `gdn_in_proj_is_merged`. Kept
free of AITER and the model's GPU imports so the CPU unit tests cover it.
"""

from __future__ import annotations

from typing import Any

import torch

from atom.quant_spec import QuantType
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


def _linear_attention_prefixes(atom_config: Any) -> list[str]:
    """`model.layers.<i>.linear_attn` for every GDN layer."""
    hf_config = atom_config.hf_config
    config = getattr(hf_config, "text_config", None) or hf_config
    layer_types = getattr(config, "layer_types", None) or []
    return [
        f"model.layers.{i}.linear_attn"
        for i, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]


def gdn_in_proj_is_merged(atom_config: Any) -> bool:
    """True when the GDN input projection is one `in_proj_qkvzba` layer.

    Both the layer construction (`Qwen3_5GatedDeltaNet.create_qkvzba_proj`) and
    the checkpoint mapping (`_apply_merged_in_proj_mapping`, and the SGLang /
    vLLM wrappers' mappings) call this, because a disagreement would load
    `in_proj_b` / `in_proj_a` into a module that does not exist, or leave the
    merged weight's last two shards unwritten.

    BF16 checkpoints merge unconditionally, as they always have. A quantized
    checkpoint merges only when, in every GDN layer, `in_proj_qkvz`,
    `in_proj_ba` and the merged `in_proj_qkvzba` name (the one the merged layer
    and its input norm resolve their spec from) all carry the same per-token
    spec: per-token activation scales are one value per row and per-channel
    weight scales one per column, so concatenating along the output dimension
    just concatenates the scales. The mapping is model-wide, so one layer that
    does not qualify keeps them all split. A block-scale checkpoint must not
    merge -- `in_proj_ba` contributes 32 columns per rank, which is not a whole
    number of 128-column scale blocks.
    """
    quant_config = atom_config.quant_config
    if quant_config is None:
        return False
    if quant_config.global_quant_config.quant_dtype == torch.bfloat16:
        return True
    if not envs.ATOM_QWEN35_GDN_MERGE_QKVZBA:
        return False
    prefixes = _linear_attention_prefixes(atom_config)
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


def _apply_merged_in_proj_mapping(mapping: dict, atom_config: Any) -> dict:
    if not gdn_in_proj_is_merged(atom_config):
        return mapping

    mapping.pop("in_proj_qkvz", None)
    mapping.pop("in_proj_ba", None)
    mapping["in_proj_qkvzba"] = ("in_proj_qkvzba", None)
    mapping.update(_MERGED_IN_PROJ_MAPPING)
    return mapping
