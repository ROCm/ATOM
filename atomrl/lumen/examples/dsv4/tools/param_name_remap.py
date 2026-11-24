# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2024 SGLang Team
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/sgl-project/sglang/blob/88c10654aedd7d3cc8ae7f16699020aff53efee0/python/sglang/srt/models/deepseek_v4.py

"""Weight name remap for DeepSeek-V4 native FP8 checkpoints (no SGLang dependency)."""

from __future__ import annotations

import json
from collections.abc import Callable


def remap_weight_name_to_dpsk_hf_format(
    name: str,
    *,
    is_nextn: bool = False,
    num_hidden_layers: int | None = None,
) -> str:
    """Vendored from SGLang ``DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format``."""
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name.startswith("hc_head_"):
        return "model." + name

    if is_nextn and name.startswith("mtp."):
        parts = name.split(".", 2)
        if len(parts) >= 3:
            rest = parts[2]
            nextn_spec_prefixes = [
                "e_proj",
                "h_proj",
                "emb",
                "enorm",
                "hnorm",
                "norm",
                "head",
                "hc_head",
            ]
            is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
            if is_nextn_spec:
                if rest.startswith("emb.tok_emb"):
                    rest = rest.replace("emb.tok_emb", "embed_tokens")
                elif rest == "norm.weight":
                    rest = "shared_head.norm.weight"
                elif rest.startswith("head."):
                    rest = "shared_head.head.weight"
                elif rest == "e_proj.scale":
                    rest = "e_proj.weight_scale_inv"
                elif rest == "h_proj.scale":
                    rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

    if name.startswith("layers."):
        name = "model." + name
    name = name.replace(".attn.", ".self_attn.")
    name = name.replace(".ffn.", ".mlp.")
    name = name.replace(".attn_norm.", ".input_layernorm.")
    name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

    if "self_attn" in name and name.endswith(".scale"):
        name = name.removesuffix(".scale") + ".weight_scale_inv"

    name = name.replace(".gate.tid2eid", ".topk.tid2eid")
    name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
    name = name.replace(".w1.", ".gate_proj.")
    name = name.replace(".w2.", ".down_proj.")
    name = name.replace(".w3.", ".up_proj.")
    if "mlp" in name and name.endswith(".scale"):
        name = name.removesuffix(".scale") + ".weight_scale_inv"

    return name


def _is_deepseek_v4_native(config_path: str, weight_map: dict) -> bool:
    with open(config_path) as f:
        is_v4_arch = "DeepseekV4ForCausalLM" in json.load(f).get("architectures", [])
    return is_v4_arch and "embed.weight" in weight_map


def get_param_name_remap(config_path: str, weight_map: dict) -> Callable[[str], str]:
    if _is_deepseek_v4_native(config_path, weight_map):
        return remap_weight_name_to_dpsk_hf_format
    return lambda name: name
