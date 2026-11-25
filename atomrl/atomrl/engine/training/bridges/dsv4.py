# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DeepSeek-V4-Flash <-> Megatron-Core GPTModel weight conversion.

Maps Hugging Face DeepSeek-V4-Flash weights to a Megatron-Core ``GPTModel``
built with the Lumen DSV4 layer spec (MLA attention, Hyper-Connection,
compressor/indexer, MoE), and back (for rollout weight sync to vLLM).

DSV4 uses Multi-Latent Attention (MLA) instead of standard GQA -- there is
no fused QKV projection.  Each attention projection (wq_a, wq_b, wkv, wo_a,
wo_b) is mapped individually.  Hyper-Connection parameters and compressor /
indexer weights are conditional on per-layer ``compress_ratios``.

MoE expert layout uses Lumen's GroupedMLP per-expert indexed format:
``linear_fc1.weight{E}`` (gate+up fused) and ``linear_fc2.weight{E}`` (down),
with router, expert bias, hash table (tid2eid), and shared expert.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from atomrl.engine.training.bridges.core import pp_layer_range, strip_module_prefix
from atomrl.engine.training.bridges.gpt import GPTDims

# Default compress_ratios for the 43-layer Flash model.
# Layers 0,1,42: ratio=0 (no compressor/indexer)
# Even layers 2,4,...,40: ratio=4 (CSA -- compressor + indexer)
# Odd layers 3,5,...,41: ratio=128 (HCA -- compressor only, no indexer)
DSV4_FLASH_COMPRESS_RATIOS: list[int] = [0, 0] + [4, 128] * 20 + [0]  # 43 values


def is_dsv4(hf: dict) -> bool:
    """True when ``config.json`` is a DeepSeek-V4 family (MLA + hyper-connections).

    ``MegatronNativeEngine.__init__`` asks this of every model it builds, so a
    Qwen3 run reaches it too: it is how the engine decides whether to take the
    DSv4 path at all.
    """
    model_type = str(hf.get("model_type") or "").lower()
    if "deepseek_v4" in model_type or "dsv4" in model_type:
        return True
    architectures = hf.get("architectures") or []
    return any("deepseekv4" in str(name).lower() for name in architectures)


@dataclass
class DSV4Dims(GPTDims):
    """GPTDims extended with DSV4-specific MLA / HC / compressor fields."""

    q_lora_rank: int = 1024
    kv_lora_rank: int = 512  # also serves as head_dim for MLA
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 512
    o_groups: int = 8
    o_lora_rank: int = 1024
    hc_mult: int = 4
    n_hash_layers: int = 3
    window_size: int = 128
    compress_ratios: list[int] = field(
        default_factory=lambda: list(DSV4_FLASH_COMPRESS_RATIOS)
    )
    moe_topk: int = 6


def _normalize_hf_keys(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize HF weight keys to deepseek-ai format.

    RedHatAI/DeepSeek-V4-Flash-BF16 uses a simplified naming:
        layers.{L}.attn.*    -> model.layers.{L}.self_attn.*
        layers.{L}.attn_norm -> model.layers.{L}.input_layernorm
        layers.{L}.ffn_norm  -> model.layers.{L}.post_attention_layernorm
        embed.weight         -> model.embed_tokens.weight
        head.weight          -> lm_head.weight
        norm.weight          -> model.norm.weight
        hc_head_*            -> model.hc_head_*

    If keys already use the deepseek-ai format (model.layers.*), returns as-is.
    """
    if any(k.startswith("model.layers.") for k in hf):
        return hf  # already in deepseek-ai format

    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in hf.items():
        new_key = key
        if key == "embed.weight":
            new_key = "model.embed_tokens.weight"
        elif key == "head.weight":
            new_key = "lm_head.weight"
        elif key == "norm.weight":
            new_key = "model.norm.weight"
        elif key.startswith("hc_head_"):
            new_key = f"model.{key}"
        elif key.startswith("layers."):
            new_key = "model." + key
            # attn -> self_attn (but NOT attn_norm or attn_sink inside self_attn)
            new_key = re.sub(r"\.attn\.", ".self_attn.", new_key)
            new_key = new_key.replace(".attn_norm.", ".input_layernorm.")
            new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")
            # MoE: ffn.* -> mlp.*
            new_key = re.sub(r"\.ffn\.gate\.tid2eid", ".mlp.topk.tid2eid", new_key)
            new_key = re.sub(
                r"\.ffn\.gate\.bias", ".mlp.gate.e_score_correction_bias", new_key
            )
            new_key = re.sub(r"\.ffn\.gate\.weight", ".mlp.gate.weight", new_key)
            # Experts: ffn.experts.{E}.w1 -> mlp.experts.{E}.gate_proj
            #          ffn.experts.{E}.w2 -> mlp.experts.{E}.down_proj
            #          ffn.experts.{E}.w3 -> mlp.experts.{E}.up_proj
            new_key = re.sub(
                r"\.ffn\.experts\.(\d+)\.w1\.", r".mlp.experts.\1.gate_proj.", new_key
            )
            new_key = re.sub(
                r"\.ffn\.experts\.(\d+)\.w2\.", r".mlp.experts.\1.down_proj.", new_key
            )
            new_key = re.sub(
                r"\.ffn\.experts\.(\d+)\.w3\.", r".mlp.experts.\1.up_proj.", new_key
            )
            # Shared experts: ffn.shared_experts.w1 -> mlp.shared_experts.gate_proj
            new_key = new_key.replace(
                ".ffn.shared_experts.w1.", ".mlp.shared_experts.gate_proj."
            )
            new_key = new_key.replace(
                ".ffn.shared_experts.w2.", ".mlp.shared_experts.down_proj."
            )
            new_key = new_key.replace(
                ".ffn.shared_experts.w3.", ".mlp.shared_experts.up_proj."
            )
        mapped[new_key] = tensor
    return mapped


def _expert_bias_or_zeros(
    hf: Mapping[str, torch.Tensor],
    layer_prefix: str,
    num_experts: int,
) -> torch.Tensor:
    bias_key = layer_prefix + "mlp.gate.e_score_correction_bias"
    if bias_key in hf:
        return hf[bias_key].float()
    router_weight = hf[layer_prefix + "mlp.gate.weight"]
    return torch.zeros(
        num_experts,
        dtype=torch.float32,
        device=router_weight.device,
    )


def denormalize_redhat_key(key: str) -> str:
    """Map a normalized DSV4 key back to the RedHat checkpoint/vLLM format."""
    if key == "model.embed_tokens.weight":
        return "embed.weight"
    if key == "lm_head.weight":
        return "head.weight"
    if key == "model.norm.weight":
        return "norm.weight"
    if key.startswith("model.hc_head_"):
        return key.removeprefix("model.")
    if not key.startswith("model.layers."):
        return key

    key = key.removeprefix("model.")
    key = key.replace(".input_layernorm.", ".attn_norm.")
    key = key.replace(".post_attention_layernorm.", ".ffn_norm.")
    key = key.replace(".self_attn.", ".attn.")
    key = key.replace(".mlp.topk.tid2eid", ".ffn.gate.tid2eid")
    key = key.replace(
        ".mlp.gate.e_score_correction_bias",
        ".ffn.gate.bias",
    )
    key = key.replace(".mlp.gate.weight", ".ffn.gate.weight")
    key = re.sub(
        r"\.mlp\.experts\.(\d+)\.gate_proj\.",
        r".ffn.experts.\1.w1.",
        key,
    )
    key = re.sub(
        r"\.mlp\.experts\.(\d+)\.down_proj\.",
        r".ffn.experts.\1.w2.",
        key,
    )
    key = re.sub(
        r"\.mlp\.experts\.(\d+)\.up_proj\.",
        r".ffn.experts.\1.w3.",
        key,
    )
    key = key.replace(".mlp.shared_experts.gate_proj.", ".ffn.shared_experts.w1.")
    key = key.replace(".mlp.shared_experts.down_proj.", ".ffn.shared_experts.w2.")
    key = key.replace(".mlp.shared_experts.up_proj.", ".ffn.shared_experts.w3.")
    return key


def hf_to_megatron(
    hf: dict[str, torch.Tensor],
    d: DSV4Dims,
    ep_rank: int = 0,
    ep_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers_per_pp_rank: list[int] | None = None,
    use_grouped_mlp: bool = True,  # ignored — kept for API compat
) -> dict[str, torch.Tensor]:
    """Return a Megatron GPTModel state_dict from HF DeepSeek-V4-Flash weights.

    For MoE models (``d.num_experts > 0``), only the local experts for the
    given ``ep_rank`` are included.

    With PP > 1, only the layers and embedding/output owned by this PP rank
    are included.  Megatron's decoder layers are numbered locally (0-based
    within the stage), while HF layers use the global index.

    FP32 parameters (HC params, attn_sink, compressor APE/norm) are preserved
    without casting to bf16.

    Supports both deepseek-ai (``model.layers.{L}.self_attn.*``) and RedHatAI
    (``layers.{L}.attn.*``) key naming via automatic normalization.
    """
    hf = _normalize_hf_keys(hf)
    m: dict[str, torch.Tensor] = {}
    is_first_pp = pp_rank == 0
    is_last_pp = pp_rank == pp_size - 1

    # -- non-layer params --
    if is_first_pp:
        m["embedding.word_embeddings.weight"] = hf["model.embed_tokens.weight"]
    if is_last_pp:
        m["decoder.final_layernorm.weight"] = hf["model.norm.weight"]
        m["output_layer.weight"] = hf["lm_head.weight"]

    # -- HC head params (FP32, on last PP rank) --
    if is_last_pp:
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            hf_key = f"model.{suffix}"
            meg_key = f"decoder.hc_head_params.{suffix}"
            if hf_key in hf:
                m[meg_key] = hf[hf_key].float()

    # -- MoE expert setup --
    is_moe = d.num_experts > 0
    if is_moe:
        num_local = d.num_experts // ep_size
        expert_offset = ep_rank * num_local

    layer_offset, num_local_layers = pp_layer_range(
        d,
        pp_rank,
        pp_size,
        layers_per_pp_rank,
    )

    for local_i in range(num_local_layers):
        global_i = layer_offset + local_i
        hp = f"model.layers.{global_i}."
        mp = f"decoder.layers.{local_i}."
        compress_ratio = d.compress_ratios[global_i] if d.compress_ratios else 0

        # -- input layernorm --
        m[mp + "input_layernorm.weight"] = hf[hp + "input_layernorm.weight"]

        # -- MLA attention projections (no fused QKV) --
        m[mp + "self_attention.wq_a.weight"] = hf[hp + "self_attn.wq_a.weight"]
        q_norm_w = hf[hp + "self_attn.q_norm.weight"]
        m[mp + "self_attention.q_norm.weight"] = q_norm_w
        m[mp + "self_attention.q_norm._norm.weight"] = q_norm_w
        m[mp + "self_attention.wq_b.weight"] = hf[hp + "self_attn.wq_b.weight"]
        m[mp + "self_attention.wkv.weight"] = hf[hp + "self_attn.wkv.weight"]
        kv_norm_w = hf[hp + "self_attn.kv_norm.weight"]
        m[mp + "self_attention.kv_norm.weight"] = kv_norm_w
        m[mp + "self_attention.kv_norm._norm.weight"] = kv_norm_w
        m[mp + "self_attention.wo_a.weight"] = hf[hp + "self_attn.wo_a.weight"]
        m[mp + "self_attention.wo_b.weight"] = hf[hp + "self_attn.wo_b.weight"]

        # -- attn_sink (FP32, TP-sharded dim0) --
        sink_key = hp + "self_attn.attn_sink"
        if sink_key in hf:
            m[mp + "self_attention.attn_sink"] = hf[sink_key].float()

        # -- pre-MLP layernorm --
        m[mp + "pre_mlp_layernorm.weight"] = hf[hp + "post_attention_layernorm.weight"]

        # -- Hyper-Connection per-layer params (FP32, duplicated) --
        for hc_suffix in (
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ):
            hf_hc_key = hp + hc_suffix
            if hf_hc_key in hf:
                m[mp + hc_suffix] = hf[hf_hc_key].float()

        # -- Compressor params (only for layers with compress_ratio > 0) --
        if compress_ratio > 0:
            comp_hf = hp + "self_attn.compressor."
            comp_mg = mp + "self_attention.compressor."
            m[comp_mg + "ape"] = hf[comp_hf + "ape"].float()
            m[comp_mg + "wkv.weight"] = hf[comp_hf + "wkv.weight"]
            m[comp_mg + "wgate.weight"] = hf[comp_hf + "wgate.weight"]
            m[comp_mg + "norm.weight"] = hf[comp_hf + "norm.weight"].float()

        # -- Indexer params (only for compress_ratio == 4 layers) --
        if compress_ratio == 4:
            # Indexer linear params
            idx_hf = hp + "self_attn.indexer."
            idx_mg = mp + "self_attention.indexer."
            m[idx_mg + "linear_wq_b.weight"] = hf[idx_hf + "wq_b.weight"]
            m[idx_mg + "linear_weights_proj.weight"] = hf[
                idx_hf + "weights_proj.weight"
            ]

            # Indexer's own compressor
            ic_hf = idx_hf + "compressor."
            ic_mg = idx_mg + "compressor."
            m[ic_mg + "ape"] = hf[ic_hf + "ape"].float()
            m[ic_mg + "wkv.weight"] = hf[ic_hf + "wkv.weight"]
            m[ic_mg + "wgate.weight"] = hf[ic_hf + "wgate.weight"]
            m[ic_mg + "norm.weight"] = hf[ic_hf + "norm.weight"].float()

        # -- MoE params --
        if is_moe:
            # Router
            m[mp + "mlp.router.weight"] = hf[hp + "mlp.gate.weight"]

            # Expert bias (e_score_correction_bias)
            m[mp + "mlp.router.expert_bias"] = _expert_bias_or_zeros(
                hf,
                hp,
                d.num_experts,
            )

            # Hash table (tid2eid) -- non-trainable
            tid2eid_key = hp + "mlp.topk.tid2eid"
            if tid2eid_key in hf:
                m[mp + "mlp.router.tid2eid"] = hf[tid2eid_key]

            # Experts — Lumen's GroupedMLP uses per-expert indexed format:
            #   mlp.experts.linear_fc1.weight{E}  (gate+up fused, [2*moe_ffn, hidden])
            #   mlp.experts.linear_fc2.weight{E}  (down, [hidden, moe_ffn])
            for local_e, e in enumerate(
                range(expert_offset, expert_offset + num_local)
            ):
                gate = hf[hp + f"mlp.experts.{e}.gate_proj.weight"]
                up = hf[hp + f"mlp.experts.{e}.up_proj.weight"]
                down = hf[hp + f"mlp.experts.{e}.down_proj.weight"]
                m[mp + f"mlp.experts.linear_fc1.weight{local_e}"] = torch.cat(
                    [gate, up],
                    dim=0,
                ).contiguous()
                m[mp + f"mlp.experts.linear_fc2.weight{local_e}"] = down

            # Shared expert
            if d.shared_expert_ffn > 0:
                shared_hp = hp + "mlp.shared_experts."
                if (shared_hp + "gate_proj.weight") in hf:
                    m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                        [
                            hf[shared_hp + "gate_proj.weight"],
                            hf[shared_hp + "up_proj.weight"],
                        ],
                        dim=0,
                    ).contiguous()
                    m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[
                        shared_hp + "down_proj.weight"
                    ]
                # Shared expert with separate naming (shared_expert vs shared_experts)
                shared_hp_alt = hp + "mlp.shared_expert."
                if (shared_hp_alt + "gate_proj.weight") in hf:
                    m[mp + "mlp.shared_experts.linear_fc1.weight"] = torch.cat(
                        [
                            hf[shared_hp_alt + "gate_proj.weight"],
                            hf[shared_hp_alt + "up_proj.weight"],
                        ],
                        dim=0,
                    ).contiguous()
                    m[mp + "mlp.shared_experts.linear_fc2.weight"] = hf[
                        shared_hp_alt + "down_proj.weight"
                    ]
        else:
            # Dense MLP fallback
            m[mp + "mlp.linear_fc1.weight"] = torch.cat(
                [hf[hp + "mlp.gate_proj.weight"], hf[hp + "mlp.up_proj.weight"]],
                dim=0,
            ).contiguous()
            m[mp + "mlp.linear_fc2.weight"] = hf[hp + "mlp.down_proj.weight"]

    return m


def megatron_to_hf(
    named_params,
    d: DSV4Dims,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers_per_pp_rank: list[int] | None = None,
    use_grouped_mlp: bool = True,  # ignored — kept for API compat
):
    """Yield ``(hf_name, tensor)`` from Megatron GPTModel named params.

    Inverse of ``hf_to_megatron``.  Used for weight sync to vLLM.

    With PP > 1, maps local decoder layer indices back to global HF
    layer indices.
    """
    if isinstance(named_params, Mapping):
        md = named_params
    else:
        md: dict[str, torch.Tensor] = {}
        for name, t in named_params:
            md[strip_module_prefix(name)] = t

    def get(n):
        return md[n]

    is_first_pp = pp_rank == 0
    is_last_pp = pp_rank == pp_size - 1

    # -- non-layer params --
    if is_first_pp:
        yield "model.embed_tokens.weight", get("embedding.word_embeddings.weight")
    if is_last_pp:
        yield "model.norm.weight", get("decoder.final_layernorm.weight")
        yield "lm_head.weight", get("output_layer.weight")

    # -- HC head params (FP32) --
    if is_last_pp:
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            meg_key = f"decoder.hc_head_params.{suffix}"
            if meg_key in md:
                yield f"model.{suffix}", get(meg_key)

    # -- MoE expert setup --
    is_moe = d.num_experts > 0
    if is_moe:
        pass

    layer_offset, num_local_layers = pp_layer_range(
        d,
        pp_rank,
        pp_size,
        layers_per_pp_rank,
    )

    for local_i in range(num_local_layers):
        global_i = layer_offset + local_i
        mp = f"decoder.layers.{local_i}."
        hp = f"model.layers.{global_i}."
        compress_ratio = d.compress_ratios[global_i] if d.compress_ratios else 0

        # -- input layernorm --
        yield hp + "input_layernorm.weight", get(mp + "input_layernorm.weight")

        # -- MLA attention projections --
        yield hp + "self_attn.wq_a.weight", get(mp + "self_attention.wq_a.weight")
        yield hp + "self_attn.q_norm.weight", get(mp + "self_attention.q_norm.weight")
        yield hp + "self_attn.wq_b.weight", get(mp + "self_attention.wq_b.weight")
        yield hp + "self_attn.wkv.weight", get(mp + "self_attention.wkv.weight")
        yield hp + "self_attn.kv_norm.weight", get(mp + "self_attention.kv_norm.weight")
        yield hp + "self_attn.wo_a.weight", get(mp + "self_attention.wo_a.weight")
        yield hp + "self_attn.wo_b.weight", get(mp + "self_attention.wo_b.weight")

        # -- attn_sink (FP32) --
        sink_key = mp + "self_attention.attn_sink"
        if sink_key in md:
            yield hp + "self_attn.attn_sink", get(sink_key)

        # -- pre-MLP layernorm --
        yield hp + "post_attention_layernorm.weight", get(
            mp + "pre_mlp_layernorm.weight"
        )

        # -- Hyper-Connection per-layer params (FP32) --
        for hc_suffix in (
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ):
            meg_hc_key = mp + hc_suffix
            if meg_hc_key in md:
                yield hp + hc_suffix, get(meg_hc_key)

        # -- Compressor params --
        if compress_ratio > 0:
            comp_mg = mp + "self_attention.compressor."
            comp_hf = hp + "self_attn.compressor."
            if (comp_mg + "ape") in md:
                yield comp_hf + "ape", get(comp_mg + "ape")
                yield comp_hf + "wkv.weight", get(comp_mg + "wkv.weight")
                yield comp_hf + "wgate.weight", get(comp_mg + "wgate.weight")
                yield comp_hf + "norm.weight", get(comp_mg + "norm.weight")

        # -- Indexer params --
        if compress_ratio == 4:
            idx_mg = mp + "self_attention.indexer."
            idx_hf = hp + "self_attn.indexer."
            if (idx_mg + "linear_wq_b.weight") in md:
                yield idx_hf + "wq_b.weight", get(idx_mg + "linear_wq_b.weight")
                yield idx_hf + "weights_proj.weight", get(
                    idx_mg + "linear_weights_proj.weight"
                )

            # Indexer's own compressor
            ic_mg = idx_mg + "compressor."
            ic_hf = idx_hf + "compressor."
            if (ic_mg + "ape") in md:
                yield ic_hf + "ape", get(ic_mg + "ape")
                yield ic_hf + "wkv.weight", get(ic_mg + "wkv.weight")
                yield ic_hf + "wgate.weight", get(ic_mg + "wgate.weight")
                yield ic_hf + "norm.weight", get(ic_mg + "norm.weight")

        # -- MoE params --
        if is_moe:
            # Router
            yield hp + "mlp.gate.weight", get(mp + "mlp.router.weight")

            # Expert bias
            ebias_key = mp + "mlp.router.expert_bias"
            if ebias_key in md:
                yield hp + "mlp.gate.e_score_correction_bias", get(ebias_key)

            # Hash table (tid2eid)
            tid2eid_key = mp + "mlp.router.tid2eid"
            if tid2eid_key in md:
                yield hp + "mlp.topk.tid2eid", get(tid2eid_key)

            # Experts — Lumen's GroupedMLP per-expert indexed format:
            #   mlp.experts.linear_fc1.weight{E} -> gate_proj + up_proj
            #   mlp.experts.linear_fc2.weight{E} -> down_proj
            for local_idx in range(d.num_experts):
                ep = hp + f"mlp.experts.{local_idx}."
                fc1 = get(mp + f"mlp.experts.linear_fc1.weight{local_idx}")
                gate_w = fc1[: d.moe_ffn].contiguous()
                up_w = fc1[d.moe_ffn :].contiguous()
                yield ep + "gate_proj.weight", gate_w
                yield ep + "up_proj.weight", up_w
                yield ep + "down_proj.weight", get(
                    mp + f"mlp.experts.linear_fc2.weight{local_idx}"
                )

            # Shared expert
            if d.shared_expert_ffn > 0:
                sfc1_key = mp + "mlp.shared_experts.linear_fc1.weight"
                if sfc1_key in md:
                    sfc1 = get(sfc1_key)
                    s_gate, s_up = (
                        sfc1[: d.shared_expert_ffn],
                        sfc1[d.shared_expert_ffn :],
                    )
                    yield hp + "mlp.shared_experts.gate_proj.weight", s_gate.contiguous()
                    yield hp + "mlp.shared_experts.up_proj.weight", s_up.contiguous()
                    yield hp + "mlp.shared_experts.down_proj.weight", get(
                        mp + "mlp.shared_experts.linear_fc2.weight"
                    )
        else:
            # Dense MLP fallback
            fc1 = get(mp + "mlp.linear_fc1.weight")
            gate, up = fc1[: d.ffn], fc1[d.ffn :]
            yield hp + "mlp.gate_proj.weight", gate.contiguous()
            yield hp + "mlp.up_proj.weight", up.contiguous()
            yield hp + "mlp.down_proj.weight", get(mp + "mlp.linear_fc2.weight")
