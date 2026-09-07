# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING

from atom.plugin.prepare import is_sglang, is_vllm
from atom.utils import envs, resolve_obj_by_qualname

if TYPE_CHECKING:
    # A return annotation only, and its module imports the attention stack --
    # keeping it out of the runtime imports is what lets the dispatch be tested
    # where there is no GPU build.
    from atom.model_ops.attentions.backends import AttentionBackend


class Family(StrEnum):
    """The attentions a backend is picked between, most specific first.

    Declared in one place because the order IS the answer for a config that
    satisfies two: DeepSeek-V4 also carries a latent rank, and so do both
    hybrids. An enum and not a string, so a name that does not exist raises
    instead of falling through to MHA.
    """

    V4 = "v4"
    KIMI_MLA = "kimi_mla"
    MLA = "mla"
    GDN = "gdn"
    MHA = "mha"

    @property
    def is_mla(self) -> bool:
        """Whether this family keeps its KV as one latent row per token.

        Two questions spelled this as `use_mla or use_kimi_mla`: whether an
        indexer has rows to ride, and what a drafter's target does.
        """
        return self in (Family.MLA, Family.KIMI_MLA)


# What makes a hybrid one is which of its layers are linear, and no field says
# that -- so these two are named, and everything else is read off the shape.
_KIMI_MLA_TYPES = ("kimi_linear", "glm5_next_text")
_GDN_TYPES = ("qwen3_next", "qwen3_next_mtp", "qwen3_5_text", "qwen3_5_moe_text")
_V4_TYPES = ("deepseek_v4", "deepseek_v4_mtp")


def attn_family(hf_text_config) -> Family:
    """Which attention a config asks for.

    A *config*, not a model: a draft is asked the same question its target is,
    and the answer is the stack it declares. Its decoding algorithm never
    appears here -- an EAGLE draft is llama or DeepSeek or whatever its trainer
    built it on, and two EAGLE drafts of one target need not agree.

    Which is why MLA is the presence of a latent rank rather than a list of
    model types. The list this replaces named six and missed `k3_dspark`, whose
    config carries `kv_lora_rank=512`; a shape cannot be missed.
    """
    model_type = getattr(hf_text_config, "model_type", None)
    # V4 reads `model_type == "deepseek_v3"`, because `_CONFIG_REGISTRY` maps
    # deepseek_v4 onto the V3 schema it reuses. `architectures` survives that
    # (`get_hf_config` preserves it) and is what tells them apart; a draft is
    # stamped `deepseek_v4_mtp` instead and has no architecture of its own.
    arches = getattr(hf_text_config, "architectures", None) or []
    if any("DeepseekV4" in str(arch) for arch in arches) or model_type in _V4_TYPES:
        return Family.V4
    if model_type in _KIMI_MLA_TYPES:
        return Family.KIMI_MLA
    if getattr(hf_text_config, "kv_lora_rank", None) is not None:
        return Family.MLA
    if model_type in _GDN_TYPES:
        return Family.GDN
    return Family.MHA


def has_mla_indexer(hf_text_config) -> bool:
    """Whether this model's MLA rows carry a sparse indexer key cache.

    Orthogonal to the family, which is why it is a second question: DeepSeek
    V3.2 is pure MLA and sparse, GLM-5.3-Flash is a hybrid and sparse.
    `index_topk` is the declaration; the family test is there because only an
    MLA pool has rows for an indexer to ride.
    """
    return hasattr(hf_text_config, "index_topk") and attn_family(hf_text_config).is_mla


def get_attn_backend(family: Family) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    return _cached_get_attn_backend(
        family=family, use_sglang=is_sglang(), use_vllm=is_vllm()
    )


@cache
def _cached_get_attn_backend(
    family: Family, use_sglang: bool, use_vllm: bool
) -> type[AttentionBackend]:

    # get device-specific attn_backend
    attention_cls = get_attn_backend_cls(
        family=family, use_sglang=use_sglang, use_vllm=use_vllm
    )
    if not attention_cls:
        raise ValueError(f"Invalid attention backend for {attention_cls}")
    return resolve_obj_by_qualname(attention_cls)


def get_attn_backend_cls(family: Family, use_sglang: bool, use_vllm: bool) -> str:
    if family is Family.V4:
        return "atom.model_ops.attentions.deepseek_v4_attn.DeepseekV4Backend"
    if family is Family.KIMI_MLA:
        return "atom.model_ops.attentions.kimi_mla_gdn_attn.KimiMLAGDNBackend"
    if family is Family.MLA:
        if envs.ATOM_USE_TRITON_MLA:
            return "atom.model_ops.attentions.triton_mla.TritonMLABackend"
        return "atom.model_ops.attentions.aiter_mla.AiterMLABackend"
    if family is Family.GDN:
        if use_vllm:
            return "atom.plugin.vllm.attention.backend.GDNAttentionBackend"
        if use_sglang:
            return (
                "atom.plugin.sglang.attention_backend.attention_gdn.GDNAttentionBackend"
            )
        return "atom.model_ops.attentions.gdn_attn.GDNAttentionBackend"
    if envs.ATOM_USE_UNIFIED_ATTN:
        return "atom.model_ops.attentions.triton_mha.TritonMHABackend"
    return "atom.model_ops.attentions.aiter_attention.AiterBackend"
