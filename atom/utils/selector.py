# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache
from typing import Type

from atom.model_ops.attentions.backends import AttentionBackend
from atom.utils import resolve_obj_by_qualname
from atom.plugin.prepare import is_sglang, is_vllm


def get_attn_backend(
    block_size: int,
    use_mla: bool = False,
    use_gdn: bool = False,
    use_v4: bool = False,
) -> Type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    return _cached_get_attn_backend(
        block_size=block_size,
        use_mla=use_mla,
        use_gdn=use_gdn,
        use_v4=use_v4,
        use_sglang=is_sglang(),
        use_vllm=is_vllm(),
    )


@cache
def _cached_get_attn_backend(
    block_size: int,
    use_mla: bool = False,
    use_gdn: bool = False,
    use_v4: bool = False,
    use_sglang: bool = False,
    use_vllm: bool = False,
) -> Type[AttentionBackend]:

    # get device-specific attn_backend
    attention_cls = get_attn_backend_cls(
        block_size, use_mla, use_gdn, use_v4, use_sglang, use_vllm
    )
    if not attention_cls:
        raise ValueError(f"Invalid attention backend for {attention_cls}")
    return resolve_obj_by_qualname(attention_cls)


def get_attn_backend_cls(
    block_size, use_mla, use_gdn, use_v4, use_sglang, use_vllm
) -> str:
    if use_v4:
        return "atom.model_ops.attentions.deepseek_v4_attn.DeepseekV4Backend"
    if use_mla:
        # if block_size == 1:
        return "atom.model_ops.attentions.aiter_mla.AiterMLABackend"  # noqa: E501
        # else:
        #     raise ValueError(
        #         f" The selected backend"
        #         f"does not support block size {block_size}."
        #         "(currently only supports block size 1)")
    if use_gdn:
        if use_vllm:
            return "atom.plugin.vllm.attention_backend.gdn_attn.GDNAttentionBackend"
        if use_sglang:
            return (
                "atom.plugin.sglang.attention_backend.attention_gdn.GDNAttentionBackend"
            )
        return "atom.model_ops.attentions.gdn_attn.GDNAttentionBackend"
    return "atom.model_ops.attentions.aiter_attention.AiterBackend"  # noqa: E501
