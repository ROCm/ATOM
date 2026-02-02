# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import nn
from typing import Optional

from .attention_mla import MLAModules
from .base_attention import BaseAttention
from atom.plugin.prepare import is_plugin_mode, is_sglang
from atom.models.utils import maybe_prefix


class RadixAttention(BaseAttention):
    """
    Attention radix implementation
    """
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        per_layer_sliding_window: Optional[int] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(num_heads=num_heads,
                         head_dim=head_dim,
                         scale=scale,
                         num_kv_heads=num_kv_heads,
                         kv_cache_dtype=kv_cache_dtype,
                         layer_num=layer_num,
                         use_mla=use_mla,
                         mla_modules=mla_modules,
                         sinks=sinks,
                         per_layer_sliding_window=per_layer_sliding_window,
                         rotary_emb=rotary_emb,
                         prefix=prefix,
                         **kwargs)

        if is_sglang():
            from sglang.srt.layers.radix_attention import RadixAttention
            self.attn = RadixAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                scaling=scale,
                num_kv_heads=num_kv_heads,
                layer_id=layer_num,
                prefix=maybe_prefix(prefix, "attn"),
            )
        else:
            raise NotImplementedError("RadixAttention is only supported for plugin mode for sglang for now")

    def forward_impl_plugin_mode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata = None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        positions: torch.Tensor = None,
        q_scale: torch.Tensor=None,
        **kwargs,
    ):
        if is_sglang():
            # for sglang, forward_batch is required
            forward_batch = kwargs.get("forward_batch", None)
            assert forward_batch is not None, "forward_batch is required for sglang"
            return self.attn(q=query,
                             k=key,
                             v=value,
                             forward_batch=forward_batch)
        else:
            raise NotImplementedError(
                "RadixAttention is only supported for plugin mode for sglang for now"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        if is_plugin_mode():
            o = self.forward_impl_plugin_mode(query=query,
                                              key=key,
                                              value=value,
                                              positions=positions,
                                              q_scale=q_scale,
                                              **kwargs)
        else:
            raise NotImplementedError(
                "RadixAttention is not supported for server mode for now"
            )
        return o
