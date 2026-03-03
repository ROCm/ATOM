# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import nn
from typing import Optional

from .attention_mla import MLAModules
from .base_attention import BaseAttention
from atom.plugin.prepare import is_plugin_mode, is_sglang
from atom.models.utils import maybe_prefix
from atom.utils import envs

from aiter.rotary_embedding import AiterFusedSetKVBufferArg


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
        super().__init__(
            num_heads=num_heads,
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
            **kwargs,
        )

        if is_sglang():
            self.rotary_emb = rotary_emb
            self.layer_num = layer_num

            self.k_scale = torch.tensor([1.0], dtype=torch.float32)
            self.v_scale = torch.tensor([1.0], dtype=torch.float32)

            # if True, save cache will be done in rope
            self.use_rope_fused_qknorm = envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION

            from sglang.srt.layers.radix_attention import RadixAttention
            self.attn = RadixAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                scaling=scale,
                num_kv_heads=num_kv_heads,
                layer_id=self.layer_num,
                prefix=maybe_prefix(prefix, "attn"),
            )
        else:
            raise NotImplementedError(
                "RadixAttention is only supported for plugin mode for sglang for now"
            )
        # if True, save cache will be done in rope
        # self.use_rope_fused_qknorm = envs.ATOM_ROPE_FUSED_QKNORM_FOR_SGL_PLUGIN_MODE

    def forward_impl_plugin_mode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata=None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        positions: torch.Tensor = None,
        q_scale: torch.Tensor = None,
        qkv: torch.Tensor = None,
        **kwargs,
    ):
        # for sglang, forward_batch is required
        forward_batch = kwargs.get("forward_batch", None)
        assert forward_batch is not None, "forward_batch is required for sglang"

        if self.use_rope_fused_qknorm:
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
                self.layer_num
            )
            block_size = 1024  # Default fallback
            if hasattr(forward_batch, "attn_backend") and hasattr(
                forward_batch.attn_backend, "page_size"
            ):
                block_size = forward_batch.attn_backend.page_size
            elif hasattr(forward_batch.token_to_kv_pool, "allocator") and hasattr(
                forward_batch.token_to_kv_pool.allocator, "page_size"
            ):
                block_size = forward_batch.token_to_kv_pool.allocator.page_size
            elif hasattr(forward_batch.token_to_kv_pool, "page_size"):
                block_size = forward_batch.token_to_kv_pool.page_size
            x = 16 // k_buffer.element_size()
            aiter_fused_set_kv_buffer_arg = AiterFusedSetKVBufferArg(
                kv_cache=(k_buffer, v_buffer),
                cache_loc=forward_batch.out_cache_loc,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
                return_kv=True,
                use_shuffle_layout=True,
                block_size=block_size,
                x=x,
            )
            q, k, v = self.rotary_emb(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                self.num_heads,
                self.num_kv_heads,
                self.q_norm.eps,
                fused_set_kv_buffer_arg=aiter_fused_set_kv_buffer_arg,
            )
        else:
            # calculate the q and k with rotary embedding
            assert self.rotary_emb is not None, "rotary_emb is required"
            q, k = self.rotary_emb(positions, query, key)
            v = value

        save_kv_cache = not self.use_rope_fused_qknorm
        return self.attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
        )


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
        qkv: torch.Tensor = None,
        **kwargs,
    ):
        if is_plugin_mode():
            o = self.forward_impl_plugin_mode(
                query=query,
                key=key,
                value=value,
                positions=positions,
                q_scale=q_scale,
                qkv=qkv,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "RadixAttention is not supported for server mode for now"
            )
        return o
