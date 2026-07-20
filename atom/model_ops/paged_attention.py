# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
from typing import Optional
import torch
from torch import nn

from .attention_mla import MLAModules
from .base_attention import BaseAttention
from atom.config import get_current_atom_config
from atom.utils.selector import get_attn_backend
from atom.plugin.prepare import is_plugin_mode


class Attention(BaseAttention):
    """
    Attention paged implementation
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        alibi_slopes: list[float] = None,
        kv_cache_dtype="bf16",
        layer_num=0,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        per_layer_sliding_window: Optional[int] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        prefix: Optional[str] = None,
        q_norm: Optional[torch.nn.Module] = None,
        k_norm: Optional[torch.nn.Module] = None,
        impl_cls: Optional[type] = None,
        **kwargs,
    ):
        assert (
            not is_plugin_mode()
        ), "ATOM native Attention is only supported for ATOM native/server mode"
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

        self.use_mla = use_mla
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.layer_num = layer_num
        self.mla_modules = mla_modules
        self.base_attention = None
        self.kv_cache = torch.tensor([])
        self.indexer = mla_modules.indexer if mla_modules is not None else None
        self.sinks = sinks

        atom_config = get_current_atom_config()
        dtype = atom_config.torch_dtype
        block_size = atom_config.kv_cache_block_size
        self.attn_backend = get_attn_backend(
            block_size,
            use_mla=self.use_mla,
        )
        # Allow a model to plug in a specialized impl (e.g. the MiniMax-M3 sparse
        # attention impl) while still reusing the backend's metadata builder.
        # Falls back to the backend default when not overridden.
        impl_cls = impl_cls or self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            mla_modules=mla_modules,
            sinks=sinks,
            sliding_window=per_layer_sliding_window,
            rotary_emb=rotary_emb,
            dtype=dtype,
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        )
        compilation_config = atom_config.compilation_config
        default_name = f"MLA_{layer_num}" if self.use_mla else f"MHA_{layer_num}"
        self.layer_name = prefix if prefix is not None else default_name
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer: {}".format(self.layer_name))
        compilation_config.static_forward_context[self.layer_name] = self

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
        output = torch.ops.aiter.unified_attention_with_output_base(
            query, q_scale, key, value, positions, self.layer_name, self.use_mla, qkv
        )
        return output
