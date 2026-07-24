# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from aiter.dist.parallel_state import get_tp_group

from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.models.flux2.flux2_transformer import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2TransformerBlock,
)

from atom.plugin.vllm_omni.diffusion.models.flux2.fused_qk_norm_rope import (
    try_fused_qk_norm_rope_2way,
)


def _gather_last_dim(hidden_states: torch.Tensor) -> torch.Tensor:
    tp_group = get_tp_group()
    if tp_group.world_size == 1:
        return hidden_states
    return tp_group.all_gather(hidden_states, dim=-1)


def _normalized_shape_size(normalized_shape) -> int:
    if isinstance(normalized_shape, tuple):
        return normalized_shape[0]
    return normalized_shape


class ATOMFlux2FeedForward(Flux2FeedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.linear_in.input_size
        inner_dim = self.linear_in.output_sizes[0]
        dim_out = self.linear_out.output_size
        bias = self.linear_in.bias is not None
        self.linear_in = MergedColumnParallelLinear(
            dim, [inner_dim, inner_dim], bias=bias
        )
        self.linear_out = RowParallelLinear(inner_dim, dim_out, bias=bias)


class ATOMFlux2Attention(Flux2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bias = self.to_qkv.bias is not None
        out_bias = self.to_out[0].bias is not None

        self.to_qkv = QKVParallelLinear(
            hidden_size=self.query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            bias=bias,
        )
        self.query_num_heads = self.to_qkv.num_heads
        self.kv_num_heads = self.to_qkv.num_kv_heads

        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(self.inner_dim, self.out_dim, bias=out_bias),
                nn.Dropout(self.dropout),
            ]
        )

        if self.added_kv_proj_dim is not None:
            added_proj_bias = self.add_kv_proj.bias is not None
            self.add_kv_proj = QKVParallelLinear(
                hidden_size=self.added_kv_proj_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                bias=added_proj_bias,
            )
            self.add_query_num_heads = self.add_kv_proj.num_heads
            self.add_kv_num_heads = self.add_kv_proj.num_kv_heads
            self.to_add_out = RowParallelLinear(
                self.inner_dim, self.query_dim, bias=out_bias
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        has_context = (
            encoder_hidden_states is not None and self.added_kv_proj_dim is not None
        )

        qkv = self.to_qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        encoder_query = encoder_key = encoder_value = None
        if has_context:
            encoder_qkv = self.add_kv_proj(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = encoder_qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (self.query_num_heads, -1))
        key = key.unflatten(-1, (self.kv_num_heads, -1))
        value = value.unflatten(-1, (self.kv_num_heads, -1))

        if has_context:
            encoder_query = encoder_query.unflatten(-1, (self.add_query_num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_num_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_num_heads, -1))

        fused_qk = None
        if image_rotary_emb is not None:
            fused_qk = try_fused_qk_norm_rope_2way(
                query=query,
                key=key,
                norm_q=self.norm_q,
                norm_k=self.norm_k,
                image_rotary_emb=image_rotary_emb,
                is_interleaved=self.rope.interleaved,
                encoder_query=encoder_query if has_context else None,
                encoder_key=encoder_key if has_context else None,
                norm_added_q=self.norm_added_q if has_context else None,
                norm_added_k=self.norm_added_k if has_context else None,
            )

        if fused_qk is not None:
            query, key = fused_qk
            if has_context:
                value = torch.cat([encoder_value, value], dim=1)
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)

            if has_context:
                encoder_query = self.norm_added_q(encoder_query)
                encoder_key = self.norm_added_k(encoder_key)

                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                cos = cos.to(query.dtype)
                sin = sin.to(query.dtype)
                query = self.rope(query, cos, sin)
                key = self.rope(key, cos, sin)

        attn_metadata = None
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if has_context:
            context_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [context_len, hidden_states.shape[1] - context_len],
                dim=1,
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states.contiguous())

        hidden_states = self.to_out[0](hidden_states.contiguous())
        hidden_states = self.to_out[1](hidden_states)

        if has_context:
            return hidden_states, encoder_hidden_states
        return hidden_states


class ATOMFlux2ParallelSelfAttention(Flux2ParallelSelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bias = self.to_qkv_mlp_proj.bias is not None
        out_bias = self.to_out.bias is not None
        self.to_qkv_mlp_proj = ColumnParallelLinear(
            self.query_dim,
            self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=bias,
        )
        self.to_out = ColumnParallelLinear(
            self.inner_dim + self.mlp_hidden_dim,
            self.out_dim,
            bias=out_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # vLLM relies on gather_output=True here; ATOM restores that explicitly.
        hidden_states = _gather_last_dim(self.to_qkv_mlp_proj(hidden_states))
        qkv, mlp_hidden_states = torch.split(
            hidden_states,
            [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        fused_qk = None
        if image_rotary_emb is not None:
            fused_qk = try_fused_qk_norm_rope_2way(
                query=query,
                key=key,
                norm_q=self.norm_q,
                norm_k=self.norm_k,
                image_rotary_emb=image_rotary_emb,
                is_interleaved=self.rope.interleaved,
            )

        if fused_qk is not None:
            query, key = fused_qk
        else:
            query = self.norm_q(query)
            key = self.norm_k(key)

            if image_rotary_emb is not None:
                cos, sin = image_rotary_emb
                cos = cos.to(query.dtype)
                sin = sin.to(query.dtype)
                query = self.rope(query, cos, sin)
                key = self.rope(key, cos, sin)

        attn_metadata = None
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        attn_output = self.attn(query, key, value, attn_metadata)
        attn_output = attn_output.flatten(2, 3).to(query.dtype)

        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = _gather_last_dim(self.to_out(hidden_states))
        return hidden_states


class ATOMFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = _normalized_shape_size(self.norm.normalized_shape)
        mlp_ratio = self.attn.mlp_hidden_dim / dim
        bias = self.attn.to_qkv_mlp_proj.bias is not None
        out_bias = self.attn.to_out.bias is not None
        self.attn = ATOMFlux2ParallelSelfAttention(
            query_dim=dim,
            heads=self.attn.heads,
            dim_head=self.attn.head_dim,
            dropout=self.attn.dropout,
            bias=bias,
            out_bias=out_bias,
            eps=self.norm.eps,
            out_dim=dim,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=self.attn.mlp_mult_factor,
        )


class ATOMFlux2TransformerBlock(Flux2TransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = _normalized_shape_size(self.norm1.normalized_shape)
        mlp_ratio = self.ff.linear_out.input_size / dim
        bias = self.attn.to_qkv.bias is not None
        added_proj_bias = self.attn.add_kv_proj.bias is not None
        out_bias = self.attn.to_out[0].bias is not None

        self.attn = ATOMFlux2Attention(
            query_dim=dim,
            heads=self.attn.heads,
            dim_head=self.attn.head_dim,
            dropout=self.attn.dropout,
            bias=bias,
            added_kv_proj_dim=self.attn.added_kv_proj_dim,
            added_proj_bias=added_proj_bias,
            out_bias=out_bias,
            eps=self.norm1.eps,
            out_dim=dim,
        )
        self.ff = ATOMFlux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=self.ff.linear_in.bias is not None,
        )
        self.ff_context = ATOMFlux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=self.ff_context.linear_in.bias is not None,
        )


class ATOMFlux2Transformer2DModel(Flux2Transformer2DModel):
    _repeated_blocks = [
        "ATOMFlux2TransformerBlock",
        "ATOMFlux2SingleTransformerBlock",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        original_transformer_blocks = list(self.transformer_blocks)
        self.transformer_blocks = nn.ModuleList(
            [
                ATOMFlux2TransformerBlock(
                    dim=_normalized_shape_size(block.norm1.normalized_shape),
                    num_attention_heads=block.attn.heads,
                    attention_head_dim=block.attn.head_dim,
                    mlp_ratio=block.ff.linear_out.input_size
                    / _normalized_shape_size(block.norm1.normalized_shape),
                    eps=block.norm1.eps,
                    bias=block.attn.to_qkv.bias is not None,
                )
                for block in original_transformer_blocks
            ]
        )

        original_single_transformer_blocks = list(self.single_transformer_blocks)
        self.single_transformer_blocks = nn.ModuleList(
            [
                ATOMFlux2SingleTransformerBlock(
                    dim=_normalized_shape_size(block.norm.normalized_shape),
                    num_attention_heads=block.attn.heads,
                    attention_head_dim=block.attn.head_dim,
                    mlp_ratio=block.attn.mlp_hidden_dim
                    / _normalized_shape_size(block.norm.normalized_shape),
                    eps=block.norm.eps,
                    bias=block.attn.to_qkv_mlp_proj.bias is not None,
                )
                for block in original_single_transformer_blocks
            ]
        )
