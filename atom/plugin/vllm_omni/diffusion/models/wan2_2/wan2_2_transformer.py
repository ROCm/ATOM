# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from vllm.logger import init_logger

from atom.model_ops.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    ColumnParallelGELU,
    WanCrossAttention,
    WanFeedForward,
    WanSelfAttention,
    WanTransformerBlock,
    WanTransformer3DModel,
    apply_rotary_emb_wan,
)

logger = init_logger(__name__)


class ATOMWanCrossAttention(WanCrossAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace vllm ColumnParallelLinear with atom versions
        self.to_q = ColumnParallelLinear(self.dim, self.inner_dim, bias=True)
        self.to_k = ColumnParallelLinear(self.dim, self.kv_inner_dim, bias=True)
        self.to_v = ColumnParallelLinear(self.dim, self.kv_inner_dim, bias=True)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = ColumnParallelLinear(self.added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = ColumnParallelLinear(self.added_kv_proj_dim, self.inner_dim, bias=True)
        else:
            self.add_k_proj = None
            self.add_v_proj = None
            self.norm_added_k = None

        # Replace vllm RowParallelLinear with atom version
        self.to_out = RowParallelLinear(self.inner_dim, self.dim, bias=True)
        # Inherited forward() works: atom Col/RowParallelLinear.forward() returns plain tensor,
        # same as vllm with return_bias=False.


class ATOMWanSelfAttention(WanSelfAttention):

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float = 1e-5, dropout: float = 0.0):
        super().__init__(dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps, dropout=dropout)
        # Replace vllm QKVParallelLinear with atom version
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim, head_size=head_dim, total_num_heads=num_heads, bias=True,
        )
        # Refresh head counts from the atom layer
        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads
        # Replace vllm RowParallelLinear with atom version
        self.to_out = RowParallelLinear(self.inner_dim, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # CRITICAL: atom QKVParallelLinear returns a plain tensor;
        # the stock WanSelfAttention.forward() does `qkv, _ = self.to_qkv(x)` (tuple unpack).
        qkv = self.to_qkv(hidden_states)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = self.norm_q(query)
        key   = self.norm_k(key)
        query = query.unflatten(2, (self.num_heads,    self.head_dim))
        key   = key.unflatten(  2, (self.num_kv_heads, self.head_dim))
        value = value.unflatten(2, (self.num_kv_heads, self.head_dim))

        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
            key   = apply_rotary_emb_wan(key,   freqs_cos, freqs_sin)

        attn_metadata = AttentionMetadata(attn_mask=attn_mask) if attn_mask is not None else None
        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = self.to_out(hidden_states)  # atom RowParallelLinear: tensor + all-reduce
        return self.dropout(hidden_states)


class ATOMWanFeedForward(WanFeedForward):

    def __init__(self, dim: int, inner_dim: int, dim_out: int | None = None, bias: bool = True):
        super().__init__(dim=dim, inner_dim=inner_dim, dim_out=dim_out, bias=bias)
        dim_out = dim_out or dim
        # Replace net_0.proj (inside ColumnParallelGELU) with atom ColumnParallelLinear.
        # ColumnParallelGELU.forward() calls self.proj(x) expecting a plain tensor —
        # atom ColumnParallelLinear.forward() satisfies this (no tuple).
        self.net_0.proj = ColumnParallelLinear(dim, inner_dim, bias=bias)
        # Replace net_2 with atom RowParallelLinear.
        self.net_2 = RowParallelLinear(inner_dim, dim_out, bias=bias)
        # forward() inherited from WanFeedForward: net_0 → net_1 (Identity) → net_2


class ATOMWanTransformerBlock(WanTransformerBlock):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
    ):
        super().__init__(
            dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, eps=eps,
            added_kv_proj_dim=added_kv_proj_dim, cross_attn_norm=cross_attn_norm,
        )
        head_dim = dim // num_heads
        self.attn1 = ATOMWanSelfAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps)
        self.attn2 = ATOMWanCrossAttention(
            dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
        )
        self.ffn = ATOMWanFeedForward(dim=dim, inner_dim=ffn_dim, dim_out=dim)
        # forward() inherited from WanTransformerBlock unchanged


class ATOMWanTransformer3DModel(WanTransformer3DModel):

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        inner_dim = num_attention_heads * attention_head_dim
        # Replace all WanTransformerBlocks with ATOMWanTransformerBlocks.
        # rope, patch_embedding, condition_embedder, norm_out, proj_out are kept from super().
        self.blocks = nn.ModuleList([
            ATOMWanTransformerBlock(
                inner_dim, ffn_dim, num_attention_heads, eps, added_kv_proj_dim, cross_attn_norm
            )
            for _ in range(num_layers)
        ])
        # forward(), load_weights(), _sp_plan, _repeated_blocks all inherited from WanTransformer3DModel
