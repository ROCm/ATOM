# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from atom.model_ops.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    FeedForward,
    QwenImageCrossAttention,
    QwenImageTransformerBlock,
    QwenImageTransformer2DModel,
)

logger = init_logger(__name__)


class ATOMFeedForward(FeedForward):

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(
            dim=dim, dim_out=dim_out, mult=mult, activation_fn=activation_fn,
            inner_dim=inner_dim, bias=bias, quant_config=quant_config, prefix=prefix,
        )
        inner_dim_val = inner_dim or int(dim * mult)
        dim_out_val = dim_out or dim
        # Replace ColumnParallelApproxGELU's inner proj with ATOM ColumnParallelLinear.
        # ColumnParallelApproxGELU.forward() calls self.proj(x) → plain tensor ✓
        self.net[0].proj = ColumnParallelLinear(dim, inner_dim_val, bias=bias)
        # Replace net[2] (RowParallelLinear) with ATOM version.
        self.net[2] = RowParallelLinear(inner_dim_val, dim_out_val, bias=bias)
        # forward() inherited: iterates self.net ✓


class ATOMQwenImageCrossAttention(QwenImageCrossAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        added_kv_proj_dim: int,
        window_size: tuple[int, int] = (-1, -1),
        out_bias: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        pre_only: bool = False,
        context_pre_only: bool = False,
        out_dim: int | None = None,
        quant_config=None,
    ):
        super().__init__(
            dim=dim, num_heads=num_heads, head_dim=head_dim,
            added_kv_proj_dim=added_kv_proj_dim, window_size=window_size,
            out_bias=out_bias, qk_norm=qk_norm, eps=eps, pre_only=pre_only,
            context_pre_only=context_pre_only, out_dim=out_dim, quant_config=quant_config,
        )
        # Replace vLLM QKVParallelLinear with ATOM versions; refresh head counts.
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim, head_size=head_dim, total_num_heads=num_heads, bias=True,
        )
        self.query_num_heads = self.to_qkv.num_heads
        self.kv_num_heads = self.to_qkv.num_kv_heads

        self.add_kv_proj = QKVParallelLinear(
            hidden_size=added_kv_proj_dim, head_size=head_dim, total_num_heads=num_heads, bias=True,
        )
        self.add_query_num_heads = self.add_kv_proj.num_heads
        self.add_kv_num_heads = self.add_kv_proj.num_kv_heads

        inner_dim = out_dim if out_dim is not None else head_dim * num_heads
        # Replace vLLM RowParallelLinear with ATOM versions.
        self.to_out = RowParallelLinear(inner_dim, dim, bias=out_bias)
        self.to_add_out = RowParallelLinear(inner_dim, dim, bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vid_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
        hidden_states_mask: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # CRITICAL: ATOM QKVParallelLinear returns a plain tensor; vLLM returns (tensor, None).
        img_qkv = self.to_qkv(hidden_states)
        q_size = self.query_num_heads * self.head_dim
        kv_size = self.kv_num_heads * self.head_dim
        img_query, img_key, img_value = img_qkv.split([q_size, kv_size, kv_size], dim=-1)

        txt_qkv = self.add_kv_proj(encoder_hidden_states)
        add_q_size = self.add_query_num_heads * self.head_dim
        add_kv_size = self.add_kv_num_heads * self.head_dim
        txt_query, txt_key, txt_value = txt_qkv.split([add_q_size, add_kv_size, add_kv_size], dim=-1)

        img_query = img_query.unflatten(-1, (self.query_num_heads, self.head_dim))
        img_key   = img_key.unflatten(  -1, (self.kv_num_heads,    self.head_dim))
        img_value = img_value.unflatten(-1, (self.kv_num_heads,    self.head_dim))

        txt_query = txt_query.unflatten(-1, (self.add_query_num_heads, self.head_dim))
        txt_key   = txt_key.unflatten(  -1, (self.add_kv_num_heads,   self.head_dim))
        txt_value = txt_value.unflatten(-1, (self.add_kv_num_heads,   self.head_dim))

        img_query = self.norm_q(img_query)
        img_key   = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key   = self.norm_added_k(txt_key)

        img_cos = vid_freqs.real.to(img_query.dtype)
        img_sin = vid_freqs.imag.to(img_query.dtype)
        txt_cos = txt_freqs.real.to(txt_query.dtype)
        txt_sin = txt_freqs.imag.to(txt_query.dtype)

        img_query = self.rope(img_query, img_cos, img_sin)
        img_key   = self.rope(img_key,   img_cos, img_sin)
        txt_query = self.rope(txt_query, txt_cos, txt_sin)
        txt_key   = self.rope(txt_key,   txt_cos, txt_sin)

        seq_len_txt = encoder_hidden_states.shape[1]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key   = torch.cat([txt_key,   img_key],   dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        if (
            self.parallel_config is not None
            and self.parallel_config.sequence_parallel_size > 1
            and not get_forward_context().split_text_embed_in_sp
        ):
            attn_metadata = AttentionMetadata(
                joint_query=txt_query,
                joint_key=txt_key,
                joint_value=txt_value,
                joint_strategy="front",
            )
            if hidden_states_mask is not None:
                attn_metadata.attn_mask = hidden_states_mask
            if encoder_hidden_states_mask is not None:
                attn_metadata.joint_attn_mask = encoder_hidden_states_mask

            joint_hidden_states = self.attn(img_query, img_key, img_value, attn_metadata)
        else:
            attn_metadata = None
            if hidden_states_mask is not None or encoder_hidden_states_mask is not None:
                mask_list: list[torch.Tensor] = []
                if encoder_hidden_states_mask is not None:
                    mask_list.append(encoder_hidden_states_mask)
                else:
                    mask_list.append(
                        torch.ones(
                            encoder_hidden_states.shape[:2],
                            dtype=torch.bool,
                            device=encoder_hidden_states.device,
                        )
                    )
                if hidden_states_mask is not None:
                    mask_list.append(hidden_states_mask)
                else:
                    mask_list.append(
                        torch.ones(
                            hidden_states.shape[:2],
                            dtype=torch.bool,
                            device=hidden_states.device,
                        )
                    )
                joint_mask = torch.cat(mask_list, dim=1) if len(mask_list) > 1 else mask_list[0]
                attn_metadata = AttentionMetadata(attn_mask=joint_mask)

            joint_hidden_states = self.attn(joint_query, joint_key, joint_value, attn_metadata)

        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_len_txt, :]
        img_attn_output = joint_hidden_states[:, seq_len_txt:, :]

        # ATOM RowParallelLinear returns plain tensor + performs all-reduce ✓
        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class ATOMQwenImageTransformerBlock(QwenImageTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
        quant_config=None,
    ):
        super().__init__(
            dim=dim, num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim, qk_norm=qk_norm,
            eps=eps, zero_cond_t=zero_cond_t, quant_config=quant_config,
        )
        # Replace joint cross-attention with ATOM version (QKV + Row parallel layers).
        # img_mod and txt_mod use ReplicatedLinear — not replaced (broadcast, not sharded).
        self.attn = ATOMQwenImageCrossAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            added_kv_proj_dim=dim,
            context_pre_only=False,
        )
        # Replace feedforward layers with ATOM versions.
        self.img_mlp = ATOMFeedForward(dim=dim, dim_out=dim)
        self.txt_mlp = ATOMFeedForward(dim=dim, dim_out=dim)
        # forward() inherited from QwenImageTransformerBlock unchanged ✓


class ATOMQwenImageTransformer2DModel(QwenImageTransformer2DModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Collect block constructor args from the already-built first block to stay DRY.
        num_attention_heads = self.transformer_blocks[0].num_attention_heads
        attention_head_dim  = self.transformer_blocks[0].attention_head_dim
        zero_cond_t         = self.transformer_blocks[0].zero_cond_t
        num_layers          = len(self.transformer_blocks)
        # Replace all QwenImageTransformerBlocks with ATOM versions.
        # img_in, txt_in, time_text_embed, norm_out.linear, proj_out use ReplicatedLinear — kept.
        import torch.nn as nn
        self.transformer_blocks = nn.ModuleList([
            ATOMQwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                zero_cond_t=zero_cond_t,
            )
            for _ in range(num_layers)
        ])
        # forward(), load_weights(), _sp_plan, _repeated_blocks all inherited ✓