# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton
import triton.language as tl
from einops import rearrange
from atom.model_ops.mamba_ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from atom.model_ops.fla_ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

# from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata
from atom.utils.forward_context import ForwardContext, get_forward_context
from torch import nn
from aiter.dist.parallel_state import get_tp_group


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


class GatedDetlaNet(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        key_dim: int,
        value_dim: int,
        dt_bias: torch.Tensor,
        A_log: torch.Tensor,
        conv1d,
        activation,
        layer_num: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.layer_num = layer_num

        self.tp_size = get_tp_group().world_size
        self.conv1d = conv1d
        self.activation = activation
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata

        fwd_ctx: ForwardContext = get_forward_context()
        gdn_metadata: GDNAttentionMetadata = fwd_ctx.attn_metadata.gdn_metadata

        if gdn_metadata is None:
            return core_attn_out
        # assert isinstance(gdn_metadata, dict)
        # gdn_metadata = gdn_metadata[self.prefix]
        # assert isinstance(gdn_metadata, GDNAttentionMetadata)

        gdn_cache = fwd_ctx.kv_cache_data
        conv_state = gdn_cache[f"layer_{self.layer_num}"].k_cache
        ssm_state = gdn_cache[f"layer_{self.layer_num}"].v_cache

        has_initial_state = gdn_metadata.has_initial_state
        spec_query_start_loc = gdn_metadata.spec_query_start_loc
        non_spec_query_start_loc = gdn_metadata.non_spec_query_start_loc
        spec_sequence_masks = gdn_metadata.spec_sequence_masks
        spec_token_indx = gdn_metadata.spec_token_indx
        non_spec_token_indx = gdn_metadata.non_spec_token_indx
        spec_state_indices_tensor = gdn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = (
            gdn_metadata.non_spec_state_indices_tensor
        )  # noqa: E501
        # non_spec_state_indices_tensor = non_spec_state_indices_tensor + ((self.layer_num + 1) % 4 - 1)

        conv_state = conv_state.transpose(-1, -2)

        # if self.prefix=="model.layers.0.linear_attn" and conv_state.device.index==0:
        #     print(f"conv_state sum {conv_state.sum()}, ssm_state sum {ssm_state.sum()}", flush=True)
        # self_kv_cache = self.kv_cache[fwd_ctx.virtual_engine]
        # conv_state = self_kv_cache[0].transpose(-1, -2)
        # ssm_state = self_kv_cache[1]
        num_actual_tokens = gdn_metadata.num_actual_tokens
        num_accepted_tokens = gdn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if gdn_metadata.num_prefills == 0 and gdn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv
        # print(f"with shape {mixed_qkv_non_spec.shape[0]}", flush=True)
        # # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            query_spec, key_spec, value_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.num_k_heads * self.head_k_dim // self.tp_size,
                self.num_v_heads * self.head_v_dim // self.tp_size,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : gdn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
            num_tokens_spec = query_spec.shape[0]
            query_spec = query_spec.view(1, num_tokens_spec, -1, self.head_k_dim)
            key_spec = key_spec.view(1, num_tokens_spec, -1, self.head_k_dim)
            value_spec = value_spec.view(1, num_tokens_spec, -1, self.head_v_dim)

        # 1.2: Process the remaining part
        if gdn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            query_non_spec, key_non_spec, value_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                k_dim_size=self.num_k_heads * self.head_k_dim // self.tp_size,
                v_dim_size=self.num_v_heads * self.head_v_dim // self.tp_size,
                metadata=gdn_metadata,
            )
        elif gdn_metadata.num_decodes > 0:
            query_non_spec, key_non_spec, value_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.num_k_heads * self.head_k_dim // self.tp_size,
                self.num_v_heads * self.head_v_dim // self.tp_size,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : gdn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        if gdn_metadata.num_prefills > 0 or gdn_metadata.num_decodes > 0:
            num_tokens_nonspec = query_non_spec.shape[0]
            query_non_spec = query_non_spec.view(
                1, num_tokens_nonspec, -1, self.head_k_dim
            )
            key_non_spec = key_non_spec.view(1, num_tokens_nonspec, -1, self.head_k_dim)
            value_non_spec = value_non_spec.view(
                1, num_tokens_nonspec, -1, self.head_v_dim
            )

        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)

        if spec_sequence_masks is not None:
            if gdn_metadata.num_prefills == 0 and gdn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: gdn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if gdn_metadata.num_prefills > 0:
            # print("ssm sstates: ", ssm_state.shape)
            # print("non_spec_state_indices_tensor: ", non_spec_state_indices_tensor.shape, flush=True)
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif gdn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[: gdn_metadata.num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output

        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

        return core_attn_out
