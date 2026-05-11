"""RTP-LLM scoped patch for ATOM GDN attention path."""

from __future__ import annotations

import logging
from typing import Any

import torch

from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.attention_gdn")

_PATCHED = False


def _extract_gdn_metadata(attn_metadata: Any, layer_name: str):
    gdn_metadata = getattr(attn_metadata, "gdn_metadata", None)
    if gdn_metadata is not None:
        return gdn_metadata

    if not isinstance(attn_metadata, dict):
        return None

    layer_md = attn_metadata.get(layer_name, None)
    if layer_md is None:
        layer_md = attn_metadata

    gdn_metadata = getattr(layer_md, "gdn_metadata", None)
    if gdn_metadata is not None:
        return gdn_metadata

    if isinstance(layer_md, dict):
        return layer_md.get("gdn_metadata", None)

    return None


def _safe_index_select(
    tensor: torch.Tensor | None, dim: int, index: torch.Tensor | None
) -> torch.Tensor | None:
    if tensor is None or index is None:
        return None
    if index.numel() == 0:
        shape = list(tensor.shape)
        shape[dim] = 0
        return tensor.new_empty(shape)
    return tensor.index_select(dim, index)


def apply_attention_gdn_rtpllm_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import atom.model_ops.attention_gdn as attention_gdn

    def _patched_gdn_forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
    ):
        from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata

        fwd_ctx = attention_gdn.get_forward_context()
        gdn_metadata: GDNAttentionMetadata = _extract_gdn_metadata(
            fwd_ctx.attn_metadata, layer_name
        )
        if gdn_metadata is None:
            raise RuntimeError(
                "RTP plugin missing GDN metadata in forward context; "
                "fallback/placeholder metadata is not allowed."
            )

        gdn_cache = fwd_ctx.kv_cache_data
        if gdn_cache is None:
            raise RuntimeError(
                "RTP plugin missing kv_cache_data in forward context; "
                "fallback/placeholder cache is not allowed."
            )

        layer_cache = gdn_cache.get(f"layer_{self.layer_num}")
        if layer_cache is None:
            raise RuntimeError(
                "RTP plugin missing GDN layer cache for "
                f"layer_{self.layer_num}; fallback path is not allowed."
            )
        conv_state = layer_cache.k_cache
        ssm_state = layer_cache.v_cache

        has_initial_state = gdn_metadata.has_initial_state
        spec_query_start_loc = gdn_metadata.spec_query_start_loc
        non_spec_query_start_loc = gdn_metadata.non_spec_query_start_loc
        spec_sequence_masks = gdn_metadata.spec_sequence_masks
        spec_token_indx = gdn_metadata.spec_token_indx
        non_spec_token_indx = gdn_metadata.non_spec_token_indx
        spec_state_indices_tensor = gdn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = gdn_metadata.non_spec_state_indices_tensor
        rtp_attn_inputs = getattr(gdn_metadata, "rtp_attn_inputs", None)
        rtp_seq_size_per_block = int(
            getattr(gdn_metadata, "rtp_seq_size_per_block", 0) or 0
        )
        if rtp_attn_inputs is not None and rtp_seq_size_per_block > 0:
            non_spec_state_indices_tensor = RTPForwardContext.state_indices_for_layer(
                attn_inputs=rtp_attn_inputs,
                is_prefill=bool(gdn_metadata.num_prefills > 0),
                device=conv_state.device,
                seq_size_per_block=rtp_seq_size_per_block,
                layer_num=int(self.layer_num),
            )

        # ModelRunner cache is [slot, state_len, conv_dim] and needs transpose.
        # RTP plugin may already pass [slot, conv_dim, state_len].
        if conv_state.size(1) != self.conv1d.weight.size(0):
            conv_state = conv_state.transpose(-1, -2)
        num_actual_tokens = gdn_metadata.num_actual_tokens
        num_accepted_tokens = gdn_metadata.num_accepted_tokens
        if num_actual_tokens <= 0:
            return core_attn_out

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        query_spec = key_spec = value_spec = None
        query_non_spec = key_non_spec = value_non_spec = None

        if spec_sequence_masks is not None:
            if gdn_metadata.num_prefills == 0 and gdn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = _safe_index_select(mixed_qkv, 0, spec_token_indx)
                mixed_qkv_non_spec = _safe_index_select(
                    mixed_qkv, 0, non_spec_token_indx
                )
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        if spec_sequence_masks is not None and mixed_qkv_spec is not None:
            query_spec, key_spec, value_spec = attention_gdn.causal_conv1d_update(
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

        if gdn_metadata.num_prefills > 0 and mixed_qkv_non_spec is not None:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            query_non_spec, key_non_spec, value_non_spec = attention_gdn.causal_conv1d_fn(
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
        elif gdn_metadata.num_decodes > 0 and mixed_qkv_non_spec is not None:
            query_non_spec, key_non_spec, value_non_spec = attention_gdn.causal_conv1d_update(
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

        if query_non_spec is not None:
            num_tokens_nonspec = query_non_spec.shape[0]
            query_non_spec = query_non_spec.view(
                1, num_tokens_nonspec, -1, self.head_k_dim
            )
            key_non_spec = key_non_spec.view(1, num_tokens_nonspec, -1, self.head_k_dim)
            value_non_spec = value_non_spec.view(
                1, num_tokens_nonspec, -1, self.head_v_dim
            )

        g, beta = attention_gdn.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        if spec_sequence_masks is not None:
            if gdn_metadata.num_prefills == 0 and gdn_metadata.num_decodes == 0:
                g_spec, beta_spec = g, beta
                g_non_spec, beta_non_spec = None, None
            else:
                g_spec = _safe_index_select(g, 1, spec_token_indx)
                beta_spec = _safe_index_select(beta, 1, spec_token_indx)
                g_non_spec = _safe_index_select(g, 1, non_spec_token_indx)
                beta_non_spec = _safe_index_select(beta, 1, non_spec_token_indx)
        else:
            g_spec = beta_spec = None
            g_non_spec, beta_non_spec = g, beta

        if spec_sequence_masks is not None and query_spec is not None:
            core_attn_out_spec, _ = attention_gdn.fused_recurrent_gated_delta_rule(
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
            core_attn_out_spec = None

        if gdn_metadata.num_prefills > 0 and query_non_spec is not None:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            core_attn_out_non_spec, last_recurrent_state = (
                attention_gdn.chunk_gated_delta_rule(
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
            )
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif gdn_metadata.num_decodes > 0 and query_non_spec is not None:
            core_attn_out_non_spec, _ = attention_gdn.fused_recurrent_gated_delta_rule(
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
        else:
            core_attn_out_non_spec = None

        if (
            spec_sequence_masks is not None
            and core_attn_out_spec is not None
            and core_attn_out_non_spec is not None
        ):
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None and core_attn_out_spec is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        elif core_attn_out_non_spec is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
        # Keep core/output semantics explicit: this is the pre-projection core output.
        return core_attn_out

    attention_gdn._extract_gdn_metadata = _extract_gdn_metadata
    attention_gdn._safe_index_select = _safe_index_select
    attention_gdn.GatedDeltaNet.forward = _patched_gdn_forward
    _PATCHED = True
    logger.info("Applied RTP patch for atom.model_ops.attention_gdn.GatedDeltaNet")
