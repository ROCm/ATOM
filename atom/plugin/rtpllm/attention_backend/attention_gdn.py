"""RTP-LLM scoped patch for ATOM GDN attention path."""

from __future__ import annotations

import logging

import torch

from atom.plugin.rtpllm.utils.forward_context import RTPForwardContext

logger = logging.getLogger("atom.plugin.rtpllm.attention_backend.attention_gdn")

_PATCHED = False


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
        del layer_name

        fwd_ctx = attention_gdn.get_forward_context()
        gdn_metadata = getattr(fwd_ctx.attn_metadata, "gdn_metadata", None)
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
        non_spec_query_start_loc = gdn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = gdn_metadata.non_spec_state_indices_tensor
        rtp_attn_inputs = getattr(gdn_metadata, "rtp_attn_inputs", None)
        rtp_seq_size_per_block = int(getattr(gdn_metadata, "rtp_seq_size_per_block", 0))
        rtp_state_indices_cache = getattr(gdn_metadata, "rtp_state_indices_cache", None)
        rtp_layer_group_map = getattr(gdn_metadata, "rtp_layer_group_map", None)
        if rtp_attn_inputs is not None and rtp_seq_size_per_block > 0:
            non_spec_state_indices_tensor = RTPForwardContext.state_indices_for_layer(
                attn_inputs=rtp_attn_inputs,
                is_prefill=bool(gdn_metadata.num_prefills > 0),
                device=conv_state.device,
                seq_size_per_block=rtp_seq_size_per_block,
                layer_num=int(self.layer_num),
                state_indices_cache=rtp_state_indices_cache,
                layer_group_map=rtp_layer_group_map,
            )

        # RTP plugin cache layout is fixed to [slot, conv_dim, state_len].
        num_actual_tokens = gdn_metadata.num_actual_tokens
        if num_actual_tokens <= 0:
            return core_attn_out

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        if gdn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv.transpose(0, 1)
            query_non_spec, key_non_spec, value_non_spec = (
                attention_gdn.causal_conv1d_fn(
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
            )
        elif gdn_metadata.num_decodes > 0:
            query_non_spec, key_non_spec, value_non_spec = (
                attention_gdn.causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    conv_weights,
                    self.num_k_heads * self.head_k_dim // self.tp_size,
                    self.num_v_heads * self.head_v_dim // self.tp_size,
                    self.conv1d.bias,
                    self.activation,
                    conv_state_indices=non_spec_state_indices_tensor,
                    validate_data=True,
                )
            )
        else:
            return core_attn_out

        num_tokens_nonspec = query_non_spec.shape[0]
        query_non_spec = query_non_spec.view(1, num_tokens_nonspec, -1, self.head_k_dim)
        key_non_spec = key_non_spec.view(1, num_tokens_nonspec, -1, self.head_k_dim)
        value_non_spec = value_non_spec.view(1, num_tokens_nonspec, -1, self.head_v_dim)

        g, beta = attention_gdn.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        if gdn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            core_attn_out_non_spec, last_recurrent_state = (
                attention_gdn.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g,
                    beta=beta,
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
        elif gdn_metadata.num_decodes > 0:
            core_attn_out_non_spec, _ = attention_gdn.fused_recurrent_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g,
                beta=beta,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[: gdn_metadata.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            return core_attn_out
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
        # Keep core/output semantics explicit: this is the pre-projection core output.
        return core_attn_out

    attention_gdn.GatedDeltaNet.forward = _patched_gdn_forward
    _PATCHED = True
    logger.info("Applied RTP patch for atom.model_ops.attention_gdn.GatedDeltaNet")
