"""RTP-LLM scoped patch for ATOM qwen3_next model path."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("atom.plugin.rtpllm.models.qwen3_next")

_PATCHED = False


def apply_qwen3_next_rtpllm_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import atom.models.qwen3_next as qwen3_next

    def _split_router_logits(self, router_logits: torch.Tensor):
        n_shared = int(getattr(self, "n_shared_experts", 0) or 0)
        if n_shared <= 0:
            n_routed = int(getattr(self, "n_routed_experts", 0) or 0)
            total_experts = int(router_logits.shape[-1])
            # Backward-compatible inference when main path has no `n_shared_experts`.
            if n_routed > 0 and total_experts > n_routed:
                n_shared = total_experts - n_routed
        if self.shared_expert is None or n_shared <= 0:
            return router_logits, None
        return torch.split(
            router_logits,
            [self.n_routed_experts, n_shared],
            dim=-1,
        )

    def _apply_shared_expert_gate(
        shared_output: torch.Tensor, shared_expert_gate_logits: torch.Tensor | None
    ) -> torch.Tensor:
        if shared_expert_gate_logits is None:
            return shared_output
        return torch.sigmoid(shared_expert_gate_logits) * shared_output

    def _patched_sparse_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        router_logits, shared_expert_gate_logits = self._split_router_logits(
            router_logits
        )
        routed_output = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if (
            not qwen3_next.is_rocm_aiter_fusion_shared_expert_enabled()
            and self.shared_expert is not None
        ):
            shared_output = self.shared_expert(hidden_states)
            shared_output = self._apply_shared_expert_gate(
                shared_output, shared_expert_gate_logits
            )
            final_hidden_states = shared_output + routed_output
        else:
            final_hidden_states = routed_output

        if self.tp_size > 1:
            final_hidden_states = qwen3_next.tensor_model_parallel_all_reduce(
                final_hidden_states
            )
        return final_hidden_states.view(orig_shape)

    def _patched_decoder_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_layernorm.use_fused_quant:
            if residual is None:
                residual = hidden_states
                hidden_states, x_scale, hidden_bf16 = self.input_layernorm(
                    hidden_states
                )
            else:
                hidden_states, x_scale, hidden_bf16, residual = self.input_layernorm(
                    hidden_states, residual
                )
        else:
            x_scale = hidden_bf16 = None
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            pre_ln_hidden = hidden_bf16 if hidden_bf16 is not None else hidden_states
            hidden_states = self.linear_attn(
                hidden_states=pre_ln_hidden,
                x_fp8=hidden_states if x_scale is not None else None,
                x_scale=x_scale,
            )
        elif self.layer_type == "full_attention":
            # RTP fused KV write path; RoPE happens inside RTP's fused kernel.
            # Slice positions as a zero-alloc view so capture does not record
            # fresh temporary allocations that may be reused before replay.
            real_num_tokens = int(hidden_states.shape[0])
            attn_positions = positions[:real_num_tokens]
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=attn_positions,
                x_scale=x_scale,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual

    def _patched_gdn_forward(
        self,
        hidden_states: torch.Tensor,
        x_fp8=None,
        x_scale=None,
    ):
        if hasattr(self, "in_proj_qkvzba"):
            projected_states_qkvzba = self.in_proj_qkvzba(hidden_states)
            ba_dim = 2 * (self.num_v_heads // self.tp_size)
            projected_states_qkvz = projected_states_qkvzba[..., :-ba_dim]
            projected_states_ba = projected_states_qkvzba[..., -ba_dim:]
            k_heads_after_tp = self.num_k_heads // self.tp_size
            v_heads_after_tp = self.num_v_heads // self.tp_size
            mixed_qkv, z, b, a, core_attn_out = (
                qwen3_next.fused_split_chunk_qwen_next_qkvzba(
                    projected_states_qkvzba,
                    k_heads_after_tp,
                    v_heads_after_tp,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            )
        else:
            if x_fp8 is not None:
                projected_states_qkvz = self.in_proj_qkvz(x_fp8, x_scale=x_scale)
            else:
                projected_states_qkvz = self.in_proj_qkvz(hidden_states)
            projected_states_ba = self.in_proj_ba(hidden_states)
            num_k_heads_tp = self.num_k_heads // self.tp_size
            num_v_heads_tp = self.num_v_heads // self.tp_size
            mixed_qkv, z, b, a, core_attn_out = (
                qwen3_next.fused_split_chunk_qwen_next_qkvz_ba(
                    projected_states_qkvz,
                    projected_states_ba,
                    num_k_heads_tp,
                    num_v_heads_tp,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            )
        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)
        core_attn_out, maybe_scale = self.norm(core_attn_out, z)
        output = self.out_proj(core_attn_out, x_scale=maybe_scale)
        return output

    cls = qwen3_next.Qwen3NextSparseMoeBlock
    # Main path references `self.shared_expert_gate` but does not always initialize it.
    # Set a class-level default so plugin mode won't crash on attribute lookup.
    cls.shared_expert_gate = None
    cls._split_router_logits = _split_router_logits
    cls._apply_shared_expert_gate = staticmethod(_apply_shared_expert_gate)
    cls.forward = _patched_sparse_moe_forward
    qwen3_next.Qwen3NextDecoderLayer.forward = _patched_decoder_forward
    qwen3_next.Qwen3NextGatedDeltaNet.forward = _patched_gdn_forward

    _PATCHED = True
    logger.info(
        "Applied RTP patch for atom.models.qwen3_next sparse_moe and decoder forward"
    )
