"""RTP-LLM scoped patch for ATOM qwen3_next model path."""

from __future__ import annotations

import os
import logging

import torch

from atom.plugin.rtpllm.utils.tensor_dump import dump_tensor as dump_atom_tensor
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom.plugin.rtpllm.models.qwen3_next")

_PATCHED = False


def _current_is_prefill() -> bool:
    fwd_ctx = get_forward_context()
    if fwd_ctx is None:
        return False
    attn_md = getattr(fwd_ctx, "attn_metadata", None)
    gdn_md = getattr(attn_md, "gdn_metadata", None)
    attn_inputs = getattr(gdn_md, "rtp_attn_inputs", None)
    return bool(getattr(attn_inputs, "is_prefill", False))


def _current_attn_inputs():
    fwd_ctx = get_forward_context()
    if fwd_ctx is None:
        return None
    attn_md = getattr(fwd_ctx, "attn_metadata", None)
    if attn_md is None:
        return None
    return getattr(attn_md, "rtp_attn_inputs", None)


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
        layer_idx = int(getattr(self, "layer_idx", -1))
        is_prefill = _current_is_prefill()
        dump_meta = {"is_prefill": is_prefill}
        if self.layer_type == "full_attention":
            dump_atom_tensor(
                tag="full_attn/hidden_states_in",
                tensor=hidden_states,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/residual_in",
                tensor=residual,
                layer=layer_idx,
                meta=dump_meta,
            )
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

        pre_ln_weight = getattr(self.input_layernorm, "weight", None)
        if pre_ln_weight is not None:
            # GemmaRMSNorm keeps delta in parameter and applies (1 + weight) in forward.
            # Dump gamma-form weight to align with RTP side RMSResNorm weight semantics.
            pre_ln_weight = pre_ln_weight + 1.0

        if self.layer_type == "linear_attention":
            pre_ln_hidden = hidden_bf16 if hidden_bf16 is not None else hidden_states
            x_scale_to_dump = x_scale
            if x_scale_to_dump is None:
                x_scale_to_dump = torch.empty(
                    (0,), dtype=torch.float32, device=pre_ln_hidden.device
                )
            dump_atom_tensor(
                tag="gdn/pre_ln_weight",
                tensor=pre_ln_weight,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/pre_ln_out",
                tensor=pre_ln_hidden,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/pre_ln_residual_out",
                tensor=residual,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/x_scale",
                tensor=x_scale_to_dump,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/hidden_states_in",
                tensor=pre_ln_hidden,
                layer=layer_idx,
                meta=dump_meta,
            )
            proj_input_fp8 = (
                hidden_states
                if x_scale is not None
                else torch.empty((0,), dtype=pre_ln_hidden.dtype, device=pre_ln_hidden.device)
            )
            proj_input_source_id = torch.tensor(
                [1 if x_scale is not None else 0],
                dtype=torch.int32,
                device=pre_ln_hidden.device,
            )
            dump_atom_tensor(
                tag="gdn/proj_input_hidden",
                tensor=pre_ln_hidden,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/proj_input_fp8",
                tensor=proj_input_fp8,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/proj_input_scale",
                tensor=x_scale_to_dump,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/proj_input_source_id",
                tensor=proj_input_source_id,
                layer=layer_idx,
                meta=dump_meta,
            )
            # Dump RTP-aligned projection tensors here as a stable fallback hook.
            # This branch is always executed for linear-attn layers in decoder forward.
            projected_qkvz = None
            projected_ba = None
            try:
                if x_scale is not None:
                    projected_qkvz = self.linear_attn.in_proj_qkvz(
                        hidden_states, x_scale=x_scale
                    )
                else:
                    projected_qkvz = self.linear_attn.in_proj_qkvz(pre_ln_hidden)
                projected_ba = self.linear_attn.in_proj_ba(pre_ln_hidden)
            except Exception:  # noqa: BLE001
                projected_qkvz = None
                projected_ba = None
            dump_atom_tensor(
                tag="gdn/projected_qkvz",
                tensor=projected_qkvz,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/projected_ba",
                tensor=projected_ba,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/mixed_qkv_input_qkvz",
                tensor=projected_qkvz,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="gdn/mixed_qkv_input_ba",
                tensor=projected_ba,
                layer=layer_idx,
                meta=dump_meta,
            )
            hidden_states = self.linear_attn(
                hidden_states=pre_ln_hidden,
                x_fp8=hidden_states if x_scale is not None else None,
                x_scale=x_scale,
            )
            dump_atom_tensor(
                tag="gdn/attn_output",
                tensor=hidden_states,
                layer=layer_idx,
                meta=dump_meta,
            )
        elif self.layer_type == "full_attention":
            attn_inputs = _current_attn_inputs()
            is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
            dump_meta = {"is_prefill": is_prefill}
            use_rtp_fused_kv_write = (
                os.getenv("ATOM_RTP_USE_RTP_FUSED_KV_WRITE", "0") == "1"
            )
            attn_positions = (
                torch.zeros_like(positions) if use_rtp_fused_kv_write else positions
            )
            dump_atom_tensor(
                tag="full_attn/pre_ln_weight",
                tensor=pre_ln_weight,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/pre_ln_out",
                tensor=hidden_states,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/pre_ln_residual_out",
                tensor=residual,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/positions",
                tensor=positions,
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/positions_for_attn",
                tensor=attn_positions,
                layer=layer_idx,
                meta={**dump_meta, "rope_mocked": use_rtp_fused_kv_write},
            )
            dump_atom_tensor(
                tag="full_attn/block_map",
                tensor=getattr(attn_inputs, "kv_cache_kernel_block_id_device", None),
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/sequence_lengths",
                tensor=getattr(attn_inputs, "sequence_lengths", None),
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/sequence_lengths_plus_1",
                tensor=getattr(attn_inputs, "sequence_lengths_plus_1_d", None),
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/prefix_lengths",
                tensor=getattr(attn_inputs, "prefix_lengths", None),
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/prefix_lengths_d",
                tensor=getattr(attn_inputs, "prefix_lengths_d", None),
                layer=layer_idx,
                meta=dump_meta,
            )
            dump_atom_tensor(
                tag="full_attn/input_lengths",
                tensor=getattr(attn_inputs, "input_lengths", None),
                layer=layer_idx,
                meta=dump_meta,
            )
            cu_seqlens_tag = (
                "full_attn/cu_seqlens_prefill"
                if bool(getattr(attn_inputs, "is_prefill", False))
                else "full_attn/cu_seqlens_decode"
            )
            dump_atom_tensor(
                tag=cu_seqlens_tag,
                tensor=getattr(attn_inputs, "cu_seqlens", None),
                layer=layer_idx,
                meta=dump_meta,
            )
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=attn_positions,
                x_scale=x_scale,
            )
            dump_atom_tensor(
                tag="full_attn/attn_output",
                tensor=hidden_states,
                layer=layer_idx,
                meta=dump_meta,
            )
        else:
            raise ValueError("Invalid layer_type")

        dump_atom_tensor(
            tag="bridge/attn_out",
            tensor=hidden_states,
            layer=layer_idx,
        )

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
        dump_atom_tensor(
            tag="bridge/post_ln_out",
            tensor=hidden_states,
            layer=layer_idx,
        )
        dump_atom_tensor(
            tag="bridge/post_ln_residual_out",
            tensor=residual,
            layer=layer_idx,
        )
        hidden_states = self.mlp(hidden_states)
        dump_atom_tensor(
            tag="bridge/mlp_out",
            tensor=hidden_states,
            layer=layer_idx,
        )

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
        layer_num = int(
            getattr(
                self,
                "layer_num",
                getattr(self, "debug_layer_idx", -1),
            )
        )
        is_prefill = _current_is_prefill()
        if hasattr(self, "in_proj_qkvzba"):
            projected_states_qkvzba = self.in_proj_qkvzba(hidden_states)
            ba_dim = 2 * (self.num_v_heads // self.tp_size)
            projected_states_qkvz = projected_states_qkvzba[..., :-ba_dim]
            projected_states_ba = projected_states_qkvzba[..., -ba_dim:]
            k_heads_after_tp = self.num_k_heads // self.tp_size
            v_heads_after_tp = self.num_v_heads // self.tp_size
            mixed_qkv, z, b, a, core_attn_out = qwen3_next.fused_split_chunk_qwen_next_qkvzba(
                projected_states_qkvzba,
                k_heads_after_tp,
                v_heads_after_tp,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            if x_fp8 is not None:
                projected_states_qkvz = self.in_proj_qkvz(x_fp8, x_scale=x_scale)
            else:
                projected_states_qkvz = self.in_proj_qkvz(hidden_states)
            projected_states_ba = self.in_proj_ba(hidden_states)
            num_k_heads_tp = self.num_k_heads // self.tp_size
            num_v_heads_tp = self.num_v_heads // self.tp_size
            mixed_qkv, z, b, a, core_attn_out = qwen3_next.fused_split_chunk_qwen_next_qkvz_ba(
                projected_states_qkvz,
                projected_states_ba,
                num_k_heads_tp,
                num_v_heads_tp,
                self.head_k_dim,
                self.head_v_dim,
            )

        dump_atom_tensor(
            tag="gdn/projected_qkvz",
            tensor=projected_states_qkvz,
            layer=layer_num,
            meta={"is_prefill": bool(is_prefill)},
        )
        dump_atom_tensor(
            tag="gdn/projected_ba",
            tensor=projected_states_ba,
            layer=layer_num,
            meta={"is_prefill": bool(is_prefill)},
        )
        layer_cache = None
        fwd_ctx = get_forward_context()
        if fwd_ctx is not None:
            kv_cache_data = getattr(fwd_ctx, "kv_cache_data", None)
            if isinstance(kv_cache_data, dict):
                layer_cache = kv_cache_data.get(f"layer_{layer_num}")
        if layer_cache is not None:
            dump_atom_tensor(
                tag="gdn/state_conv_pre",
                tensor=getattr(layer_cache, "k_cache", None),
                layer=layer_num,
                meta={"is_prefill": bool(is_prefill)},
            )
            dump_atom_tensor(
                tag="gdn/state_ssm_pre",
                tensor=getattr(layer_cache, "v_cache", None),
                layer=layer_num,
                meta={"is_prefill": bool(is_prefill)},
            )
        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)
        if layer_cache is not None:
            dump_atom_tensor(
                tag="gdn/state_conv_post",
                tensor=getattr(layer_cache, "k_cache", None),
                layer=layer_num,
                meta={"is_prefill": bool(is_prefill)},
            )
            dump_atom_tensor(
                tag="gdn/state_ssm_post",
                tensor=getattr(layer_cache, "v_cache", None),
                layer=layer_num,
                meta={"is_prefill": bool(is_prefill)},
            )
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

