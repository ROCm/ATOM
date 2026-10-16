"""RTP-only Qwen4Exp MTP state updates, bound to the target model instance."""

from __future__ import annotations

from dataclasses import replace
from types import MethodType

import torch

from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.models.qwen4_exp import Qwen4ExpDecoderLayer
from atom.utils.forward_context import get_forward_context


def _rtp_spec_conv1d_update(
    self: GatedDeltaNet,
    mixed_qkv_spec: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    gdn_metadata,
    num_accepted_tokens,
    use_replayssm,
):
    block_map = getattr(gdn_metadata, "rtp_target_verify_block_map", None)
    if block_map is None:
        return GatedDeltaNet._spec_conv1d_update(
            self,
            mixed_qkv_spec,
            conv_state,
            conv_weights,
            gdn_metadata,
            num_accepted_tokens,
            use_replayssm,
        )

    from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
        causal_conv1d_update,
    )

    batch_size = gdn_metadata.num_spec_decodes
    query_len = gdn_metadata.spec_state_indices_tensor.shape[1]
    mixed_width = mixed_qkv_spec.shape[-1]
    conv_out = causal_conv1d_update(
        mixed_qkv_spec.reshape(batch_size, query_len, mixed_width)
        .transpose(1, 2)
        .contiguous(),
        conv_state,
        conv_weights,
        bias=self.conv1d.bias,
        activation=self.activation,
        block_map=block_map,
        seq_size_per_block=gdn_metadata.rtp_seq_size_per_block,
        sequence_lengths=gdn_metadata.rtp_target_verify_sequence_lengths,
    )
    conv_out = conv_out.transpose(1, 2).reshape(-1, mixed_width)
    return torch.split(
        conv_out,
        [
            self.num_k_heads * self.head_k_dim // self.tp_size,
            self.num_k_heads * self.head_k_dim // self.tp_size,
            self.num_v_heads * self.head_v_dim // self.tp_size,
        ],
        dim=-1,
    )


def _rtp_apply_ple(self: Qwen4ExpDecoderLayer, hidden_states, input_ids):
    ple_metadata = get_forward_context().attn_metadata.ple_metadata
    spec_slots = getattr(ple_metadata, "rtp_spec_state_indices", None)
    if spec_slots is None:
        return Qwen4ExpDecoderLayer._apply_ple(self, hidden_states, input_ids)

    # The input block can alias the first output block, so write shorter
    # prefixes in descending order after computing the full verify window.
    batch_size, query_len = spec_slots.shape
    if query_len not in (2, 3, 4):
        raise ValueError("Qwen4Exp RTP MTP requires 2-4 verify tokens")
    contribution = self.ple.forward_with_state(hidden_states, input_ids, ple_metadata)
    hidden_by_request = hidden_states.reshape(batch_size, query_len, -1)
    ids_by_request = input_ids.reshape(batch_size, query_len)
    for prefix_len in range(query_len - 1, 0, -1):
        prefix_metadata = replace(
            ple_metadata,
            query_start_loc=torch.arange(
                0,
                (batch_size + 1) * prefix_len,
                prefix_len,
                device=hidden_states.device,
                dtype=torch.int32,
            ),
            state_indices_out=spec_slots[:, prefix_len - 1],
        )
        self.ple.forward_with_state(
            hidden_by_request[:, :prefix_len]
            .reshape(batch_size * prefix_len, -1)
            .contiguous(),
            ids_by_request[:, :prefix_len].reshape(-1).contiguous(),
            prefix_metadata,
        )
    return hidden_states + contribution


def bind_qwen4_exp_mtp_hooks(model: torch.nn.Module) -> None:
    """Install MTP hooks only on this RTP target, never on shared classes."""
    for layer in model.modules():
        if not isinstance(layer, Qwen4ExpDecoderLayer):
            continue
        if layer.ple is not None:
            layer._apply_ple = MethodType(_rtp_apply_ple, layer)
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn.attn.impl
            if not isinstance(gdn, GatedDeltaNet):
                raise TypeError("Qwen4Exp RTP MTP requires GatedDeltaNet")
            gdn._spec_conv1d_update = MethodType(_rtp_spec_conv1d_update, gdn)
