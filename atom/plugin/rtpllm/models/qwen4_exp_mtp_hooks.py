"""RTP-only Qwen4Exp MTP state updates, bound to the target model instance."""

from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl

from atom.model_ops.attention_gdn import GatedDeltaNet
from atom.model_ops.qwen4_exp.ops.ple import advance_ngram_state
from atom.models.qwen4_exp import Qwen4ExpDecoderLayer
from atom.utils.forward_context import get_forward_context


@triton.jit
def _write_ple_ngram_candidates(
    Ids,
    Context,
    State,
    Slots,
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_COL_STRIDE: tl.constexpr,
    SLOT_ROW_STRIDE: tl.constexpr,
    SLOT_COL_STRIDE: tl.constexpr,
    QUERY: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    for reverse in tl.static_range(1, QUERY):
        prefix = QUERY - reverse
        dst = tl.load(Slots + req * SLOT_ROW_STRIDE + (prefix - 1) * SLOT_COL_STRIDE)
        position = prefix - WIDTH + j
        old = tl.load(
            Context + req * WIDTH + WIDTH + position, (j < WIDTH) & (position < 0), 0
        )
        token = tl.load(Ids + req * QUERY + position, (j < WIDTH) & (position >= 0), 0)
        tl.store(
            State + dst * STATE_SLOT_STRIDE + j * STATE_COL_STRIDE,
            tl.where(position >= 0, token, old),
            (dst >= 0) & (j < WIDTH),
        )


@triton.jit
def _write_ple_conv_candidates(
    Inputs,
    State,
    In,
    Has,
    Slots,
    CHANNELS: tl.constexpr,
    QUERY: tl.constexpr,
    HISTORY: tl.constexpr,
    STATE_SLOT_STRIDE: tl.constexpr,
    STATE_CHANNEL_STRIDE: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    SLOT_ROW_STRIDE: tl.constexpr,
    SLOT_COL_STRIDE: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
):
    req = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    j = tl.arange(0, BLOCK_HISTORY)
    src = tl.load(In + req * IN_STRIDE)
    has = tl.load(Has + req) & (src >= 0)
    for reverse in tl.static_range(1, QUERY):
        prefix = QUERY - reverse
        dst = tl.load(Slots + req * SLOT_ROW_STRIDE + (prefix - 1) * SLOT_COL_STRIDE)
        position = prefix - HISTORY + j[None, :]
        valid = (dst >= 0) & (c[:, None] < CHANNELS) & (j[None, :] < HISTORY)
        old = tl.load(
            State
            + src * STATE_SLOT_STRIDE
            + c[:, None] * STATE_CHANNEL_STRIDE
            + HISTORY
            + position,
            valid & has & (position < 0),
            0,
        )
        token = tl.load(
            Inputs + (req * QUERY + position) * CHANNELS + c[:, None],
            valid & (position >= 0),
            0,
        )
        tl.store(
            State
            + dst * STATE_SLOT_STRIDE
            + c[:, None] * STATE_CHANNEL_STRIDE
            + j[None, :],
            tl.where(position >= 0, token, old),
            valid,
        )


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
        mixed_qkv_spec.reshape(batch_size, query_len, mixed_width).transpose(1, 2),
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

    # The input block can alias the first output block. Compute the full
    # contribution once, then write shorter snapshots in descending order so
    # the source history is not overwritten until the final candidate.
    batch_size, query_len = spec_slots.shape
    if query_len not in (2, 3, 4):
        raise ValueError("Qwen4Exp RTP MTP requires 2-4 verify tokens")
    if ple_metadata.num_accepted_tokens is not None:
        raise ValueError("Qwen4Exp RTP MTP verify cannot commit accepted tokens early")
    if hidden_states.shape[0] != batch_size * query_len:
        raise ValueError("Qwen4Exp RTP MTP PLE token count does not match slots")
    ple = self.ple
    input_ids = input_ids.reshape(-1).contiguous()
    width = ple.ple_embedding.ngram_size - 1
    ngram_context = advance_ngram_state(
        input_ids,
        ple_metadata.query_start_loc,
        ple_metadata.ngram_state,
        ple_metadata.state_indices_in,
        ple_metadata.state_indices_out,
        ple_metadata.has_initial_state,
        ple.ple_embedding.eos_token_id,
        history_width=width,
        num_accepted_tokens=ple_metadata.num_accepted_tokens,
    )
    gated = ple.gated_memory(
        hidden_states, input_ids, ple_metadata.query_start_loc, ngram_context
    )
    normalized = ple._apply_norm(ple.norm_conv, gated).flatten(-2).contiguous()
    conv_output = ple._short_conv(
        normalized,
        ple_metadata.query_start_loc,
        ple_metadata.conv_state,
        ple_metadata.state_indices_in,
        ple_metadata.state_indices_out,
        ple_metadata.has_initial_state,
        ple_metadata.num_accepted_tokens,
    )
    if batch_size:
        _write_ple_ngram_candidates[(batch_size,)](
            input_ids,
            ngram_context,
            ple_metadata.ngram_state,
            spec_slots,
            ple_metadata.ngram_state.stride(0),
            ple_metadata.ngram_state.stride(1),
            spec_slots.stride(0),
            spec_slots.stride(1),
            query_len,
            width,
            triton.next_power_of_2(width),
            num_warps=1,
        )
        channels = normalized.shape[1]
        history = ple.conv_state_len
        _write_ple_conv_candidates[(batch_size, triton.cdiv(channels, 32))](
            normalized,
            ple_metadata.conv_state,
            ple_metadata.state_indices_in,
            ple_metadata.has_initial_state,
            spec_slots,
            channels,
            query_len,
            history,
            ple_metadata.conv_state.stride(0),
            ple_metadata.conv_state.stride(1),
            ple_metadata.state_indices_in.stride(0),
            spec_slots.stride(0),
            spec_slots.stride(1),
            32,
            triton.next_power_of_2(history),
        )
    return hidden_states + gated.flatten(-2) + conv_output


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
