"""Qwen3.8-Flash-Next PLE layer: an n-gram memory read, gated against the hidden state.

Port of `qwen3_8_flash_next/nvidia/ple_layer.py:Qwen3_8FlashNextPLELayer`.

Present on exactly one layer (`ple_layer_ids` is 1-based, so `[2]` puts it on
`layers.1`). Its output is added to the wide `[tokens, hc_count * hidden]`
residual at the very top of the block, before the attention hyper-connection,
so an error here propagates through every layer above it.

    emb    = ngram_embedding(input_ids)            # [T, ple_embed_dim]
    key    = norm_key(key_proj(emb))               # [T, hc, H]
    query  = norm_query(hidden)                    # [T, hc, H]
    gate   = sigmoid(signed_sqrt(<key, query> / sqrt(H)))
    gated  = gate * value_proj(emb)                # [T, hc, H]
    out    = gated + silu(dilated_depthwise_conv(norm_conv(gated)))

The convolution is depthwise over all `hc_count * hidden` channels with kernel
`ple_conv_kernel_size` and dilation `ngram_size`, i.e. it reaches back
`(kernel - 1) * dilation` tokens. That window has to survive chunked prefill
and decode, so it carries a per-request conv state exactly like a Mamba short
convolution.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.qwen3_8_flash_next.ngram import Qwen3_8FlashNextNGramEmbedding
from atom.model_ops.qwen3_8_flash_next.norm import Qwen3_8FlashNextGroupedGemmaRMSNorm


class Qwen3_8FlashNextPLELayer(nn.Module):
    def __init__(
        self,
        config,
        max_total_tokens: int,
        max_num_reqs: int,
        ple_dense_layer_id: int = 0,
        num_spec_tokens: int = 0,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        # How many past tokens the dilated kernel reaches back over.
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = num_spec_tokens

        ple_embed_dim = int(getattr(config, "ple_embed_dim", None) or self.hidden_size)
        self.ple_embedding = Qwen3_8FlashNextNGramEmbedding(
            config,
            ple_embed_dim,
            ple_dense_layer_id,
            max_total_tokens,
            max_num_reqs,
            prefix=f"{prefix}.ple_embedding",
        )
        self.key_proj = ReplicatedLinear(
            ple_embed_dim,
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            ple_embed_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (self.hc_hidden_size, config.rms_norm_eps, self.hidden_size)
        self.norm_key = Qwen3_8FlashNextGroupedGemmaRMSNorm(*norm_args)
        self.norm_query = Qwen3_8FlashNextGroupedGemmaRMSNorm(*norm_args)
        self.norm_conv = Qwen3_8FlashNextGroupedGemmaRMSNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)

    @property
    def state_shape(self) -> tuple[int, int]:
        """Per-request conv state: `(state_len + spec slack, channels)`."""
        return (self.conv_state_len + self.num_spec_tokens, self.hc_hidden_size)

    def _apply_norm(
        self, norm: Qwen3_8FlashNextGroupedGemmaRMSNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def short_conv_whole_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        """Causal dilated depthwise conv over a complete sequence, then SiLU.

        Correct only when every token of the request is present in `inputs`
        (single-chunk prefill, and the reference's own profiling path). The
        engine-facing path is `short_conv_with_state`.
        """
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = self.conv1d(inputs_t)[..., : inputs_t.size(-1)]
        return F.silu(output).squeeze(0).transpose(0, 1)

    def short_conv_with_state(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decode step per row, carrying the dilated window in `state`.

        `state` is `[rows, channels, conv_state_len]` holding the previous
        `conv_state_len` inputs; rows without history are zeroed so the
        convolution sees left padding rather than stale tokens.

        Returns the activations and the updated state.
        """
        weights = self.conv1d.weight.squeeze(1).to(inputs.dtype)
        history = inputs.unsqueeze(-1)
        if self.conv_state_len > 0:
            initial = torch.where(
                has_initial_state.view(-1, 1, 1),
                state.to(inputs.dtype),
                torch.zeros_like(state, dtype=inputs.dtype),
            )
            history = torch.cat((initial, history), dim=-1)
        output = F.conv1d(
            history,
            weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        ).squeeze(-1)
        next_state = (
            history[..., -self.conv_state_len :] if self.conv_state_len > 0 else state
        )
        return F.silu(output), next_state

    def short_conv_prefill(
        self,
        inputs: torch.Tensor,
        query_start_loc: torch.Tensor,
        max_query_len: int,
        state: torch.Tensor,
        state_indices_in: torch.Tensor,
        state_indices_out: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """Chunked-prefill convolution: pack ragged requests, then carry state.

        `inputs` is the flat `[tokens, channels]` batch, `query_start_loc` the
        usual `[reqs + 1]` offsets. Requests are padded into a dense
        `[reqs, max_query_len, channels]` box so one grouped conv covers the
        batch (the width comes from the host so no device sync is needed);
        the `conv_state_len` inputs preceding each request are prepended from
        `state` (zeroed where the request has no history), and the last
        `conv_state_len` inputs of each request are written back.

        Port of the reference's `_short_conv_dilated_prefill_batched`.
        """
        num_reqs = int(state_indices_out.shape[0])
        channels = inputs.shape[1]
        starts = query_start_loc[: num_reqs + 1].to(torch.int64)
        lengths = starts[1:] - starts[:-1]
        num_tokens = inputs.shape[0]
        if num_reqs == 0 or num_tokens == 0:
            return torch.empty_like(inputs)

        positions = torch.arange(num_tokens, device=inputs.device, dtype=torch.int64)
        req_indices = torch.searchsorted(starts[1:], positions, right=True)
        col_indices = positions - starts[req_indices]

        packed = inputs.new_zeros((num_reqs, max_query_len, channels))
        packed[req_indices, col_indices] = inputs
        packed = packed.transpose(1, 2).contiguous()

        indices_in = state_indices_in.to(device=state.device, dtype=torch.int64)
        indices_out = state_indices_out.to(device=state.device, dtype=torch.int64)
        valid = indices_out >= 0
        safe_in = torch.where(indices_in >= 0, indices_in, torch.zeros_like(indices_in))
        safe_out = torch.where(valid, indices_out, torch.zeros_like(indices_out))

        if self.conv_state_len > 0:
            previous = state.index_select(0, safe_in)[
                ..., : self.conv_state_len
            ].to(inputs.dtype)
            keep = (valid & has_initial_state.to(state.device)).view(num_reqs, 1, 1)
            history = torch.cat(
                (torch.where(keep, previous, torch.zeros_like(previous)), packed),
                dim=-1,
            )
        else:
            history = packed

        conv_out = F.conv1d(
            history,
            self.conv1d.weight.squeeze(1).unsqueeze(1).to(inputs.dtype).contiguous(),
            groups=channels,
            dilation=self.short_conv_dilation,
        )
        conv_out = F.silu(conv_out).transpose(1, 2).contiguous()
        output = conv_out[req_indices, col_indices]

        if self.conv_state_len > 0:
            offsets = torch.arange(
                self.conv_state_len, device=history.device, dtype=torch.int64
            ).view(1, 1, -1)
            gather = (lengths.view(num_reqs, 1, 1) + offsets).expand(
                -1, channels, -1
            )
            next_state = history.gather(dim=2, index=gather)
            existing = state.index_select(0, safe_out)
            update = (valid & (lengths.to(state.device) > 0)).view(num_reqs, 1, 1)
            existing[..., : self.conv_state_len] = torch.where(
                update, next_state.to(state.dtype), existing[..., : self.conv_state_len]
            )
            state.index_copy_(0, safe_out, existing)
        return output

    def short_conv_decode(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        state_indices_in: torch.Tensor,
        state_indices_out: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """One token per request, carrying the dilated window through `state`."""
        num_rows = inputs.shape[0]
        indices_in = state_indices_in.to(device=state.device, dtype=torch.int64)
        indices_out = state_indices_out.to(device=state.device, dtype=torch.int64)
        valid = indices_out >= 0
        safe_in = torch.where(indices_in >= 0, indices_in, torch.zeros_like(indices_in))
        safe_out = torch.where(valid, indices_out, torch.zeros_like(indices_out))

        history = inputs.unsqueeze(-1)
        if self.conv_state_len > 0:
            previous = state.index_select(0, safe_in)[..., : self.conv_state_len].to(
                inputs.dtype
            )
            keep = (valid & has_initial_state.to(state.device)).view(num_rows, 1, 1)
            history = torch.cat(
                (torch.where(keep, previous, torch.zeros_like(previous)), history),
                dim=-1,
            )

        output = F.silu(
            F.conv1d(
                history,
                self.conv1d.weight.squeeze(1)
                .unsqueeze(1)
                .to(inputs.dtype)
                .contiguous(),
                groups=history.size(1),
                dilation=self.short_conv_dilation,
            ).squeeze(-1)
        )
        output = output * valid.view(-1, 1).to(output.dtype)

        if self.conv_state_len > 0:
            existing = state.index_select(0, safe_out)
            existing[..., : self.conv_state_len] = torch.where(
                valid.view(num_rows, 1, 1),
                history[..., -self.conv_state_len :].to(state.dtype),
                existing[..., : self.conv_state_len],
            )
            state.index_copy_(0, safe_out, existing)
        return output

    def gated_memory(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        max_query_len: int | None = None,
    ) -> torch.Tensor:
        """Everything up to the convolution: `[T, hc*H]` gated memory read."""
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same token "
                f"length, got {input_ids.shape[0]} and {hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(
            input_ids, query_start_loc, ngram_context, max_query_len
        )
        token_count = hidden_states.shape[0]
        # ATOM's Linear defaults its output to bf16; pass the working dtype so
        # the projections never silently downcast (a no-op in bf16 serving).
        otype = embeddings.dtype
        key = self.key_proj(embeddings, otype=otype).reshape(
            token_count, self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings, otype=otype)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        # Signed square root before the sigmoid: squashes large scores without
        # losing their sign. clamp_min keeps the gradient-free sqrt finite at 0.
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        return gate * value.unsqueeze(-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        conv_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """PLE contribution to add to the wide residual, `[T, hc*H]`.

        `conv_output` lets the engine supply a state-aware convolution result;
        when omitted the whole-sequence path runs, which is only valid if every
        request in the batch is present in full.
        """
        gated_value = self.gated_memory(
            hidden_states, input_ids, query_start_loc, ngram_context
        )
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        if conv_output is None:
            conv_output = self.short_conv_whole_sequence(normalized)
        return gated_value.flatten(-2) + conv_output

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        metadata,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """Engine path: PLE contribution with a per-request convolution window.

        `metadata` is the builder's `Qwen3_8FlashNextPLEMetadata`. Prefill and decode
        take different conv paths only because their token layouts differ --
        ragged runs per request versus one row each.
        """
        gated_value = self.gated_memory(
            hidden_states,
            input_ids,
            metadata.query_start_loc,
            metadata.ngram_context,
            metadata.max_query_len,
        )
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        if metadata.is_prefill:
            conv_output = self.short_conv_prefill(
                normalized,
                metadata.query_start_loc,
                metadata.max_query_len,
                conv_state,
                metadata.state_indices_in,
                metadata.state_indices_out,
                metadata.has_initial_state,
            )
        else:
            conv_output = self.short_conv_decode(
                normalized,
                conv_state,
                metadata.state_indices_in,
                metadata.state_indices_out,
                metadata.has_initial_state,
            )
        return gated_value.flatten(-2) + conv_output
