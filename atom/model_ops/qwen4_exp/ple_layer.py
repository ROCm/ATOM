# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next PLE: n-gram memory, gating and stateful dilated convolution.

Port of `qwen3_8_flash_next/nvidia/ple_layer.py`:
`Qwen3_8FlashNextPLELayer` and `Qwen3_8FlashNextNGramEmbedding`.

Present on exactly one layer (`ple_layer_ids` is 1-based, so `[2]` puts it on
`layers.1`). Its output is added to the wide `[tokens, hc_count * hidden]`
residual before the attention hyper-connection.

    emb    = ngram_embedding(input_ids)            # [T, ple_embed_dim]
    key    = norm_key(key_proj(emb))               # [T, hc, H]
    query  = norm_query(hidden)                    # [T, hc, H]
    gate   = sigmoid(signed_sqrt(<key, query> / sqrt(H)))
    gated  = gate * value_proj(emb)                # [T, hc, H]
    out    = gated + silu(dilated_depthwise_conv(norm_conv(gated)))

N-gram token history and convolution history share backend-owned request slots
across chunked prefill and decode. The depthwise convolution uses kernel size
`ple_conv_kernel_size` and dilation `ngram_size`, retaining
`(kernel - 1) * dilation` previous tokens per channel.
"""

import math

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.qwen4_exp.hyperconnection import Qwen4ExpGroupedRMSNorm
from atom.model_ops.qwen4_exp.ops.ple import (
    advance_ngram_state,
    compute_ngram_ids,
    dilated_causal_conv1d,
    fp8_embedding_lookup,
    ple_gate,
)
from atom.model_ops.utils import atom_parameter

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def is_prime_64(value: int) -> bool:
    """Deterministic Miller-Rabin, exact for every 64-bit input."""
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class _Qwen4ExpFP8Embedding(nn.Module):
    """Vocab-sharded FP8 table with one checkpoint scale for the whole table.

    Only selected rows are dequantized; the table stays in checkpoint E4M3FN
    storage. The TP reduction operates on activation-dtype vectors.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.tp_size = get_tp_group().world_size
        if num_embeddings % self.tp_size:
            raise ValueError("PLE vocabulary size must be divisible by TP size")
        rows = num_embeddings // self.tp_size
        self.vocab_start_idx = rows * get_tp_group().rank_in_group
        self.vocab_end_idx = self.vocab_start_idx + rows
        self.output_dtype = torch.get_default_dtype()
        self.weight = atom_parameter(
            torch.empty(rows, embedding_dim, dtype=torch.float8_e4m3fn)
        )
        self.weight_scale = atom_parameter(
            torch.full((1,), float("nan"), dtype=torch.float32)
        )
        self.weight_scale.weight_loader = self._scale_loader

    @staticmethod
    def _scale_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.numel() != 1:
            raise ValueError("PLE FP8 embedding requires one global weight_scale")
        param.data.copy_(loaded_weight.reshape_as(param))

    def process_weights_after_loading(self) -> None:
        scale = self.weight_scale.item()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"{self.prefix}.weight_scale is missing or is not finite and positive"
            )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        output = fp8_embedding_lookup(
            ids,
            self.weight,
            self.weight_scale,
            self.vocab_start_idx,
            self.vocab_end_idx,
            self.output_dtype,
        )
        if ids.numel() and self.tp_size > 1:
            output = get_tp_group().all_reduce(output, ca_fp8_quant=False)
        return output


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hashed n-gram lookup producing `[tokens, ple_embed_dim]`.

    Each (n-gram order, head) owns a disjoint prime-sized range of the table.
    Hash multipliers and range sizes are derived from the config and layer ID,
    then verified against the checkpoint during loading. They are parameters,
    not buffers, so the weight loader can perform that verification.
    """

    def __init__(
        self,
        config,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str = "",
        quant_config=None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        eos = config.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        # Per-n-gram odd multipliers, derived from (seed, ple layer id).
        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = int(getattr(config, "seed", 1234)) + (
            _PLE_LAYER_PRIME * ple_dense_layer_id
        )
        multipliers = [
            2 * (splitmix64(base_seed + _SPLITMIX_GAMMA * (index + 1)) % half_bound) + 1
            for index in range(self.ngram_size)
        ]
        self.layer_multipliers = self._derived_constant(multipliers)

        # Disjoint prime-sized slice per head.
        base = int(config.ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = nth_prime_after(base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.ngram_heads_vocab_sizes = self._derived_constant(sizes)
        self.ngram_heads_offsets = self._derived_constant(offsets)

        divisor = int(config.make_ngram_vocab_size_divisible_by)
        self.table_rows = ((offset + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        policy = (
            quant_config.get_layer_quant_config(embedding_prefix)
            if quant_config is not None
            else None
        )
        embedding_cls = VocabParallelEmbedding
        if policy is not None and policy.is_quantized:
            if quant_config.quant_method != "fp8":
                raise ValueError("PLE embedding supports BF16 or FP8 checkpoints")
            # PLE uses a global scalar, independently of the linear block scales.
            embedding_cls = _Qwen4ExpFP8Embedding
        self.ngram_embedding = embedding_cls(
            self.table_rows, self.head_dim, prefix=embedding_prefix
        )
        # The 128 checkpoint shards are slices of the PADDED table, so the
        # shard width divides it exactly.
        self.checkpoint_shard_rows = (
            self.table_rows + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        self.ngram_embedding.weight.weight_loader = self._embedding_shard_loader

        self.max_total_tokens = max_total_tokens
        self.max_num_reqs = max_num_reqs

    @staticmethod
    def _derived_constant(values: list[int]) -> nn.Parameter:
        """A derived int64 constant the checkpoint is allowed to confirm.

        Registered as a (gradient-free) parameter rather than a buffer so the
        weight loader sees it and `_verify_derived` runs; nothing here is
        learned.
        """
        tensor = torch.tensor(values, dtype=torch.long)
        param = nn.Parameter(tensor, requires_grad=False)
        param.weight_loader = Qwen4ExpNGramEmbedding._verify_derived
        return param

    @staticmethod
    def _verify_derived(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Fail loudly if our derivation disagrees with the checkpoint."""
        expected = param.data.to(device=loaded_weight.device)
        if expected.shape != loaded_weight.shape or not torch.equal(
            expected, loaded_weight.to(expected.dtype)
        ):
            raise ValueError(
                "Qwen3.8-Flash-Next PLE hash constants do not match the checkpoint: "
                f"derived {expected.tolist()[:4]}..., "
                f"checkpoint {loaded_weight.tolist()[:4]}.... Every n-gram "
                "lookup would read the wrong table row."
            )

    def _embedding_shard_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_index: int = 0
    ) -> None:
        """Copy one checkpoint shard's overlap with this rank's vocab range."""
        embedding = self.ngram_embedding
        if loaded_weight.dtype != param.dtype:
            raise ValueError(
                f"{embedding.prefix}: expected {param.dtype} PLE weights, "
                f"got {loaded_weight.dtype}; check the checkpoint quantization config"
            )
        checkpoint_start = shard_index * self.checkpoint_shard_rows
        expected_rows = min(
            self.checkpoint_shard_rows, self.table_rows - checkpoint_start
        )
        if not 0 <= shard_index < self.split_ngram_parts or loaded_weight.shape != (
            expected_rows,
            self.head_dim,
        ):
            raise ValueError(
                f"invalid n-gram checkpoint shard {shard_index}: {tuple(loaded_weight.shape)}"
            )
        tp_start = embedding.vocab_start_idx
        tp_end = embedding.vocab_end_idx
        overlap_start = max(checkpoint_start, tp_start)
        overlap_end = min(checkpoint_start + loaded_weight.shape[0], tp_end)
        if overlap_start >= overlap_end:
            return
        rows = overlap_end - overlap_start
        source = loaded_weight.narrow(0, overlap_start - checkpoint_start, rows)
        target = param.data.narrow(0, overlap_start - tp_start, rows)
        target.copy_(source)

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Hash tokens using this layer's constants and capacity limits."""
        if (
            input_ids.numel() > self.max_total_tokens
            or query_start_loc.numel() - 1 > self.max_num_reqs
        ):
            raise ValueError("PLE batch exceeds configured token/request capacity")
        return compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            self.ngram_size,
            self.heads_per_ngram,
            self.eos_token_id,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        # Look the heads up as one flat batch: the TP path returns `[N, dim]`
        # for an N-element index whatever its shape, so a 2D index would come
        # back already flattened on one path and 3D on the other.
        rows = self.ngram_embedding(ngram_ids.reshape(-1))
        return rows.reshape(ngram_ids.shape[0], self.embedding_dim)


class Qwen4ExpPLELayer(nn.Module):
    def __init__(
        self,
        config,
        max_total_tokens: int,
        max_num_reqs: int,
        ple_dense_layer_id: int = 0,
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

        ple_embed_dim = int(getattr(config, "ple_embed_dim", None) or self.hidden_size)
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            ple_embed_dim,
            ple_dense_layer_id,
            max_total_tokens,
            max_num_reqs,
            prefix=f"{prefix}.ple_embedding",
            quant_config=quant_config,
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
        norm_args = {
            "hidden_size": self.hc_hidden_size,
            "eps": config.rms_norm_eps,
            "group_size": self.hidden_size,
        }
        self.norm_key = Qwen4ExpGroupedRMSNorm(**norm_args)
        self.norm_query = Qwen4ExpGroupedRMSNorm(**norm_args)
        self.norm_conv = Qwen4ExpGroupedRMSNorm(**norm_args)
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

    def _apply_norm(
        self, norm: Qwen4ExpGroupedRMSNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv(
        self,
        inputs: torch.Tensor,
        query_start_loc: torch.Tensor,
        state: torch.Tensor,
        state_indices_in: torch.Tensor,
        state_indices_out: torch.Tensor,
        has_initial_state: torch.Tensor,
    ) -> torch.Tensor:
        """One flat varlen path for prefill, decode and graph padding."""
        return dilated_causal_conv1d(
            inputs,
            self.conv1d.weight.squeeze(1),
            state,
            query_start_loc,
            state_indices_in,
            state_indices_out,
            has_initial_state,
            self.short_conv_dilation,
        )

    def gated_memory(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Everything up to the convolution: `[T, hc*H]` gated memory read."""
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same token "
                f"length, got {input_ids.shape[0]} and {hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
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
        return ple_gate(key, query, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Whole-sequence path for profiling before the state pool exists."""
        gated_value = self.gated_memory(
            hidden_states, input_ids, query_start_loc, ngram_context
        )
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        requests = query_start_loc.numel() - 1
        slots = torch.arange(requests, device=hidden_states.device, dtype=torch.int32)
        state = normalized.new_zeros(
            (requests, self.hc_hidden_size, self.conv_state_len)
        )
        conv_output = self._short_conv(
            normalized,
            query_start_loc,
            state,
            slots,
            slots,
            torch.zeros_like(slots, dtype=torch.bool),
        )
        return gated_value.flatten(-2) + conv_output

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        metadata,
    ) -> torch.Tensor:
        """PLE contribution with persistent n-gram and convolution windows."""
        input_ids = input_ids.reshape(-1)
        ngram_context = advance_ngram_state(
            input_ids,
            metadata.query_start_loc,
            metadata.ngram_state,
            metadata.state_indices_in,
            metadata.state_indices_out,
            metadata.has_initial_state,
            self.ple_embedding.eos_token_id,
        )
        gated_value = self.gated_memory(
            hidden_states, input_ids, metadata.query_start_loc, ngram_context
        )
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = self._short_conv(
            normalized,
            metadata.query_start_loc,
            metadata.conv_state,
            metadata.state_indices_in,
            metadata.state_indices_out,
            metadata.has_initial_state,
        )
        return gated_value.flatten(-2) + conv_output
