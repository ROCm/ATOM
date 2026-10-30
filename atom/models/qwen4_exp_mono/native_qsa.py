# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native QSA operators used at Mono's projection and attention boundaries.

The native model keeps its original forward API. This local seam publishes
index caches once when Mono replaces K1, and preserves Native BF16/PTPC endpoints.
"""

from dataclasses import dataclass

import torch

from atom.model_ops.qwen4_exp.ops.gated import (
    is_per_token_fp8_linear,
    sigmoid_mul,
    sigmoid_mul_quant,
)
from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_apply_mrope,
    qsa_compress_groups,
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
    qsa_store_quantized_rows,
    qsa_store_rows,
)
from atom.model_ops.qwen4_exp.qsa_attention import canonical_rope_positions
from atom.models.qwen4_exp_mono.kernels.batch_qsa_epilogue import (
    qsa_norm_rope_cache,
)
from atom.utils.forward_context import get_forward_context


@dataclass(frozen=True)
class PreparedQsa:
    """Mono-owned K1 -> native attention seam for one QSA layer.

    The native path keeps this as tensor views.  Qwen MonoKernel owns the same
    fields in graph-stable buffers, allowing K1 to move into FlyDSL without
    repeating projections or cache writes in the temporary native K2 path.
    """

    gate: torch.Tensor
    query: torch.Tensor
    qsa: object | None
    index_query: torch.Tensor | None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    positions: torch.Tensor | None = None


def _prepare_index(
    module,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    qsa,
) -> torch.Tensor:
    """Project the index plane and publish its raw/compressed cache rows."""
    index_q, raw_key = module.indexer.project_qk(hidden_states, positions)
    qsa_store_rows(module.raw_key_cache, qsa.slot_mapping, raw_key)
    if module.rope_position_cache is not None:
        qsa_store_rows(
            module.rope_position_cache,
            qsa.slot_mapping,
            canonical_rope_positions(positions[..., : hidden_states.shape[0]]),
        )

    pooled, first_positions = qsa_compress_groups(
        module.raw_key_cache,
        qsa.block_tables,
        qsa.token_to_req,
        qsa.logical_positions,
        qsa.compressed_slot_mapping,
        module.indexer.compress_ratio,
        position_cache=module.rope_position_cache,
        # Draft RoPE is one ahead of its logical cache slots.
        rope_position_offset=int(get_forward_context().context.is_draft),
    )
    normalized = module.indexer.normalize_compressed_keys(pooled, first_positions)
    if module.compressed_key_scale is None:
        qsa_store_rows(
            module.compressed_key_cache, qsa.compressed_slot_mapping, normalized
        )
    else:
        qsa_store_quantized_rows(
            module.compressed_key_cache,
            module.compressed_key_scale,
            qsa.compressed_slot_mapping,
            normalized,
        )

    return index_q


def _select_prepared(module, index_q: torch.Tensor, qsa) -> torch.Tensor:
    """Start QSA K2 from the query and caches published by K1."""

    return qsa_select_paged_tokens(
        index_q,
        module.compressed_key_cache,
        qsa.block_tables,
        qsa.token_to_req,
        qsa.logical_positions,
        qsa.seq_lens,
        module.indexer.token_topk,
        module.indexer.compress_ratio,
        max_seq_len=qsa.max_seq_len,
        allow_flydsl=bool(getattr(get_forward_context().context, "is_prefill", False)),
        compressed_k_scale=module.compressed_key_scale,
    )


def prepare(
    module,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    *,
    fuse_decode_index: bool = False,
    index_query_length: int = 1,
    projected_qkv: torch.Tensor | None = None,
    projected_index_qk: torch.Tensor | None = None,
    quantized: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> PreparedQsa:
    """Prepare QSA projections and fuse norm/RoPE/cache when metadata is present."""

    num_tokens = hidden_states.shape[0]
    if projected_qkv is not None:
        qkv = projected_qkv
    elif quantized is not None:
        qkv = module.qkv_proj(quantized[0], x_scale=quantized[1])
    else:
        qkv = module.qkv_proj(hidden_states)
    gate, q, k, v = qkv.split(
        [module.q_size, module.q_size, module.kv_size, module.kv_size], -1
    )
    forward_context = get_forward_context()
    qsa = forward_context.attn_metadata.qsa_metadata
    if qsa is not None:
        qsa = qsa.for_tokens(num_tokens)
    value = v.view(num_tokens, module.num_kv_heads, module.head_dim)
    if qsa is not None:
        query = qsa_norm_rope_cache(
            q.view(num_tokens, module.num_heads, module.head_dim),
            k.view(num_tokens, module.num_kv_heads, module.head_dim),
            value,
            module.qk_norm,
            module.rotary_emb,
            positions,
            module.k_cache,
            module.v_cache,
            qsa.slot_mapping,
            module.k_scale_cache,
            module.v_scale_cache,
        )
    else:
        q, k = module.qk_norm(q, k)
        query, key = qsa_apply_mrope(
            module.rotary_emb,
            positions,
            q.view(num_tokens, module.num_heads, module.head_dim),
            k.view(num_tokens, module.num_kv_heads, module.head_dim),
        )
        return PreparedQsa(
            gate, query, None, None, key, value, hidden_states, positions
        )
    if fuse_decode_index:
        from atom.models.qwen4_exp_mono.kernels.quark_qsa_index import prepare

        index_query = prepare(
            module,
            hidden_states,
            positions,
            qsa,
            projected=projected_index_qk,
            query_length=index_query_length,
        )
    else:
        index_query = _prepare_index(module, hidden_states, positions, qsa)
    return PreparedQsa(gate, query, qsa, index_query)


def attend(
    module,
    prepared: PreparedQsa,
    selected: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run QSA selection and attention from the prepared K1 seam."""

    if prepared.qsa is None:
        if any(
            value is None
            for value in (
                prepared.key,
                prepared.value,
                prepared.hidden_states,
                prepared.positions,
            )
        ):
            raise RuntimeError("profile QSA seam is missing K1 tensors")
        attn_out = module._profile_attention(
            prepared.query,
            prepared.key,
            prepared.value,
            prepared.hidden_states,
            prepared.positions,
        )
    else:
        if prepared.index_query is None:
            raise RuntimeError("decode QSA seam is missing the index query")
        if selected is None:
            selected = _select_prepared(module, prepared.index_query, prepared.qsa)
        forward_context = get_forward_context()
        attn_out = qsa_sparse_paged_gqa(
            prepared.query,
            module.k_cache,
            module.v_cache,
            selected,
            prepared.qsa.block_tables,
            prepared.qsa.token_to_req,
            softmax_scale=module.scaling,
            num_decode_requests=(
                None
                if forward_context.context.is_prefill
                else prepared.qsa.block_tables.shape[0]
            ),
            k_scale=module.k_scale_cache,
            v_scale=module.v_scale_cache,
        )

    return attn_out


def finish(
    module,
    prepared: PreparedQsa,
    selected: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the native output gate and projection to QSA attention."""
    attn_out = attend(module, prepared, selected)
    if is_per_token_fp8_linear(module.o_proj):
        gated, scale = sigmoid_mul_quant(
            attn_out.reshape(prepared.gate.shape[0], -1),
            prepared.gate,
            module.o_proj.params_dtype,
        )
        return module.o_proj(gated, x_scale=scale)
    gated = sigmoid_mul(attn_out.reshape(prepared.gate.shape[0], -1), prepared.gate)
    return module.o_proj(gated)
