from typing import Optional

import aiter
import torch

from atom.utils import mark_spliting_op


def _get_layer_context(layer_name: str):
    from vllm.forward_context import get_forward_context

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata.get(layer_name)
    layer = forward_context.no_compile_layers[layer_name]
    return layer, attn_metadata, layer.kv_cache


def atom_vllm_mha_attention_fake(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    layer_name: str,
    positions: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    qkv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


@mark_spliting_op(
    is_custom=True,
    gen_fake=atom_vllm_mha_attention_fake,
    mutates_args=[],
)
def atom_vllm_mha_attention(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    layer_name: str,
    positions: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    qkv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    layer, attn_metadata, kv_cache = _get_layer_context(layer_name)
    return layer.forward_impl(
        query,
        key,
        value,
        kv_cache,
        attn_metadata=attn_metadata,
        position=positions,
        q_scale=q_scale,
        qkv=qkv,
    )


def atom_vllm_mla_attention_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    output_hidden_size: int,
) -> torch.Tensor:
    return q.new_empty((q.shape[0], output_hidden_size))


@mark_spliting_op(
    is_custom=True,
    gen_fake=atom_vllm_mla_attention_fake,
    mutates_args=[],
)
def atom_vllm_mla_attention(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    output_hidden_size: int,
) -> torch.Tensor:
    layer, attn_metadata, kv_cache = _get_layer_context(layer_name)
    output = torch.empty(
        (q.shape[0], output_hidden_size),
        dtype=q.dtype,
        device=q.device,
    )
    layer.forward_impl(
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )
    return output


def minimax_m3_sparse_attention_preproc_fake(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    num_index_heads: int,
    kv_cache: torch.Tensor,
    index_cache: torch.Tensor,
    q_out: torch.Tensor,
    index_q_out: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
) -> None:
    del (
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        kv_cache,
        index_cache,
        q_out,
        index_q_out,
        layer_name,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    return None


@mark_spliting_op(
    is_custom=True,
    gen_fake=minimax_m3_sparse_attention_preproc_fake,
    mutates_args=["qkv", "kv_cache", "index_cache", "q_out", "index_q_out"],
)
def minimax_m3_sparse_attention_preproc(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    num_index_heads: int,
    kv_cache: torch.Tensor,
    index_cache: torch.Tensor,
    q_out: torch.Tensor,
    index_q_out: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
) -> None:
    from vllm.forward_context import get_forward_context

    slot_mapping = None
    index_slot_mapping = None
    kv_cache_input = None
    index_cache_input = None
    block_size = 0
    layer, main_meta, _ = _get_layer_context(layer_name)
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        index_meta = attn_metadata.get(layer.index_cache.prefix)
        if kv_cache.numel() > 0 and index_cache.numel() > 0:
            slot_mapping = main_meta.slot_mapping
            index_slot_mapping = index_meta.slot_mapping
            kv_cache_input = kv_cache
            index_cache_input = index_cache
            block_size = kv_cache.shape[2]

    aiter.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        kv_cache_input,
        index_cache_input,
        block_size,
        q_out,
        index_q_out,
        index_slot_mapping,
        kv_cache_dtype,
        k_scale if kv_cache_input is not None else None,
        v_scale if kv_cache_input is not None else None,
    )
    return None


def minimax_m3_sparse_attention_insert_kv_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    index_key: torch.Tensor,
    kv_cache: torch.Tensor,
    index_kv_cache: torch.Tensor,
    layer_name: str,
) -> None:
    del key, value, index_key, kv_cache, index_kv_cache, layer_name
    return None


@mark_spliting_op(
    is_custom=True,
    gen_fake=minimax_m3_sparse_attention_insert_kv_fake,
    mutates_args=["kv_cache", "index_kv_cache"],
)
def minimax_m3_sparse_attention_insert_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    index_key: torch.Tensor,
    kv_cache: torch.Tensor,
    index_kv_cache: torch.Tensor,
    layer_name: str,
) -> None:
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None
    layer, main_meta, _ = _get_layer_context(layer_name)
    index_meta = attn_metadata.get(layer.index_cache.prefix)
    layer._insert_kv(
        key,
        value,
        index_key,
        kv_cache,
        index_kv_cache,
        main_meta,
        index_meta,
    )
    return None


def minimax_m3_sparse_attention_fake(
    query: torch.Tensor,
    index_query: torch.Tensor,
    kv_cache: torch.Tensor,
    index_kv_cache: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del index_query, kv_cache, index_kv_cache, layer_name
    return torch.empty_like(query).contiguous()


@mark_spliting_op(
    is_custom=True,
    gen_fake=minimax_m3_sparse_attention_fake,
    mutates_args=["kv_cache", "index_kv_cache"],
)
def minimax_m3_sparse_attention(
    query: torch.Tensor,
    index_query: torch.Tensor,
    kv_cache: torch.Tensor,
    index_kv_cache: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    from vllm.forward_context import get_forward_context

    layer, main_meta, _ = _get_layer_context(layer_name)
    attn_metadata = get_forward_context().attn_metadata
    index_meta = None
    if isinstance(attn_metadata, dict):
        index_meta = attn_metadata.get(layer.index_cache.prefix)
    return layer.minimax_m3_sparse_attention_forward(
        query,
        index_query,
        kv_cache,
        index_kv_cache,
        main_meta,
        index_meta,
    )
