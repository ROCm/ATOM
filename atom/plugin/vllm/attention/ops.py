from typing import Optional

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
