from typing import Optional

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

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
    kv_cache: torch.Tensor,
    layer_name: str,
    positions: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    qkv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


@mark_spliting_op(
    is_custom=True,
    gen_fake=atom_vllm_mha_attention_fake,
    mutates_args=["kv_cache"],
)
def atom_vllm_mha_attention(
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    kv_cache: torch.Tensor,
    layer_name: str,
    positions: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    qkv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    layer, attn_metadata, _ = _get_layer_context(layer_name)
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
    output: torch.Tensor,
) -> None:
    return None


@eager_break_during_capture
def _mla_attention_run(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    output: torch.Tensor,
) -> None:
    """Run MLA outside the piecewise cudagraph, writing into ``output``.

    Everything this layer reads that varies per batch -- ``attn_metadata`` and
    its block tables, sequence lengths and DCP work descriptors -- is fetched
    here rather than passed in, because the breakable cudagraph records this
    callable once and replays it. An argument would be pinned to the object
    that existed at capture time; a lookup re-reads the live batch. That is
    also why the tensors stay as arguments: those are cudagraph-pool buffers
    whose addresses the surrounding segments depend on.

    Without the break the whole layer is captured, and every replay indexes
    the KV cache with the capture batch's tables -- an illegal access as soon
    as a later batch is shaped differently.

    Full decode graphs are untouched: the decorator hands the call straight
    through when the runtime mode is ``FULL``.
    """
    layer, attn_metadata, kv_cache = _get_layer_context(layer_name)
    layer.forward_impl(
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata=attn_metadata,
        output=output,
    )


@mark_spliting_op(
    is_custom=True,
    gen_fake=atom_vllm_mla_attention_fake,
    mutates_args=["output"],
)
def atom_vllm_mla_attention(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
    output: torch.Tensor,
) -> None:
    """Opaque splitting-op boundary for the MLA attention layer.

    Takes ``output`` from the caller instead of allocating it: a tensor
    allocated in here would land at a fresh address on every replay, while the
    captured segments downstream keep reading the address they saw at capture.
    """
    _mla_attention_run(q, kv_c_normed, k_pe, layer_name, output)
