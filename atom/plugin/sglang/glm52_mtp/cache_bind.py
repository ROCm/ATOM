"""Bind SGLang KV pool views to ATOM GLM-5.2 sparse MLA modules."""

from __future__ import annotations

import torch

from atom.plugin.sglang.glm52_mtp.common import (
    EMPTY_VALUE_CACHE_ATTR,
    INDEXER_PAGE_SIZE_ATTR,
    SHARED_SPARSE_INDICES_ATTR,
)


def bind_glm52_dsa_cache_views(model, token_to_kv_pool) -> bool:
    if token_to_kv_pool is None or not hasattr(token_to_kv_pool, "get_key_buffer"):
        return False
    if not hasattr(token_to_kv_pool, "get_index_k_with_scale_buffer"):
        return False

    from atom.config import KVCacheTensor
    from atom.models.deepseek_v2 import DeepseekV2MLAAttention
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    shared_sparse = getattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, None)
    if shared_sparse is None:
        return False

    page_size = int(
        getattr(
            token_to_kv_pool,
            INDEXER_PAGE_SIZE_ATTR,
            getattr(token_to_kv_pool, "page_size", 1),
        )
    )
    empty_value_cache = getattr(token_to_kv_pool, EMPTY_VALUE_CACHE_ATTR, None)
    if empty_value_cache is None or empty_value_cache.device != shared_sparse.device:
        empty_value_cache = torch.empty(0, device=shared_sparse.device)
        setattr(token_to_kv_pool, EMPTY_VALUE_CACHE_ATTR, empty_value_cache)
    kv_cache_data = {}
    for module in model.modules():
        if not isinstance(module, DeepseekV2MLAAttention):
            continue

        layer_id = int(module.layer_num)
        mla_attn = module.mla_attn
        kv_cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=token_to_kv_pool.get_key_buffer(layer_id),
            v_cache=empty_value_cache,
            k_scale=getattr(mla_attn, "_k_scale", None),
            v_scale=getattr(mla_attn, "_k_scale", None),
        )

        indexer = getattr(module, "indexer", None)
        if indexer is not None:
            index_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
            index_entry_dim = int(getattr(indexer, "head_dim")) + 4
            indexer.k_cache.kv_cache[0] = index_cache.view(
                -1, page_size, index_entry_dim
            )
            indexer.sparse_kv_indices_buffer = shared_sparse

        if hasattr(mla_attn, "sparse_kv_indices_buffer"):
            mla_attn.sparse_kv_indices_buffer = shared_sparse

    if not kv_cache_data:
        return False

    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data
    return True
