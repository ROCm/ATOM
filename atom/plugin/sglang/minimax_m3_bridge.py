"""Bridge SGLang ForwardBatch metadata to ATOM MiniMax-M3 sparse attention."""

from __future__ import annotations

from typing import Any

import torch

from atom.model_ops.minimax_m3.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    make_sparse_decode_metadata,
    make_sparse_prefill_metadata,
)


def is_minimax_m3_config(config: Any) -> bool:
    archs = getattr(config, "architectures", None) or []
    model_type = str(getattr(config, "model_type", "")).lower()
    return any("MiniMaxM3" in str(arch) for arch in archs) or "minimax_m3" in model_type


def maybe_get_minimax_m3_pools_from_sglang_batch(forward_batch=None):
    from atom.plugin.sglang.runtime.attention_backend_resolver import (
        resolve_sglang_runtime,
    )

    try:
        runtime = resolve_sglang_runtime(forward_batch)
    except RuntimeError:
        return None, None
    return runtime.token_to_kv_pool, runtime.req_to_token_pool


def _page_size(token_to_kv_pool) -> int:
    page_size = int(getattr(token_to_kv_pool, "page_size", 1))
    if page_size != SPARSE_BLOCK_SIZE:
        raise ValueError(
            "MiniMax-M3 native sparse attention requires SGLang page size "
            f"{SPARSE_BLOCK_SIZE}, got {page_size}. Launch with --page-size "
            f"{SPARSE_BLOCK_SIZE}."
        )
    return page_size


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _seq_lens(forward_batch, bs: int) -> torch.Tensor:
    return forward_batch.seq_lens[:bs].to(dtype=torch.int32)


def _extend_lens(forward_batch, positions: torch.Tensor, bs: int) -> torch.Tensor:
    extend_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens is not None:
        return extend_lens[:bs].to(device=positions.device, dtype=torch.int32)

    extend_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens_cpu is not None:
        return torch.as_tensor(
            extend_lens_cpu[:bs], dtype=torch.int32, device=positions.device
        )

    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is None:
        tokens_per_req = max(1, int(positions.numel()) // max(1, bs))
    return torch.full(
        (bs,), int(tokens_per_req), dtype=torch.int32, device=positions.device
    )


def _is_target_verify(forward_batch) -> bool:
    return bool(
        getattr(forward_batch.forward_mode, "is_target_verify", lambda: False)()
    )


def _target_verify_lens(
    forward_batch,
    positions: torch.Tensor,
    bs: int,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    spec_info = getattr(forward_batch, "spec_info", None)
    if spec_info is None:
        raise RuntimeError("MiniMax-M3 target_verify requires speculative metadata")
    draft_num = int(getattr(spec_info, "draft_token_num", 0) or 0)
    if draft_num <= 0:
        # Keep this robust for future SGLang variants that expose only the flat
        # token tensor shape.
        draft_num = max(1, int(positions.numel()) // max(1, bs))
    query_lens = torch.full(
        (bs,),
        draft_num,
        dtype=torch.int32,
        device=positions.device,
    )
    return seq_lens + query_lens, query_lens, draft_num


def _build_block_table(
    forward_batch,
    req_to_token_pool,
    *,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor | None,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    bs = int(forward_batch.batch_size)
    if max_seq_len is None:
        # This path is prefill/eager only. Decode graph capture must pass a static
        # max_seq_len because GPU scalar reads (`.item()`) are illegal in capture.
        max_seq_len = int(seq_lens.max().item()) if bs else 0
    max_blocks = max(1, (max_seq_len + page_size - 1) // page_size)
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    token_table = req_to_token_pool.req_to_token[
        req_pool_indices, : max_blocks * page_size
    ].clone()

    if extend_lens is not None:
        prefix_lens = seq_lens - extend_lens
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None:
            columns = torch.arange(token_table.shape[1], device=token_table.device)
            rel_pos = columns.unsqueeze(0) - prefix_lens.unsqueeze(1)
            mask = (rel_pos >= 0) & (rel_pos < extend_lens.unsqueeze(1))
            query_offsets = torch.cumsum(extend_lens, dim=0) - extend_lens
            src_idx = query_offsets.unsqueeze(1) + rel_pos
            max_src_idx = max(int(out_cache_loc.numel()) - 1, 0)
            src_idx = src_idx.clamp_min(0).clamp_max(max_src_idx).to(torch.long)
            src_values = out_cache_loc.gather(0, src_idx.reshape(-1)).view_as(src_idx)
            token_table = torch.where(mask, src_values, token_table)

    return (
        (token_table[:, : max_blocks * page_size : page_size] // page_size)
        .to(dtype=torch.int32)
        .contiguous()
    )


def build_atom_minimax_m3_attention_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    max_model_len: int,
):
    from atom.utils.forward_context import AttentionMetaData

    page_size = _page_size(token_to_kv_pool)
    bs = int(forward_batch.batch_size)
    seq_lens = _seq_lens(forward_batch, bs)
    forward_mode = forward_batch.forward_mode
    is_prefill = bool(getattr(forward_mode, "is_prefill", lambda: False)())
    is_target_verify = _is_target_verify(forward_batch)
    max_context_len = int(req_to_token_pool.req_to_token.shape[1])

    if is_target_verify:
        seq_lens, query_lens, tokens_per_req = _target_verify_lens(
            forward_batch,
            positions,
            bs,
            seq_lens,
        )
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if not torch.is_tensor(out_cache_loc):
            raise RuntimeError("MiniMax-M3 target_verify requires out_cache_loc")

        # Keep a full model-length block table so its shape matches CUDA graph
        # capture and native ATOM. In eager mode, use the live context maximum
        # for max_seqlen_k, matching native ATOM target verify.
        block_table_max_seq_len = min(max_context_len, max_model_len)
        max_seq_len = (
            block_table_max_seq_len
            if _is_stream_capturing()
            else (int(seq_lens.max().item()) if bs else 0)
        )
        block_table = _build_block_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=query_lens,
            page_size=page_size,
            max_seq_len=block_table_max_seq_len,
        )
        slot_mapping = out_cache_loc[: bs * tokens_per_req].to(dtype=torch.int64)
        sparse_md = make_sparse_decode_metadata(
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            max_seq_len=max_seq_len,
            max_query_len=tokens_per_req,
        )
        cu_q = torch.arange(
            0,
            bs * tokens_per_req + 1,
            tokens_per_req,
            dtype=torch.int32,
            device=positions.device,
        )
        metadata = AttentionMetaData(
            cu_seqlens_q=cu_q,
            max_seqlen_q=tokens_per_req,
            max_seqlen_k=max_seq_len,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            block_tables=block_table,
        )
        metadata.sparse_attention_metadata = sparse_md
        return metadata

    if is_prefill:
        extend_lens = _extend_lens(forward_batch, positions, bs)
        cu_q = torch.empty(bs + 1, dtype=torch.int32, device=positions.device)
        cu_q[0] = 0
        torch.cumsum(extend_lens, dim=0, out=cu_q[1:])
        total_tokens = int(cu_q[-1].item()) if bs else 0
        block_table = _build_block_table(
            forward_batch,
            req_to_token_pool,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            page_size=page_size,
        )
        max_query_len = int(extend_lens.max().item()) if bs else 0
        max_seq_len = int(seq_lens.max().item()) if bs else 0
        slot_mapping = getattr(forward_batch, "out_cache_loc", None)
        slot_mapping = (
            slot_mapping[:total_tokens]
            if torch.is_tensor(slot_mapping)
            else torch.empty(0, dtype=torch.int64, device=positions.device)
        )
        sparse_md = make_sparse_prefill_metadata(
            cu_seqlens_q=cu_q,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            num_prefills=bs,
            num_prefill_tokens=total_tokens,
        )
        sparse_md.prefill.qo_indptr = torch.arange(
            total_tokens + 1,
            dtype=torch.int32,
            device=positions.device,
        )
        md = AttentionMetaData(
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_q,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_seq_len,
            slot_mapping=slot_mapping,
            context_lens=seq_lens,
            block_tables=block_table,
        )
        md.sparse_attention_metadata = sparse_md
        return md

    tokens_per_req = max(1, int(positions.numel()) // max(1, bs)) if bs else 1
    # Decode CUDA graph capture cannot synchronize on seq_lens.max(). Use the
    # static req_to_token table capacity, matching the fixed-shape graph contract.
    max_seq_len = (
        max_context_len
        if _is_stream_capturing()
        else (int(seq_lens.max().item()) if bs else 0)
    )
    block_table = _build_block_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=None,
        page_size=page_size,
        max_seq_len=max_seq_len,
    )
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_cache_loc):
        slot_mapping = out_cache_loc[: bs * tokens_per_req]
    else:
        scratch = max(0, int(getattr(token_to_kv_pool, "size", 1)) - 1)
        slot_mapping = torch.full(
            (bs * tokens_per_req,),
            scratch,
            dtype=torch.int64,
            device=positions.device,
        )
    sparse_md = make_sparse_decode_metadata(
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        max_seq_len=max_seq_len,
        max_query_len=tokens_per_req,
    )
    cu_q = torch.arange(
        0,
        bs * tokens_per_req + 1,
        tokens_per_req,
        dtype=torch.int32,
        device=positions.device,
    )
    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        max_seqlen_q=tokens_per_req,
        max_seqlen_k=max_seq_len,
        slot_mapping=slot_mapping,
        context_lens=seq_lens,
        block_tables=block_table,
    )
    md.sparse_attention_metadata = sparse_md
    return md


def _iter_m3_attention_layers(model):
    from atom.models.minimax_m3 import MiniMaxM3Attention, MiniMaxM3SparseAttention

    for module in model.modules():
        if isinstance(module, (MiniMaxM3Attention, MiniMaxM3SparseAttention)):
            attn = getattr(module, "attn", None)
            impl = getattr(attn, "impl", None)
            if impl is not None:
                yield module, attn, impl, isinstance(module, MiniMaxM3SparseAttention)


def _get_index_cache_view(
    token_to_kv_pool,
    layer_id: int,
    *,
    k_buffer: torch.Tensor,
    index_dim: int,
) -> torch.Tensor:
    """Return SGLang MiniMaxSparseKVPool's existing index-K cache view."""

    if not hasattr(token_to_kv_pool, "get_index_k_buffer"):
        raise RuntimeError(
            "MiniMax-M3 native sparse attention requires SGLang MiniMaxSparseKVPool "
            "with get_index_k_buffer(); refusing to allocate a side index cache."
        )
    page_size = _page_size(token_to_kv_pool)
    num_slots = int(k_buffer.shape[0])
    num_blocks = max(1, num_slots // page_size)
    expected = (num_blocks, page_size, int(index_dim))
    index_buffer = token_to_kv_pool.get_index_k_buffer(layer_id)
    return index_buffer[: num_blocks * page_size].view(*expected)


def bind_minimax_m3_sparse_cache_views(model, token_to_kv_pool) -> bool:
    if token_to_kv_pool is None or not hasattr(token_to_kv_pool, "get_kv_buffer"):
        return False

    from atom.config import KVCacheTensor
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    page_size = _page_size(token_to_kv_pool)
    kv_cache_data = {}
    bound = False
    for _, attn, impl, is_sparse_layer in _iter_m3_attention_layers(model):
        layer_id = int(impl.layer_num)
        k_buffer, v_buffer = token_to_kv_pool.get_kv_buffer(layer_id)
        num_slots, num_kv_heads, head_dim = k_buffer.shape
        num_blocks = max(1, int(num_slots) // page_size)
        live_slots = num_blocks * page_size
        x = 16 // k_buffer.element_size()
        k_cache = k_buffer[:live_slots].view(
            num_blocks, num_kv_heads, head_dim // x, page_size, x
        )
        v_cache = v_buffer[:live_slots].view(
            num_blocks, num_kv_heads, page_size // x, head_dim, x
        )
        if hasattr(token_to_kv_pool, "get_kv_scale_buffer"):
            k_scale, v_scale = token_to_kv_pool.get_kv_scale_buffer(layer_id)
        else:
            k_scale = None
            v_scale = None
        if not is_sparse_layer:
            pass
        elif getattr(impl, "skip_index_topk", False):
            # Shared-index sparse layers reuse the previous full-index layer's
            # top-k result and never write/read their own index-K cache.
            impl.index_cache = None
        else:
            impl.index_cache = _get_index_cache_view(
                token_to_kv_pool,
                layer_id,
                k_buffer=k_buffer,
                index_dim=impl.index_head_dim,
            )
        impl.max_model_len = int(getattr(token_to_kv_pool, "size", live_slots))
        if is_sparse_layer:
            impl.index_topk_cache_state = getattr(
                token_to_kv_pool, "_atom_minimax_m3_topk_cache_state", None
            )
            if impl.index_topk_cache_state is None:
                impl.index_topk_cache_state = {}
                setattr(
                    token_to_kv_pool,
                    "_atom_minimax_m3_topk_cache_state",
                    impl.index_topk_cache_state,
                )
        kv_cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        attn.k_cache = k_cache
        attn.v_cache = v_cache
        attn.k_scale = k_scale
        attn.v_scale = v_scale
        bound = True

    if not bound:
        return False

    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data
    return True
